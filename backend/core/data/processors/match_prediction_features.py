"""
Match prediction feature engineering for NBA 2K25 esports.
Creates features for predicting individual player scores in upcoming matches.
"""

import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent.parent.parent
sys.path.append(str(backend_dir))

from utils.logging import log_execution_time, log_exceptions
from config.logging_config import get_data_fetcher_logger
from config.settings import OUTPUT_DIR

logger = get_data_fetcher_logger()


class MatchPredictionFeatureEngineer:
    """
    Feature engineering for match prediction models.
    Creates comprehensive features for predicting individual player scores.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.output_dir = Path(OUTPUT_DIR)
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def create_training_dataset(self, matches: List[Dict], player_stats: Dict) -> pd.DataFrame:
        """
        Create training dataset from historical matches.
        
        Args:
            matches: List of historical match data
            player_stats: Player statistics dictionary
            
        Returns:
            DataFrame with features and target variables
        """
        logger.info(f"Creating training dataset from {len(matches)} matches")
        
        training_data = []
        
        for match in matches:
            # Skip matches without scores
            if 'homeScore' not in match or 'awayScore' not in match:
                continue
                
            home_player_id = str(match['homePlayer']['id'])
            away_player_id = str(match['awayPlayer']['id'])
            
            # Skip if we don't have stats for either player
            if home_player_id not in player_stats or away_player_id not in player_stats:
                continue
            
            # Create features for this match
            features = self._create_match_features(
                home_player_id, away_player_id, player_stats, match
            )
            
            # Add target variables
            features.update({
                'home_score': match['homeScore'],
                'away_score': match['awayScore'],
                'home_win': 1 if match['homeScore'] > match['awayScore'] else 0,
                'total_score': match['homeScore'] + match['awayScore']
            })
            
            training_data.append(features)
        
        df = pd.DataFrame(training_data)
        logger.info(f"Created training dataset with {len(df)} samples and {len(df.columns)} features")
        
        return df
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def create_prediction_features(self, upcoming_matches: List[Dict], player_stats: Dict) -> pd.DataFrame:
        """
        Create features for predicting upcoming matches.
        
        Args:
            upcoming_matches: List of upcoming match data
            player_stats: Player statistics dictionary
            
        Returns:
            DataFrame with features for prediction
        """
        logger.info(f"Creating prediction features for {len(upcoming_matches)} upcoming matches")
        
        prediction_data = []
        
        for match in upcoming_matches:
            home_player_id = str(match['homePlayer']['id'])
            away_player_id = str(match['awayPlayer']['id'])
            
            # Skip if we don't have stats for either player
            if home_player_id not in player_stats or away_player_id not in player_stats:
                logger.warning(f"Missing stats for match {match.get('id', 'unknown')}")
                continue
            
            # Create features for this match
            features = self._create_match_features(
                home_player_id, away_player_id, player_stats, match
            )
            
            # Add match metadata for tracking
            features.update({
                'match_id': match.get('id', match.get('fixtureId')),
                'home_player_id': home_player_id,
                'away_player_id': away_player_id,
                'home_player_name': match['homePlayer']['name'],
                'away_player_name': match['awayPlayer']['name'],
                'fixture_start': match.get('fixtureStart')
            })
            
            prediction_data.append(features)
        
        df = pd.DataFrame(prediction_data)
        logger.info(f"Created prediction dataset with {len(df)} samples and {len(df.columns)} features")
        
        return df
    
    def _create_match_features(self, home_player_id: str, away_player_id: str, 
                              player_stats: Dict, match: Dict) -> Dict:
        """
        Create comprehensive features for a single match.
        
        Args:
            home_player_id: ID of home player
            away_player_id: ID of away player
            player_stats: Player statistics dictionary
            match: Match data dictionary
            
        Returns:
            Dictionary of features
        """
        home_stats = player_stats[home_player_id]
        away_stats = player_stats[away_player_id]
        
        features = {}
        
        # === INDIVIDUAL PLAYER FEATURES ===
        
        # Home player features
        features.update(self._get_individual_player_features(home_stats, 'home'))
        
        # Away player features
        features.update(self._get_individual_player_features(away_stats, 'away'))
        
        # === HEAD-TO-HEAD FEATURES ===
        features.update(self._get_head_to_head_features(home_stats, away_stats, home_player_id, away_player_id))
        
        # === RELATIVE FEATURES ===
        features.update(self._get_relative_features(home_stats, away_stats))
        
        # === TEAM FEATURES ===
        features.update(self._get_team_features(home_stats, away_stats, match))
        
        return features
    
    def _get_individual_player_features(self, player_stats: Dict, prefix: str) -> Dict:
        """Get individual player performance features."""
        features = {}
        
        # Basic performance metrics
        features[f'{prefix}_avg_score'] = player_stats.get('avg_score', 0)
        features[f'{prefix}_win_rate'] = player_stats.get('win_rate', 0)
        features[f'{prefix}_total_matches'] = player_stats.get('total_matches', 0)
        
        # Recent form (last 5 matches)
        features[f'{prefix}_recent_win_rate'] = player_stats.get('recent_win_rate', 0)
        features[f'{prefix}_recent_avg_score'] = player_stats.get('recent_avg_score', 0)
        features[f'{prefix}_momentum'] = player_stats.get('momentum', 0)
        
        # Consistency metrics
        features[f'{prefix}_score_variance'] = player_stats.get('score_variance', 0)
        features[f'{prefix}_score_std'] = player_stats.get('score_std', 0)
        
        # Experience features
        features[f'{prefix}_matches_played'] = len(player_stats.get('match_history', []))
        
        # Last match performance
        last_5_matches = player_stats.get('last_5_matches', [])
        if last_5_matches:
            last_match = last_5_matches[-1]
            features[f'{prefix}_last_match_score'] = last_match.get('score', 0)
            features[f'{prefix}_last_match_win'] = 1 if last_match.get('win', False) else 0
        else:
            features[f'{prefix}_last_match_score'] = 0
            features[f'{prefix}_last_match_win'] = 0
        
        return features
    
    def _get_head_to_head_features(self, home_stats: Dict, away_stats: Dict, 
                                   home_player_id: str, away_player_id: str) -> Dict:
        """Get head-to-head performance features."""
        features = {}
        
        # Home player vs away player
        home_vs_away = home_stats.get('opponents_faced', {}).get(away_player_id, {})
        features['h2h_home_matches'] = home_vs_away.get('matches', 0)
        features['h2h_home_wins'] = home_vs_away.get('wins', 0)
        features['h2h_home_win_rate'] = home_vs_away.get('win_rate', 0)
        features['h2h_home_avg_score'] = home_vs_away.get('avg_score', 0)
        features['h2h_home_avg_score_against'] = home_vs_away.get('avg_score_against', 0)
        
        # Away player vs home player
        away_vs_home = away_stats.get('opponents_faced', {}).get(home_player_id, {})
        features['h2h_away_matches'] = away_vs_home.get('matches', 0)
        features['h2h_away_wins'] = away_vs_home.get('wins', 0)
        features['h2h_away_win_rate'] = away_vs_home.get('win_rate', 0)
        features['h2h_away_avg_score'] = away_vs_home.get('avg_score', 0)
        features['h2h_away_avg_score_against'] = away_vs_home.get('avg_score_against', 0)
        
        # H2H summary
        total_h2h_matches = features['h2h_home_matches']
        features['h2h_total_matches'] = total_h2h_matches
        features['h2h_have_played'] = 1 if total_h2h_matches > 0 else 0
        
        return features
    
    def _get_relative_features(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """Get relative performance features between players."""
        features = {}
        
        # Score differences
        home_avg = home_stats.get('avg_score', 0)
        away_avg = away_stats.get('avg_score', 0)
        features['score_diff_home_advantage'] = home_avg - away_avg
        
        # Win rate differences
        home_wr = home_stats.get('win_rate', 0)
        away_wr = away_stats.get('win_rate', 0)
        features['win_rate_diff_home_advantage'] = home_wr - away_wr
        
        # Recent form differences
        home_recent = home_stats.get('recent_avg_score', 0)
        away_recent = away_stats.get('recent_avg_score', 0)
        features['recent_score_diff_home_advantage'] = home_recent - away_recent
        
        # Momentum differences
        home_momentum = home_stats.get('momentum', 0)
        away_momentum = away_stats.get('momentum', 0)
        features['momentum_diff_home_advantage'] = home_momentum - away_momentum
        
        # Experience differences
        home_matches = home_stats.get('total_matches', 0)
        away_matches = away_stats.get('total_matches', 0)
        features['experience_diff_home_advantage'] = home_matches - away_matches
        
        # Consistency differences (lower variance = more consistent)
        home_std = home_stats.get('score_std', 0)
        away_std = away_stats.get('score_std', 0)
        features['consistency_diff_home_advantage'] = away_std - home_std  # Higher = home more consistent
        
        return features
    
    def _get_team_features(self, home_stats: Dict, away_stats: Dict, match: Dict) -> Dict:
        """Get team-related features."""
        features = {}
        
        # Extract team information
        home_team_id = str(match.get('homeTeam', {}).get('id', ''))
        away_team_id = str(match.get('awayTeam', {}).get('id', ''))
        
        # Home player's performance with their team
        home_teams = home_stats.get('teams_used', {})
        if home_team_id in home_teams:
            team_stats = home_teams[home_team_id]
            features['home_team_matches'] = team_stats.get('matches', 0)
            features['home_team_win_rate'] = team_stats.get('win_rate', 0)
            features['home_team_avg_score'] = team_stats.get('avg_score', 0)
        else:
            features['home_team_matches'] = 0
            features['home_team_win_rate'] = 0
            features['home_team_avg_score'] = 0
        
        # Away player's performance with their team
        away_teams = away_stats.get('teams_used', {})
        if away_team_id in away_teams:
            team_stats = away_teams[away_team_id]
            features['away_team_matches'] = team_stats.get('matches', 0)
            features['away_team_win_rate'] = team_stats.get('win_rate', 0)
            features['away_team_avg_score'] = team_stats.get('avg_score', 0)
        else:
            features['away_team_matches'] = 0
            features['away_team_win_rate'] = 0
            features['away_team_avg_score'] = 0
        
        # Team experience features
        features['home_team_familiarity'] = 1 if features['home_team_matches'] > 0 else 0
        features['away_team_familiarity'] = 1 if features['away_team_matches'] > 0 else 0
        
        return features
    
    @log_exceptions(logger)
    def save_features(self, df: pd.DataFrame, filename: str):
        """Save features to file."""
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved features to {filepath}")
    
    @log_exceptions(logger)
    def load_features(self, filename: str) -> pd.DataFrame:
        """Load features from file."""
        filepath = self.output_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            logger.info(f"Loaded features from {filepath}")
            return df
        else:
            logger.warning(f"Features file {filepath} does not exist")
            return pd.DataFrame()
