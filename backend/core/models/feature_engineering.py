"""
Feature engineering module for prediction models.
"""

import numpy as np
from datetime import datetime
from collections import defaultdict

from config.logging_config import get_model_tuning_logger
from utils.logging import log_execution_time, log_exceptions

logger = get_model_tuning_logger()


class FeatureEngineer:
    """
    Enhanced feature engineering for NBA 2K25 eSports match prediction.
    """

    def __init__(self, feature_config=None):
        """
        Initialize the feature engineer.
        
        Args:
            feature_config (dict): Feature configuration dictionary
        """
        # Default configuration
        default_config = {
            "use_basic_features": True,
            "use_team_features": True,
            "use_h2h_features": True,
            "use_recent_form": True,
            "use_advanced_features": True,
            "use_temporal_features": True,
            "use_momentum_features": True,  # New
            "use_streak_features": True,   # New
            "use_efficiency_features": True,  # New
            "recent_matches_window": 5,
            "momentum_window": 5,  # New
            "streak_window": 5     # New
        }
        
        # Merge provided config with defaults
        self.feature_config = default_config.copy()
        if feature_config:
            self.feature_config.update(feature_config)
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def extract_features(self, player_stats, matches, for_score_prediction=True):
        """
        Extract features from match data.
        
        Args:
            player_stats (dict): Player statistics dictionary
            matches (list): List of match data dictionaries
            for_score_prediction (bool): Whether to extract features for score prediction
            
        Returns:
            tuple: Features and labels
        """
        features = []
        
        if for_score_prediction:
            home_scores = []
            away_scores = []
        else:
            labels = []
        
        # Process each match
        for match in matches:
            # Skip matches without scores
            if 'homeScore' not in match or 'awayScore' not in match:
                continue
            
            # Extract player and team IDs
            home_player_id = str(match['homePlayer']['id'])
            away_player_id = str(match['awayPlayer']['id'])
            
            # Skip if player stats not available
            if home_player_id not in player_stats or away_player_id not in player_stats:
                continue
            
            home_player = player_stats[home_player_id]
            away_player = player_stats[away_player_id]
            
            home_team_id = str(match['homeTeam']['id'])
            away_team_id = str(match['awayTeam']['id'])
            
            # Get match date
            match_date = self._parse_match_date(match)
            
            # Get previous matches before this one
            prev_matches = self._get_previous_matches(matches, match, match_date)
            
            # Extract features
            match_features = []
            
            # Basic player features
            if self.feature_config["use_basic_features"]:
                basic_features = self._extract_basic_features(home_player, away_player)
                match_features.extend(basic_features)
            
            # Team-specific features
            if self.feature_config["use_team_features"]:
                team_features = self._extract_team_features(
                    home_player, away_player, home_team_id, away_team_id
                )
                match_features.extend(team_features)
            
            # Head-to-head features
            if self.feature_config["use_h2h_features"]:
                h2h_features = self._extract_h2h_features(
                    home_player, away_player, home_player_id, away_player_id
                )
                match_features.extend(h2h_features)
            
            # Recent form features
            if self.feature_config["use_recent_form"]:
                recent_form_features = self._extract_recent_form_features(
                    home_player_id, away_player_id, prev_matches, 
                    self.feature_config["recent_matches_window"]
                )
                match_features.extend(recent_form_features)
            
            # Advanced features
            if self.feature_config["use_advanced_features"]:
                advanced_features = self._extract_advanced_features(
                    home_player, away_player, home_team_id, away_team_id,
                    home_player_id, away_player_id, prev_matches
                )
                match_features.extend(advanced_features)
            
            # Temporal features
            if self.feature_config["use_temporal_features"]:
                temporal_features = self._extract_temporal_features(match_date, prev_matches)
                match_features.extend(temporal_features)
            
            # NEW: Momentum features
            if self.feature_config["use_momentum_features"]:
                momentum_features = self._extract_momentum_features(
                    home_player_id, away_player_id, prev_matches,
                    self.feature_config["momentum_window"]
                )
                match_features.extend(momentum_features)
            
            # NEW: Streak features
            if self.feature_config["use_streak_features"]:
                streak_features = self._extract_streak_features(
                    home_player_id, away_player_id, prev_matches,
                    self.feature_config["streak_window"]
                )
                match_features.extend(streak_features)
            
            # NEW: Efficiency features
            if self.feature_config["use_efficiency_features"]:
                efficiency_features = self._extract_efficiency_features(
                    home_player, away_player, home_player_id, away_player_id, prev_matches
                )
                match_features.extend(efficiency_features)
            
            features.append(match_features)
            
            # Extract labels
            if for_score_prediction:
                home_scores.append(match['homeScore'])
                away_scores.append(match['awayScore'])
            else:
                # 1 if home win, 0 if away win
                home_score = match['homeScore']
                away_score = match['awayScore']
                label = 1 if home_score > away_score else 0
                labels.append(label)
        
        if for_score_prediction:
            return np.array(features), np.array(home_scores), np.array(away_scores)
        else:
            return np.array(features), np.array(labels)
    
    @log_exceptions(logger)
    def _parse_match_date(self, match):
        """
        Parse match date from match data.
        
        Args:
            match (dict): Match data dictionary
            
        Returns:
            datetime: Match date
        """
        try:
            # Try to parse date from match data
            if 'date' in match:
                return datetime.strptime(match['date'], "%Y-%m-%d")
            elif 'startTime' in match:
                return datetime.strptime(match['startTime'].split('T')[0], "%Y-%m-%d")
            else:
                # Default to current date if not available
                return datetime.now()
        except (ValueError, TypeError):
            # Default to current date if parsing fails
            return datetime.now()
    
    @log_exceptions(logger)
    def _get_previous_matches(self, all_matches, current_match, current_date):
        """
        Get previous matches before the current match.
        
        Args:
            all_matches (list): List of all match data dictionaries
            current_match (dict): Current match data dictionary
            current_date (datetime): Current match date
            
        Returns:
            list: List of previous match data dictionaries
        """
        prev_matches = []
        
        for match in all_matches:
            # Skip matches without scores
            if 'homeScore' not in match or 'awayScore' not in match:
                continue
            
            # Skip the current match
            if match == current_match:
                continue
            
            # Get match date
            match_date = self._parse_match_date(match)
            
            # Only include matches before the current match
            if match_date < current_date:
                prev_matches.append(match)
        
        return prev_matches
    
    @log_exceptions(logger)
    def _extract_basic_features(self, home_player, away_player):
        """
        Extract basic player features.
        
        Args:
            home_player (dict): Home player statistics dictionary
            away_player (dict): Away player statistics dictionary
            
        Returns:
            list: Basic player features
        """
        return [
            # Player overall stats
            home_player.get('win_rate', 0),
            away_player.get('win_rate', 0),
            home_player.get('avg_score', 0),
            away_player.get('avg_score', 0),
            home_player.get('total_matches', 0),
            away_player.get('total_matches', 0)
        ]
    
    @log_exceptions(logger)
    def _extract_team_features(self, home_player, away_player, home_team_id, away_team_id):
        """
        Extract team-specific features.
        
        Args:
            home_player (dict): Home player statistics dictionary
            away_player (dict): Away player statistics dictionary
            home_team_id (str): Home team ID
            away_team_id (str): Away team ID
            
        Returns:
            list: Team-specific features
        """
        # Get team stats
        home_team_win_rate = self._get_team_win_rate(home_player, home_team_id)
        away_team_win_rate = self._get_team_win_rate(away_player, away_team_id)
        home_team_avg_score = self._get_team_avg_score(home_player, home_team_id)
        away_team_avg_score = self._get_team_avg_score(away_player, away_team_id)
        home_team_matches = self._get_team_matches(home_player, home_team_id)
        away_team_matches = self._get_team_matches(away_player, away_team_id)
        
        # Calculate team experience ratio (how often the player uses this team)
        home_team_exp_ratio = home_team_matches / max(home_player.get('total_matches', 1), 1)
        away_team_exp_ratio = away_team_matches / max(away_player.get('total_matches', 1), 1)
        
        # Calculate team performance relative to overall performance
        home_team_rel_win_rate = home_team_win_rate - home_player.get('win_rate', 0)
        away_team_rel_win_rate = away_team_win_rate - away_player.get('win_rate', 0)
        home_team_rel_avg_score = home_team_avg_score - home_player.get('avg_score', 0)
        away_team_rel_avg_score = away_team_avg_score - away_player.get('avg_score', 0)
        
        return [
            # Team-specific stats
            home_team_win_rate,
            away_team_win_rate,
            home_team_avg_score,
            away_team_avg_score,
            home_team_matches,
            away_team_matches,
            
            # Team experience ratio
            home_team_exp_ratio,
            away_team_exp_ratio,
            
            # Team performance relative to overall
            home_team_rel_win_rate,
            away_team_rel_win_rate,
            home_team_rel_avg_score,
            away_team_rel_avg_score
        ]
    
    @log_exceptions(logger)
    def _extract_h2h_features(self, home_player, away_player, home_player_id, away_player_id):
        """
        Extract head-to-head features.
        
        Args:
            home_player (dict): Home player statistics dictionary
            away_player (dict): Away player statistics dictionary
            home_player_id (str): Home player ID
            away_player_id (str): Away player ID
            
        Returns:
            list: Head-to-head features
        """
        # Get head-to-head stats
        home_h2h_win_rate = self._get_h2h_win_rate(home_player, away_player_id)
        away_h2h_win_rate = self._get_h2h_win_rate(away_player, home_player_id)
        home_h2h_matches = self._get_h2h_matches(home_player, away_player_id)
        away_h2h_matches = self._get_h2h_matches(away_player, home_player_id)
        
        # Calculate head-to-head score stats
        home_h2h_avg_score = self._get_avg_score_against(home_player, away_player_id)
        away_h2h_avg_score = self._get_avg_score_against(away_player, home_player_id)
        
        # Calculate head-to-head win rate difference
        h2h_win_rate_diff = home_h2h_win_rate - away_h2h_win_rate
        
        # Calculate head-to-head score difference
        h2h_score_diff = home_h2h_avg_score - away_h2h_avg_score
        
        return [
            # Head-to-head stats
            home_h2h_win_rate,
            away_h2h_win_rate,
            home_h2h_matches,
            away_h2h_matches,
            home_h2h_avg_score,
            away_h2h_avg_score,
            h2h_win_rate_diff,
            h2h_score_diff
        ]
    
    @log_exceptions(logger)
    def _extract_recent_form_features(self, home_player_id, away_player_id, prev_matches, window_size=5):
        """
        Extract recent form features.
        
        Args:
            home_player_id (str): Home player ID
            away_player_id (str): Away player ID
            prev_matches (list): List of previous match data dictionaries
            window_size (int): Number of recent matches to consider
            
        Returns:
            list: Recent form features
        """
        # Get recent matches for each player
        home_recent_matches = self._get_player_recent_matches(home_player_id, prev_matches, window_size)
        away_recent_matches = self._get_player_recent_matches(away_player_id, prev_matches, window_size)
        
        # Calculate recent win rates
        home_recent_win_rate = self._calculate_recent_win_rate(home_player_id, home_recent_matches)
        away_recent_win_rate = self._calculate_recent_win_rate(away_player_id, away_recent_matches)
        
        # Calculate recent average scores
        home_recent_avg_score = self._calculate_recent_avg_score(home_player_id, home_recent_matches)
        away_recent_avg_score = self._calculate_recent_avg_score(away_player_id, away_recent_matches)
        
        # Calculate recent score variance (consistency)
        home_recent_score_var = self._calculate_recent_score_variance(home_player_id, home_recent_matches)
        away_recent_score_var = self._calculate_recent_score_variance(away_player_id, away_recent_matches)
        
        # Calculate momentum (trend in recent performance)
        home_momentum = self._calculate_momentum(home_player_id, home_recent_matches)
        away_momentum = self._calculate_momentum(away_player_id, away_recent_matches)
        
        return [
            # Recent form stats
            home_recent_win_rate,
            away_recent_win_rate,
            home_recent_avg_score,
            away_recent_avg_score,
            home_recent_score_var,
            away_recent_score_var,
            home_momentum,
            away_momentum
        ]
    
    @log_exceptions(logger)
    def _extract_advanced_features(self, home_player, away_player, home_team_id, away_team_id, 
                                  home_player_id, away_player_id, prev_matches):
        """
        Extract advanced features.
        
        Args:
            home_player (dict): Home player statistics dictionary
            away_player (dict): Away player statistics dictionary
            home_team_id (str): Home team ID
            away_team_id (str): Away team ID
            home_player_id (str): Home player ID
            away_player_id (str): Away player ID
            prev_matches (list): List of previous match data dictionaries
            
        Returns:
            list: Advanced features
        """
        # Calculate win rate difference
        win_rate_diff = home_player.get('win_rate', 0) - away_player.get('win_rate', 0)
        
        # Calculate average score difference
        avg_score_diff = home_player.get('avg_score', 0) - away_player.get('avg_score', 0)
        
        # Calculate experience difference (total matches)
        exp_diff = home_player.get('total_matches', 0) - away_player.get('total_matches', 0)
        
        # Calculate team win rate difference
        team_win_rate_diff = (self._get_team_win_rate(home_player, home_team_id) - 
                             self._get_team_win_rate(away_player, away_team_id))
        
        # Calculate team average score difference
        team_avg_score_diff = (self._get_team_avg_score(home_player, home_team_id) - 
                              self._get_team_avg_score(away_player, away_team_id))
        
        # Calculate team experience difference
        team_exp_diff = (self._get_team_matches(home_player, home_team_id) - 
                        self._get_team_matches(away_player, away_team_id))
        
        # Calculate home court advantage
        home_advantage = self._calculate_home_advantage(prev_matches)
        
        # Calculate player consistency (variance in scores)
        home_consistency = self._calculate_player_consistency(home_player_id, prev_matches)
        away_consistency = self._calculate_player_consistency(away_player_id, prev_matches)
        
        return [
            # Advanced stats
            win_rate_diff,
            avg_score_diff,
            exp_diff,
            team_win_rate_diff,
            team_avg_score_diff,
            team_exp_diff,
            home_advantage,
            home_consistency,
            away_consistency
        ]
    
    @log_exceptions(logger)
    def _extract_temporal_features(self, match_date, prev_matches):
        """
        Extract temporal features.
        
        Args:
            match_date (datetime): Match date
            prev_matches (list): List of previous match data dictionaries
            
        Returns:
            list: Temporal features
        """
        # Calculate day of week (0-6, Monday is 0)
        day_of_week = match_date.weekday()
        
        # Calculate month of year (1-12)
        month = match_date.month
        
        # Calculate weekend indicator (1 if weekend, 0 otherwise)
        is_weekend = 1 if day_of_week >= 5 else 0
        
        return [
            # Temporal features
            day_of_week / 6,  # Normalize to [0, 1]
            month / 12,  # Normalize to [0, 1]
            is_weekend
        ]
    
    @log_exceptions(logger)
    def _get_team_win_rate(self, player, team_id):
        """
        Get a player's win rate with a specific team.
        
        Args:
            player (dict): Player statistics dictionary
            team_id (str): Team ID
            
        Returns:
            float: Win rate
        """
        teams_used = player.get('teams_used', {})
        if team_id in teams_used:
            return teams_used[team_id].get('win_rate', 0)
        return 0
    
    @log_exceptions(logger)
    def _get_team_avg_score(self, player, team_id):
        """
        Get a player's average score with a specific team.
        
        Args:
            player (dict): Player statistics dictionary
            team_id (str): Team ID
            
        Returns:
            float: Average score
        """
        teams_used = player.get('teams_used', {})
        if team_id in teams_used:
            return teams_used[team_id].get('avg_score', 0)
        return 0
    
    @log_exceptions(logger)
    def _get_team_matches(self, player, team_id):
        """
        Get a player's number of matches with a specific team.
        
        Args:
            player (dict): Player statistics dictionary
            team_id (str): Team ID
            
        Returns:
            int: Number of matches
        """
        teams_used = player.get('teams_used', {})
        if team_id in teams_used:
            return teams_used[team_id].get('matches', 0)
        return 0
    
    @log_exceptions(logger)
    def _get_h2h_win_rate(self, player, opponent_id):
        """
        Get a player's win rate against a specific opponent.
        
        Args:
            player (dict): Player statistics dictionary
            opponent_id (str): Opponent player ID
            
        Returns:
            float: Win rate
        """
        opponents_faced = player.get('opponents_faced', {})
        if opponent_id in opponents_faced:
            return opponents_faced[opponent_id].get('win_rate', 0)
        return 0
    
    @log_exceptions(logger)
    def _get_h2h_matches(self, player, opponent_id):
        """
        Get a player's number of matches against a specific opponent.
        
        Args:
            player (dict): Player statistics dictionary
            opponent_id (str): Opponent player ID
            
        Returns:
            int: Number of matches
        """
        opponents_faced = player.get('opponents_faced', {})
        if opponent_id in opponents_faced:
            return opponents_faced[opponent_id].get('matches', 0)
        return 0
    
    @log_exceptions(logger)
    def _get_avg_score_against(self, player, opponent_id):
        """
        Get a player's average score against a specific opponent.
        
        Args:
            player (dict): Player statistics dictionary
            opponent_id (str): Opponent player ID
            
        Returns:
            float: Average score
        """
        opponents_faced = player.get('opponents_faced', {})
        if opponent_id in opponents_faced:
            total_score = opponents_faced[opponent_id].get('total_score', 0)
            matches = opponents_faced[opponent_id].get('matches', 0)
            if matches > 0:
                return total_score / matches
        return 0
    
    @log_exceptions(logger)
    def _get_player_recent_matches(self, player_id, matches, window_size=5):
        """
        Get a player's recent matches.
        
        Args:
            player_id (str): Player ID
            matches (list): List of match data dictionaries
            window_size (int): Number of recent matches to consider
            
        Returns:
            list: List of recent match data dictionaries
        """
        player_matches = []
        
        for match in matches:
            home_player_id = str(match['homePlayer']['id'])
            away_player_id = str(match['awayPlayer']['id'])
            
            if home_player_id == player_id or away_player_id == player_id:
                player_matches.append(match)
        
        # Sort by date (most recent first)
        player_matches.sort(key=lambda m: self._parse_match_date(m), reverse=True)
        
        # Return the most recent matches
        return player_matches[:window_size]
    
    @log_exceptions(logger)
    def _calculate_recent_win_rate(self, player_id, recent_matches):
        """
        Calculate a player's recent win rate.
        
        Args:
            player_id (str): Player ID
            recent_matches (list): List of recent match data dictionaries
            
        Returns:
            float: Recent win rate
        """
        if not recent_matches:
            return 0
        
        wins = 0
        
        for match in recent_matches:
            home_player_id = str(match['homePlayer']['id'])
            away_player_id = str(match['awayPlayer']['id'])
            
            home_score = match['homeScore']
            away_score = match['awayScore']
            
            if home_player_id == player_id and home_score > away_score:
                wins += 1
            elif away_player_id == player_id and away_score > home_score:
                wins += 1
        
        return wins / len(recent_matches)
    
    @log_exceptions(logger)
    def _calculate_recent_avg_score(self, player_id, recent_matches):
        """
        Calculate a player's recent average score.
        
        Args:
            player_id (str): Player ID
            recent_matches (list): List of recent match data dictionaries
            
        Returns:
            float: Recent average score
        """
        if not recent_matches:
            return 0
        
        total_score = 0
        
        for match in recent_matches:
            home_player_id = str(match['homePlayer']['id'])
            away_player_id = str(match['awayPlayer']['id'])
            
            if home_player_id == player_id:
                total_score += match['homeScore']
            elif away_player_id == player_id:
                total_score += match['awayScore']
        
        return total_score / len(recent_matches)
    
    @log_exceptions(logger)
    def _calculate_recent_score_variance(self, player_id, recent_matches):
        """
        Calculate a player's recent score variance.
        
        Args:
            player_id (str): Player ID
            recent_matches (list): List of recent match data dictionaries
            
        Returns:
            float: Recent score variance
        """
        if not recent_matches or len(recent_matches) < 2:
            return 0
        
        scores = []
        
        for match in recent_matches:
            home_player_id = str(match['homePlayer']['id'])
            away_player_id = str(match['awayPlayer']['id'])
            
            if home_player_id == player_id:
                scores.append(match['homeScore'])
            elif away_player_id == player_id:
                scores.append(match['awayScore'])
        
        if not scores:
            return 0
        
        return np.var(scores)
    
    @log_exceptions(logger)
    def _calculate_momentum(self, player_id, recent_matches):
        """
        Calculate a player's momentum (trend in recent performance).
        
        Args:
            player_id (str): Player ID
            recent_matches (list): List of recent match data dictionaries
            
        Returns:
            float: Momentum
        """
        if not recent_matches or len(recent_matches) < 2:
            return 0
        
        # Calculate weighted win rate (more recent matches have higher weight)
        total_weight = 0
        weighted_wins = 0
        
        for i, match in enumerate(recent_matches):
            home_player_id = str(match['homePlayer']['id'])
            away_player_id = str(match['awayPlayer']['id'])
            
            home_score = match['homeScore']
            away_score = match['awayScore']
            
            # Weight is inversely proportional to recency (most recent has highest weight)
            weight = len(recent_matches) - i
            total_weight += weight
            
            if home_player_id == player_id and home_score > away_score:
                weighted_wins += weight
            elif away_player_id == player_id and away_score > home_score:
                weighted_wins += weight
        
        if total_weight == 0:
            return 0
        
        weighted_win_rate = weighted_wins / total_weight
        
        # Calculate unweighted win rate
        unweighted_win_rate = self._calculate_recent_win_rate(player_id, recent_matches)
        
        # Momentum is the difference between weighted and unweighted win rates
        return weighted_win_rate - unweighted_win_rate
    
    @log_exceptions(logger)
    def _calculate_home_advantage(self, matches):
        """
        Calculate home court advantage.
        
        Args:
            matches (list): List of match data dictionaries
            
        Returns:
            float: Home court advantage
        """
        if not matches:
            return 0
        
        home_wins = 0
        
        for match in matches:
            home_score = match['homeScore']
            away_score = match['awayScore']
            
            if home_score > away_score:
                home_wins += 1
        
        return home_wins / len(matches)
    
    @log_exceptions(logger)
    def _calculate_player_consistency(self, player_id, matches):
        """
        Calculate a player's consistency (inverse of score variance).
        
        Args:
            player_id (str): Player ID
            matches (list): List of match data dictionaries
            
        Returns:
            float: Consistency
        """
        player_matches = []
        
        for match in matches:
            home_player_id = str(match['homePlayer']['id'])
            away_player_id = str(match['awayPlayer']['id'])
            
            if home_player_id == player_id or away_player_id == player_id:
                player_matches.append(match)
        
        if not player_matches or len(player_matches) < 2:
            return 0
        
        scores = []
        
        for match in player_matches:
            home_player_id = str(match['homePlayer']['id'])
            away_player_id = str(match['awayPlayer']['id'])
            
            if home_player_id == player_id:
                scores.append(match['homeScore'])
            elif away_player_id == player_id:
                scores.append(match['awayScore'])
        
        if not scores:
            return 0
        
        # Consistency is the inverse of variance (normalized to [0, 1])
        variance = np.var(scores)
        if variance == 0:
            return 1  # Perfect consistency
        
        # Normalize using a reasonable maximum variance (e.g., 100)
        max_variance = 100
        normalized_variance = min(variance / max_variance, 1)
        
        return 1 - normalized_variance

    def _get_default_config(self):
        """
        Get default feature configuration.
    
        Returns:
            dict: Default feature configuration
        """
        return {
            "use_basic_features": True,
            "use_team_features": True,
            "use_h2h_features": True,
            "use_recent_form": True,
            "use_advanced_features": True,
            "use_temporal_features": True,
            "use_momentum_features": True,  # New
            "use_streak_features": True,   # New
            "use_efficiency_features": True,  # New
            "recent_matches_window": 5,
            "momentum_window": 10,  # New
            "streak_window": 15     # New
        }
    
    @log_exceptions(logger)
    def _calculate_score_trend(self, player_id, matches):
        """
        Calculate weighted score trend for momentum.
        
        Args:
            player_id (str): Player ID
            matches (list): List of recent matches
            
        Returns:
            float: Score trend (positive = improving, negative = declining)
        """
        if len(matches) < 2:
            return 0.0
        
        scores = []
        weights = []
        
        for i, match in enumerate(matches[-10:]):  # Last 10 matches
            if str(match['homePlayer']['id']) == player_id:
                score = match.get('homeScore', 0)
            elif str(match['awayPlayer']['id']) == player_id:
                score = match.get('awayScore', 0)
            else:
                continue
            
            scores.append(score)
            weights.append(i + 1)  # More recent matches have higher weight
        
        if len(scores) < 2:
            return 0.0
        
        # Calculate weighted linear regression slope
        weighted_avg_x = sum(w * i for i, w in enumerate(weights)) / sum(weights)
        weighted_avg_y = sum(w * s for s, w in zip(scores, weights)) / sum(weights)
        
        numerator = sum(w * (i - weighted_avg_x) * (s - weighted_avg_y) 
                       for i, s, w in zip(range(len(scores)), scores, weights))
        denominator = sum(w * (i - weighted_avg_x) ** 2 
                         for i, w in zip(range(len(scores)), weights))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @log_exceptions(logger)
    def _extract_momentum_features(self, home_player_id, away_player_id, prev_matches, momentum_window):
        """
        Extract momentum-based features.
        
        Args:
            home_player_id (str): Home player ID
            away_player_id (str): Away player ID
            prev_matches (list): Previous matches data
            momentum_window (int): Window size for momentum calculation
            
        Returns:
            list: Momentum features
        """
        home_recent_matches = self._get_player_recent_matches(home_player_id, prev_matches, momentum_window)
        away_recent_matches = self._get_player_recent_matches(away_player_id, prev_matches, momentum_window)
        
        home_momentum = self._calculate_momentum(home_player_id, home_recent_matches)
        away_momentum = self._calculate_momentum(away_player_id, away_recent_matches)
        
        home_score_trend = self._calculate_score_trend(home_player_id, home_recent_matches)
        away_score_trend = self._calculate_score_trend(away_player_id, away_recent_matches)
        
        return [
            home_momentum,
            away_momentum,
            home_score_trend,
            away_score_trend,
            home_momentum - away_momentum,  # Momentum differential
            home_score_trend - away_score_trend  # Score trend differential
        ]
    
    @log_exceptions(logger)
    def _extract_streak_features(self, home_player_id, away_player_id, prev_matches, streak_window):
        """
        Extract streak-based features.
        
        Args:
            home_player_id (str): Home player ID
            away_player_id (str): Away player ID
            prev_matches (list): Previous matches data
            streak_window (int): Window size for streak calculation
            
        Returns:
            list: Streak features
        """
        home_recent_matches = self._get_player_recent_matches(home_player_id, prev_matches, streak_window)
        away_recent_matches = self._get_player_recent_matches(away_player_id, prev_matches, streak_window)
        
        home_win_streak = self._calculate_current_streak(home_player_id, home_recent_matches, 'win')
        away_win_streak = self._calculate_current_streak(away_player_id, away_recent_matches, 'win')
        
        home_win_trend = self._calculate_win_trend(home_player_id, home_recent_matches)
        away_win_trend = self._calculate_win_trend(away_player_id, away_recent_matches)
        
        return [
            home_win_streak,
            away_win_streak,
            home_win_trend,
            away_win_trend,
            home_win_streak - away_win_streak,  # Streak differential
            home_win_trend - away_win_trend  # Win trend differential
        ]
    
    @log_exceptions(logger)
    def _extract_efficiency_features(self, home_player, away_player, home_player_id, away_player_id, prev_matches):
        """
        Extract efficiency-based features.
        
        Args:
            home_player (dict): Home player data
            away_player (dict): Away player data
            home_player_id (str): Home player ID
            away_player_id (str): Away player ID
            prev_matches (list): Previous matches data
            
        Returns:
            list: Efficiency features
        """
        home_recent_matches = self._get_player_recent_matches(home_player_id, prev_matches, 10)
        away_recent_matches = self._get_player_recent_matches(away_player_id, prev_matches, 10)
        
        home_score_efficiency = self._calculate_ppg_efficiency({'id': home_player_id})
        away_score_efficiency = self._calculate_ppg_efficiency({'id': away_player_id})
        
        home_consistency = self._calculate_consistency_efficiency(home_player_id, home_recent_matches)
        away_consistency = self._calculate_consistency_efficiency(away_player_id, away_recent_matches)
        
        home_dominance = self._calculate_dominance_factor(home_player_id, home_recent_matches)
        away_dominance = self._calculate_dominance_factor(away_player_id, away_recent_matches)
        
        return [
            home_score_efficiency,
            away_score_efficiency,
            home_consistency,
            away_consistency,
            home_dominance,
            away_dominance,
            home_score_efficiency - away_score_efficiency,  # Score efficiency differential
            home_consistency - away_consistency,  # Consistency differential
            home_dominance - away_dominance  # Dominance differential
        ]
    
    @log_exceptions(logger)
    def _calculate_win_trend(self, player_id, matches):
        """
        Calculate win rate trend for momentum.
        
        Args:
            player_id (str): Player ID
            matches (list): List of recent matches
            
        Returns:
            float: Win trend (positive = improving, negative = declining)
        """
        if len(matches) < 2:
            return 0.0
        
        wins = []
        weights = []
        
        for i, match in enumerate(matches[-10:]):  # Last 10 matches
            if str(match['homePlayer']['id']) == player_id:
                win = 1 if match.get('homeScore', 0) > match.get('awayScore', 0) else 0
            elif str(match['awayPlayer']['id']) == player_id:
                win = 1 if match.get('awayScore', 0) > match.get('homeScore', 0) else 0
            else:
                continue
            
            wins.append(win)
            weights.append(i + 1)
        
        if len(wins) < 2:
            return 0.0
        
        # Calculate weighted linear regression slope
        weighted_avg_x = sum(w * i for i, w in enumerate(weights)) / sum(weights)
        weighted_avg_y = sum(w * w_val for w_val, w in zip(wins, weights)) / sum(weights)
        
        numerator = sum(w * (i - weighted_avg_x) * (w_val - weighted_avg_y) 
                       for i, w_val, w in zip(range(len(wins)), wins, weights))
        denominator = sum(w * (i - weighted_avg_x) ** 2 
                         for i, w in zip(range(len(wins)), weights))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @log_exceptions(logger)
    def _calculate_vs_expectation(self, player_id, matches):
        """
        Calculate performance vs expectation.
        
        Args:
            player_id (str): Player ID
            matches (list): List of recent matches
            
        Returns:
            float: Performance vs expectation ratio
        """
        if not matches:
            return 0.0
        
        total_performance = 0
        total_expectation = 0
        count = 0
        
        for match in matches[-10:]:
            if str(match['homePlayer']['id']) == player_id:
                actual_score = match.get('homeScore', 0)
                opponent_avg = match.get('awayPlayer', {}).get('average_score', 50)
                expected_score = max(50, 100 - opponent_avg)  # Simple expectation model
            elif str(match['awayPlayer']['id']) == player_id:
                actual_score = match.get('awayScore', 0)
                opponent_avg = match.get('homePlayer', {}).get('average_score', 50)
                expected_score = max(50, 100 - opponent_avg)
            else:
                continue
            
            total_performance += actual_score
            total_expectation += expected_score
            count += 1
        
        if count == 0 or total_expectation == 0:
            return 0.0
        
        return (total_performance / total_expectation) - 1.0
    
    @log_exceptions(logger)
    def _calculate_clutch_performance(self, player_id, matches):
        """
        Calculate clutch performance in close games.
        
        Args:
            player_id (str): Player ID
            matches (list): List of recent matches
            
        Returns:
            float: Clutch performance ratio
        """
        close_games = 0
        clutch_wins = 0
        
        for match in matches[-15:]:
            home_score = match.get('homeScore', 0)
            away_score = match.get('awayScore', 0)
            
            # Consider games within 10 points as "close"
            if abs(home_score - away_score) <= 10:
                close_games += 1
                
                if str(match['homePlayer']['id']) == player_id and home_score > away_score:
                    clutch_wins += 1
                elif str(match['awayPlayer']['id']) == player_id and away_score > home_score:
                    clutch_wins += 1
        
        if close_games == 0:
            return 0.0
        
        return clutch_wins / close_games
    
    @log_exceptions(logger)
    def _calculate_current_streak(self, player_id, matches, streak_type):
        """
        Calculate current win/loss streak.
        
        Args:
            player_id (str): Player ID
            matches (list): List of recent matches
            streak_type (str): 'win' or 'loss'
            
        Returns:
            int: Current streak length
        """
        if not matches:
            return 0
        
        streak = 0
        target_result = (streak_type == 'win')
        
        # Go through matches from most recent to oldest
        for match in reversed(matches[-15:]):
            if str(match['homePlayer']['id']) == player_id:
                won = match.get('homeScore', 0) > match.get('awayScore', 0)
            elif str(match['awayPlayer']['id']) == player_id:
                won = match.get('awayScore', 0) > match.get('homeScore', 0)
            else:
                continue
            
            if won == target_result:
                streak += 1
            else:
                break
        
        return streak
    
    @log_exceptions(logger)
    def _calculate_longest_streak(self, player_id, matches, streak_type):
        """
        Calculate longest win/loss streak in window.
        
        Args:
            player_id (str): Player ID
            matches (list): List of recent matches
            streak_type (str): 'win' or 'loss'
            
        Returns:
            int: Longest streak length
        """
        if not matches:
            return 0
        
        longest_streak = 0
        current_streak = 0
        target_result = (streak_type == 'win')
        
        for match in matches[-15:]:
            if str(match['homePlayer']['id']) == player_id:
                won = match.get('homeScore', 0) > match.get('awayScore', 0)
            elif str(match['awayPlayer']['id']) == player_id:
                won = match.get('awayScore', 0) > match.get('homeScore', 0)
            else:
                continue
            
            if won == target_result:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 0
        
        return longest_streak
    
    @log_exceptions(logger)
    def _calculate_ppg_efficiency(self, player):
        """
        Calculate points per game efficiency.
        
        Args:
            player (dict): Player statistics
            
        Returns:
            float: PPG efficiency score
        """
        avg_score = player.get('average_score', 0)
        games_played = player.get('games_played', 1)
        
        # Normalize by games played to account for sample size
        efficiency = avg_score * min(1.0, games_played / 10.0)
        
        return efficiency / 100.0  # Normalize to 0-1 range
    
    @log_exceptions(logger)
    def _calculate_score_vs_opponents(self, player_id, matches):
        """
        Calculate average score differential vs opponents.
        
        Args:
            player_id (str): Player ID
            matches (list): List of recent matches
            
        Returns:
            float: Average score differential
        """
        if not matches:
            return 0.0
        
        differentials = []
        
        for match in matches[-10:]:
            if str(match['homePlayer']['id']) == player_id:
                player_score = match.get('homeScore', 0)
                opponent_score = match.get('awayScore', 0)
            elif str(match['awayPlayer']['id']) == player_id:
                player_score = match.get('awayScore', 0)
                opponent_score = match.get('homeScore', 0)
            else:
                continue
            
            differentials.append(player_score - opponent_score)
        
        if not differentials:
            return 0.0
        
        return sum(differentials) / len(differentials)
    
    @log_exceptions(logger)
    def _calculate_consistency_efficiency(self, player_id, matches):
        """
        Calculate consistency efficiency (inverse of score variance).
        
        Args:
            player_id (str): Player ID
            matches (list): List of recent matches
            
        Returns:
            float: Consistency efficiency score
        """
        if not matches:
            return 0.0
        
        scores = []
        
        for match in matches[-10:]:
            if str(match['homePlayer']['id']) == player_id:
                score = match.get('homeScore', 0)
            elif str(match['awayPlayer']['id']) == player_id:
                score = match.get('awayScore', 0)
            else:
                continue
            
            scores.append(score)
        
        if len(scores) < 2:
            return 0.0
        
        variance = np.var(scores)
        max_variance = 1000  # Reasonable max variance for normalization
        
        return 1.0 - min(variance / max_variance, 1.0)
    
    @log_exceptions(logger)
    def _calculate_comeback_rate(self, player_id, matches):
        """
        Calculate comeback ability rate.
        
        Args:
            player_id (str): Player ID
            matches (list): List of recent matches
            
        Returns:
            float: Comeback rate (0-1)
        """
        # For now, use a simplified metric based on wins in close games
        # In a real implementation, this would analyze game progression data
        return self._calculate_clutch_performance(player_id, matches)
    
    @log_exceptions(logger)
    def _calculate_dominance_factor(self, player_id, matches):
        """
        Calculate dominance factor (average margin of victory).
        
        Args:
            player_id (str): Player ID
            matches (list): List of recent matches
            
        Returns:
            float: Average margin of victory
        """
        if not matches:
            return 0.0
        
        victory_margins = []
        
        for match in matches[-10:]:
            if str(match['homePlayer']['id']) == player_id:
                player_score = match.get('homeScore', 0)
                opponent_score = match.get('awayScore', 0)
                if player_score > opponent_score:
                    victory_margins.append(player_score - opponent_score)
            elif str(match['awayPlayer']['id']) == player_id:
                player_score = match.get('awayScore', 0)
                opponent_score = match.get('homeScore', 0)
                if player_score > opponent_score:
                    victory_margins.append(player_score - opponent_score)
        
        if not victory_margins:
            return 0.0
        
        return sum(victory_margins) / len(victory_margins)
