"""
Match prediction service for NBA 2K25 esports.
Integrates feature engineering and prediction model for end-to-end predictions.
"""

import json
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from utils.logging import log_execution_time, log_exceptions
from config.logging_config import get_data_fetcher_logger
from config.settings import OUTPUT_DIR
from core.data.processors.match_prediction_features import MatchPredictionFeatureEngineer
from core.data.processors.match_prediction_model import MatchPredictionModel
from core.data.processors.player_stats import PlayerStatsProcessor
from core.data.fetchers.match_history import MatchHistoryFetcher
from core.data.fetchers.upcoming_matches import UpcomingMatchesFetcher

logger = get_data_fetcher_logger()


class MatchPredictionService:
    """
    Complete service for predicting NBA 2K25 esports match outcomes.
    Handles data preparation, feature engineering, model training, and predictions.
    """
    
    def __init__(self, model_name: str = "nba2k_match_predictor"):
        """
        Initialize the prediction service.
        
        Args:
            model_name: Name for saving/loading the model
        """
        self.model_name = model_name
        self.feature_engineer = MatchPredictionFeatureEngineer()
        self.model = MatchPredictionModel()
        self.player_stats_processor = PlayerStatsProcessor()
        
        # Data fetchers
        self.match_history_fetcher = MatchHistoryFetcher()
        self.upcoming_matches_fetcher = UpcomingMatchesFetcher()
        
        # Paths
        self.output_dir = Path(OUTPUT_DIR)
        
    @log_execution_time(logger)
    @log_exceptions(logger)
    def prepare_training_data(self, days_back: int = 30, min_matches_per_player: int = 5) -> pd.DataFrame:
        """
        Prepare training data by fetching matches and calculating player stats.
        
        Args:
            days_back: Number of days of match history to fetch
            min_matches_per_player: Minimum matches required for a player to be included
            
        Returns:
            DataFrame ready for model training
        """
        logger.info(f"Preparing training data with {days_back} days of history")
        
        # Fetch match history
        logger.info("Fetching match history...")
        self.match_history_fetcher.days_back = days_back
        matches = self.match_history_fetcher.fetch_match_history(save_to_file=True)
        
        if not matches:
            raise ValueError("No match history data available")
        
        # Calculate player statistics
        logger.info("Calculating player statistics...")
        player_stats = self.player_stats_processor.calculate_player_stats(matches, save_to_file=True)
        
        # Filter players with sufficient match history
        filtered_player_stats = {
            player_id: stats for player_id, stats in player_stats.items()
            if stats.get('total_matches', 0) >= min_matches_per_player
        }
        
        logger.info(f"Filtered to {len(filtered_player_stats)} players with >= {min_matches_per_player} matches")
        
        # Create training features
        logger.info("Engineering features...")
        training_df = self.feature_engineer.create_training_dataset(matches, filtered_player_stats)
        
        # Save training data
        self.feature_engineer.save_features(training_df, "training_features.csv")
        
        logger.info(f"Training data prepared: {len(training_df)} samples with {len(training_df.columns)} features")
        return training_df
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def train_model(self, training_df: Optional[pd.DataFrame] = None, 
                   validation_split: float = 0.2, save_model: bool = True) -> Dict:
        """
        Train the prediction model.
        
        Args:
            training_df: Training data (if None, will load from file)
            validation_split: Fraction for validation
            save_model: Whether to save the trained model
            
        Returns:
            Training metrics
        """
        if training_df is None:
            logger.info("Loading training data from file...")
            training_df = self.feature_engineer.load_features("training_features.csv")
            
            if training_df.empty:
                raise ValueError("No training data available. Run prepare_training_data() first.")
        
        logger.info(f"Training model on {len(training_df)} samples")
        
        # Train the model
        metrics = self.model.train(training_df, validation_split=validation_split)
        
        # Save model if requested
        if save_model:
            self.model.save_model(self.model_name)
        
        # Log training results
        logger.info("Training completed!")
        logger.info(f"Winner Prediction Accuracy: {metrics['val_winner_accuracy']:.3f}")
        logger.info(f"Home Score MAE: {metrics['val_home_mae']:.2f}")
        logger.info(f"Away Score MAE: {metrics['val_away_mae']:.2f}")
        logger.info(f"Total Score MAE: {metrics['val_total_mae']:.2f}")
        
        return metrics
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def predict_upcoming_matches(self, load_model: bool = True) -> pd.DataFrame:
        """
        Predict outcomes for upcoming matches.
        
        Args:
            load_model: Whether to load saved model (if False, assumes model is already trained)
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Predicting upcoming matches...")
        
        # Load model if requested
        if load_model:
            try:
                self.model.load_model(self.model_name)
                logger.info(f"Loaded model: {self.model_name}")
            except FileNotFoundError:
                raise ValueError(f"Model {self.model_name} not found. Train the model first.")
        
        # Fetch upcoming matches
        logger.info("Fetching upcoming matches...")
        upcoming_matches = self.upcoming_matches_fetcher.fetch_upcoming_matches(save_to_file=True)
        
        if not upcoming_matches:
            logger.warning("No upcoming matches found")
            return pd.DataFrame()
        
        # Load player statistics
        logger.info("Loading player statistics...")
        player_stats = self.player_stats_processor.load_from_file()
        
        if not player_stats:
            raise ValueError("No player statistics available. Run prepare_training_data() first.")
        
        # Create prediction features
        logger.info("Engineering prediction features...")
        prediction_df = self.feature_engineer.create_prediction_features(upcoming_matches, player_stats)
        
        if prediction_df.empty:
            logger.warning("No prediction features could be created")
            return pd.DataFrame()
        
        # Make predictions
        logger.info("Generating predictions...")
        predictions = self.model.predict(prediction_df)
        
        # Save predictions
        predictions_file = self.output_dir / "match_predictions.csv"
        predictions.to_csv(predictions_file, index=False)
        logger.info(f"Predictions saved to {predictions_file}")
        
        return predictions
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def evaluate_model(self, test_days_back: int = 7) -> Dict:
        """
        Evaluate model performance on recent matches.
        
        Args:
            test_days_back: Number of recent days to use for testing
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating model on last {test_days_back} days of matches")
        
        # Load model
        try:
            self.model.load_model(self.model_name)
        except FileNotFoundError:
            raise ValueError(f"Model {self.model_name} not found. Train the model first.")
        
        # Fetch recent matches for testing
        self.match_history_fetcher.days_back = test_days_back
        test_matches = self.match_history_fetcher.fetch_match_history(save_to_file=False)
        
        if not test_matches:
            raise ValueError("No test data available")
        
        # Calculate player stats (excluding test period)
        # Note: In a real scenario, you'd want to calculate stats only on data before the test period
        player_stats = self.player_stats_processor.load_from_file()
        
        if not player_stats:
            raise ValueError("No player statistics available")
        
        # Create test features
        test_df = self.feature_engineer.create_training_dataset(test_matches, player_stats)
        
        if test_df.empty:
            raise ValueError("No test features could be created")
        
        # Make predictions
        predictions = self.model.predict(test_df)
        
        # Calculate evaluation metrics
        actual_home_wins = test_df['home_win'].values
        predicted_home_wins = predictions['predicted_home_win'].values
        
        accuracy = (actual_home_wins == predicted_home_wins).mean()
        
        metrics = {
            'test_samples': len(test_df),
            'winner_accuracy': accuracy,
            'home_score_mae': abs(test_df['home_score'] - predictions['predicted_home_score']).mean(),
            'away_score_mae': abs(test_df['away_score'] - predictions['predicted_away_score']).mean(),
            'total_score_mae': abs(test_df['total_score'] - predictions['predicted_total_score']).mean()
        }
        
        logger.info(f"Evaluation completed. Winner accuracy: {accuracy:.3f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        try:
            self.model.load_model(self.model_name)
        except FileNotFoundError:
            raise ValueError(f"Model {self.model_name} not found. Train the model first.")
        
        return self.model.get_feature_importance()
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def retrain_with_new_data(self, days_back: int = 60) -> Dict:
        """
        Retrain the model with fresh data.
        
        Args:
            days_back: Number of days of history to include
            
        Returns:
            Training metrics
        """
        logger.info(f"Retraining model with {days_back} days of fresh data")
        
        # Prepare fresh training data
        training_df = self.prepare_training_data(days_back=days_back)
        
        # Train new model
        metrics = self.train_model(training_df=training_df, save_model=True)
        
        logger.info("Model retrained successfully")
        return metrics
    
    @log_exceptions(logger)
    def get_prediction_summary(self, predictions: pd.DataFrame) -> Dict:
        """
        Get a summary of predictions for easy interpretation.
        
        Args:
            predictions: DataFrame with predictions
            
        Returns:
            Summary dictionary
        """
        if predictions.empty:
            return {'message': 'No predictions available'}
        
        summary = {
            'total_matches': len(predictions),
            'predictions': [],
            'average_confidence': predictions['prediction_confidence'].mean(),
            'high_confidence_matches': len(predictions[predictions['prediction_confidence'] > 0.3]),
            'predicted_total_score_avg': predictions['predicted_total_score'].mean()
        }
        
        # Create detailed predictions
        for _, row in predictions.iterrows():
            match_pred = {
                'match_id': row.get('match_id', 'unknown'),
                'home_player': row.get('home_player_name', 'unknown'),
                'away_player': row.get('away_player_name', 'unknown'),
                'predicted_winner': row['predicted_winner'],
                'home_win_probability': round(row['home_win_probability'], 3),
                'predicted_scores': {
                    'home': round(row['predicted_home_score'], 1),
                    'away': round(row['predicted_away_score'], 1),
                    'total': round(row['predicted_total_score'], 1)
                },
                'confidence': round(row['prediction_confidence'], 3),
                'fixture_start': row.get('fixture_start', 'unknown')
            }
            summary['predictions'].append(match_pred)
        
        return summary
