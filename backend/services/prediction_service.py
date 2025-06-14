"""
Prediction service for generating match predictions.
"""

import json
from pathlib import Path

from config.settings import PREDICTIONS_FILE, PREDICTION_HISTORY_FILE
from config.logging_config import get_prediction_refresh_logger
from utils.logging import log_execution_time, log_exceptions
from utils.time import get_current_time, format_datetime, parse_datetime
from core.models.registry import ModelRegistry, ScoreModelRegistry
from core.models.winner_prediction import WinnerPredictionModel
from core.models.score_prediction import ScorePredictionModel

logger = get_prediction_refresh_logger()


class PredictionService:
    """
    Service for generating and managing predictions.
    """
    
    def __init__(self):
        """
        Initialize the prediction service.
        """
        self.winner_model_registry = ModelRegistry()
        self.score_model_registry = ScoreModelRegistry()
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def generate_predictions(self, player_stats, upcoming_matches):
        """
        Generate predictions for upcoming matches.
        
        Args:
            player_stats (dict): Player statistics dictionary
            upcoming_matches (list): List of upcoming match dictionaries
            
        Returns:
            list: List of prediction dictionaries
        """
        logger.info(f"Generating predictions for {len(upcoming_matches)} upcoming matches")
        
        # Get best winner prediction model
        winner_model_info = self.winner_model_registry.get_best_model_info()
        if not winner_model_info:
            logger.error("No winner prediction model available")
            return []
        
        # Load winner prediction model
        winner_model = WinnerPredictionModel.load(
            winner_model_info.get("model_path"),
            winner_model_info.get("info_path")
        )
        
        # Get best score prediction model
        score_model_info = self.score_model_registry.get_best_model_info()
        if not score_model_info:
            logger.error("No score prediction model available")
            return []
        
        # Load score prediction model
        score_model = ScorePredictionModel.load(
            score_model_info.get("model_path"),
            score_model_info.get("info_path")
        )
        
        # Generate predictions
        predictions = []
        
        for match in upcoming_matches:
            # Generate winner prediction
            winner_prediction = winner_model.predict(player_stats, match)
            
            # Generate score prediction
            score_prediction = score_model.predict(player_stats, match)
            
            # Create prediction object
            prediction = {
                "fixtureId": match.get("id"),
                "homePlayer": match.get("homePlayer"),
                "awayPlayer": match.get("awayPlayer"),
                "homeTeam": match.get("homeTeam"),
                "awayTeam": match.get("awayTeam"),
                "fixtureStart": match.get("fixtureStart"),
                "prediction": winner_prediction,
                "score_prediction": score_prediction,
                "generated_at": format_datetime(get_current_time())
            }
            
            predictions.append(prediction)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def save_predictions(self, predictions):
        """
        Save predictions to file.
        
        Args:
            predictions (list): List of prediction dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Saving {len(predictions)} predictions")
        
        try:
            # Create directory if it doesn't exist
            Path(PREDICTIONS_FILE).parent.mkdir(parents=True, exist_ok=True)
            
            # Save predictions
            with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2)
            
            logger.info(f"Successfully saved predictions to {PREDICTIONS_FILE}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            return False
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def update_prediction_history(self, predictions):
        """
        Update prediction history with new predictions.
        Prevents duplicates based on fixture ID.
        
        Args:
            predictions (list): List of prediction dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Updating prediction history")
        
        try:
            # Load existing prediction history
            history = []
            if Path(PREDICTION_HISTORY_FILE).exists():
                try:
                    with open(PREDICTION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.error(f"Error loading prediction history: {str(e)}")
            
            # Create a set of existing fixture IDs for deduplication
            existing_fixtures = {p.get('fixtureId') for p in history if p.get('fixtureId')}
            logger.info(f"Found {len(existing_fixtures)} existing fixtures in history")
            
            # Add timestamp to predictions and filter duplicates
            timestamped_predictions = []
            duplicates_skipped = 0
            for prediction in predictions:
                fixture_id = prediction.get('fixtureId')
                if fixture_id not in existing_fixtures:
                    prediction_copy = prediction.copy()
                    prediction_copy["saved_at"] = format_datetime(get_current_time())
                    timestamped_predictions.append(prediction_copy)
                    existing_fixtures.add(fixture_id)
                else:
                    duplicates_skipped += 1
            
            # Only append if we have new predictions
            if timestamped_predictions:
                history.extend(timestamped_predictions)
                
                # Create directory if it doesn't exist
                Path(PREDICTION_HISTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
                
                # Save updated history
                with open(PREDICTION_HISTORY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2)
                
                logger.info(f"Prediction history updated with {len(timestamped_predictions)} new predictions")
            else:
                logger.info("No new predictions to add to history")
                
            if duplicates_skipped > 0:
                logger.info(f"Skipped {duplicates_skipped} duplicate predictions")
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating prediction history: {str(e)}")
            return False
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def get_predictions(self, filter_future=True):
        """
        Get predictions from file.
        
        Args:
            filter_future (bool): Whether to filter for future matches only
            
        Returns:
            list: List of prediction dictionaries
        """
        logger.info("Getting predictions")
        
        try:
            # Check if predictions file exists
            if not Path(PREDICTIONS_FILE).exists():
                logger.warning(f"Predictions file {PREDICTIONS_FILE} does not exist")
                return []
            
            # Load predictions
            with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            
            # Filter for future matches if requested
            if filter_future:
                now = get_current_time()
                predictions = [
                    p for p in predictions
                    if parse_datetime(p.get("fixtureStart", "")) > now
                ]
            
            logger.info(f"Retrieved {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            return []
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def get_prediction_history(self, player_filter=None, date_filter=None):
        """
        Get prediction history with filtering.
        
        Args:
            player_filter (str): Filter by player name
            date_filter (str): Filter by date
            
        Returns:
            list: List of prediction history dictionaries
        """
        logger.info("Getting prediction history")
        
        try:
            # Check if prediction history file exists
            if not Path(PREDICTION_HISTORY_FILE).exists():
                logger.warning(f"Prediction history file {PREDICTION_HISTORY_FILE} does not exist")
                return []
            
            # Load prediction history
            with open(PREDICTION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # Apply filters
            filtered_history = history
            
            if player_filter:
                filtered_history = [
                    p for p in filtered_history
                    if player_filter.lower() in p.get('homePlayer', {}).get('name', '').lower() or
                       player_filter.lower() in p.get('awayPlayer', {}).get('name', '').lower()
                ]
            
            if date_filter:
                filtered_history = [
                    p for p in filtered_history
                    if date_filter in p.get('fixtureStart', '')
                ]
            
            logger.info(f"Retrieved {len(filtered_history)} prediction history entries")
            return filtered_history
            
        except Exception as e:
            logger.error(f"Error getting prediction history: {str(e)}")
            return []
