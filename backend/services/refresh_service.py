"""
Refresh service for updating predictions.
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path

# Add the parent directory to the Python path so we can import our modules
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from config.settings import (
    MATCH_HISTORY_FILE, PLAYER_STATS_FILE, UPCOMING_MATCHES_FILE,
    PREDICTIONS_FILE, PREDICTION_HISTORY_FILE, MATCH_HISTORY_DAYS,
    UPCOMING_MATCHES_DAYS
)
from config.logging_config import get_prediction_refresh_logger
from utils.logging import log_execution_time, log_exceptions
from utils.time import get_current_time, format_datetime
from services.prediction_validation_service import PredictionValidationService
from core.data.fetchers import TokenFetcher
from core.data.fetchers.match_history import MatchHistoryFetcher
from core.data.fetchers.upcoming_matches import UpcomingMatchesFetcher
from core.data.processors.player_stats import PlayerStatsProcessor
from core.models.registry import ModelRegistry, ScoreModelRegistry
from core.models.winner_prediction import WinnerPredictionModel
from core.models.score_prediction import ScorePredictionModel
from services.refresh_status_service import get_refresh_status_service

logger = get_prediction_refresh_logger()


class RefreshService:
    """
    Service for refreshing data and predictions.
    """

    def __init__(self):
        """
        Initialize the refresh service.
        """
        self.token_fetcher = TokenFetcher()
        self.match_history_fetcher = MatchHistoryFetcher(days_back=MATCH_HISTORY_DAYS)
        # Use the updated UPCOMING_MATCHES_DAYS value (now 30 days)
        self.upcoming_matches_fetcher = UpcomingMatchesFetcher(days_forward=UPCOMING_MATCHES_DAYS)
        self.player_stats_processor = PlayerStatsProcessor()
        self.winner_model_registry = ModelRegistry()
        self.score_model_registry = ScoreModelRegistry()
        self.validation_service = PredictionValidationService()

    @log_execution_time(logger)
    @log_exceptions(logger)
    def refresh_data(self):
        """
        Refresh all data (match history, upcoming matches, player stats).

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting data refresh")

        try:
            # Get authentication token
            token = self.token_fetcher.get_token()
            if not token:
                logger.error("Failed to retrieve authentication token")
                return False

            # Fetch match history
            logger.info("Fetching match history")
            matches = self.match_history_fetcher.fetch_match_history()
            if not matches:
                logger.error("Failed to fetch match history")
                return False

            # Calculate player statistics
            logger.info("Calculating player statistics")
            player_stats = self.player_stats_processor.calculate_player_stats(matches)
            if not player_stats:
                logger.error("Failed to calculate player statistics")
                return False

            # Fetch upcoming matches for the next 30 days
            logger.info(f"Fetching upcoming matches for the next {UPCOMING_MATCHES_DAYS} days")
            upcoming_matches = self.upcoming_matches_fetcher.fetch_upcoming_matches()
            if not upcoming_matches:
                logger.error("Failed to fetch upcoming matches")
                return False

            # Log detailed information about the upcoming matches
            logger.info(f"Successfully fetched {len(upcoming_matches)} upcoming matches")
            for i, match in enumerate(upcoming_matches[:10]):  # Log first 10 matches for debugging
                logger.info(f"Match {i+1}: ID={match.get('id')}, Start={match.get('fixtureStart')}, "
                           f"Home={match.get('homePlayer', {}).get('name')}, "
                           f"Away={match.get('awayPlayer', {}).get('name')}")

            logger.info("Data refresh completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during data refresh: {str(e)}")
            return False    @log_execution_time(logger)
    @log_exceptions(logger)
    def refresh_predictions(self, status_service=None):
        """
        Refresh predictions for upcoming matches.

        Args:
            status_service: Optional status service for progress updates

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting prediction refresh")

        try:
            # Stage: Loading models
            if status_service:
                status_service.update_stage("winner_models", "Loading winner prediction models...")

            # Load player statistics
            player_stats = self.player_stats_processor.load_from_file()
            if not player_stats:
                logger.error("Failed to load player statistics")
                return False

            # Load upcoming matches
            upcoming_matches = self.upcoming_matches_fetcher.load_from_file()
            if not upcoming_matches:
                logger.error("Failed to load upcoming matches")
                return False

            # Get all winner prediction models
            winner_models = self.winner_model_registry.list_models()
            if not winner_models:
                logger.error("No winner prediction models available")
                return False

            # Use the best model (highest accuracy)
            best_winner_model_info = self.winner_model_registry.get_best_model_info()
            if not best_winner_model_info:
                # Fallback to most recent model if best model is not set
                winner_models.sort(key=lambda x: x.get("model_id", 0), reverse=True)
                best_winner_model_info = winner_models[0]

            logger.info(f"Using winner prediction model {best_winner_model_info.get('model_id')} with accuracy {best_winner_model_info.get('accuracy')}")

            # Load winner prediction model
            try:
                winner_model = WinnerPredictionModel.load(
                    best_winner_model_info.get("model_path"),
                    best_winner_model_info.get("info_path")
                )
                logger.info(f"Successfully loaded winner prediction model from {best_winner_model_info.get('model_path')}")
            except Exception as e:
                logger.error(f"Error loading winner prediction model: {str(e)}")
                return False

            if status_service:
                status_service.complete_stage("winner_models")
                status_service.update_stage("score_models", "Loading score prediction models...")

            # Get all score prediction models
            score_models = self.score_model_registry.list_models()
            if not score_models:
                logger.error("No score prediction models available")
                return False

            # Use the best model (lowest MAE)
            best_score_model_info = self.score_model_registry.get_best_model_info()
            if not best_score_model_info:
                # Fallback to most recent model if best model is not set
                score_models.sort(key=lambda x: x.get("model_id", 0), reverse=True)
                best_score_model_info = score_models[0]

            logger.info(f"Using score prediction model {best_score_model_info.get('model_id')} with MAE {best_score_model_info.get('total_score_mae')}")

            # Load score prediction model
            try:
                score_model = ScorePredictionModel.load(
                    best_score_model_info.get("model_path"),
                    best_score_model_info.get("info_path")
                )
                logger.info(f"Successfully loaded score prediction model from {best_score_model_info.get('model_path')}")
            except Exception as e:
                logger.error(f"Error loading score prediction model: {str(e)}")
                return False

            # Generate predictions
            logger.info(f"Generating predictions for {len(upcoming_matches)} upcoming matches")
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

            # Save predictions
            logger.info(f"Saving {len(predictions)} predictions")
            with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2)

            # Update prediction history
            self._update_prediction_history(predictions)

            logger.info("Prediction refresh completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during prediction refresh: {str(e)}")
            return False

    @log_exceptions(logger)
    def _update_prediction_history(self, predictions):
        """
        Update prediction history with new predictions.
        Prevents duplicates based on fixture ID.

        Args:
            predictions (list): List of prediction dictionaries
        """
        logger.info("Updating prediction history")

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
            
            # Save updated history
            with open(PREDICTION_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Prediction history updated with {len(timestamped_predictions)} new predictions")
        else:
            logger.info("No new predictions to add to history")
            
        if duplicates_skipped > 0:
            logger.info(f"Skipped {duplicates_skipped} duplicate predictions")


@log_execution_time(logger)
@log_exceptions(logger)
def refresh_predictions():
    """
    Refresh data and predictions.

    Returns:
        bool: True if successful, False otherwise
    """
    refresh_status_service = get_refresh_status_service()
    
    try:
        # Start the refresh process
        refresh_status_service.start_refresh()
        logger.info("Starting prediction refresh with status tracking")

        # Initialize the refresh service
        refresh_service = RefreshService()
        
        # Stage 1: Authentication
        refresh_status_service.update_stage("auth", "Fetching authentication token...")
        try:
            token = refresh_service.token_fetcher.get_token()
            if not token:
                raise Exception("Failed to fetch authentication token")
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            refresh_status_service.complete_refresh(False, f"Authentication failed: {str(e)}")
            return False
        refresh_status_service.complete_stage("auth")

        # Stage 2: Data refresh
        refresh_status_service.update_stage("player_stats", "Refreshing match history and player data...")
        try:
            data_success = refresh_service.refresh_data()
            if not data_success:
                raise Exception("Failed to refresh data")
        except Exception as e:
            logger.error(f"Data refresh failed: {str(e)}")
            refresh_status_service.complete_refresh(False, f"Data refresh failed: {str(e)}")
            return False
        refresh_status_service.complete_stage("player_stats")

        # Stage 3: Loading upcoming matches
        refresh_status_service.update_stage("upcoming_matches", "Loading upcoming matches...")
        try:
            # This is already done in refresh_data, but we'll update the status
            refresh_status_service.update_stage("upcoming_matches", "Upcoming matches loaded successfully", 100)
        except Exception as e:
            logger.error(f"Failed to load upcoming matches: {str(e)}")
            refresh_status_service.complete_refresh(False, f"Failed to load upcoming matches: {str(e)}")
            return False
        refresh_status_service.complete_stage("upcoming_matches")        # Stage 4: Prediction refresh
        refresh_status_service.update_stage("generating_predictions", "Generating match predictions...")
        try:
            prediction_success = refresh_service.refresh_predictions(refresh_status_service)
            if not prediction_success:
                raise Exception("Failed to generate predictions")
        except Exception as e:
            logger.error(f"Prediction refresh failed: {str(e)}")
            refresh_status_service.complete_refresh(False, f"Prediction refresh failed: {str(e)}")
            return False
        refresh_status_service.complete_stage("generating_predictions")

        # Stage 5: Validation
        refresh_status_service.update_stage("validation", "Validating predictions...")
        try:
            # Run prediction validation
            validation_service = PredictionValidationService()
            validation_service.validate_recent_predictions()
            refresh_status_service.update_stage("validation", "Predictions validated successfully", 100)
        except Exception as e:
            logger.warning(f"Prediction validation failed (non-critical): {str(e)}")
            # Don't fail the entire process for validation errors
        refresh_status_service.complete_stage("validation")

        # Complete successfully
        refresh_status_service.complete_refresh(True)
        logger.info("Prediction refresh completed successfully")
        return True

    except Exception as e:
        logger.error(f"Unexpected error during refresh: {str(e)}")
        refresh_status_service.complete_refresh(False, f"Unexpected error: {str(e)}")
        return False
