import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from core.data.fetchers.match_history import MatchHistoryFetcher
from utils.logging import log_execution_time, log_exceptions
from utils.time import get_current_time, format_datetime
from config.settings import PREDICTION_HISTORY_FILE

logger = logging.getLogger(__name__)

class PredictionValidationService:
    """
    Service for validating predictions against completed match results.
    """
    
    def __init__(self):
        self.match_history_fetcher = MatchHistoryFetcher(days_back=30)  # Look back 30 days for completed matches
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def validate_predictions(self) -> bool:
        """
        Validate all pending predictions against completed match results.
        
        Returns:
            bool: True if validation was successful, False otherwise
        """
        logger.info("Starting prediction validation")
        
        # Load existing prediction history
        predictions = self._load_prediction_history()
        if not predictions:
            logger.info("No prediction history found")
            return True
        
        logger.info(f"Loaded {len(predictions)} predictions from history")
        
        # Count pending predictions
        pending_predictions = [p for p in predictions if p.get('prediction_correct') is None]
        logger.info(f"Found {len(pending_predictions)} pending predictions to validate")
        
        # Get completed matches from the last 30 days
        completed_matches = self._get_completed_matches()
        if not completed_matches:
            logger.warning("No completed matches found - validation cannot proceed")
            return True
        
        logger.info(f"Found {len(completed_matches)} completed matches")
        
        # Log sample of completed match fixture IDs for debugging
        sample_fixture_ids = [str(match.get('id', 'NO_ID')) for match in completed_matches[:5]]
        logger.info(f"Sample completed match fixture IDs: {sample_fixture_ids}")
        
        # Create a lookup dictionary for completed matches
        completed_matches_dict = {match['id']: match for match in completed_matches}
        
        # Log fixture ID matching analysis
        self._log_fixture_id_analysis(pending_predictions, completed_matches_dict)
        
        # Validate predictions
        updated_count = 0
        skipped_count = 0
        
        for prediction in predictions:
            if self._should_validate_prediction(prediction, completed_matches_dict):
                if self._validate_single_prediction(prediction, completed_matches_dict):
                    updated_count += 1
            else:
                skipped_count += 1
        
        logger.info(f"Validation summary: {updated_count} updated, {skipped_count} skipped")
        
        # Save updated predictions if any were modified
        if updated_count > 0:
            self._save_prediction_history(predictions)
            logger.info(f"Updated validation status for {updated_count} predictions")
        else:
            logger.info("No predictions required validation updates")
        
        return True
    
    def _log_fixture_id_analysis(self, pending_predictions: List[Dict], completed_matches_dict: Dict) -> None:
        """
        Log detailed analysis of fixture ID matching for debugging.
        
        Args:
            pending_predictions (List[Dict]): List of pending predictions
            completed_matches_dict (Dict): Dictionary of completed matches by fixture ID
        """
        if not pending_predictions:
            logger.info("No pending predictions to analyze")
            return
        
        # Sample pending prediction fixture IDs
        pending_fixture_ids = [str(p.get('fixtureId', 'NO_ID')) for p in pending_predictions[:10]]
        logger.info(f"Sample pending prediction fixture IDs: {pending_fixture_ids}")
        
        # Check for matches
        matched_count = 0
        unmatched_sample = []
        
        for prediction in pending_predictions[:10]:  # Check first 10 for detailed logging
            fixture_id = prediction.get('fixtureId')
            if fixture_id in completed_matches_dict:
                matched_count += 1
                logger.debug(f"MATCH FOUND: Fixture ID {fixture_id} has completed match")
            else:
                unmatched_sample.append(str(fixture_id))
                logger.debug(f"NO MATCH: Fixture ID {fixture_id} not in completed matches")
        
        logger.info(f"Fixture ID matching: {matched_count}/{len(pending_predictions[:10])} sample predictions have completed matches")
        if unmatched_sample:
            logger.info(f"Sample unmatched fixture IDs: {unmatched_sample[:5]}")
    
    def _load_prediction_history(self) -> List[Dict]:
        """
        Load prediction history from file.
        
        Returns:
            List[Dict]: List of prediction dictionaries
        """
        if not Path(PREDICTION_HISTORY_FILE).exists():
            return []
        
        try:
            with open(PREDICTION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Error loading prediction history: {str(e)}")
            return []
    
    def _save_prediction_history(self, predictions: List[Dict]) -> None:
        """
        Save prediction history to file.
        
        Args:
            predictions (List[Dict]): List of prediction dictionaries
        """
        try:
            with open(PREDICTION_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving prediction history: {str(e)}")
            raise
    
    def _get_completed_matches(self) -> List[Dict]:
        """
        Get completed matches from the match history fetcher.
        
        Returns:
            List[Dict]: List of completed match dictionaries
        """
        try:
            matches = self.match_history_fetcher.fetch_match_history(save_to_file=False)
            logger.info(f"Fetched {len(matches)} total matches from API")
            
            # Log sample match structure for debugging
            if matches:
                sample_match = matches[0]
                logger.debug(f"Sample match structure: {list(sample_match.keys())}")
                logger.debug(f"Sample match data: id={sample_match.get('id')}, homeScore={sample_match.get('homeScore')}, awayScore={sample_match.get('awayScore')}")
            
            # Filter for matches that have scores (completed matches)
            completed_matches = [
                match for match in matches 
                if match.get('homeScore') is not None and match.get('awayScore') is not None
            ]
            
            # Log filtering results
            incomplete_matches = len(matches) - len(completed_matches)
            if incomplete_matches > 0:
                logger.info(f"Filtered out {incomplete_matches} incomplete matches (no scores)")
            
            logger.info(f"Found {len(completed_matches)} completed matches with scores")
            return completed_matches
        except Exception as e:
            logger.error(f"Error fetching completed matches: {str(e)}")
            return []
    
    def _should_validate_prediction(self, prediction: Dict, completed_matches_dict: Dict) -> bool:
        """
        Check if a prediction should be validated.
        
        Args:
            prediction (Dict): Prediction dictionary
            completed_matches_dict (Dict): Dictionary of completed matches by fixture ID
        
        Returns:
            bool: True if prediction should be validated
        """
        fixture_id = prediction.get('fixtureId')
        
        # Skip if no fixture ID
        if not fixture_id:
            logger.debug(f"Skipping prediction: No fixture ID found")
            return False
        
        # Skip if prediction_correct is already set (unless it's None/undefined)
        if prediction.get('prediction_correct') is not None:
            logger.debug(f"Skipping prediction {fixture_id}: Already validated (prediction_correct={prediction.get('prediction_correct')})")
            return False
        
        # Skip if match is not in completed matches
        if fixture_id not in completed_matches_dict:
            logger.debug(f"Skipping prediction {fixture_id}: No completed match found")
            return False
        
        logger.debug(f"Prediction {fixture_id} ready for validation")
        return True
    
    def _validate_single_prediction(self, prediction: Dict, completed_matches_dict: Dict) -> bool:
        """
        Validate a single prediction against completed match result.
        
        Args:
            prediction (Dict): Prediction dictionary
            completed_matches_dict (Dict): Dictionary of completed matches by fixture ID
        
        Returns:
            bool: True if prediction was updated, False otherwise
        """
        fixture_id = prediction.get('fixtureId')
        completed_match = completed_matches_dict.get(fixture_id)
        
        if not completed_match:
            logger.warning(f"No completed match found for fixture {fixture_id}")
            return False
        
        # Get actual match result
        home_score = completed_match.get('homeScore')
        away_score = completed_match.get('awayScore')
        
        if home_score is None or away_score is None:
            logger.warning(f"Incomplete score data for fixture {fixture_id}: home={home_score}, away={away_score}")
            return False
        
        # Determine actual winner
        if home_score > away_score:
            actual_winner = 'home_win'
        elif away_score > home_score:
            actual_winner = 'away_win'
        else:
            actual_winner = 'draw'
        
        # Get predicted winner
        predicted_winner = prediction.get('prediction', {}).get('predicted_winner')
        
        if not predicted_winner:
            logger.warning(f"No predicted winner found for fixture {fixture_id}")
            return False
        
        # Normalize prediction format (handle different formats)
        original_predicted_winner = predicted_winner
        if predicted_winner == 'home':
            predicted_winner = 'home_win'
        elif predicted_winner == 'away':
            predicted_winner = 'away_win'
        
        # Update prediction with validation results
        is_correct = predicted_winner == actual_winner
        prediction['prediction_correct'] = is_correct
        prediction['homeScore'] = home_score
        prediction['awayScore'] = away_score
        prediction['result'] = actual_winner
        prediction['validated_at'] = format_datetime(get_current_time())
        
        # Calculate score prediction errors if score prediction exists
        score_prediction = prediction.get('score_prediction', {})
        if score_prediction:
            predicted_home_score = score_prediction.get('home_score')
            predicted_away_score = score_prediction.get('away_score')
            
            if predicted_home_score is not None and predicted_away_score is not None:
                prediction['home_score_error'] = abs(predicted_home_score - home_score)
                prediction['away_score_error'] = abs(predicted_away_score - away_score)
                prediction['total_score_error'] = abs(
                    (predicted_home_score + predicted_away_score) - (home_score + away_score)
                )
                logger.debug(f"Score prediction errors for fixture {fixture_id}: home={prediction['home_score_error']}, away={prediction['away_score_error']}, total={prediction['total_score_error']}")
        
        logger.info(f"Validated prediction for fixture {fixture_id}: predicted='{original_predicted_winner}' ({predicted_winner}), actual='{actual_winner}', correct={is_correct}, score={home_score}-{away_score}")
        return True


@log_execution_time(logger)
@log_exceptions(logger)
def validate_predictions():
    """
    Validate predictions against completed match results.
    
    Returns:
        bool: True if successful, False otherwise
    """
    service = PredictionValidationService()
    return service.validate_predictions()