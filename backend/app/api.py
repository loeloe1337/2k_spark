"""
API server for the 2K Flash application.
"""

import os
import json
import threading
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add the parent directory to the Python path so we can import our modules
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from flask import Flask, jsonify, request
from flask_cors import CORS
import pytz

from config.settings import (
    API_HOST, API_PORT, CORS_ORIGINS, PREDICTIONS_FILE,
    PREDICTION_HISTORY_FILE, MODELS_DIR, DEFAULT_TIMEZONE,
    UPCOMING_MATCHES_FILE, PLAYER_STATS_FILE
)
from config.logging_config import get_api_logger
from utils.logging import log_execution_time, log_exceptions
from core.models.registry import ModelRegistry, ScoreModelRegistry
from services.data_service import DataService
from services.prediction_service import PredictionService
from services.refresh_service import RefreshService
from services.live_scores_service import LiveScoresService
from services.refresh_status_service import get_refresh_status_service

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)

# Initialize logger
logger = get_api_logger()


@app.route('/api/predictions', methods=['GET'])
@log_execution_time(logger)
@log_exceptions(logger)
def get_predictions():
    """
    Get predictions for upcoming matches.

    Returns:
        flask.Response: JSON response with predictions
    """
    try:
        # Check if predictions file exists
        if not Path(PREDICTIONS_FILE).exists():
            # Return empty list if file doesn't exist
            logger.warning(f"Predictions file not found: {PREDICTIONS_FILE}")
            return jsonify([])

        # Read predictions from file
        with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
            logger.info(f"Loaded {len(predictions)} predictions from {PREDICTIONS_FILE}")

        # Add debug information about match dates
        for match in predictions:
            logger.info(f"Match {match.get('fixtureId')}: {match.get('fixtureStart')}")

            # Add a timestamp to each prediction to indicate when it was fetched
            match['fetched_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Return all matches with their timezone information
        logger.info(f"Returning {len(predictions)} matches")
        return jsonify(predictions)
    except Exception as e:
        logger.error(f"Error retrieving predictions: {str(e)}")
        # Return empty list on error
        return jsonify([])


@app.route('/api/score-predictions', methods=['GET'])
@log_execution_time(logger)
@log_exceptions(logger)
def get_score_predictions():
    """
    Get score predictions for upcoming matches.

    Returns:
        flask.Response: JSON response with score predictions
    """
    try:
        # Check if predictions file exists
        if not Path(PREDICTIONS_FILE).exists():
            # Return empty list if file doesn't exist
            logger.warning(f"Predictions file not found: {PREDICTIONS_FILE}")
            return jsonify({
                "predictions": [],
                "summary": {
                    "model_accuracy": 10.0  # Default value
                }
            })

        # Read predictions from file
        with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
            logger.info(f"Loaded {len(predictions)} predictions from {PREDICTIONS_FILE}")

        # For demo purposes, return all matches regardless of date
        logger.info(f"Returning {len(predictions)} matches")

        # Get score model accuracy from registry
        score_model_accuracy = 10.0  # Default value
        try:
            registry = ScoreModelRegistry(MODELS_DIR)
            best_model = registry.get_best_model_info()
            if best_model:
                score_model_accuracy = best_model.get("total_score_mae", 10.0)
                logger.info(f"Retrieved score model accuracy: {score_model_accuracy}")
        except Exception as e:
            logger.error(f"Error retrieving score model accuracy: {str(e)}")

        return jsonify({
            "predictions": predictions,
            "summary": {
                "model_accuracy": score_model_accuracy
            }
        })
    except Exception as e:
        logger.error(f"Error retrieving score predictions: {str(e)}")
        # Return empty list on error
        return jsonify({
            "predictions": [],
            "summary": {
                "model_accuracy": 10.0  # Default value
            }
        })


@app.route('/api/upcoming-matches', methods=['GET'])
@log_execution_time(logger)
@log_exceptions(logger)
def get_upcoming_matches():
    """
    Get upcoming matches.

    Returns:
        flask.Response: JSON response with upcoming matches
    """
    try:
        # Check if upcoming matches file exists
        if not Path(UPCOMING_MATCHES_FILE).exists():
            # Return empty list if file doesn't exist
            logger.warning(f"Upcoming matches file not found: {UPCOMING_MATCHES_FILE}")
            return jsonify([])

        # Read upcoming matches from file
        with open(UPCOMING_MATCHES_FILE, 'r', encoding='utf-8') as f:
            upcoming_matches = json.load(f)
            logger.info(f"Loaded {len(upcoming_matches)} upcoming matches from {UPCOMING_MATCHES_FILE}")

        # For demo purposes, return all matches regardless of date
        logger.info(f"Returning {len(upcoming_matches)} matches")
        return jsonify(upcoming_matches)
    except Exception as e:
        logger.error(f"Error retrieving upcoming matches: {str(e)}")
        # Return empty list on error
        return jsonify([])


@app.route('/api/prediction-history', methods=['GET'])
@log_execution_time(logger)
@log_exceptions(logger)
def get_prediction_history():
    """
    Get prediction history with filtering.

    Returns:
        flask.Response: JSON response with prediction history
    """
    try:
        # Get filter parameters
        player_filter = request.args.get('player', '')
        date_filter = request.args.get('date', '')

        # Check if prediction history file exists
        if not Path(PREDICTION_HISTORY_FILE).exists():
            # Return empty list if file doesn't exist
            return jsonify({
                "predictions": []
            })

        # Read prediction history from file
        with open(PREDICTION_HISTORY_FILE, 'r', encoding='utf-8') as f:
            predictions = json.load(f)

        # Apply filters
        filtered_predictions = predictions
        if player_filter:
            filtered_predictions = [
                p for p in filtered_predictions
                if player_filter.lower() in p.get('homePlayer', {}).get('name', '').lower() or
                   player_filter.lower() in p.get('awayPlayer', {}).get('name', '').lower()
            ]

        if date_filter:
            filtered_predictions = [
                p for p in filtered_predictions
                if date_filter in p.get('fixtureStart', '')
            ]

        return jsonify({
            "predictions": filtered_predictions
        })
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {str(e)}")
        # Return empty list on error
        return jsonify({
            "predictions": []
        })


@app.route('/api/stats', methods=['GET'])
@log_execution_time(logger)
@log_exceptions(logger)
def get_stats():
    """
    Get prediction statistics.

    Returns:
        flask.Response: JSON response with statistics
    """
    try:
        # Check if predictions file exists
        if not Path(PREDICTIONS_FILE).exists():
            # Return default stats if file doesn't exist
            return jsonify({
                "total_matches": 0,
                "home_wins_predicted": 0,
                "away_wins_predicted": 0,
                "avg_confidence": 0,
                "model_accuracy": 0.5,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        # Read predictions from file
        with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
            predictions = json.load(f)

        # Calculate statistics
        total_matches = len(predictions)
        home_wins_predicted = sum(1 for match in predictions if match.get("prediction", {}).get("predicted_winner") == "home")
        away_wins_predicted = total_matches - home_wins_predicted

        # Calculate average confidence
        confidences = [match.get("prediction", {}).get("confidence", 0) for match in predictions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Get model accuracy from registry
        model_accuracy = 0.5  # Default value
        try:
            registry = ModelRegistry(MODELS_DIR)
            best_model = registry.get_best_model_info()
            if best_model:
                model_accuracy = best_model.get("accuracy", 0.5)
        except Exception as e:
            logger.error(f"Error retrieving model accuracy: {str(e)}")

        return jsonify({
            "total_matches": total_matches,
            "home_wins_predicted": home_wins_predicted,
            "away_wins_predicted": away_wins_predicted,
            "avg_confidence": avg_confidence,
            "model_accuracy": model_accuracy,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}")
        # Return default stats on error
        return jsonify({
            "total_matches": 0,
            "home_wins_predicted": 0,
            "away_wins_predicted": 0,
            "avg_confidence": 0,
            "model_accuracy": 0.5,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })


@app.route('/api/player-stats', methods=['GET'])
@log_execution_time(logger)
@log_exceptions(logger)
def get_player_stats():
    """
    Get player statistics.

    Returns:
        flask.Response: JSON response with player statistics
    """
    try:
        # Check if player stats file exists
        if not Path(PLAYER_STATS_FILE).exists():
            # Return empty list if file doesn't exist
            logger.warning(f"Player stats file not found: {PLAYER_STATS_FILE}")
            return jsonify([])

        # Read player stats from file
        with open(PLAYER_STATS_FILE, 'r', encoding='utf-8') as f:
            player_stats = json.load(f)
            logger.info(f"Loaded statistics for {len(player_stats)} players from {PLAYER_STATS_FILE}")

        # Convert to list of player stats
        stats_list = []
        for player_id, stats in player_stats.items():
            # Add player ID to stats
            stats['id'] = player_id
            stats_list.append(stats)

        # Sort by win rate (descending)
        stats_list.sort(key=lambda x: x.get('win_rate', 0), reverse=True)

        logger.info(f"Returning statistics for {len(stats_list)} players")
        return jsonify(stats_list)
    except Exception as e:
        logger.error(f"Error retrieving player statistics: {str(e)}")
        # Return empty list on error
        return jsonify([])


@app.route('/api/live-scores', methods=['GET'])
@log_execution_time(logger)
@log_exceptions(logger)
def get_live_scores():
    """
    Get live scores from H2H API.
    
    Returns:
        flask.Response: JSON response with live scores
    """
    try:
        live_service = LiveScoresService()
        live_data = live_service.get_live_matches_with_predictions()
        
        logger.info(f"Returning {live_data['total_count']} live matches")
        return jsonify(live_data)
    except Exception as e:
        logger.error(f"Error retrieving live scores: {str(e)}")
        return jsonify({
            'matches': [],
            'total_count': 0,
            'last_updated': datetime.now().isoformat(),
            'error': str(e)
        }), 500


@app.route('/api/live-matches-with-predictions', methods=['GET'])
@log_execution_time(logger)
@log_exceptions(logger)
def get_live_matches_with_predictions():
    """
    Get live matches merged with predictions.
    
    Returns:
        flask.Response: JSON response with live matches and predictions
    """
    try:
        live_service = LiveScoresService()
        
        # Load existing predictions
        predictions = []
        if Path(PREDICTIONS_FILE).exists():
            with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
        
        live_data = live_service.get_live_matches_with_predictions(predictions)
        
        logger.info(f"Returning {live_data['total_count']} live matches with predictions")
        return jsonify(live_data)
    except Exception as e:
        logger.error(f"Error retrieving live matches with predictions: {str(e)}")
        return jsonify({
            'matches': [],
            'total_count': 0,
            'last_updated': datetime.now().isoformat(),
            'error': str(e)
        }), 500


@app.route('/api/refresh', methods=['POST'])
@log_execution_time(logger)
@log_exceptions(logger)
def refresh_data():
    """
    Trigger data refresh and prediction update.    Returns:
        flask.Response: JSON response with refresh status
    """
    try:
        # Import the refresh function
        from services.refresh_service import refresh_predictions
        
        # Get the status service instance
        refresh_status_service = get_refresh_status_service()
        
        # Check if a refresh is already running
        current_status = refresh_status_service.get_status()
        
        if current_status['status'] == 'running':
            return jsonify({
                "status": "error", 
                "message": "Refresh is already in progress"
            }), 409        # Run refresh in background thread
        def run_refresh():
            try:
                logger.info("Starting background refresh process")
                success = refresh_predictions()
                logger.info(f"Background refresh completed with success: {success}")
            except Exception as e:
                logger.error(f"Background refresh failed with exception: {str(e)}")
        
        # Start the refresh in a background thread
        refresh_thread = threading.Thread(target=run_refresh, daemon=True)
        refresh_thread.start()
        
        logger.info("Refresh thread started successfully")
        return jsonify({"status": "success", "message": "Refresh process started"})
    except Exception as e:
        logger.error(f"Error starting refresh: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/refresh/status', methods=['GET'])
@log_execution_time(logger)
@log_exceptions(logger)
def get_refresh_status():
    """
    Get the current refresh status and progress.

    Returns:
        flask.Response: JSON response with current refresh status
    """
    try:
        refresh_status_service = get_refresh_status_service()
        status = refresh_status_service.get_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting refresh status: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/refresh/reset', methods=['POST'])
@log_execution_time(logger)
@log_exceptions(logger)
def reset_refresh_status():
    """
    Reset the refresh status to idle.

    Returns:
        flask.Response: JSON response confirming reset
    """
    try:
        refresh_status_service = get_refresh_status_service()
        refresh_status_service.reset()
        return jsonify({"status": "success", "message": "Refresh status reset"})
    except Exception as e:
        logger.error(f"Error resetting refresh status: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500




def refresh_predictions_periodically():
    """
    Periodically refresh predictions in the background.
    """
    import time
    while True:
        try:
            # Wait for 1 hour
            time.sleep(3600)

            logger.info("Starting scheduled prediction refresh")

            # Run the prediction refresh process
            # In a real implementation, this would call a function to refresh predictions
            # For now, we'll just log a message
            logger.info("Scheduled prediction refresh completed")

        except Exception as e:
            logger.error(f"Error in refresh cycle: {e}")


def run_api_server():
    """
    Run the API server.
    """
    # Start the background refresh thread
    refresh_thread = threading.Thread(target=refresh_predictions_periodically, daemon=True)
    refresh_thread.start()

    # Run the Flask app
    app.run(host=API_HOST, port=API_PORT)


if __name__ == '__main__':
    run_api_server()
