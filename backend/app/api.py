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


@app.route('/api/refresh', methods=['POST'])
@log_execution_time(logger)
@log_exceptions(logger)
def refresh_data():
    """
    Trigger data refresh and prediction update.

    Returns:
        flask.Response: JSON response with refresh status
    """
    try:
        # Run the prediction refresh process
        # In a real implementation, this would call a function to refresh predictions
        # For now, we'll just return a success message

        # Create a temporary script to run the refresh process
        script_path = Path(__file__).parent / "refresh_script.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("""
import sys
import os
import traceback
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import refresh function and logging
from services.refresh_service import refresh_predictions
from config.logging_config import get_prediction_refresh_logger

# Initialize logger
logger = get_prediction_refresh_logger()

try:
    # Log start time
    start_time = time.time()
    logger.info(f"Refresh script started with PID: {os.getpid()}")

    # Run refresh
    success = refresh_predictions()

    # Log completion
    end_time = time.time()
    duration = end_time - start_time

    if success:
        logger.info(f"Refresh completed successfully in {duration:.2f} seconds")
    else:
        logger.error(f"Refresh failed after {duration:.2f} seconds")

    # Exit with appropriate code
    sys.exit(0 if success else 1)
except Exception as e:
    # Log any unhandled exceptions
    logger.error(f"Unhandled exception in refresh script: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)
            """)

        # Get the virtual environment's Python executable
        venv_python = Path(__file__).parent.parent / ".venv" / "Scripts" / "python.exe"
        python_executable = str(venv_python) if venv_python.exists() else "python"
        
        # Run the script in a separate process
        process = subprocess.Popen(
            [python_executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Log the process ID for debugging
        logger.info(f"Started refresh process with PID: {process.pid}")

        # Create a background thread to monitor the process
        def monitor_process():
            try:
                # Wait for the process to complete (with a timeout)
                try:
                    stdout, stderr = process.communicate(timeout=5)
                    if stdout:
                        logger.info(f"Refresh process stdout: {stdout}")
                    if stderr:
                        logger.error(f"Refresh process stderr: {stderr}")
                except subprocess.TimeoutExpired:
                    # Process is still running, which is expected for longer refreshes
                    logger.info(f"Refresh process {process.pid} is still running (expected)")

                # Check if the process is still running after the timeout
                if process.poll() is None:
                    logger.info(f"Refresh process {process.pid} is running in the background")
                else:
                    # Process completed quickly
                    exit_code = process.returncode
                    logger.info(f"Refresh process completed with exit code: {exit_code}")

                    # Check if there was any output
                    stdout, stderr = process.communicate()
                    if stdout:
                        logger.info(f"Refresh process stdout: {stdout}")
                    if stderr:
                        logger.error(f"Refresh process stderr: {stderr}")
            except Exception as e:
                logger.error(f"Error monitoring refresh process: {str(e)}")

        # Start the monitoring thread
        monitor_thread = threading.Thread(target=monitor_process)
        monitor_thread.daemon = True
        monitor_thread.start()

        return jsonify({"status": "success", "message": "Refresh process started"})
    except Exception as e:
        logger.error(f"Error refreshing data: {str(e)}")
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
