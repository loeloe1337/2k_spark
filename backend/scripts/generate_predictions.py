"""
Script to generate predictions for upcoming matches.
"""

import json
import sys
from pathlib import Path
import random
from datetime import datetime

# Add the parent directory to the Python path so we can import our modules
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from config.settings import (
    PLAYER_STATS_FILE, UPCOMING_MATCHES_FILE, PREDICTIONS_FILE, PREDICTION_HISTORY_FILE
)
from config.logging_config import get_prediction_refresh_logger

logger = get_prediction_refresh_logger()

def load_player_stats():
    """
    Load player statistics from file.
    
    Returns:
        dict: Player statistics dictionary
    """
    with open(PLAYER_STATS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_upcoming_matches():
    """
    Load upcoming matches from file.
    
    Returns:
        list: List of upcoming match data dictionaries
    """
    with open(UPCOMING_MATCHES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_predictions(player_stats, upcoming_matches):
    """
    Generate predictions for upcoming matches.
    
    Args:
        player_stats (dict): Player statistics dictionary
        upcoming_matches (list): List of upcoming match data dictionaries
        
    Returns:
        list: List of match predictions
    """
    predictions = []
    
    for match in upcoming_matches:
        home_player_id = str(match['homePlayer']['id'])
        away_player_id = str(match['awayPlayer']['id'])
        
        # Get player stats if available
        home_player_stats = player_stats.get(home_player_id, {})
        away_player_stats = player_stats.get(away_player_id, {})
        
        # Get win rates
        home_win_rate = home_player_stats.get('win_rate', 0.5)
        away_win_rate = away_player_stats.get('win_rate', 0.5)
        
        # Normalize win rates
        total_win_rate = home_win_rate + away_win_rate
        if total_win_rate > 0:
            home_win_probability = home_win_rate / total_win_rate
            away_win_probability = away_win_rate / total_win_rate
        else:
            home_win_probability = 0.5
            away_win_probability = 0.5
        
        # Determine predicted winner
        predicted_winner = "home" if home_win_probability > away_win_probability else "away"
        confidence = max(home_win_probability, away_win_probability)
        
        # Generate score prediction
        home_avg_score = home_player_stats.get('avg_score', 60)
        away_avg_score = away_player_stats.get('avg_score', 60)
        
        # Add some randomness to the scores
        home_score = max(0, round(home_avg_score + random.uniform(-5, 5)))
        away_score = max(0, round(away_avg_score + random.uniform(-5, 5)))
        
        # Create prediction object
        prediction = {
            "fixtureId": match.get('id'),
            "homePlayer": match.get('homePlayer'),
            "awayPlayer": match.get('awayPlayer'),
            "homeTeam": match.get('homeTeam'),
            "awayTeam": match.get('awayTeam'),
            "fixtureStart": match.get('fixtureStart'),
            "prediction": {
                "home_win_probability": round(home_win_probability, 2),
                "away_win_probability": round(away_win_probability, 2),
                "predicted_winner": predicted_winner,
                "confidence": round(confidence, 2)
            },
            "score_prediction": {
                "home_score": home_score,
                "away_score": away_score,
                "total_score": home_score + away_score,
                "score_diff": home_score - away_score
            },
            "generated_at": datetime.now().isoformat()
        }
        
        predictions.append(prediction)
    
    return predictions

def save_predictions(predictions):
    """
    Save predictions to file and add to prediction history for future validation.
    
    Args:
        predictions (list): List of match predictions
    """
    # Save current predictions
    with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"Saved {len(predictions)} predictions to {PREDICTIONS_FILE}")
    
    # Add predictions to history for future validation
    add_predictions_to_history(predictions)

def add_predictions_to_history(new_predictions):
    """
    Add new predictions to the prediction history file for future validation.
    
    Args:
        new_predictions (list): List of new predictions to add
    """
    # Load existing prediction history
    existing_predictions = []
    if Path(PREDICTION_HISTORY_FILE).exists():
        try:
            with open(PREDICTION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                existing_predictions = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Error loading existing prediction history: {str(e)}")
            existing_predictions = []
    
    # Get existing fixture IDs to avoid duplicates
    existing_fixture_ids = {pred.get('fixtureId') for pred in existing_predictions}
    
    # Add new predictions that don't already exist
    added_count = 0
    for prediction in new_predictions:
        fixture_id = prediction.get('fixtureId')
        if fixture_id not in existing_fixture_ids:
            # Add fields that will be filled during validation
            prediction_with_validation_fields = prediction.copy()
            prediction_with_validation_fields.update({
                'homeScore': None,
                'awayScore': None,
                'result': None,
                'prediction_correct': None,
                'home_score_error': None,
                'away_score_error': None,
                'total_score_error': None
            })
            existing_predictions.append(prediction_with_validation_fields)
            existing_fixture_ids.add(fixture_id)
            added_count += 1
    
    # Save updated prediction history
    if added_count > 0:
        with open(PREDICTION_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_predictions, f, indent=2)
        logger.info(f"Added {added_count} new predictions to history for future validation")
    else:
        logger.info("No new predictions to add to history")

def main():
    """
    Main function.
    """
    # Load player stats
    player_stats = load_player_stats()
    logger.info(f"Loaded statistics for {len(player_stats)} players")
    
    # Load upcoming matches
    upcoming_matches = load_upcoming_matches()
    logger.info(f"Loaded {len(upcoming_matches)} upcoming matches")
    
    # Generate predictions
    predictions = generate_predictions(player_stats, upcoming_matches)
    logger.info(f"Generated predictions for {len(predictions)} matches")
    
    # Save predictions
    save_predictions(predictions)

if __name__ == "__main__":
    main()
