#!/usr/bin/env python3
"""
Script to clean up duplicate entries in prediction history.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add the parent directory to the Python path so we can import our modules
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from config.settings import PREDICTION_HISTORY_FILE
from config.logging_config import get_prediction_refresh_logger

logger = get_prediction_refresh_logger()

def cleanup_duplicate_history():
    """
    Remove duplicate entries from prediction history based on fixture ID.
    Keeps the first occurrence of each fixture.
    """
    if not Path(PREDICTION_HISTORY_FILE).exists():
        logger.warning(f"Prediction history file {PREDICTION_HISTORY_FILE} does not exist")
        return
    
    logger.info("Loading prediction history for cleanup")
    
    try:
        with open(PREDICTION_HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Error loading prediction history: {str(e)}")
        return
    
    original_count = len(history)
    logger.info(f"Original history contains {original_count} entries")
    
    # Track seen fixture IDs and keep only first occurrence
    seen_fixtures = set()
    cleaned_history = []
    duplicates_removed = 0
    
    for prediction in history:
        fixture_id = prediction.get('fixtureId')
        
        if fixture_id is None:
            # Keep entries without fixture ID (shouldn't happen but be safe)
            cleaned_history.append(prediction)
            logger.warning("Found prediction without fixture ID")
        elif fixture_id not in seen_fixtures:
            # First occurrence of this fixture
            seen_fixtures.add(fixture_id)
            cleaned_history.append(prediction)
        else:
            # Duplicate - skip it
            duplicates_removed += 1
    
    final_count = len(cleaned_history)
    logger.info(f"Cleaned history contains {final_count} entries")
    logger.info(f"Removed {duplicates_removed} duplicate entries")
    
    if duplicates_removed > 0:
        # Create backup of original file
        backup_file = PREDICTION_HISTORY_FILE.with_suffix('.backup.json')
        logger.info(f"Creating backup at {backup_file}")
        
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return
        
        # Save cleaned history
        try:
            with open(PREDICTION_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(cleaned_history, f, indent=2)
            logger.info(f"Successfully cleaned prediction history")
        except Exception as e:
            logger.error(f"Error saving cleaned history: {str(e)}")
            return
    else:
        logger.info("No duplicates found - no cleanup needed")
    
    # Show statistics by fixture ID
    fixture_stats = defaultdict(int)
    for prediction in cleaned_history:
        fixture_id = prediction.get('fixtureId', 'unknown')
        fixture_stats[fixture_id] += 1
    
    logger.info(f"Final history contains {len(fixture_stats)} unique fixtures")

def main():
    """
    Main function.
    """
    logger.info("Starting prediction history cleanup")
    cleanup_duplicate_history()
    logger.info("Prediction history cleanup completed")

if __name__ == "__main__":
    main()