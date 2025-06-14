#!/usr/bin/env python3
"""
Script to clear mock prediction history and prepare for real prediction tracking.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add the parent directory to the Python path so we can import our modules
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from config.settings import PREDICTION_HISTORY_FILE
from config.logging_config import get_prediction_refresh_logger

logger = get_prediction_refresh_logger()

def backup_existing_history():
    """
    Create a backup of the existing prediction history before clearing it.
    """
    if Path(PREDICTION_HISTORY_FILE).exists():
        backup_file = PREDICTION_HISTORY_FILE.with_suffix('.backup.json')
        
        # Read existing data
        with open(PREDICTION_HISTORY_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # Save backup
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"Backed up {len(existing_data)} existing predictions to {backup_file}")
        return len(existing_data)
    else:
        logger.info("No existing prediction history file found")
        return 0

def clear_prediction_history():
    """
    Clear the prediction history file to prepare for real predictions.
    """
    # Create empty prediction history
    empty_history = []
    
    with open(PREDICTION_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(empty_history, f, indent=2)
    
    logger.info(f"Cleared prediction history file: {PREDICTION_HISTORY_FILE}")

def main():
    """
    Main function to clear mock predictions and prepare for real tracking.
    """
    logger.info("Starting mock prediction cleanup process")
    
    # Backup existing data
    backup_count = backup_existing_history()
    
    # Clear the prediction history
    clear_prediction_history()
    
    logger.info("Mock prediction cleanup completed successfully")
    logger.info("The system is now ready to track real predictions")
    logger.info("Next steps:")
    logger.info("1. Generate new predictions using generate_predictions.py")
    logger.info("2. Wait for matches to complete")
    logger.info("3. Run validate_predictions.py to validate real predictions")
    
    print("✅ Mock prediction cleanup completed successfully")
    print(f"📦 Backed up {backup_count} mock predictions")
    print("🎯 System is now ready to track real predictions")
    print("")
    print("Next steps:")
    print("1. Generate new predictions: python scripts/generate_predictions.py")
    print("2. Wait for matches to complete")
    print("3. Validate predictions: python scripts/validate_predictions.py")

if __name__ == "__main__":
    main()