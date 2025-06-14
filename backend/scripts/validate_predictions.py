#!/usr/bin/env python3
"""
Script to validate predictions against completed match results.

This script fetches completed matches and validates existing predictions
to determine if they were correct or incorrect.
"""

import sys
import logging
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.prediction_validation_service import validate_predictions

def main():
    """
    Main function to run prediction validation.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting prediction validation script")
    
    try:
        # Run prediction validation
        success = validate_predictions()
        
        if success:
            logger.info("Prediction validation completed successfully")
            print("✅ Prediction validation completed successfully")
        else:
            logger.error("Prediction validation failed")
            print("❌ Prediction validation failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during prediction validation: {str(e)}")
        print(f"❌ Error during prediction validation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()