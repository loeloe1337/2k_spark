"""
API server for the 2K Flash application.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add the parent directory to the Python path so we can import our modules
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import (
    API_HOST, API_PORT, CORS_ORIGINS,
    UPCOMING_MATCHES_FILE, PLAYER_STATS_FILE
)
from config.logging_config import get_api_logger
from utils.logging import log_execution_time, log_exceptions
from services.live_scores_service import LiveScoresService
from services.data_service import DataService

# Initialize FastAPI app
app = FastAPI(title="2K Flash API", description="API server for the 2K Flash application", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger and data service
logger = get_api_logger()
data_service = DataService()


@app.get('/api/upcoming-matches')
@log_execution_time(logger)
@log_exceptions(logger)
def get_upcoming_matches():
    """
    Get upcoming matches.

    Returns:
        list: JSON response with upcoming matches
    """
    try:
        # First try to get from file
        upcoming_matches = data_service.get_upcoming_matches()
        
        # If no data in file, try to fetch fresh data
        if not upcoming_matches:
            logger.info("No upcoming matches found in file, fetching fresh data...")
            upcoming_matches = data_service.fetch_upcoming_matches()
            
        # If still no data, return empty list
        if not upcoming_matches:
            logger.warning("No upcoming matches available")
            return []

        logger.info(f"Returning {len(upcoming_matches)} upcoming matches")
        return upcoming_matches
    except Exception as e:
        logger.error(f"Error retrieving upcoming matches: {str(e)}")
        # Return empty list on error
        return []


@app.get('/api/player-stats')
@log_execution_time(logger)
@log_exceptions(logger)
def get_player_stats():
    """
    Get player statistics.

    Returns:
        list: JSON response with player statistics
    """
    try:
        # First try to get from file
        player_stats = data_service.get_player_stats()
        
        # If no data in file, try to calculate from match history
        if not player_stats:
            logger.info("No player stats found in file, calculating from match history...")
            # First ensure we have match history
            matches = data_service.get_match_history()
            if not matches:
                logger.info("No match history found, fetching fresh data...")
                matches = data_service.fetch_match_history()
            
            if matches:
                player_stats = data_service.calculate_player_stats(matches)
        
        # If still no data, return empty list
        if not player_stats:
            logger.warning("No player statistics available")
            return []

        # Convert to list of player stats
        stats_list = []
        for player_id, stats in player_stats.items():
            # Add player ID to stats
            stats['id'] = player_id
            stats_list.append(stats)

        # Sort by win rate (descending)
        stats_list.sort(key=lambda x: x.get('win_rate', 0), reverse=True)

        logger.info(f"Returning statistics for {len(stats_list)} players")
        return stats_list
    except Exception as e:
        logger.error(f"Error retrieving player statistics: {str(e)}")
        # Return empty list on error
        return []


@app.get('/api/live-scores')
@log_execution_time(logger)
@log_exceptions(logger)
def get_live_scores():
    """
    Get live NBA scores.
    
    Returns:
        dict: JSON response with live scores
    """
    try:
        live_service = LiveScoresService()
        live_data = live_service.get_live_matches()
        
        logger.info(f"Returning {live_data['total_count']} live matches")
        return live_data
    except Exception as e:
        logger.error(f"Error retrieving live scores: {str(e)}")
        raise HTTPException(status_code=500, detail={
            'matches': [],
            'total_count': 0,
            'last_updated': datetime.now().isoformat(),
            'error': str(e)
        })


def run_api_server():
    """
    Run the API server.
    """
    import uvicorn
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == '__main__':
    run_api_server()
