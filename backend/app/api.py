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


@app.get('/api/health')
def health_check():
    """
    Health check endpoint for monitoring and Docker health checks.
    
    Returns:
        dict: Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "2K Flash API",
        "version": "1.0.0"
    }


@app.get('/api/system-status')
@log_execution_time(logger)
@log_exceptions(logger)
def get_system_status():
    """
    Get comprehensive system status including database connectivity.
    
    Returns:
        dict: System status information
    """
    try:
        status = data_service.get_system_status()
        status.update({
            "timestamp": datetime.now().isoformat(),
            "service": "2K Flash API",
            "version": "1.0.0"
        })
        return status
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@app.get('/api/system/status')
@log_execution_time(logger)
@log_exceptions(logger)
def system_status():
    """
    Get system status including database connectivity.
    
    Returns:
        dict: System status information
    """
    status = {
        "api": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "2K Flash API",
        "version": "1.0.0",
        "database": {
            "connected": data_service.supabase_service.is_connected(),
            "stats": {}
        }
    }
    
    if data_service.supabase_service.is_connected():
        try:
            status["database"]["stats"] = data_service.supabase_service.get_database_stats()
            status["database"]["test_connection"] = data_service.supabase_service.test_connection()
        except Exception as e:
            status["database"]["error"] = str(e)
    
    return status


@app.post('/api/system/setup-database')
@log_execution_time(logger)
@log_exceptions(logger)
def setup_database():
    """
    Setup database schema and initial data.
    
    Returns:
        dict: Setup result
    """
    if not data_service.supabase_service.is_connected():
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        # This is a simple endpoint - in production you'd want proper authentication
        # For now, we'll just test the connection and return the setup status
        connection_test = data_service.supabase_service.test_connection()
        
        return {
            "status": "success" if connection_test else "failed",
            "message": "Database connection verified" if connection_test else "Database connection failed",
            "timestamp": datetime.now().isoformat(),
            "next_steps": [
                "1. Go to your Supabase project dashboard",
                "2. Navigate to SQL Editor",
                "3. Run the schema.sql file to create tables",
                "4. Optionally run the migration script to import existing data"
            ]
        }
    except Exception as e:
        logger.error(f"Error setting up database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/upcoming-matches')
@log_execution_time(logger)
@log_exceptions(logger)
def get_upcoming_matches():
    """
    Get upcoming matches from database with file fallback.

    Returns:
        list: JSON response with upcoming matches
    """
    try:
        # Try to get from database first, fallback to file
        upcoming_matches = data_service.get_data_from_database_or_file('upcoming')
        
        # If no data found, try to fetch fresh data
        if not upcoming_matches:
            logger.info("No upcoming matches found, fetching fresh data...")
            upcoming_matches = data_service.fetch_and_save_upcoming_matches()
            
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
    Get player statistics from database with file fallback.

    Returns:
        list: JSON response with player statistics
    """
    try:
        # Try to get from database first, fallback to file
        player_stats = data_service.get_data_from_database_or_file('stats')
        
        # If no data found, try to calculate from match history
        if not player_stats:
            logger.info("No player stats found, calculating from match history...")
            player_stats = data_service.calculate_and_save_player_stats()
        
        # If still no data, return empty list
        if not player_stats:
            logger.warning("No player statistics available")
            return []

        # If data is already in list format from database, return as is
        if isinstance(player_stats, list):
            logger.info(f"Returning {len(player_stats)} player statistics")
            return player_stats

        # Convert dict format to list if needed
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
