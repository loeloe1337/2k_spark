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

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from config.settings import (
    API_HOST, API_PORT, CORS_ORIGINS,
    UPCOMING_MATCHES_FILE, PLAYER_STATS_FILE
)
from config.logging_config import get_api_logger
from utils.logging import log_execution_time, log_exceptions
from services.live_scores_service import LiveScoresService
from services.data_service import DataService
from services.enhanced_prediction_service import EnhancedMatchPredictionService
# Import CLI functionality for pipeline
from core.data.fetchers import TokenFetcher
from core.data.fetchers.match_history import MatchHistoryFetcher
from core.data.fetchers.upcoming_matches import UpcomingMatchesFetcher
from core.data.processors.player_stats import PlayerStatsProcessor

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
prediction_service = EnhancedMatchPredictionService()


@app.get('/')
def root():
    """
    Root endpoint providing basic API information.
    
    Returns:
        dict: API information
    """
    return {
        "message": "2K Flash API",
        "version": "1.0.0",
        "description": "API server for the 2K Flash application",
        "endpoints": {
            "health": "/api/health",
            "status": "/api/system-status",
            "upcoming_matches": "/api/upcoming-matches",
            "player_stats": "/api/data/player-stats",
            "run_pipeline": "/api/run-pipeline",
            "docs": "/docs"
        }
    }


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


@app.post('/api/data/fetch-matches')
@log_execution_time(logger)
@log_exceptions(logger)
def fetch_match_history():
    """
    Fetch and store historical match data.
    
    Returns:
        dict: Fetch result with count of matches processed
    """
    try:
        logger.info("Starting match history fetch")
        
        # Fetch match history using the data service
        result = data_service.fetch_match_history()
        
        if result:
            # If we have database connection, also save to database
            if data_service.supabase_service.is_connected():
                # Read the fetched data from JSON file and save to database
                import json
                from config.settings import MATCH_HISTORY_FILE
                
                if MATCH_HISTORY_FILE.exists():
                    with open(MATCH_HISTORY_FILE, 'r') as f:
                        matches_data = json.load(f)
                    
                    # Save to database
                    saved = data_service.save_match_history_to_db(matches_data)
                    
                    return {
                        "status": "success",
                        "message": f"Fetched {len(matches_data)} matches",
                        "saved_to_database": saved,
                        "timestamp": datetime.now().isoformat()
                    }
            
            return {
                "status": "success", 
                "message": "Match history fetched to file",
                "database_save": "skipped - no database connection",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "failed",
                "message": "Failed to fetch match history",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error fetching match history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/data/calculate-stats')
@log_execution_time(logger) 
@log_exceptions(logger)
def calculate_player_statistics():
    """
    Calculate player statistics from match data.
    
    Returns:
        dict: Calculation result with count of players processed
    """
    try:
        logger.info("Starting player statistics calculation")
        
        # Calculate player stats using the data service
        result = data_service.calculate_player_stats()
        
        if result:
            # If we have database connection, also save to database
            if data_service.supabase_service.is_connected():
                # Read the calculated stats from JSON file and save to database
                import json
                from config.settings import PLAYER_STATS_FILE
                
                if PLAYER_STATS_FILE.exists():
                    with open(PLAYER_STATS_FILE, 'r') as f:
                        stats_data = json.load(f)
                    
                    # Save to database
                    saved = data_service.save_player_stats_to_db(stats_data)
                    
                    return {
                        "status": "success",
                        "message": f"Calculated stats for {len(stats_data)} players",
                        "saved_to_database": saved,
                        "timestamp": datetime.now().isoformat()
                    }
            
            return {
                "status": "success",
                "message": "Player statistics calculated to file", 
                "database_save": "skipped - no database connection",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "failed",
                "message": "Failed to calculate player statistics",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error calculating player statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/data/refresh-all')
@log_execution_time(logger)
@log_exceptions(logger) 
def refresh_all_data():
    """
    Refresh all data: fetch matches, calculate stats, and update database.
    
    Returns:
        dict: Complete refresh result
    """
    try:
        logger.info("Starting complete data refresh")
        results = {}
        
        # Step 1: Fetch match history
        match_result = data_service.fetch_match_history()
        results["match_fetch"] = "success" if match_result else "failed"
        
        # Step 2: Calculate player stats (only if matches were fetched)
        if match_result:
            stats_result = data_service.calculate_player_stats()
            results["stats_calculation"] = "success" if stats_result else "failed"
            
            # Step 3: Save to database if connected
            if data_service.supabase_service.is_connected():
                import json
                from config.settings import MATCH_HISTORY_FILE, PLAYER_STATS_FILE
                
                # Save matches to database
                if MATCH_HISTORY_FILE.exists():
                    with open(MATCH_HISTORY_FILE, 'r') as f:
                        matches_data = json.load(f)
                    results["matches_saved"] = data_service.save_match_history_to_db(matches_data)
                
                # Save stats to database
                if PLAYER_STATS_FILE.exists() and stats_result:
                    with open(PLAYER_STATS_FILE, 'r') as f:
                        stats_data = json.load(f)
                    results["stats_saved"] = data_service.save_player_stats_to_db(stats_data)
        
        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during complete data refresh: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/ml/train')
def train_prediction_model(days_back: int = 60, min_matches_per_player: int = 5):
    """
    Train the match prediction model with historical data.
    
    Args:
        days_back: Number of days of match history to use for training
        min_matches_per_player: Minimum matches required for a player to be included
        
    Returns:
        dict: Training results and metrics
    """
    try:
        logger.info(f"Training prediction model with {days_back} days of data")
        
        # Prepare training data
        training_df = prediction_service.prepare_training_data(
            days_back=days_back,
            min_matches_per_player=min_matches_per_player
        )
          # Train the model with versioning
        version, metrics = prediction_service.train_model_with_versioning(
            training_df=training_df, 
            auto_activate=True,
            performance_threshold=0.6
        )
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "model_version": version,
            "training_samples": len(training_df),
            "metrics": {
                "winner_accuracy": round(metrics['val_winner_accuracy'], 3),
                "home_score_mae": round(metrics['val_home_mae'], 2),
                "away_score_mae": round(metrics['val_away_mae'], 2),
                "total_score_mae": round(metrics['val_total_mae'], 2)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training prediction model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/ml/predictions')
def get_match_predictions():
    """
    Get predictions for upcoming matches.
    
    Returns:
        dict: Match predictions with winner and score predictions
    """
    try:
        logger.info("Generating match predictions")
          # Generate predictions using best model
        predictions_df = prediction_service.predict_with_best_model(load_model=True)
        
        if predictions_df.empty:
            return {
                "status": "success",
                "message": "No upcoming matches to predict",
                "predictions": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Get prediction summary
        summary = prediction_service.get_prediction_summary(predictions_df)
        
        return {
            "status": "success",
            "message": f"Generated predictions for {summary['total_matches']} matches",
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/ml/model-performance')
def get_model_performance(test_days: int = 7):
    """
    Evaluate model performance on recent matches.
    
    Args:
        test_days: Number of recent days to use for evaluation
        
    Returns:
        dict: Model performance metrics
    """
    try:
        logger.info(f"Evaluating model performance on last {test_days} days")
        
        # Evaluate model
        metrics = prediction_service.evaluate_model(test_days_back=test_days)
        
        return {
            "status": "success",
            "evaluation_period_days": test_days,
            "metrics": {
                "test_samples": metrics['test_samples'],
                "winner_accuracy": round(metrics['winner_accuracy'], 3),
                "home_score_mae": round(metrics['home_score_mae'], 2),
                "away_score_mae": round(metrics['away_score_mae'], 2),
                "total_score_mae": round(metrics['total_score_mae'], 2)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/ml/feature-importance')
def get_feature_importance():
    """
    Get feature importance from the trained model.
    
    Returns:
        dict: Feature importance rankings
    """
    try:
        logger.info("Getting feature importance")
        
        # Get feature importance
        importance_df = prediction_service.get_feature_importance()
        
        # Convert to list of dictionaries
        importance_list = importance_df.to_dict('records')
        
        return {
            "status": "success",
            "feature_count": len(importance_list),
            "features": importance_list,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/ml/retrain')
def retrain_model(days_back: int = 60):
    """
    Retrain the model with fresh data.
    
    Args:
        days_back: Number of days of history to include in retraining
        
    Returns:
        dict: Retraining results
    """
    try:
        logger.info(f"Retraining model with {days_back} days of fresh data")
        
        # Retrain model
        metrics = prediction_service.retrain_with_new_data(days_back=days_back)
        
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "metrics": {
                "winner_accuracy": round(metrics['val_winner_accuracy'], 3),
                "home_score_mae": round(metrics['val_home_mae'], 2),
                "away_score_mae": round(metrics['val_away_mae'], 2),
                "total_score_mae": round(metrics['val_total_mae'], 2)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/ml/predictions/summary')
def get_predictions_summary():
    """
    Get a simplified summary of current predictions.
    
    Returns:
        dict: Simplified prediction summary
    """
    try:
        # Try to load existing predictions first
        predictions_file = Path("output/match_predictions.csv")
        
        if predictions_file.exists():
            import pandas as pd
            predictions_df = pd.read_csv(predictions_file)
            summary = prediction_service.get_prediction_summary(predictions_df)
        else:
            # Generate fresh predictions
            predictions_df = prediction_service.predict_upcoming_matches(load_model=True)
            summary = prediction_service.get_prediction_summary(predictions_df)
        
        # Create simplified summary
        simplified_predictions = []
        for pred in summary.get('predictions', []):
            simplified_predictions.append({
                'match': f"{pred['home_player']} vs {pred['away_player']}",
                'predicted_winner': pred['predicted_winner'],
                'confidence': pred['confidence'],
                'predicted_total_score': pred['predicted_scores']['total']
            })
        
        return {
            "status": "success",
            "total_matches": summary.get('total_matches', 0),
            "average_confidence": round(summary.get('average_confidence', 0), 3),
            "predictions": simplified_predictions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting predictions summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/ml/models')
def list_model_versions():
    """
    List all available model versions with their performance metrics.
    
    Returns:
        dict: List of model versions and their metrics
    """
    try:
        logger.info("Listing model versions")
        
        versions_df = prediction_service.list_model_versions()
        
        if versions_df.empty:
            return {
                "status": "success",
                "message": "No trained models found",
                "models": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Convert to list of dictionaries
        models_list = versions_df.to_dict('records')
        
        return {
            "status": "success",
            "model_count": len(models_list),
            "models": models_list,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing model versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/ml/models/{version}/activate')
def activate_model_version(version: str):
    """
    Activate a specific model version.
    
    Args:
        version: Model version to activate (e.g., v1.0.1)
        
    Returns:
        dict: Activation result
    """
    try:
        logger.info(f"Activating model version: {version}")
        
        prediction_service.activate_model_version(version)
        
        return {
            "status": "success",
            "message": f"Successfully activated model version: {version}",
            "active_version": version,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.error(f"Error activating model version {version}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error activating model version {version}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/ml/models/compare/{version1}/{version2}')
def compare_model_versions(version1: str, version2: str):
    """
    Compare performance between two model versions.
    
    Args:
        version1: First model version (e.g., v1.0.1)
        version2: Second model version (e.g., v1.0.2)
        
    Returns:
        dict: Comparison results
    """
    try:
        logger.info(f"Comparing model versions: {version1} vs {version2}")
        
        comparison = prediction_service.compare_model_versions(version1, version2)
        
        return {
            "status": "success",
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.error(f"Error comparing model versions: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error comparing model versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/ml/models/active')
def get_active_model():
    """
    Get information about the currently active model.
    
    Returns:
        dict: Active model information
    """
    try:
        logger.info("Getting active model information")
        
        active_version = prediction_service._get_active_model_version()
        
        if not active_version:
            return {
                "status": "success",
                "message": "No active model found",
                "active_model": None,
                "timestamp": datetime.now().isoformat()
            }
        
        # Get model metadata
        metadata = prediction_service._load_model_metadata(active_version)
        
        return {
            "status": "success",
            "active_model": {
                "version": active_version,
                "training_date": metadata.get("training_date", "Unknown"),
                "performance_metrics": metadata.get("performance_metrics", {}),
                "training_info": metadata.get("training_info", {})
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting active model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Pydantic models for request bodies
class PipelineRequest(BaseModel):
    """
    Request model for the pipeline endpoint.
    """
    train_new_model: bool = False
    refresh_token: bool = False
    history_days: int = 90
    training_days: int = 60
    min_matches: int = 5


@app.post('/api/run-pipeline')
@log_execution_time(logger)
@log_exceptions(logger)
def run_pipeline_endpoint(request: PipelineRequest):
    """
    Run the complete prediction pipeline with fresh data via API.
    
    This endpoint executes the full workflow:
    1. Fetch authentication token (if needed)
    2. Fetch latest match history
    3. Fetch latest player statistics
    4. Fetch upcoming matches
    5. Train new model (if requested)
    6. Generate predictions using the best model
    
    Args:
        request (PipelineRequest): Pipeline configuration parameters
        
    Returns:
        dict: Pipeline execution results and summary
    """
    try:
        pipeline_results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "steps_completed": [],
            "steps_failed": [],
            "summary": {},
            "errors": []
        }
        
        logger.info("🚀 Starting API Pipeline execution...")
        
        # Step 1: Authentication token
        try:
            logger.info("📝 Step 1/6: Checking authentication token...")
            token_fetcher = TokenFetcher()
            token = token_fetcher.get_token(force_refresh=request.refresh_token)
            pipeline_results["steps_completed"].append("authentication_token")
            logger.info("✅ Authentication token ready")
        except Exception as e:
            error_msg = f"Token fetch failed: {str(e)}"
            pipeline_results["steps_failed"].append("authentication_token")
            pipeline_results["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Step 2: Fetch match history
        try:
            logger.info("📊 Step 2/6: Fetching latest match history...")
            match_fetcher = MatchHistoryFetcher(days_back=request.history_days)
            matches = match_fetcher.fetch_match_history(save_to_file=True)
            pipeline_results["steps_completed"].append("match_history")
            pipeline_results["summary"]["matches_fetched"] = len(matches) if matches else 0
            logger.info("✅ Match history updated")
        except Exception as e:
            error_msg = f"Match history fetch failed: {str(e)}"
            pipeline_results["steps_failed"].append("match_history")
            pipeline_results["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Step 3: Calculate player statistics
        try:
            logger.info("🏀 Step 3/6: Calculating player statistics...")
            stats_processor = PlayerStatsProcessor()
            match_fetcher = MatchHistoryFetcher()
            matches = match_fetcher.load_from_file()
            player_stats = stats_processor.calculate_player_stats(matches, save_to_file=True)
            pipeline_results["steps_completed"].append("player_statistics")
            pipeline_results["summary"]["players_processed"] = len(player_stats) if player_stats else 0
            logger.info("✅ Player statistics updated")
        except Exception as e:
            error_msg = f"Player stats calculation failed: {str(e)}"
            pipeline_results["steps_failed"].append("player_statistics")
            pipeline_results["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Step 4: Fetch upcoming matches
        try:
            logger.info("🔮 Step 4/6: Fetching upcoming matches...")
            upcoming_fetcher = UpcomingMatchesFetcher()
            upcoming_matches = upcoming_fetcher.fetch_upcoming_matches(save_to_file=True)
            pipeline_results["steps_completed"].append("upcoming_matches")
            pipeline_results["summary"]["upcoming_matches"] = len(upcoming_matches) if upcoming_matches else 0
            logger.info("✅ Upcoming matches updated")
        except Exception as e:
            error_msg = f"Upcoming matches fetch failed: {str(e)}"
            pipeline_results["steps_failed"].append("upcoming_matches")
            pipeline_results["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Step 5: Train new model (if requested)
        if request.train_new_model:
            try:
                logger.info("🤖 Step 5/6: Training new prediction model...")
                prediction_service = EnhancedMatchPredictionService()
                
                # Prepare training data
                training_df = prediction_service.prepare_training_data(
                    days_back=request.training_days,
                    min_matches_per_player=request.min_matches
                )
                
                # Train new model
                version, metrics = prediction_service.train_model_with_versioning(
                    training_df=training_df, 
                    auto_activate=True,
                    performance_threshold=0.6
                )
                
                pipeline_results["steps_completed"].append("model_training")
                pipeline_results["summary"]["model_version"] = version
                pipeline_results["summary"]["model_accuracy"] = metrics.get('val_winner_accuracy', 0)
                pipeline_results["summary"]["home_mae"] = metrics.get('val_home_mae', 0)
                pipeline_results["summary"]["away_mae"] = metrics.get('val_away_mae', 0)
                pipeline_results["summary"]["training_samples"] = len(training_df)
                
                logger.info(f"✅ New model trained: {version}")
                logger.info(f"   Accuracy: {metrics.get('val_winner_accuracy', 0):.1%}")
                
            except Exception as e:
                error_msg = f"Model training failed: {str(e)}"
                pipeline_results["steps_failed"].append("model_training")
                pipeline_results["errors"].append(error_msg)
                logger.error(f"❌ {error_msg}")
                # Continue with existing model instead of failing
                logger.info("⚠️  Continuing with existing model...")
        else:
            logger.info("⏭️  Step 5/6: Skipping model training")
            pipeline_results["steps_completed"].append("model_training_skipped")
        
        # Step 6: Generate predictions
        try:
            logger.info("🎯 Step 6/6: Generating predictions...")
            prediction_service = EnhancedMatchPredictionService()
            
            predictions_df = prediction_service.predict_upcoming_matches(load_model=True)
            
            if not predictions_df.empty:
                summary = prediction_service.get_prediction_summary(predictions_df)
                pipeline_results["steps_completed"].append("predictions")
                pipeline_results["summary"]["total_predictions"] = summary["total_matches"]
                pipeline_results["summary"]["average_confidence"] = summary["average_confidence"]
                pipeline_results["summary"]["high_confidence_matches"] = summary["high_confidence_matches"]
            else:
                pipeline_results["summary"]["total_predictions"] = 0
                pipeline_results["summary"]["message"] = "No upcoming matches to predict"
            
            logger.info("✅ Predictions generated successfully!")
            
        except Exception as e:
            error_msg = f"Prediction generation failed: {str(e)}"
            pipeline_results["steps_failed"].append("predictions")
            pipeline_results["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Final summary
        pipeline_results["summary"]["total_steps"] = 6
        pipeline_results["summary"]["completed_steps"] = len(pipeline_results["steps_completed"])
        pipeline_results["summary"]["failed_steps"] = len(pipeline_results["steps_failed"])
        
        logger.info("🎉 API Pipeline completed successfully!")
        
        return pipeline_results
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        error_msg = f"Pipeline failed with unexpected error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


def run_api_server():
    """
    Run the API server.
    """
    import uvicorn
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == '__main__':
    run_api_server()
