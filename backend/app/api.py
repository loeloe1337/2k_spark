"""
API server for the 2K Flash application.
"""

import json
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
from uuid import uuid4

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
from services.supabase_service import SupabaseService
from services.job_service import job_service, JobType, JobStatus
from services.model_training_handler import model_training_handler
from services.health_monitor import health_monitor
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
supabase_service = SupabaseService()


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
            "player_stats": "/api/player-stats",
            "run_pipeline": "/api/run-pipeline",
            "pipeline_status": "/api/pipeline-status",
            "pipeline_results": "/api/pipeline-results",
            "predictions": "/api/ml/predictions",
            "predictions_summary": "/api/ml/predictions/summary",
            "model_training": "/api/ml/train",
            "job_management": "/api/jobs",
            "system_health": "/api/system/health",
            "docs": "/docs"
        },
        "job_system": {
            "description": "All long-running operations use the job system for background processing",
            "supported_job_types": [
                "model_training",
                "data_pipeline", 
                "quick_data_refresh",
                "prediction_generation"
            ],
            "monitoring": {
                "job_status": "/api/jobs/{job_id}",
                "job_list": "/api/jobs",
                "system_health": "/api/system/health"
            }
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
        # If data is already in list format from database, return only a sample for logging and response
        if isinstance(player_stats, list):
            sample = player_stats[:10]  # Only return first 10 for preview
            logger.info(f"Returning {len(player_stats)} player statistics. Sample: {sample}")
            return player_stats[:50]  # Only return first 50 players in API response
        # Convert dict format to list if needed
        stats_list = []
        for player_id, stats in player_stats.items():
            stats['id'] = player_id
            stats_list.append(stats)
        stats_list.sort(key=lambda x: x.get('win_rate', 0), reverse=True)
        sample = stats_list[:10]
        logger.info(f"Returning statistics for {len(stats_list)} players. Sample: {sample}")
        return stats_list[:50]  # Only return first 50 players in API response
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
def train_prediction_model(days_back: int = 30, min_matches_per_player: int = 3, 
                          performance_threshold: float = 0.5):
    """
    Start model training using the new job system.
    Returns a job_id for status tracking.
    """
    try:
        # Create training job payload
        payload = {
            "days_back": days_back,
            "min_matches_per_player": min_matches_per_player,
            "performance_threshold": performance_threshold,
            "requested_at": datetime.now().isoformat()
        }
        
        # Create job in queue
        job_id = job_service.create_job(
            job_type=JobType.MODEL_TRAINING,
            payload=payload,
            priority=1  # High priority for training jobs
        )
        
        if not job_id:
            raise HTTPException(status_code=500, detail="Failed to create training job")
        
        # Start job execution
        success = job_service.start_job(job_id, model_training_handler.train_model_job)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start training job")
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Model training job created and started. Check /api/ml/train-status?job_id=... for updates.",
            "estimated_duration": "5-10 minutes",
            "check_status_url": f"/api/ml/train-status?job_id={job_id}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/ml/train-status')
def get_train_status(job_id: Optional[str] = None):
    """
    Get the status of a model training job by job_id.
    """
    if not job_id:
        return {"status": "error", "message": "Missing job_id parameter."}
    
    try:
        # Get job status from new job system
        job_status = job_service.get_job_status(job_id)
        
        if not job_status:
            return {"status": "not_found", "message": f"No training job found for job_id {job_id}"}
        
        # Format response to match expected format
        response = {
            "job_id": job_status.get('job_id'),
            "status": job_status.get('status'),
            "progress": job_status.get('progress', 0),
            "message": job_status.get('error_message') if job_status.get('status') == 'failed' else "Job in progress",
            "created_at": job_status.get('created_at'),
            "updated_at": job_status.get('updated_at'),
            "started_at": job_status.get('started_at'),
            "completed_at": job_status.get('completed_at')
        }
        
        # Add result details if completed
        if job_status.get('result'):
            response.update(job_status['result'])
        
        return response
        
    except Exception as e:
        logger.error(f"Error fetching training job status: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get('/api/ml/predictions')
def get_match_predictions():
    """
    Get predictions for upcoming matches.
    
    Returns:
        dict: Match predictions with winner and score predictions
    """
    try:
        logger.info("Generating match predictions")
        predictions_df = prediction_service.predict_with_best_model(load_model=True)
        if predictions_df.empty:
            return {
                "status": "success",
                "message": "No upcoming matches to predict",
                "predictions": [],
                "timestamp": datetime.now().isoformat()
            }
        summary = prediction_service.get_prediction_summary(predictions_df)
        # --- Upsert predictions to DB ---
        if supabase_service.is_connected():
            try:
                active_model_version = prediction_service._get_active_model_version() or "v1.0.0"
                db_predictions = []
                for pred in summary.get('predictions', []):
                    home_score = float(pred['predicted_scores']['home'])
                    away_score = float(pred['predicted_scores']['away'])
                    home_win_prob = home_score / (home_score + away_score) if (home_score + away_score) > 0 else 0.5
                    db_pred = {
                        "match_id": f"{pred['home_player']}_{pred['away_player']}_{datetime.now().strftime('%Y%m%d')}",
                        "model_version": active_model_version,
                        "home_player": pred['home_player'],
                        "away_player": pred['away_player'],
                        "predicted_home_score": home_score,
                        "predicted_away_score": away_score,
                        "predicted_total_score": float(pred['predicted_scores']['total']),
                        "predicted_winner": pred['predicted_winner'],
                        "home_win_probability": home_win_prob,
                        "confidence_score": float(pred['confidence'])
                    }
                    db_predictions.append(db_pred)
                supabase_service.upsert_match_predictions(db_predictions)
            except Exception as db_e:
                logger.warning(f"[WARN] Database upsert failed for predictions: {str(db_e)}")
        return {
            "status": "success",
            "message": f"Generated predictions for {summary['total_matches']} matches",
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        return {"status": "error", "message": str(e)}


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
    return_predictions: bool = True  # New parameter to include predictions in response


@app.post('/api/run-pipeline')
@log_execution_time(logger)
@log_exceptions(logger)
def run_pipeline_endpoint(request: PipelineRequest):
    """
    Run the complete prediction pipeline using the job system.
    
    This endpoint creates a background job for the full workflow:
    1. Fetch authentication token (if needed)
    2. Fetch latest match history  
    3. Fetch latest player statistics
    4. Fetch upcoming matches
    5. Train new model (if requested)
    6. Generate predictions using the best model
    
    Args:
        request (PipelineRequest): Pipeline configuration parameters
        
    Returns:
        dict: Job information for tracking pipeline execution
    """
    try:
        logger.info("[PIPELINE] Starting pipeline job creation...")
        
        # Prepare job payload
        payload = {
            "history_days": request.history_days,
            "train_model": request.train_new_model,
            "return_predictions": request.return_predictions,
            "refresh_token": request.refresh_token,
            "training_days_back": getattr(request, 'training_days', 60),
            "min_matches_per_player": getattr(request, 'min_matches', 5),
            "performance_threshold": getattr(request, 'performance_threshold', 0.5),
            "auto_activate": getattr(request, 'auto_activate', True),
            "requested_at": datetime.now().isoformat()
        }
        
        # Create and start pipeline job
        job_id = job_service.create_and_start_job(
            job_type=JobType.DATA_PIPELINE,
            payload=payload,
            priority=2  # High priority for full pipeline
        )
        
        if not job_id:
            raise HTTPException(status_code=500, detail="Failed to create pipeline job")
        
        return {
            "status": "started",
            "job_id": job_id,
            "job_type": "data_pipeline",
            "message": "Pipeline job created and started. This will run all data fetching, processing, and optional model training.",
            "estimated_duration": "5-15 minutes depending on options",
            "monitoring": {
                "job_status_url": f"/api/jobs/{job_id}",
                "job_list_url": "/api/jobs",
                "system_health_url": "/api/system/health"
            },
            "configuration": {
                "history_days": request.history_days,
                "train_model": request.train_new_model,
                "return_predictions": request.return_predictions,
                "refresh_token": request.refresh_token
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"[PIPELINE] Error starting pipeline job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def run_pipeline_background(request_data: dict):
    """
    Run the actual pipeline in the background to avoid timeouts.
    
    Args:
        request_data (dict): Pipeline configuration parameters
    """
    pipeline_results = {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "steps_completed": [],
        "steps_failed": [],
        "summary": {},
        "errors": [],
        "predictions": None
    }
    
    try:
        logger.info("[RUN] Starting background pipeline execution...")
        
        # Step 2: Fetch authentication token (with timeout protection)
        try:
            logger.info("[AUTH] Fetching authentication token...")
            token_fetcher = TokenFetcher()
            # Set shorter timeout for deployment environment
            token_fetcher.timeout = 30  # Reduce from default
            token = token_fetcher.get_token(force_refresh=request_data.get('refresh_token', False))
            logger.info("[OK] Authentication token retrieved")
            pipeline_results["steps_completed"].append("token_fetch")
        except Exception as e:
            logger.error(f"[ERROR] Token fetch failed: {str(e)}")
            logger.info("[WARN]  Continuing without fresh token...")
            pipeline_results["steps_failed"].append("token_fetch")
            pipeline_results["errors"].append(f"Token fetch failed: {str(e)}")          # Step 3: Fetch match history (use existing data if fetch fails)
        try:
            logger.info("[FETCH] Fetching match history...")
            match_fetcher = MatchHistoryFetcher(days_back=request_data.get('history_days', 30))  # Reduce days
            matches = match_fetcher.fetch_match_history(save_to_file=True)
            logger.info(f"[OK] Match history updated: {len(matches) if matches else 0} matches")
            pipeline_results["steps_completed"].append("match_history")
            pipeline_results["summary"]["match_history_count"] = len(matches) if matches else 0            # Save to database
            if matches and supabase_service.is_connected():
                try:
                    logger.info(">>> Saving match history to database...")
                    supabase_service.save_match_history(matches)
                    logger.info("=== Match history saved to database")
                except Exception as db_e:
                    logger.warning(f"!!! Database save failed for match history: {str(db_e)}")
            
        except Exception as e:
            logger.error(f"[ERROR] Match history fetch failed: {str(e)}")
            logger.info("[WARN]  Using existing match data...")
            pipeline_results["steps_failed"].append("match_history")
            pipeline_results["errors"].append(f"Match history fetch failed: {str(e)}")
        
        # Step 4: Calculate player statistics
        try:
            logger.info("[STATS] Calculating player statistics...")
            stats_processor = PlayerStatsProcessor()
            match_fetcher = MatchHistoryFetcher()
            matches = match_fetcher.load_from_file()
            player_stats = stats_processor.calculate_player_stats(matches, save_to_file=True)
            logger.info(f"[OK] Player statistics updated: {len(player_stats) if player_stats else 0} players")
            pipeline_results["steps_completed"].append("player_stats")
            pipeline_results["summary"]["player_stats_count"] = len(player_stats) if player_stats else 0
            
            # Save to database
            if player_stats and supabase_service.is_connected():
                try:
                    logger.info("[SAVE] Saving player statistics to database...")
                    supabase_service.save_player_stats(player_stats)
                    logger.info("[OK] Player statistics saved to database")
                except Exception as db_e:
                    logger.warning(f"[WARN] Database save failed for player stats: {str(db_e)}")
            
        except Exception as e:
            logger.error(f"[ERROR] Player stats calculation failed: {str(e)}")
            pipeline_results["steps_failed"].append("player_stats")
            pipeline_results["errors"].append(f"Player stats calculation failed: {str(e)}")
        
        # Step 5: Fetch upcoming matches
        try:
            logger.info("[FETCH] Fetching upcoming matches...")
            upcoming_fetcher = UpcomingMatchesFetcher()
            upcoming_matches = upcoming_fetcher.fetch_upcoming_matches(save_to_file=True)
            logger.info(f"[OK] Upcoming matches updated: {len(upcoming_matches) if upcoming_matches else 0} matches")
            pipeline_results["steps_completed"].append("upcoming_matches")
            pipeline_results["summary"]["upcoming_matches_count"] = len(upcoming_matches) if upcoming_matches else 0
            
            # Save to database
            if upcoming_matches and supabase_service.is_connected():
                try:
                    logger.info("[SAVE] Saving upcoming matches to database...")
                    supabase_service.save_upcoming_matches(upcoming_matches)
                    logger.info("[OK] Upcoming matches saved to database")
                except Exception as db_e:
                    logger.warning(f"[WARN] Database save failed for upcoming matches: {str(db_e)}")
            
        except Exception as e:
            logger.error(f"[ERROR] Upcoming matches fetch failed: {str(e)}")
            pipeline_results["steps_failed"].append("upcoming_matches")
            pipeline_results["errors"].append(f"Upcoming matches fetch failed: {str(e)}")
        
        # Step 6: Train model (only if explicitly requested and we have resources)
        if request_data.get('train_new_model', False):
            try:
                logger.info("[MODEL] Training prediction model...")
                prediction_service = EnhancedMatchPredictionService()
                training_df = prediction_service.prepare_training_data(
                    days_back=request_data.get('training_days', 30),  # Reduce for resource constraints
                    min_matches_per_player=request_data.get('min_matches', 3)  # Reduce threshold
                )
                version, metrics = prediction_service.train_model_with_versioning(
                    training_df=training_df, 
                    auto_activate=True,
                    performance_threshold=0.5  # Lower threshold for deployment
                )
                logger.info(f"[OK] Model trained: {version}")
                pipeline_results["steps_completed"].append("model_training")
                pipeline_results["summary"]["model_version"] = str(version)
                # Convert metrics to JSON-serializable format
                json_metrics = {}
                for key, value in metrics.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        json_metrics[key] = float(value.item())
                    elif isinstance(value, (int, float, str, bool)):
                        json_metrics[key] = value
                    else:
                        json_metrics[key] = str(value)
                pipeline_results["summary"]["model_metrics"] = json_metrics
                
                # Save model metadata to database
                if supabase_service.is_connected():
                    try:
                        logger.info("[SAVE] Saving model metadata to database...")
                        model_data = {
                            "model_name": "nba2k_match_predictor",
                            "model_version": str(version),
                            "model_type": "unified_prediction",
                            "performance_metrics": json_metrics,
                            "is_active": True,
                            "training_samples": len(training_df) if 'training_df' in locals() else 0
                        }
                        supabase_service.save_model_registry(model_data)
                        logger.info("[OK] Model metadata saved to database")
                    except Exception as db_e:
                        logger.warning(f"[WARN] Database save failed for model metadata: {str(db_e)}")
                        
            except Exception as e:
                logger.error(f"[ERROR] Model training failed: {str(e)}")
                pipeline_results["steps_failed"].append("model_training")
                pipeline_results["errors"].append(f"Model training failed: {str(e)}")
          # Step 7: Generate predictions (if requested)
        if request_data.get('return_predictions', True):
            try:
                logger.info("[PREDICT] Generating predictions for upcoming matches...")
                prediction_service = EnhancedMatchPredictionService()
                predictions_df = prediction_service.predict_with_best_model(load_model=True)
                
                if not predictions_df.empty:
                    summary = prediction_service.get_prediction_summary(predictions_df)
                    
                    # Create simplified predictions for the response
                    simplified_predictions = []
                    for pred in summary.get('predictions', []):
                        simplified_predictions.append({
                            'match': f"{pred['home_player']} vs {pred['away_player']}",
                            'predicted_winner': pred['predicted_winner'],
                            'confidence': float(pred['confidence']),  # Convert to regular Python float
                            'predicted_total_score': float(pred['predicted_scores']['total']),
                            'home_score': float(pred['predicted_scores']['home']),
                            'away_score': float(pred['predicted_scores']['away'])
                        })
                    
                    pipeline_results["predictions"] = {
                        "total_matches": int(summary.get('total_matches', 0)),
                        "average_confidence": float(summary.get('average_confidence', 0)),
                        "matches": simplified_predictions
                    }
                    pipeline_results["steps_completed"].append("predictions")
                    logger.info(f"[OK] Generated predictions for {len(simplified_predictions)} matches")
                      # Save predictions to database
                    if supabase_service.is_connected():
                        try:
                            logger.info("[SAVE] Saving predictions to database...")
                            
                            # Get the active model version from prediction service
                            try:
                                active_model_version = prediction_service._get_active_model_version()
                                if not active_model_version:
                                    active_model_version = "v1.0.0"  # Default fallback
                            except:
                                active_model_version = pipeline_results["summary"].get("model_version", "v1.0.0")
                            
                            # Convert predictions for database storage
                            db_predictions = []
                            for pred in summary.get('predictions', []):
                                # Calculate home win probability from confidence
                                home_score = float(pred['predicted_scores']['home'])
                                away_score = float(pred['predicted_scores']['away'])
                                home_win_prob = home_score / (home_score + away_score) if (home_score + away_score) > 0 else 0.5
                                
                                db_pred = {
                                    "match_id": f"{pred['home_player']}_{pred['away_player']}_{datetime.now().strftime('%Y%m%d')}",
                                    "model_version": active_model_version,
                                    "home_player": pred['home_player'],
                                    "away_player": pred['away_player'],
                                    "predicted_home_score": float(pred['predicted_scores']['home']),
                                    "predicted_away_score": float(pred['predicted_scores']['away']),
                                    "predicted_total_score": float(pred['predicted_scores']['total']),
                                    "predicted_winner": pred['predicted_winner'],
                                    "home_win_probability": float(home_win_prob),
                                    "confidence_score": float(pred['confidence'])
                                }
                                db_predictions.append(db_pred)
                            
                            supabase_service.save_match_predictions(db_predictions)
                            logger.info("[OK] Predictions saved to database")
                        except Exception as db_e:
                            logger.warning(f"[WARN] Database save failed for predictions: {str(db_e)}")
                    
                else:
                    pipeline_results["predictions"] = {
                        "total_matches": 0,
                        "average_confidence": 0,
                        "matches": [],
                        "message": "No upcoming matches found to predict"
                    }
                    pipeline_results["steps_completed"].append("predictions")
                    logger.info("[OK] No upcoming matches to predict")
                    
            except Exception as e:
                logger.error(f"[ERROR] Prediction generation failed: {str(e)}")
                pipeline_results["steps_failed"].append("predictions")
                pipeline_results["errors"].append(f"Prediction generation failed: {str(e)}")
                pipeline_results["predictions"] = None
        
        # Final status
        pipeline_results["status"] = "completed"
        pipeline_results["completion_timestamp"] = datetime.now().isoformat()
        
        # Save results to a temporary file for retrieval
        results_file = Path("output/pipeline_results.json")
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        logger.info("[DONE] Background pipeline completed!")
        
    except Exception as e:
        logger.error(f"[ERROR] Background pipeline failed: {str(e)}")
        pipeline_results["status"] = "failed"
        pipeline_results["completion_timestamp"] = datetime.now().isoformat()
        pipeline_results["errors"].append(f"Pipeline failed: {str(e)}")
        
        # Save error results
        results_file = Path("output/pipeline_results.json")
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)


@app.post('/api/data/quick-refresh')
@log_execution_time(logger)
@log_exceptions(logger)
def quick_data_refresh(refresh_token: bool = False, return_predictions: bool = True):
    """
    Quick data refresh without model training.
    
    Args:
        refresh_token: Whether to refresh authentication token
        return_predictions: Whether to generate predictions
        
    Returns:
        dict: Job information for tracking
    """
    try:
        payload = {
            "history_days": 20,  # Reduced for quick refresh
            "train_model": False,
            "return_predictions": return_predictions,
            "refresh_token": refresh_token,
            "requested_at": datetime.now().isoformat()
        }
        
        job_id = job_service.create_and_start_job(
            job_type=JobType.QUICK_DATA_REFRESH,
            payload=payload,
            priority=1
        )
        
        if not job_id:
            raise HTTPException(status_code=500, detail="Failed to create quick refresh job")
        
        return {
            "status": "started",
            "job_id": job_id,
            "job_type": "quick_data_refresh",
            "message": "Quick data refresh job started",
            "estimated_duration": "2-5 minutes",
            "job_status_url": f"/api/jobs/{job_id}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting quick refresh job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/ml/generate-predictions')
@log_execution_time(logger)
@log_exceptions(logger)  
def generate_predictions_job():
    """
    Generate predictions using existing data and models.
    
    Returns:
        dict: Job information for tracking
    """
    try:
        payload = {
            "requested_at": datetime.now().isoformat()
        }
        
        job_id = job_service.create_and_start_job(
            job_type=JobType.PREDICTION_GENERATION,
            payload=payload,
            priority=0  # Lower priority
        )
        
        if not job_id:
            raise HTTPException(status_code=500, detail="Failed to create prediction job")
        
        return {
            "status": "started",
            "job_id": job_id,
            "job_type": "prediction_generation",
            "message": "Prediction generation job started",
            "estimated_duration": "1-3 minutes",
            "job_status_url": f"/api/jobs/{job_id}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting prediction job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/system/health')
@log_execution_time(logger)
@log_exceptions(logger)
def system_health_check():
    """
    Comprehensive system health check.
    
    Returns:
        dict: Detailed system health status
    """
    try:
        return health_monitor.check_system_health()
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "overall_status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Health check service failed"
        }


@app.get('/api/system/health/history')
@log_execution_time(logger)
@log_exceptions(logger)
def system_health_history(hours: int = 24):
    """
    Get system health history.
    
    Args:
        hours: Number of hours of history to retrieve
        
    Returns:
        dict: Health check history
    """
    try:
        if hours < 1 or hours > 168:  # Limit to 1 week
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
        
        return health_monitor.get_health_history(hours)
    except Exception as e:
        logger.error(f"Health history retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/pipeline-status')
def get_pipeline_status():
    """
    Get the current status of pipeline operations.
    
    Returns:
        dict: Current pipeline status and recent log entries
    """
    try:
        # Check if files have been recently updated (indication of pipeline activity)
        from config.settings import MATCH_HISTORY_FILE, PLAYER_STATS_FILE, UPCOMING_MATCHES_FILE
        import os
        from datetime import datetime, timedelta
        
        recent_activity = []
        current_time = datetime.now()
        
        for file_path, name in [
            (MATCH_HISTORY_FILE, "Match History"),
            (PLAYER_STATS_FILE, "Player Stats"), 
            (UPCOMING_MATCHES_FILE, "Upcoming Matches")
        ]:
            if file_path.exists():
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                minutes_ago = (current_time - file_time).total_seconds() / 60
                recent_activity.append({
                    "file": name,
                    "last_updated": file_time.isoformat(),
                    "minutes_ago": round(minutes_ago, 1),
                    "recent": minutes_ago < 30  # Updated in last 30 minutes
                })
        
        return {
            "status": "active" if any(item["recent"] for item in recent_activity) else "idle",
            "timestamp": datetime.now().isoformat(),
            "recent_activity": recent_activity,
            "message": "Check file timestamps to see pipeline activity"
        }
        
    except Exception as e:
        logger.error(f"Error getting pipeline status: {str(e)}")
        return {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@app.get('/api/pipeline-results')
@log_execution_time(logger)
@log_exceptions(logger)
def get_pipeline_results():
    """
    Get the results from the most recent pipeline execution.
    
    Returns:
        dict: Pipeline execution results including predictions if available
    """
    try:
        results_file = Path("output/pipeline_results.json")
        
        if not results_file.exists():
            return {
                "status": "no_results",
                "message": "No pipeline results found. Run the pipeline first.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Load the results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Add additional metadata
        import os
        file_time = datetime.fromtimestamp(os.path.getmtime(results_file))
        results["results_file_timestamp"] = file_time.isoformat()
        results["results_age_minutes"] = round((datetime.now() - file_time).total_seconds() / 60, 1)
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting pipeline results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Job Management Endpoints
@app.get('/api/jobs')
def list_jobs(job_type: Optional[str] = None, status: Optional[str] = None, limit: int = 20):
    """
    List jobs with optional filtering.
    
    Args:
        job_type: Filter by job type (model_training, data_pipeline, etc.)
        status: Filter by status (pending, running, completed, failed)
        limit: Maximum number of jobs to return
        
    Returns:
        dict: List of jobs with their status
    """
    try:
        if not supabase_service.is_connected():
            raise HTTPException(status_code=503, detail="Database not connected")
        
        query = supabase_service.client.table('job_queue').select('*')
        
        if job_type:
            query = query.eq('job_type', job_type)
        if status:
            query = query.eq('status', status)
            
        result = query.order('created_at', desc=True).limit(limit).execute()
        
        jobs = result.data if result.data else []
        
        return {
            "status": "success",
            "jobs": jobs,
            "total": len(jobs),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/jobs/{job_id}')
def get_job_details(job_id: str):
    """
    Get detailed information about a specific job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        dict: Job details including status, progress, and results
    """
    try:
        job_details = job_service.get_job_status(job_id)
        
        if not job_details:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return {
            "status": "success",
            "job": job_details,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id: str):
    """
    Cancel a running or pending job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        dict: Cancellation result
    """
    try:
        success = job_service.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to cancel job {job_id}")
        
        return {
            "status": "success",
            "message": f"Job {job_id} has been cancelled",
            "job_id": job_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/api/jobs/cleanup')
def cleanup_old_jobs(days: int = 7):
    """
    Clean up completed jobs older than specified days.
    
    Args:
        days: Number of days old jobs to keep
        
    Returns:
        dict: Cleanup result
    """
    try:
        success = job_service.cleanup_old_jobs(days_old=days)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to cleanup old jobs")
        
        return {
            "status": "success",
            "message": f"Cleaned up jobs older than {days} days",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/system/job-stats')
def get_job_statistics():
    """
    Get statistics about job queue and execution.
    
    Returns:
        dict: Job statistics
    """
    try:
        if not supabase_service.is_connected():
            raise HTTPException(status_code=503, detail="Database not connected")
        
        # Get job counts by status
        stats = {}
        for status in ['pending', 'running', 'completed', 'failed', 'cancelled']:
            result = supabase_service.client.table('job_queue').select('id').eq('status', status).execute()
            stats[f"{status}_jobs"] = len(result.data) if result.data else 0
        
        # Get recent jobs (last 24 hours)
        recent_cutoff = datetime.now().replace(hour=datetime.now().hour - 24)
        recent_result = supabase_service.client.table('job_queue').select('id').gte(
            'created_at', recent_cutoff.isoformat()
        ).execute()
        stats['recent_jobs_24h'] = len(recent_result.data) if recent_result.data else 0
        
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def run_api_server():
    """
    Run the API server.
    """
    import uvicorn
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == '__main__':
    run_api_server()
