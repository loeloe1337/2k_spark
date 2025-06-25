"""
Data pipeline job handlers for background processing.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
import asyncio
import threading
import time

# Add the parent directory to Python path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from config.logging_config import get_api_logger
from services.supabase_service import SupabaseService
from services.enhanced_prediction_service import EnhancedMatchPredictionService
from services.job_service import JobStatus
from core.data.fetchers import TokenFetcher
from core.data.fetchers.match_history import MatchHistoryFetcher
from core.data.fetchers.upcoming_matches import UpcomingMatchesFetcher
from core.data.processors.player_stats import PlayerStatsProcessor

logger = get_api_logger()


class DataPipelineHandler:
    """
    Handler for data pipeline jobs with cloud optimization.
    """
    
    def __init__(self):
        self.supabase = SupabaseService()
        self.prediction_service = EnhancedMatchPredictionService()
        self.max_timeout = 600  # 10 minutes max for any pipeline job
        
    def run_full_pipeline_job(self, job_id: str, job_service) -> Dict[str, Any]:
        """
        Execute the complete data pipeline as a background job.
        
        Args:
            job_id: Unique job identifier
            job_service: Reference to job service
            
        Returns:
            Dict with success status and results
        """
        try:
            start_time = time.time()
            
            # Get job details
            job_details = job_service.get_job_status(job_id)
            if not job_details:
                return {"success": False, "error": "Job not found"}
            
            payload = job_details.get('payload', {})
            
            # Initialize result tracking
            result = {
                "success": True,
                "steps_completed": [],
                "steps_failed": [],
                "summary": {},
                "errors": [],
                "data": {}
            }
            
            logger.info(f"[JOB {job_id}] Starting full data pipeline")
            
            # Step 1: Token Authentication (5% progress)
            job_service.update_job_status(job_id, JobStatus.RUNNING, progress=5)
            token_result = self._fetch_token_step(job_id, payload, result)
            
            # Step 2: Fetch Match History (25% progress)
            job_service.update_job_status(job_id, JobStatus.RUNNING, progress=15)
            matches_result = self._fetch_matches_step(job_id, payload, result)
            
            # Step 3: Process Player Stats (45% progress)
            job_service.update_job_status(job_id, JobStatus.RUNNING, progress=35)
            stats_result = self._process_stats_step(job_id, payload, result)
            
            # Step 4: Fetch Upcoming Matches (65% progress)
            job_service.update_job_status(job_id, JobStatus.RUNNING, progress=55)
            upcoming_result = self._fetch_upcoming_step(job_id, payload, result)
            
            # Step 5: Generate Predictions (85% progress)
            job_service.update_job_status(job_id, JobStatus.RUNNING, progress=75)
            predictions_result = self._generate_predictions_step(job_id, payload, result)
            
            # Step 6: Model Training (if requested) (95% progress)
            if payload.get('train_model', False):
                job_service.update_job_status(job_id, JobStatus.RUNNING, progress=85)
                training_result = self._train_model_step(job_id, payload, result)
            
            # Finalize
            total_time = time.time() - start_time
            result["execution_time_seconds"] = round(total_time, 2)
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
            
            # Log pipeline summary
            logger.info(f"[JOB {job_id}] Pipeline completed in {total_time:.2f}s")
            logger.info(f"[JOB {job_id}] Steps completed: {len(result['steps_completed'])}")
            logger.info(f"[JOB {job_id}] Steps failed: {len(result['steps_failed'])}")
            
            return result
            
        except Exception as e:
            error_msg = f"Data pipeline job failed: {str(e)}"
            logger.error(f"[JOB {job_id}] {error_msg}")
            return {"success": False, "error": error_msg}
    
    def _fetch_token_step(self, job_id: str, payload: Dict, result: Dict) -> bool:
        """Fetch authentication token with timeout protection."""
        try:
            logger.info(f"[JOB {job_id}] Step 1/6: Fetching authentication token...")
            
            # Use shorter timeout for cloud deployment
            token_fetcher = TokenFetcher()
            token_fetcher.timeout = 30  # 30 second timeout
            
            token = token_fetcher.get_token(force_refresh=payload.get('refresh_token', False))
            
            if token:
                result["steps_completed"].append("token_fetch")
                result["summary"]["token_status"] = "success"
                logger.info(f"[JOB {job_id}] Token fetched successfully")
                return True
            else:
                result["steps_failed"].append("token_fetch")
                result["errors"].append("Token fetch returned None")
                result["summary"]["token_status"] = "failed"
                logger.warning(f"[JOB {job_id}] Token fetch failed but continuing...")
                return False
                
        except Exception as e:
            error_msg = f"Token fetch failed: {str(e)}"
            result["steps_failed"].append("token_fetch")
            result["errors"].append(error_msg)
            result["summary"]["token_status"] = "error"
            logger.error(f"[JOB {job_id}] {error_msg}")
            return False
    
    def _fetch_matches_step(self, job_id: str, payload: Dict, result: Dict) -> bool:
        """Fetch match history with optimizations."""
        try:
            logger.info(f"[JOB {job_id}] Step 2/6: Fetching match history...")
            
            days_back = payload.get('history_days', 30)  # Reduced from 60
            match_fetcher = MatchHistoryFetcher(days_back=days_back)
            
            matches = match_fetcher.fetch_match_history(save_to_file=True)
            
            if matches:
                result["steps_completed"].append("match_history")
                result["summary"]["match_history_count"] = len(matches)
                result["data"]["matches"] = len(matches)
                
                # Save to database if connected
                if self.supabase.is_connected():
                    try:
                        saved_count = self.supabase.save_matches_batch(matches)
                        result["summary"]["matches_saved_to_db"] = saved_count
                        logger.info(f"[JOB {job_id}] Saved {saved_count} matches to database")
                    except Exception as e:
                        logger.warning(f"[JOB {job_id}] Database save failed: {str(e)}")
                
                logger.info(f"[JOB {job_id}] Match history fetched: {len(matches)} matches")
                return True
            else:
                result["steps_failed"].append("match_history")
                result["errors"].append("No matches returned from API")
                result["summary"]["match_history_count"] = 0
                logger.warning(f"[JOB {job_id}] No matches returned from API")
                return False
                
        except Exception as e:
            error_msg = f"Match history fetch failed: {str(e)}"
            result["steps_failed"].append("match_history")
            result["errors"].append(error_msg)
            result["summary"]["match_history_count"] = 0
            logger.error(f"[JOB {job_id}] {error_msg}")
            return False
    
    def _process_stats_step(self, job_id: str, payload: Dict, result: Dict) -> bool:
        """Process player statistics."""
        try:
            logger.info(f"[JOB {job_id}] Step 3/6: Processing player statistics...")
            
            processor = PlayerStatsProcessor()
            
            # Check if we have match data to process
            if self.supabase.is_connected():
                # Process from database
                stats_data = processor.calculate_enhanced_player_stats()
            else:
                # Process from files
                from config.settings import MATCH_HISTORY_FILE
                if Path(MATCH_HISTORY_FILE).exists():
                    stats_data = processor.calculate_enhanced_player_stats()
                else:
                    raise Exception("No match data available for stats processing")
            
            if stats_data and len(stats_data) > 0:
                result["steps_completed"].append("player_stats")
                result["summary"]["player_stats_count"] = len(stats_data)
                result["data"]["player_stats"] = len(stats_data)
                
                # Save to database if connected
                if self.supabase.is_connected():
                    try:
                        saved_count = self.supabase.save_player_stats_batch(stats_data)
                        result["summary"]["stats_saved_to_db"] = saved_count
                        logger.info(f"[JOB {job_id}] Saved {saved_count} player stats to database")
                    except Exception as e:
                        logger.warning(f"[JOB {job_id}] Stats database save failed: {str(e)}")
                
                logger.info(f"[JOB {job_id}] Player stats processed: {len(stats_data)} players")
                return True
            else:
                result["steps_failed"].append("player_stats")
                result["errors"].append("No player stats calculated")
                result["summary"]["player_stats_count"] = 0
                logger.warning(f"[JOB {job_id}] No player stats calculated")
                return False
                
        except Exception as e:
            error_msg = f"Player stats processing failed: {str(e)}"
            result["steps_failed"].append("player_stats")
            result["errors"].append(error_msg)
            result["summary"]["player_stats_count"] = 0
            logger.error(f"[JOB {job_id}] {error_msg}")
            return False
    
    def _fetch_upcoming_step(self, job_id: str, payload: Dict, result: Dict) -> bool:
        """Fetch upcoming matches."""
        try:
            logger.info(f"[JOB {job_id}] Step 4/6: Fetching upcoming matches...")
            
            upcoming_fetcher = UpcomingMatchesFetcher()
            upcoming_matches = upcoming_fetcher.fetch_upcoming_matches(save_to_file=True)
            
            if upcoming_matches:
                result["steps_completed"].append("upcoming_matches")
                result["summary"]["upcoming_matches_count"] = len(upcoming_matches)
                result["data"]["upcoming_matches"] = len(upcoming_matches)
                
                # Save to database if connected
                if self.supabase.is_connected():
                    try:
                        saved_count = self.supabase.save_upcoming_matches_batch(upcoming_matches)
                        result["summary"]["upcoming_saved_to_db"] = saved_count
                        logger.info(f"[JOB {job_id}] Saved {saved_count} upcoming matches to database")
                    except Exception as e:
                        logger.warning(f"[JOB {job_id}] Upcoming matches database save failed: {str(e)}")
                
                logger.info(f"[JOB {job_id}] Upcoming matches fetched: {len(upcoming_matches)} matches")
                return True
            else:
                result["steps_failed"].append("upcoming_matches")
                result["errors"].append("No upcoming matches returned from API")
                result["summary"]["upcoming_matches_count"] = 0
                logger.warning(f"[JOB {job_id}] No upcoming matches returned from API")
                return False
                
        except Exception as e:
            error_msg = f"Upcoming matches fetch failed: {str(e)}"
            result["steps_failed"].append("upcoming_matches")
            result["errors"].append(error_msg)
            result["summary"]["upcoming_matches_count"] = 0
            logger.error(f"[JOB {job_id}] {error_msg}")
            return False
    
    def _generate_predictions_step(self, job_id: str, payload: Dict, result: Dict) -> bool:
        """Generate match predictions."""
        try:
            logger.info(f"[JOB {job_id}] Step 5/6: Generating predictions...")
            
            if not payload.get('return_predictions', True):
                logger.info(f"[JOB {job_id}] Predictions not requested, skipping...")
                result["steps_completed"].append("predictions_skipped")
                return True
            
            predictions_df = self.prediction_service.predict_with_best_model(load_model=True)
            
            if not predictions_df.empty:
                result["steps_completed"].append("predictions")
                result["summary"]["predictions_count"] = len(predictions_df)
                result["data"]["predictions"] = len(predictions_df)
                
                # Convert to JSON-serializable format
                predictions_list = []
                for _, row in predictions_df.iterrows():
                    pred = {
                        "match_id": str(row.get('match_id', '')),
                        "home_player": str(row.get('home_player', '')),
                        "away_player": str(row.get('away_player', '')),
                        "predicted_winner": str(row.get('predicted_winner', '')),
                        "home_win_probability": float(row.get('home_win_probability', 0)),
                        "predicted_home_score": float(row.get('predicted_home_score', 0)),
                        "predicted_away_score": float(row.get('predicted_away_score', 0)),
                        "confidence_score": float(row.get('confidence_score', 0))
                    }
                    predictions_list.append(pred)
                
                result["data"]["predictions_details"] = predictions_list
                
                logger.info(f"[JOB {job_id}] Predictions generated: {len(predictions_df)} matches")
                return True
            else:
                result["steps_failed"].append("predictions")
                result["errors"].append("No predictions generated - no upcoming matches or model issues")
                result["summary"]["predictions_count"] = 0
                logger.warning(f"[JOB {job_id}] No predictions generated")
                return False
                
        except Exception as e:
            error_msg = f"Predictions generation failed: {str(e)}"
            result["steps_failed"].append("predictions")
            result["errors"].append(error_msg)
            result["summary"]["predictions_count"] = 0
            logger.error(f"[JOB {job_id}] {error_msg}")
            return False
    
    def _train_model_step(self, job_id: str, payload: Dict, result: Dict) -> bool:
        """Train new model if requested."""
        try:
            logger.info(f"[JOB {job_id}] Step 6/6: Training new model...")
            
            # Use optimized parameters for cloud deployment
            training_df = self.prediction_service.prepare_training_data(
                days_back=payload.get('training_days_back', 20),  # Reduced
                min_matches_per_player=payload.get('min_matches_per_player', 2)  # Reduced
            )
            
            if training_df.empty:
                result["steps_failed"].append("model_training")
                result["errors"].append("No training data available")
                logger.warning(f"[JOB {job_id}] No training data available")
                return False
            
            # Train with timeout protection
            version, metrics = self.prediction_service.train_model_with_versioning(
                training_df=training_df,
                auto_activate=payload.get('auto_activate', True),
                performance_threshold=payload.get('performance_threshold', 0.5),
                max_training_time=300,  # 5 minutes max
                early_stopping=True,
                reduced_complexity=True
            )
            
            result["steps_completed"].append("model_training")
            result["summary"]["model_version"] = str(version)
            result["summary"]["training_samples"] = len(training_df)
            
            # Convert metrics to JSON-serializable format
            json_metrics = {}
            for key, value in metrics.items():
                if hasattr(value, 'item'):
                    json_metrics[key] = float(value.item())
                elif isinstance(value, (int, float, str, bool)):
                    json_metrics[key] = value
                else:
                    json_metrics[key] = str(value)
            
            result["summary"]["model_metrics"] = json_metrics
            
            logger.info(f"[JOB {job_id}] Model training completed: {version}")
            return True
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            result["steps_failed"].append("model_training")
            result["errors"].append(error_msg)
            logger.error(f"[JOB {job_id}] {error_msg}")
            return False


# Create data pipeline job handlers for different types
class QuickDataRefreshHandler(DataPipelineHandler):
    """Handler for quick data refresh jobs (no model training)."""
    
    def run_quick_refresh_job(self, job_id: str, job_service) -> Dict[str, Any]:
        """Run a quick data refresh without model training."""
        try:
            job_details = job_service.get_job_status(job_id)
            if not job_details:
                return {"success": False, "error": "Job not found"}
            
            payload = job_details.get('payload', {})
            
            # Override to skip model training
            payload['train_model'] = False
            payload['return_predictions'] = payload.get('return_predictions', True)
            
            # Run the full pipeline but skip training
            return self.run_full_pipeline_job(job_id, job_service)
            
        except Exception as e:
            error_msg = f"Quick data refresh job failed: {str(e)}"
            logger.error(f"[JOB {job_id}] {error_msg}")
            return {"success": False, "error": error_msg}


class PredictionGenerationHandler(DataPipelineHandler):
    """Handler for prediction generation only."""
    
    def run_prediction_job(self, job_id: str, job_service) -> Dict[str, Any]:
        """Generate predictions using existing data and models."""
        try:
            start_time = time.time()
            
            job_details = job_service.get_job_status(job_id)
            if not job_details:
                return {"success": False, "error": "Job not found"}
            
            payload = job_details.get('payload', {})
            
            result = {
                "success": True,
                "steps_completed": [],
                "steps_failed": [],
                "summary": {},
                "errors": [],
                "data": {}
            }
            
            logger.info(f"[JOB {job_id}] Starting prediction generation job")
            job_service.update_job_status(job_id, JobStatus.RUNNING, progress=10)
            
            # Generate predictions
            predictions_success = self._generate_predictions_step(job_id, payload, result)
            
            job_service.update_job_status(job_id, JobStatus.RUNNING, progress=90)
            
            # Finalize
            total_time = time.time() - start_time
            result["execution_time_seconds"] = round(total_time, 2)
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"[JOB {job_id}] Prediction generation completed in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Prediction generation job failed: {str(e)}"
            logger.error(f"[JOB {job_id}] {error_msg}")
            return {"success": False, "error": error_msg}


# Global handler instances
data_pipeline_handler = DataPipelineHandler()
quick_refresh_handler = QuickDataRefreshHandler()
prediction_handler = PredictionGenerationHandler()
