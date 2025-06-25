"""
Model training job handler with optimization for cloud deployment.
"""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from config.logging_config import get_api_logger
from services.enhanced_prediction_service import EnhancedMatchPredictionService
from services.supabase_service import SupabaseService
from services.job_service import JobStatus

logger = get_api_logger()


class ModelTrainingHandler:
    """
    Optimized model training handler for cloud deployment.
    """
    
    def __init__(self):
        self.prediction_service = EnhancedMatchPredictionService()
        self.supabase = SupabaseService()
    
    def train_model_job(self, job_id: str, job_service) -> Dict[str, Any]:
        """
        Execute model training job with cloud optimizations.
        
        Args:
            job_id: Unique job identifier
            job_service: Reference to job service for progress updates
            
        Returns:
            Dict with success status and results
        """
        try:
            # Get job details
            job_details = job_service.get_job_status(job_id)
            if not job_details:
                return {"success": False, "error": "Job not found"}
            
            payload = job_details.get('payload', {})
            days_back = payload.get('days_back', 30)  # Reduced default for faster training
            min_matches_per_player = payload.get('min_matches_per_player', 3)  # Reduced threshold
            performance_threshold = payload.get('performance_threshold', 0.5)  # Lower threshold for deployment
            
            logger.info(f"[JOB {job_id}] Starting model training with optimized parameters")
            
            # Step 1: Prepare training data (10% progress)
            job_service.update_job_status(job_id, JobStatus.RUNNING, progress=10)
            logger.info(f"[JOB {job_id}] Preparing training data...")
            
            training_df = self.prediction_service.prepare_training_data(
                days_back=days_back,
                min_matches_per_player=min_matches_per_player
            )
            
            if training_df.empty:
                return {"success": False, "error": "No training data available"}
            
            logger.info(f"[JOB {job_id}] Training data prepared: {len(training_df)} samples")
            
            # Step 2: Train model with optimizations (50% progress)
            job_service.update_job_status(job_id, JobStatus.RUNNING, progress=30)
            logger.info(f"[JOB {job_id}] Training model...")
            
            # Use optimized training parameters for cloud deployment
            version, metrics = self.prediction_service.train_model_with_versioning(
                training_df=training_df,
                auto_activate=True,
                performance_threshold=performance_threshold,
                # Additional optimizations can be added here
                max_training_time=300,  # 5 minutes max training time
                early_stopping=True,
                reduced_complexity=True
            )
            
            job_service.update_job_status(job_id, JobStatus.RUNNING, progress=70)
            logger.info(f"[JOB {job_id}] Model training completed: {version}")
            
            # Step 3: Upload model to cloud storage (80% progress)
            logger.info(f"[JOB {job_id}] Uploading model to cloud storage...")
            
            model_uploaded = False
            try:
                model_path = self.prediction_service.get_model_path(version)
                if model_path and model_path.exists():
                    model_uploaded = self.supabase.upload_model_file(model_path, version)
                    if model_uploaded:
                        logger.info(f"[JOB {job_id}] Model file uploaded successfully")
                    else:
                        logger.warning(f"[JOB {job_id}] Model file upload failed")
            except Exception as e:
                logger.warning(f"[JOB {job_id}] Model file upload failed: {str(e)}")
            
            job_service.update_job_status(job_id, JobStatus.RUNNING, progress=90)
            
            # Step 4: Save model metadata to database
            logger.info(f"[JOB {job_id}] Saving model metadata...")
            
            # Convert metrics to JSON-serializable format
            json_metrics = {}
            for key, value in metrics.items():
                if hasattr(value, 'item'):  # numpy scalar
                    json_metrics[key] = float(value.item())
                elif isinstance(value, (int, float, str, bool)):
                    json_metrics[key] = value
                else:
                    json_metrics[key] = str(value)
            
            # Save model metadata
            model_metadata_saved = False
            if self.supabase.is_connected():
                try:
                    model_data = {
                        "model_name": "nba2k_match_predictor",
                        "model_version": str(version),
                        "model_type": "unified_prediction",
                        "performance_metrics": json_metrics,
                        "is_active": True,
                        "training_samples": len(training_df),
                        "model_path": f"models/model_{version}.pkl" if model_uploaded else None,
                        "training_date": datetime.now(timezone.utc).isoformat()
                    }
                    model_metadata_saved = self.supabase.save_model_registry(model_data)
                    logger.info(f"[JOB {job_id}] Model metadata saved to database")
                except Exception as e:
                    logger.warning(f"[JOB {job_id}] Model metadata save failed: {str(e)}")
            
            # Prepare result
            result = {
                "success": True,
                "model_version": str(version),
                "training_samples": len(training_df),
                "model_uploaded": model_uploaded,
                "metadata_saved": model_metadata_saved,
                "metrics": {
                    "winner_accuracy": round(json_metrics.get('val_winner_accuracy', 0), 3),
                    "home_score_mae": round(json_metrics.get('val_home_mae', 0), 2),
                    "away_score_mae": round(json_metrics.get('val_away_mae', 0), 2),
                    "total_score_mae": round(json_metrics.get('val_total_mae', 0), 2)
                },
                "training_duration": payload.get('training_duration', 'unknown'),
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"[JOB {job_id}] Model training job completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Model training job failed: {str(e)}"
            logger.error(f"[JOB {job_id}] {error_msg}")
            return {"success": False, "error": error_msg}
    
    def optimize_training_parameters(self, available_memory: int = None, 
                                   time_limit: int = 300) -> Dict[str, Any]:
        """
        Optimize training parameters based on available resources.
        
        Args:
            available_memory: Available memory in MB
            time_limit: Maximum training time in seconds
            
        Returns:
            Optimized parameters
        """
        params = {
            "days_back": 30,  # Reduced from 60
            "min_matches_per_player": 3,  # Reduced from 5
            "performance_threshold": 0.5,  # Lowered threshold
            "max_training_time": time_limit,
            "early_stopping": True,
            "reduced_complexity": True
        }
        
        # Adjust based on memory constraints
        if available_memory and available_memory < 512:  # Less than 512MB
            params["days_back"] = 20
            params["min_matches_per_player"] = 2
            params["reduced_complexity"] = True
        
        return params


# Global handler instance
model_training_handler = ModelTrainingHandler()
