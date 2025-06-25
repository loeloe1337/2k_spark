"""
Background job processing service for long-running tasks.
"""

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from uuid import uuid4

from config.logging_config import get_api_logger
from services.supabase_service import SupabaseService

logger = get_api_logger()


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    MODEL_TRAINING = "model_training"
    DATA_PIPELINE = "data_pipeline"
    PREDICTION_GENERATION = "prediction_generation"
    QUICK_DATA_REFRESH = "quick_data_refresh"
    MATCH_HISTORY_FETCH = "match_history_fetch"
    PLAYER_STATS_CALCULATION = "player_stats_calculation"


class JobService:
    """
    Service for managing background jobs with database persistence.
    """
    
    def __init__(self):
        self.supabase = SupabaseService()
        self.workers = {}  # Store running worker threads
        self.job_handlers = {}  # Registry of job type handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register job handlers for different job types."""
        try:
            # Import handlers lazily to avoid circular imports
            from services.model_training_handler import model_training_handler
            from services.data_pipeline_handler import (
                data_pipeline_handler, 
                quick_refresh_handler, 
                prediction_handler
            )
            
            # Register handlers
            self.job_handlers = {
                JobType.MODEL_TRAINING: model_training_handler.train_model_job,
                JobType.DATA_PIPELINE: data_pipeline_handler.run_full_pipeline_job,
                JobType.QUICK_DATA_REFRESH: quick_refresh_handler.run_quick_refresh_job,
                JobType.PREDICTION_GENERATION: prediction_handler.run_prediction_job,
            }
            logger.info("Job handlers registered successfully")
            
        except Exception as e:
            logger.warning(f"Failed to register some job handlers: {str(e)}")
            self.job_handlers = {}
    
    def create_job(self, job_type: JobType, payload: Dict[str, Any], priority: int = 0) -> str:
        """
        Create a new background job.
        
        Args:
            job_type: Type of job to create
            payload: Job parameters
            priority: Job priority (higher = more priority)
            
        Returns:
            job_id: Unique identifier for the job
        """
        job_id = str(uuid4())
        
        try:
            job_data = {
                'job_id': job_id,
                'job_type': job_type.value,
                'status': JobStatus.PENDING.value,
                'priority': priority,
                'payload': payload,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            if self.supabase.is_connected():
                self.supabase.client.table('job_queue').insert(job_data).execute()
                logger.info(f"Created job {job_id} of type {job_type.value}")
            else:
                logger.error("Cannot create job: Supabase not connected")
                return None
                
            return job_id
            
        except Exception as e:
            logger.error(f"Error creating job: {str(e)}")
            return None
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and details."""
        try:
            if not self.supabase.is_connected():
                return None
                
            result = self.supabase.client.table('job_queue').select('*').eq('job_id', job_id).limit(1).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            return None
    
    def update_job_status(self, job_id: str, status: JobStatus, progress: int = None, 
                         result: Dict = None, error_message: str = None):
        """Update job status in database."""
        try:
            if not self.supabase.is_connected():
                return False
                
            update_data = {
                'status': status.value,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            if progress is not None:
                update_data['progress'] = progress
            if result is not None:
                update_data['result'] = result
            if error_message is not None:
                update_data['error_message'] = error_message
            if status == JobStatus.RUNNING and 'started_at' not in update_data:
                update_data['started_at'] = datetime.now(timezone.utc).isoformat()
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                update_data['completed_at'] = datetime.now(timezone.utc).isoformat()
                
            self.supabase.client.table('job_queue').update(update_data).eq('job_id', job_id).execute()
            logger.info(f"Updated job {job_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating job status: {str(e)}")
            return False
    
    def start_job(self, job_id: str, handler_func: Callable):
        """Start executing a job in a background thread."""
        if job_id in self.workers:
            logger.warning(f"Job {job_id} is already running")
            return False
            
        def worker():
            try:
                logger.info(f"Starting job {job_id}")
                self.update_job_status(job_id, JobStatus.RUNNING, progress=0)
                
                # Execute the job handler
                result = handler_func(job_id, self)
                
                if result.get('success', False):
                    self.update_job_status(job_id, JobStatus.COMPLETED, progress=100, result=result)
                    logger.info(f"Job {job_id} completed successfully")
                else:
                    self.update_job_status(job_id, JobStatus.FAILED, 
                                         error_message=result.get('error', 'Unknown error'))
                    logger.error(f"Job {job_id} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Job {job_id} failed with exception: {str(e)}")
                self.update_job_status(job_id, JobStatus.FAILED, error_message=str(e))
            finally:
                # Clean up worker reference
                if job_id in self.workers:
                    del self.workers[job_id]
        
        # Start worker thread
        thread = threading.Thread(target=worker, daemon=True)
        self.workers[job_id] = thread
        thread.start()
        return True
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        try:
            if job_id in self.workers:
                # Note: Python threads can't be forcefully killed, 
                # so we just mark as cancelled in DB
                logger.warning(f"Job {job_id} marked as cancelled but thread may continue")
            
            return self.update_job_status(job_id, JobStatus.CANCELLED)
            
        except Exception as e:
            logger.error(f"Error cancelling job: {str(e)}")
            return False
    
    def get_pending_jobs(self, job_type: JobType = None) -> List[Dict[str, Any]]:
        """Get list of pending jobs, optionally filtered by type."""
        try:
            if not self.supabase.is_connected():
                return []
                
            query = self.supabase.client.table('job_queue').select('*').eq('status', JobStatus.PENDING.value)
            
            if job_type:
                query = query.eq('job_type', job_type.value)
                
            result = query.order('priority', desc=True).order('created_at', desc=False).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error getting pending jobs: {str(e)}")
            return []
    
    def cleanup_old_jobs(self, days_old: int = 7) -> bool:
        """Clean up completed/failed jobs older than specified days."""
        try:
            if not self.supabase.is_connected():
                return False
                
            cutoff_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_old)
            
            result = self.supabase.client.table('job_queue').delete().in_(
                'status', [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]
            ).lt('completed_at', cutoff_date.isoformat()).execute()
            
            deleted_count = len(result.data) if result.data else 0
            logger.info(f"Cleaned up {deleted_count} old jobs")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old jobs: {str(e)}")
            return False
    
    def start_job_by_type(self, job_id: str, job_type: JobType) -> bool:
        """Start a job using its registered handler."""
        if job_type not in self.job_handlers:
            logger.error(f"No handler registered for job type: {job_type.value}")
            return False
        
        handler_func = self.job_handlers[job_type]
        return self.start_job(job_id, handler_func)
    
    def create_and_start_job(self, job_type: JobType, payload: Dict[str, Any], priority: int = 0) -> str:
        """Create and immediately start a job."""
        job_id = self.create_job(job_type, payload, priority)
        if job_id:
            success = self.start_job_by_type(job_id, job_type)
            if not success:
                logger.error(f"Failed to start job {job_id}")
                # Mark job as failed
                self.update_job_status(job_id, JobStatus.FAILED, error_message="Failed to start job execution")
                return None
            return job_id
        return None


# Global job service instance
job_service = JobService()
