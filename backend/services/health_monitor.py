"""
System health monitoring service for cloud deployment.
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any, List
from pathlib import Path

from config.logging_config import get_api_logger
from services.supabase_service import SupabaseService

logger = get_api_logger()

# Try to import psutil, fall back to basic checks if not available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available - resource monitoring will be limited")


class HealthMonitorService:
    """
    Service for monitoring system health and performance.
    """
    
    def __init__(self):
        self.supabase = SupabaseService()
        
    def check_system_health(self) -> Dict[str, Any]:
        """
        Comprehensive system health check.
        
        Returns:
            Dict with health status and metrics
        """
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "metrics": {},
            "warnings": [],
            "errors": []
        }
        
        # Check database connectivity
        db_status = self._check_database_health()
        health_status["checks"]["database"] = db_status
        if db_status["status"] != "healthy":
            health_status["errors"].append(f"Database: {db_status['message']}")
        
        # Check system resources
        resource_status = self._check_resource_health()
        health_status["checks"]["resources"] = resource_status
        health_status["metrics"].update(resource_status["metrics"])
        if resource_status["status"] == "warning":
            health_status["warnings"].extend(resource_status.get("warnings", []))
        elif resource_status["status"] == "error":
            health_status["errors"].extend(resource_status.get("errors", []))
        
        # Check job queue health
        job_status = self._check_job_queue_health()
        health_status["checks"]["job_queue"] = job_status
        if job_status["status"] != "healthy":
            if job_status["status"] == "warning":
                health_status["warnings"].append(f"Job Queue: {job_status['message']}")
            else:
                health_status["errors"].append(f"Job Queue: {job_status['message']}")
        
        # Check model availability
        model_status = self._check_model_health()
        health_status["checks"]["models"] = model_status
        if model_status["status"] != "healthy":
            health_status["warnings"].append(f"Models: {model_status['message']}")
        
        # Check data freshness
        data_status = self._check_data_freshness()
        health_status["checks"]["data_freshness"] = data_status
        if data_status["status"] != "healthy":
            health_status["warnings"].append(f"Data: {data_status['message']}")
        
        # Determine overall status
        if health_status["errors"]:
            health_status["overall_status"] = "error"
        elif health_status["warnings"]:
            health_status["overall_status"] = "warning"
        
        # Save health check to database
        self._save_health_check(health_status)
        
        return health_status
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            if not self.supabase.is_connected():
                return {
                    "status": "error",
                    "message": "Database not connected",
                    "details": "Supabase connection failed"
                }
            
            # Test database query performance
            start_time = time.time()
            result = self.supabase.client.table('job_queue').select('count').limit(1).execute()
            query_time = time.time() - start_time
            
            if query_time > 5.0:  # Slow query warning
                return {
                    "status": "warning",
                    "message": f"Database queries are slow ({query_time:.2f}s)",
                    "query_time_seconds": query_time
                }
            
            return {
                "status": "healthy",
                "message": "Database connection is healthy",
                "query_time_seconds": round(query_time, 3)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Database health check failed: {str(e)}",
                "details": str(e)
            }
    
    def _check_resource_health(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            if not HAS_PSUTIL:
                return {
                    "status": "warning",
                    "message": "Resource monitoring not available - psutil not installed",
                    "metrics": {
                        "cpu_percent": "N/A",
                        "memory_percent": "N/A", 
                        "disk_percent": "N/A"
                    },
                    "warnings": ["Resource monitoring requires psutil package"],
                    "errors": []
                }
            
            # Get system metrics with psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(memory.percent, 1),
                "memory_used_mb": round(memory.used / 1024 / 1024, 1),
                "memory_available_mb": round(memory.available / 1024 / 1024, 1),
                "disk_percent": round(disk.percent, 1),
                "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 1)
            }
            
            warnings = []
            errors = []
            
            # Check thresholds
            if cpu_percent > 90:
                errors.append(f"CPU usage is critically high: {cpu_percent}%")
            elif cpu_percent > 80:
                warnings.append(f"CPU usage is high: {cpu_percent}%")
            
            if memory.percent > 90:
                errors.append(f"Memory usage is critically high: {memory.percent}%")
            elif memory.percent > 80:
                warnings.append(f"Memory usage is high: {memory.percent}%")
            
            if disk.percent > 90:
                errors.append(f"Disk usage is critically high: {disk.percent}%")
            elif disk.percent > 85:
                warnings.append(f"Disk usage is high: {disk.percent}%")
            
            # Determine status
            if errors:
                status = "error"
                message = f"Critical resource issues: {len(errors)} errors"
            elif warnings:
                status = "warning"
                message = f"Resource warnings: {len(warnings)} warnings"
            else:
                status = "healthy"
                message = "System resources are healthy"
            
            return {
                "status": status,
                "message": message,
                "metrics": metrics,
                "warnings": warnings,
                "errors": errors
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Resource check failed: {str(e)}",
                "metrics": {},
                "errors": [str(e)]
            }
    
    def _check_job_queue_health(self) -> Dict[str, Any]:
        """Check job queue status and performance."""
        try:
            if not self.supabase.is_connected():
                return {
                    "status": "warning",
                    "message": "Cannot check job queue - database not connected"
                }
            
            # Get job statistics
            result = self.supabase.client.table('job_queue').select('status').execute()
            
            if not result.data:
                return {
                    "status": "healthy",
                    "message": "Job queue is empty",
                    "job_counts": {}
                }
            
            # Count jobs by status
            job_counts = {}
            for job in result.data:
                status = job.get('status', 'unknown')
                job_counts[status] = job_counts.get(status, 0) + 1
            
            # Check for issues
            warnings = []
            running_jobs = job_counts.get('running', 0)
            failed_jobs = job_counts.get('failed', 0)
            pending_jobs = job_counts.get('pending', 0)
            
            if failed_jobs > 5:
                warnings.append(f"High number of failed jobs: {failed_jobs}")
            
            if pending_jobs > 10:
                warnings.append(f"High number of pending jobs: {pending_jobs}")
            
            if running_jobs > 5:
                warnings.append(f"High number of running jobs: {running_jobs}")
            
            status = "warning" if warnings else "healthy"
            message = f"Job queue status: {', '.join([f'{k}: {v}' for k, v in job_counts.items()])}"
            
            return {
                "status": status,
                "message": message,
                "job_counts": job_counts,
                "warnings": warnings
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Job queue check failed: {str(e)}",
                "details": str(e)
            }
    
    def _check_model_health(self) -> Dict[str, Any]:
        """Check model availability and performance."""
        try:
            if not self.supabase.is_connected():
                return {
                    "status": "warning",
                    "message": "Cannot check models - database not connected"
                }
            
            # Get active models
            result = self.supabase.client.table('model_registry').select('*').eq('is_active', True).execute()
            
            if not result.data:
                return {
                    "status": "warning",
                    "message": "No active models found",
                    "active_models": 0
                }
            
            active_models = result.data
            model_count = len(active_models)
            
            # Check model age and performance
            warnings = []
            for model in active_models:
                training_date = model.get('training_date')
                if training_date:
                    # Check if model is more than 7 days old
                    try:
                        model_date = datetime.fromisoformat(training_date.replace('Z', '+00:00'))
                        days_old = (datetime.now(timezone.utc) - model_date).days
                        if days_old > 7:
                            warnings.append(f"Model {model.get('model_version')} is {days_old} days old")
                    except:
                        pass
                
                # Check model performance
                accuracy = model.get('validation_accuracy')
                if accuracy and accuracy < 0.6:
                    warnings.append(f"Model {model.get('model_version')} has low accuracy: {accuracy}")
            
            status = "warning" if warnings else "healthy"
            message = f"Found {model_count} active model(s)"
            
            return {
                "status": status,
                "message": message,
                "active_models": model_count,
                "model_details": [
                    {
                        "version": m.get('model_version'),
                        "accuracy": m.get('validation_accuracy'),
                        "training_date": m.get('training_date')
                    } for m in active_models
                ],
                "warnings": warnings
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Model check failed: {str(e)}",
                "details": str(e)
            }
    
    def _check_data_freshness(self) -> Dict[str, Any]:
        """Check data freshness and availability."""
        try:
            if not self.supabase.is_connected():
                return {
                    "status": "warning",
                    "message": "Cannot check data freshness - database not connected"
                }
            
            # Check match data freshness
            match_result = self.supabase.client.table('matches').select('updated_at').order('updated_at', desc=True).limit(1).execute()
            
            # Check player stats freshness
            stats_result = self.supabase.client.table('player_stats').select('updated_at').order('updated_at', desc=True).limit(1).execute()
            
            # Check upcoming matches
            upcoming_result = self.supabase.client.table('upcoming_matches').select('updated_at').order('updated_at', desc=True).limit(1).execute()
            
            warnings = []
            data_status = {}
            
            # Check each data type
            for name, result in [
                ("matches", match_result),
                ("player_stats", stats_result),
                ("upcoming_matches", upcoming_result)
            ]:
                if result.data and result.data[0].get('updated_at'):
                    try:
                        last_update = datetime.fromisoformat(result.data[0]['updated_at'].replace('Z', '+00:00'))
                        hours_old = (datetime.now(timezone.utc) - last_update).total_seconds() / 3600
                        data_status[name] = {
                            "last_updated": last_update.isoformat(),
                            "hours_old": round(hours_old, 1)
                        }
                        
                        if hours_old > 24:
                            warnings.append(f"{name} data is {hours_old:.1f} hours old")
                    except:
                        warnings.append(f"Cannot parse {name} update time")
                else:
                    warnings.append(f"No {name} data found")
                    data_status[name] = {"last_updated": None, "hours_old": None}
            
            status = "warning" if warnings else "healthy"
            message = "Data freshness check completed"
            
            return {
                "status": status,
                "message": message,
                "data_status": data_status,
                "warnings": warnings
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Data freshness check failed: {str(e)}",
                "details": str(e)
            }
    
    def _save_health_check(self, health_status: Dict[str, Any]):
        """Save health check results to database."""
        try:
            if not self.supabase.is_connected():
                return
            
            # Save overall health status
            health_data = {
                "check_type": "system_overall",
                "status": health_status["overall_status"],
                "message": f"Health check completed with {len(health_status['errors'])} errors and {len(health_status['warnings'])} warnings",
                "metrics": {
                    "error_count": len(health_status["errors"]),
                    "warning_count": len(health_status["warnings"]),
                    "checks_performed": len(health_status["checks"])
                },
                "checked_at": health_status["timestamp"]
            }
            
            self.supabase.client.table('system_health').insert(health_data).execute()
            
            # Save individual check results
            for check_name, check_result in health_status["checks"].items():
                check_data = {
                    "check_type": check_name,
                    "status": check_result["status"],
                    "message": check_result["message"],
                    "metrics": check_result.get("metrics", {}),
                    "checked_at": health_status["timestamp"]
                }
                self.supabase.client.table('system_health').insert(check_data).execute()
            
        except Exception as e:
            logger.warning(f"Failed to save health check to database: {str(e)}")
    
    def get_health_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get health check history for the specified time period."""
        try:
            if not self.supabase.is_connected():
                return {"error": "Database not connected"}
            
            cutoff_time = datetime.now(timezone.utc).replace(
                hour=datetime.now(timezone.utc).hour - hours,
                minute=0, second=0, microsecond=0
            )
            
            result = self.supabase.client.table('system_health').select('*').gte(
                'checked_at', cutoff_time.isoformat()
            ).order('checked_at', desc=True).execute()
            
            return {
                "time_period_hours": hours,
                "health_checks": result.data if result.data else [],
                "total_checks": len(result.data) if result.data else 0
            }
            
        except Exception as e:
            return {"error": f"Failed to get health history: {str(e)}"}


# Global health monitor instance
health_monitor = HealthMonitorService()
