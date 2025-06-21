"""
Enhanced Match Prediction Service with Model Versioning and Best Model Selection
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import json

from services.match_prediction_service import MatchPredictionService
from core.data.processors.match_prediction_model import MatchPredictionModel
from utils.logging import log_execution_time, log_exceptions
from config.logging_config import get_data_fetcher_logger
from config.settings import OUTPUT_DIR

logger = get_data_fetcher_logger()


class EnhancedMatchPredictionService(MatchPredictionService):
    """
    Enhanced prediction service with model versioning and automatic best model selection.
    """
    
    def __init__(self, model_name: str = "nba2k_match_predictor"):
        super().__init__(model_name)
        self.models_dir = Path(OUTPUT_DIR) / "models"
        self.models_dir.mkdir(exist_ok=True)
        
    def _generate_model_version(self) -> str:
        """Generate a new version number for the model."""
        existing_versions = self._get_existing_versions()
        if not existing_versions:
            return "v1.0.0"
        
        # Get the latest version and increment patch number
        latest_version = max(existing_versions, key=lambda v: self._version_to_tuple(v))
        major, minor, patch = self._version_to_tuple(latest_version)
        return f"v{major}.{minor}.{patch + 1}"
    
    def _version_to_tuple(self, version: str) -> Tuple[int, int, int]:
        """Convert version string to tuple for comparison."""
        version_clean = version.replace('v', '')
        parts = version_clean.split('.')
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    
    def _get_existing_versions(self) -> List[str]:
        """Get list of existing model versions."""
        versions = []
        for model_file in self.models_dir.glob(f"{self.model_name}_v*.joblib"):
            version = model_file.stem.replace(f"{self.model_name}_", "")
            versions.append(version)
        return versions
    
    def _get_model_metadata_path(self, version: str) -> Path:
        """Get path for model metadata file."""
        return self.models_dir / f"{self.model_name}_{version}_metadata.json"
    
    def _save_model_metadata(self, version: str, metrics: Dict, training_info: Dict):
        """Save model metadata including performance metrics."""
        metadata = {
            "version": version,
            "model_name": self.model_name,
            "training_date": datetime.now().isoformat(),
            "performance_metrics": metrics,
            "training_info": training_info,
            "is_active": False
        }
        
        metadata_path = self._get_model_metadata_path(version)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_model_metadata(self, version: str) -> Dict:
        """Load model metadata."""
        metadata_path = self._get_model_metadata_path(version)
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def _get_best_model_version(self, metric: str = "val_winner_accuracy") -> Optional[str]:
        """
        Find the best performing model version based on specified metric.
        
        Args:
            metric: Metric to optimize (val_winner_accuracy, val_home_mae, etc.)
            
        Returns:
            Best model version string or None if no models found
        """
        versions = self._get_existing_versions()
        if not versions:
            return None
        
        best_version = None
        best_score = None
        
        for version in versions:
            metadata = self._load_model_metadata(version)
            metrics = metadata.get("performance_metrics", {})
            
            if metric not in metrics:
                continue
                
            score = metrics[metric]
            
            # For accuracy metrics, higher is better
            # For error metrics (MAE, MSE), lower is better
            is_accuracy_metric = "accuracy" in metric.lower()
            
            if best_score is None or (
                (is_accuracy_metric and score > best_score) or 
                (not is_accuracy_metric and score < best_score)
            ):
                best_score = score
                best_version = version
        
        return best_version
    
    def _set_active_model(self, version: str):
        """Set a model version as active and deactivate others."""
        versions = self._get_existing_versions()
        
        for v in versions:
            metadata = self._load_model_metadata(v)
            metadata["is_active"] = (v == version)
            
            metadata_path = self._get_model_metadata_path(v)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _get_active_model_version(self) -> Optional[str]:
        """Get the currently active model version."""
        versions = self._get_existing_versions()
        
        for version in versions:
            metadata = self._load_model_metadata(version)
            if metadata.get("is_active", False):
                return version
        
        # If no active model, return the best one
        return self._get_best_model_version()
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def train_model_with_versioning(self, training_df: pd.DataFrame, 
                                   auto_activate: bool = True,
                                   performance_threshold: float = 0.6) -> Tuple[str, Dict]:
        """
        Train a new model version and optionally activate it if it performs well.
        
        Args:
            training_df: Training data
            auto_activate: Whether to automatically activate if model performs well
            performance_threshold: Minimum accuracy threshold for auto-activation
            
        Returns:
            Tuple of (version, metrics)
        """
        # Generate new version
        version = self._generate_model_version()
        logger.info(f"Training new model version: {version}")
        
        # Train model
        model = MatchPredictionModel()
        metrics = model.train(training_df)
        
        # Save model with version
        versioned_model_name = f"{self.model_name}_{version}"
        model.save_model(versioned_model_name)
        
        # Save metadata
        training_info = {
            "training_samples": len(training_df),
            "feature_count": len(training_df.columns) - 2,  # Subtract target columns
            "days_back": getattr(self, '_last_days_back', None),
            "min_matches_per_player": getattr(self, '_last_min_matches', None)
        }
        
        self._save_model_metadata(version, metrics, training_info)
        
        # Auto-activate if performance is good enough
        val_accuracy = metrics.get("val_winner_accuracy", 0)
        if auto_activate and val_accuracy >= performance_threshold:
            # Compare with current best model
            current_best = self._get_best_model_version()
            if current_best is None:
                # First model, activate it
                self._set_active_model(version)
                logger.info(f"Activated model {version} as first model (accuracy: {val_accuracy:.3f})")
            else:
                # Compare with current best
                current_best_metadata = self._load_model_metadata(current_best)
                current_best_accuracy = current_best_metadata.get("performance_metrics", {}).get("val_winner_accuracy", 0)
                
                if val_accuracy > current_best_accuracy:
                    self._set_active_model(version)
                    logger.info(f"Activated model {version} as new best (accuracy: {val_accuracy:.3f} > {current_best_accuracy:.3f})")
                else:
                    logger.info(f"Model {version} trained but not activated (accuracy: {val_accuracy:.3f} <= {current_best_accuracy:.3f})")
        else:
            logger.info(f"Model {version} trained but not activated (accuracy: {val_accuracy:.3f} < threshold: {performance_threshold})")
        
        return version, metrics
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def predict_with_best_model(self, load_model: bool = True) -> pd.DataFrame:
        """
        Make predictions using the best available model.
        
        Args:
            load_model: Whether to load the model first
            
        Returns:
            DataFrame with predictions
        """
        # Get the active or best model version
        active_version = self._get_active_model_version()
        if not active_version:
            raise ValueError("No trained models found. Train a model first.")
        
        best_version = self._get_best_model_version()
        
        # Use active model if it exists, otherwise use best model
        model_version = active_version or best_version
        logger.info(f"Using model version: {model_version}")
        
        # Load the specific model version
        if load_model:
            versioned_model_name = f"{self.model_name}_{model_version}"
            try:
                self.model.load_model(versioned_model_name)
                logger.info(f"Loaded model: {versioned_model_name}")
            except FileNotFoundError:
                raise ValueError(f"Model {versioned_model_name} not found.")
        
        # Make predictions using parent class method
        return super().predict_upcoming_matches(load_model=False)  # Don't reload model
    
    def list_model_versions(self) -> pd.DataFrame:
        """
        List all available model versions with their performance metrics.
        
        Returns:
            DataFrame with model versions and metrics
        """
        versions = self._get_existing_versions()
        if not versions:
            return pd.DataFrame()
        
        models_info = []
        for version in versions:
            metadata = self._load_model_metadata(version)
            
            info = {
                "version": version,
                "training_date": metadata.get("training_date", "Unknown"),
                "is_active": metadata.get("is_active", False),
                "val_winner_accuracy": metadata.get("performance_metrics", {}).get("val_winner_accuracy", None),
                "val_home_mae": metadata.get("performance_metrics", {}).get("val_home_mae", None),
                "val_away_mae": metadata.get("performance_metrics", {}).get("val_away_mae", None),
                "training_samples": metadata.get("training_info", {}).get("training_samples", None),
                "feature_count": metadata.get("training_info", {}).get("feature_count", None)
            }
            models_info.append(info)
        
        df = pd.DataFrame(models_info)
        return df.sort_values("val_winner_accuracy", ascending=False) if not df.empty else df
    
    def activate_model_version(self, version: str):
        """Manually activate a specific model version."""
        if version not in self._get_existing_versions():
            raise ValueError(f"Model version {version} not found")
        
        self._set_active_model(version)
        logger.info(f"Activated model version: {version}")
    
    def compare_model_versions(self, version1: str, version2: str) -> Dict:
        """
        Compare performance between two model versions.
        
        Args:
            version1: First model version
            version2: Second model version
            
        Returns:
            Comparison results
        """
        if version1 not in self._get_existing_versions():
            raise ValueError(f"Model version {version1} not found")
        if version2 not in self._get_existing_versions():
            raise ValueError(f"Model version {version2} not found")
        
        metadata1 = self._load_model_metadata(version1)
        metadata2 = self._load_model_metadata(version2)
        
        metrics1 = metadata1.get("performance_metrics", {})
        metrics2 = metadata2.get("performance_metrics", {})
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "winner_accuracy_diff": metrics1.get("val_winner_accuracy", 0) - metrics2.get("val_winner_accuracy", 0),
            "home_mae_diff": metrics1.get("val_home_mae", 0) - metrics2.get("val_home_mae", 0),
            "away_mae_diff": metrics1.get("val_away_mae", 0) - metrics2.get("val_away_mae", 0),
            "better_model": version1 if metrics1.get("val_winner_accuracy", 0) > metrics2.get("val_winner_accuracy", 0) else version2
        }
        
        return comparison
