"""
Match prediction model for NBA 2K25 esports.
Unified model that predicts individual player scores to derive winner and total score.
"""

import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
from datetime import datetime

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent.parent.parent
sys.path.append(str(backend_dir))

# ML imports
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False

from utils.logging import log_execution_time, log_exceptions
from config.logging_config import get_data_fetcher_logger
from config.settings import OUTPUT_DIR

logger = get_data_fetcher_logger()


class MatchPredictionModel:
    """
    Unified prediction model for NBA 2K25 esports matches.
    Predicts individual player scores to derive consistent winner and total score predictions.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the prediction model.
        
        Args:
            model_dir: Directory to save/load models
        """
        if not HAS_ML_LIBS:
            raise ImportError("Required ML libraries not installed. Run: pip install xgboost scikit-learn")
        
        self.model_dir = Path(model_dir or OUTPUT_DIR) / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        
        # Model configuration
        self.model_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def train(self, training_df: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """
        Train the prediction model.
        
        Args:
            training_df: Training dataset with features and targets
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training model on {len(training_df)} samples")
        
        # Prepare features and targets
        feature_cols = [col for col in training_df.columns if col not in [
            'home_score', 'away_score', 'home_win', 'total_score',
            'match_id', 'home_player_id', 'away_player_id', 
            'home_player_name', 'away_player_name', 'fixture_start'
        ]]
        
        X = training_df[feature_cols].fillna(0)
        y = training_df[['home_score', 'away_score']].values
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train multi-output XGBoost model
        base_model = xgb.XGBRegressor(**self.model_params)
        self.model = MultiOutputRegressor(base_model)
        
        logger.info("Training XGBoost multi-output model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, train_pred, y_val, val_pred, 
                                        training_df.iloc[X_train.index], 
                                        training_df.iloc[X_val.index])
        
        self.is_trained = True
        logger.info(f"Model training completed. Validation accuracy: {metrics['val_winner_accuracy']:.3f}")
        
        return metrics
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def predict(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            prediction_df: DataFrame with features for prediction
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions for {len(prediction_df)} matches")
        
        # Prepare features
        X = prediction_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict scores
        score_predictions = self.model.predict(X_scaled)
        
        # Create results DataFrame
        results = prediction_df.copy()
        
        # Add score predictions
        results['predicted_home_score'] = score_predictions[:, 0]
        results['predicted_away_score'] = score_predictions[:, 1]
        
        # Derive winner predictions
        results['predicted_home_win'] = (score_predictions[:, 0] > score_predictions[:, 1]).astype(int)
        results['predicted_winner'] = results.apply(
            lambda row: row['home_player_name'] if row['predicted_home_win'] == 1 else row['away_player_name'],
            axis=1
        )
        
        # Calculate win probabilities based on score difference
        score_diff = score_predictions[:, 0] - score_predictions[:, 1]
        results['home_win_probability'] = self._score_diff_to_probability(score_diff)
        results['away_win_probability'] = 1 - results['home_win_probability']
        
        # Total score predictions
        results['predicted_total_score'] = score_predictions[:, 0] + score_predictions[:, 1]
        
        # Confidence metrics
        results['prediction_confidence'] = np.abs(score_diff) / (score_predictions[:, 0] + score_predictions[:, 1])
        
        logger.info(f"Generated predictions for {len(results)} matches")
        
        return results
    
    def _score_diff_to_probability(self, score_diff: np.ndarray) -> np.ndarray:
        """
        Convert score difference to win probability using sigmoid function.
        
        Args:
            score_diff: Array of score differences (home - away)
            
        Returns:
            Array of home win probabilities
        """
        # Use sigmoid function to convert score difference to probability
        # Adjust the scaling factor based on typical score ranges in your data
        scaling_factor = 0.1  # Adjust this based on your data
        probabilities = 1 / (1 + np.exp(-scaling_factor * score_diff))
        return probabilities
    
    def _calculate_metrics(self, y_train: np.ndarray, train_pred: np.ndarray,
                          y_val: np.ndarray, val_pred: np.ndarray,
                          train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive model metrics."""
        metrics = {}
        
        # Score prediction metrics
        metrics['train_home_mse'] = mean_squared_error(y_train[:, 0], train_pred[:, 0])
        metrics['train_away_mse'] = mean_squared_error(y_train[:, 1], train_pred[:, 1])
        metrics['val_home_mse'] = mean_squared_error(y_val[:, 0], val_pred[:, 0])
        metrics['val_away_mse'] = mean_squared_error(y_val[:, 1], val_pred[:, 1])
        
        metrics['train_home_mae'] = mean_absolute_error(y_train[:, 0], train_pred[:, 0])
        metrics['train_away_mae'] = mean_absolute_error(y_train[:, 1], train_pred[:, 1])
        metrics['val_home_mae'] = mean_absolute_error(y_val[:, 0], val_pred[:, 0])
        metrics['val_away_mae'] = mean_absolute_error(y_val[:, 1], val_pred[:, 1])
        
        # Winner prediction accuracy
        train_actual_winners = (y_train[:, 0] > y_train[:, 1]).astype(int)
        train_pred_winners = (train_pred[:, 0] > train_pred[:, 1]).astype(int)
        val_actual_winners = (y_val[:, 0] > y_val[:, 1]).astype(int)
        val_pred_winners = (val_pred[:, 0] > val_pred[:, 1]).astype(int)
        
        metrics['train_winner_accuracy'] = accuracy_score(train_actual_winners, train_pred_winners)
        metrics['val_winner_accuracy'] = accuracy_score(val_actual_winners, val_pred_winners)
        
        # Total score prediction metrics
        train_actual_total = y_train[:, 0] + y_train[:, 1]
        train_pred_total = train_pred[:, 0] + train_pred[:, 1]
        val_actual_total = y_val[:, 0] + y_val[:, 1]
        val_pred_total = val_pred[:, 0] + val_pred[:, 1]
        
        metrics['train_total_mse'] = mean_squared_error(train_actual_total, train_pred_total)
        metrics['val_total_mse'] = mean_squared_error(val_actual_total, val_pred_total)
        metrics['train_total_mae'] = mean_absolute_error(train_actual_total, train_pred_total)
        metrics['val_total_mae'] = mean_absolute_error(val_actual_total, val_pred_total)
        
        return metrics
    
    @log_exceptions(logger)
    def save_model(self, model_name: str = "match_prediction_model"):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_path = self.model_dir / f"{model_name}.joblib"
        scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
        features_path = self.model_dir / f"{model_name}_features.json"
        
        # Save model components
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f)
        
        logger.info(f"Model saved to {model_path}")
    
    @log_exceptions(logger)
    def load_model(self, model_name: str = "match_prediction_model"):
        """Load a trained model."""
        model_path = self.model_dir / f"{model_name}.joblib"
        scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
        features_path = self.model_dir / f"{model_name}_features.json"
        
        if not all(path.exists() for path in [model_path, scaler_path, features_path]):
            raise FileNotFoundError("Model files not found")
        
        # Load model components
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(features_path, 'r') as f:
            self.feature_columns = json.load(f)
        
        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get importance from first estimator (they should be similar for both outputs)
        importance_scores = self.model.estimators_[0].feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def cross_validate(self, training_df: pd.DataFrame, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation to assess model performance.
        
        Args:
            training_df: Training dataset
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation metrics
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Prepare data
        feature_cols = [col for col in training_df.columns if col not in [
            'home_score', 'away_score', 'home_win', 'total_score',
            'match_id', 'home_player_id', 'away_player_id', 
            'home_player_name', 'away_player_name', 'fixture_start'
        ]]
        
        X = training_df[feature_cols].fillna(0)
        y = training_df[['home_score', 'away_score']].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create model
        base_model = xgb.XGBRegressor(**self.model_params)
        model = MultiOutputRegressor(base_model)
        
        # Cross-validation for score prediction
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, 
                                   scoring='neg_mean_squared_error')
        
        return {
            'cv_mse_mean': -cv_scores.mean(),
            'cv_mse_std': cv_scores.std(),
            'cv_scores': -cv_scores
        }
