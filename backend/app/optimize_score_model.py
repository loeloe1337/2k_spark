"""
Script to optimize the score prediction model using Bayesian optimization.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from datetime import datetime

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from config.settings import (
    PLAYER_STATS_FILE, MATCH_HISTORY_FILE, MODELS_DIR, DEFAULT_RANDOM_STATE
)
from config.logging_config import get_model_tuning_logger
from utils.logging import log_execution_time, log_exceptions
from core.models.score_prediction import ScorePredictionModel
from core.models.registry import ScoreModelRegistry
from core.optimization.bayesian_optimizer import BayesianOptimizer

logger = get_model_tuning_logger()


@log_execution_time(logger)
@log_exceptions(logger)
def load_data():
    """
    Load player stats and match history data.

    Returns:
        tuple: (player_stats, matches)
    """
    logger.info(f"Loading player stats from {PLAYER_STATS_FILE}")
    with open(PLAYER_STATS_FILE, 'r', encoding='utf-8') as f:
        player_stats = json.load(f)

    logger.info(f"Loading match history from {MATCH_HISTORY_FILE}")
    with open(MATCH_HISTORY_FILE, 'r', encoding='utf-8') as f:
        matches = json.load(f)

    return player_stats, matches


@log_execution_time(logger)
@log_exceptions(logger)
def optimize_score_model(n_trials=50, test_size=0.2, random_state=DEFAULT_RANDOM_STATE):
    """
    Optimize the score prediction model using Bayesian optimization.

    Args:
        n_trials (int): Number of optimization trials
        test_size (float): Proportion of data to use for testing
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (best_params, best_score, best_model)
    """
    # Load data
    player_stats, matches = load_data()

    # Define parameter space for score prediction model
    param_space = {
        # Ensemble type selection
        'ensemble_type': {
            'type': 'categorical',
            'choices': ['stacking', 'voting']
        },
        
        # Advanced validation option
        'use_advanced_validation': {
            'type': 'categorical',
            'choices': [True, False]
        },
        
        # Feature engineering options
        'use_momentum_features': {
            'type': 'categorical',
            'choices': [True, False]
        },
        'use_streak_features': {
            'type': 'categorical',
            'choices': [True, False]
        },
        'use_efficiency_features': {
            'type': 'categorical',
            'choices': [True, False]
        },
        'momentum_window': {
            'type': 'integer',
            'low': 3,
            'high': 10
        },
        'streak_window': {
            'type': 'integer',
            'low': 3,
            'high': 8
        },
        
        # XGBoost parameters for home model
        'xgb_home_n_estimators': {
            'type': 'integer',
            'low': 50,
            'high': 500
        },
        'xgb_home_learning_rate': {
            'type': 'real',
            'low': 0.01,
            'high': 0.3,
            'prior': 'log-uniform'
        },
        'xgb_home_max_depth': {
            'type': 'integer',
            'low': 3,
            'high': 10
        },
        'xgb_home_subsample': {
            'type': 'real',
            'low': 0.5,
            'high': 1.0
        },
        'xgb_home_colsample_bytree': {
            'type': 'real',
            'low': 0.5,
            'high': 1.0
        },

        # Gradient Boosting parameters for home model
        'gb_home_n_estimators': {
            'type': 'integer',
            'low': 50,
            'high': 500
        },
        'gb_home_learning_rate': {
            'type': 'real',
            'low': 0.01,
            'high': 0.3,
            'prior': 'log-uniform'
        },
        'gb_home_max_depth': {
            'type': 'integer',
            'low': 3,
            'high': 10
        },
        'gb_home_subsample': {
            'type': 'real',
            'low': 0.5,
            'high': 1.0
        },

        # Ridge parameters for home model
        'ridge_home_alpha': {
            'type': 'real',
            'low': 0.01,
            'high': 10.0,
            'prior': 'log-uniform'
        },

        # Lasso parameters for home model
        'lasso_home_alpha': {
            'type': 'real',
            'low': 0.001,
            'high': 1.0,
            'prior': 'log-uniform'
        },

        # Final estimator parameters for home model
        'final_home_alpha': {
            'type': 'real',
            'low': 0.01,
            'high': 10.0,
            'prior': 'log-uniform'
        },

        # XGBoost parameters for away model
        'xgb_away_n_estimators': {
            'type': 'integer',
            'low': 50,
            'high': 500
        },
        'xgb_away_learning_rate': {
            'type': 'real',
            'low': 0.01,
            'high': 0.3,
            'prior': 'log-uniform'
        },
        'xgb_away_max_depth': {
            'type': 'integer',
            'low': 3,
            'high': 10
        },
        'xgb_away_subsample': {
            'type': 'real',
            'low': 0.5,
            'high': 1.0
        },
        'xgb_away_colsample_bytree': {
            'type': 'real',
            'low': 0.5,
            'high': 1.0
        },

        # Gradient Boosting parameters for away model
        'gb_away_n_estimators': {
            'type': 'integer',
            'low': 50,
            'high': 500
        },
        'gb_away_learning_rate': {
            'type': 'real',
            'low': 0.01,
            'high': 0.3,
            'prior': 'log-uniform'
        },
        'gb_away_max_depth': {
            'type': 'integer',
            'low': 3,
            'high': 10
        },
        'gb_away_subsample': {
            'type': 'real',
            'low': 0.5,
            'high': 1.0
        },

        # Ridge parameters for away model
        'ridge_away_alpha': {
            'type': 'real',
            'low': 0.01,
            'high': 10.0,
            'prior': 'log-uniform'
        },

        # Lasso parameters for away model
        'lasso_away_alpha': {
            'type': 'real',
            'low': 0.001,
            'high': 1.0,
            'prior': 'log-uniform'
        },

        # Final estimator parameters for away model
        'final_away_alpha': {
            'type': 'real',
            'low': 0.01,
            'high': 10.0,
            'prior': 'log-uniform'
        }
    }

    # Create optimizer
    optimizer = BayesianOptimizer(
        model_class=OptimizedScorePredictionModel,
        param_space=param_space,
        random_state=random_state
    )

    # Run optimization
    best_params, best_score, best_model = optimizer.optimize(
        player_stats=player_stats,
        matches=matches,
        n_trials=n_trials,
        test_size=test_size,
        scoring='neg_mean_absolute_error'
    )

    # Save the best model
    model_path = os.path.join(MODELS_DIR, f"optimized_score_model_{best_model.model_id}.pkl")
    info_path = os.path.join(MODELS_DIR, f"optimized_score_model_info_{best_model.model_id}.json")
    best_model.save(model_path, info_path)

    # Update model registry
    registry = ScoreModelRegistry(MODELS_DIR)
    registry.add_model(
        model_id=best_model.model_id,
        model_path=model_path,
        info_path=info_path,
        total_score_mae=-best_score  # Convert back to positive MAE
    )

    return best_params, best_score, best_model


class OptimizedScorePredictionModel(ScorePredictionModel):
    """
    Score prediction model with optimized hyperparameters.
    """

    def __init__(
        self,
        model_id=None,
        random_state=DEFAULT_RANDOM_STATE,
        ensemble_type='stacking',
        use_advanced_validation=False,
        use_momentum_features=True,
        use_streak_features=True,
        use_efficiency_features=True,
        momentum_window=5,
        streak_window=5,
        xgb_home_n_estimators=100,
        xgb_home_learning_rate=0.1,
        xgb_home_max_depth=5,
        xgb_home_subsample=0.8,
        xgb_home_colsample_bytree=0.8,
        gb_home_n_estimators=100,
        gb_home_learning_rate=0.1,
        gb_home_max_depth=5,
        gb_home_subsample=0.8,
        ridge_home_alpha=1.0,
        lasso_home_alpha=0.1,
        final_home_alpha=0.5,
        xgb_away_n_estimators=100,
        xgb_away_learning_rate=0.1,
        xgb_away_max_depth=5,
        xgb_away_subsample=0.8,
        xgb_away_colsample_bytree=0.8,
        gb_away_n_estimators=100,
        gb_away_learning_rate=0.1,
        gb_away_max_depth=5,
        gb_away_subsample=0.8,
        ridge_away_alpha=1.0,
        lasso_away_alpha=0.1,
        final_away_alpha=0.5
    ):
        """
        Initialize the optimized score prediction model.

        Args:
            model_id (str): Model ID
            random_state (int): Random state for reproducibility
            xgb_home_n_estimators (int): Number of estimators for home XGBoost model
            xgb_home_learning_rate (float): Learning rate for home XGBoost model
            xgb_home_max_depth (int): Maximum depth for home XGBoost model
            xgb_home_subsample (float): Subsample ratio for home XGBoost model
            xgb_home_colsample_bytree (float): Column subsample ratio for home XGBoost model
            gb_home_n_estimators (int): Number of estimators for home Gradient Boosting model
            gb_home_learning_rate (float): Learning rate for home Gradient Boosting model
            gb_home_max_depth (int): Maximum depth for home Gradient Boosting model
            gb_home_subsample (float): Subsample ratio for home Gradient Boosting model
            ridge_home_alpha (float): Alpha for home Ridge model
            lasso_home_alpha (float): Alpha for home Lasso model
            final_home_alpha (float): Alpha for home final estimator
            xgb_away_n_estimators (int): Number of estimators for away XGBoost model
            xgb_away_learning_rate (float): Learning rate for away XGBoost model
            xgb_away_max_depth (int): Maximum depth for away XGBoost model
            xgb_away_subsample (float): Subsample ratio for away XGBoost model
            xgb_away_colsample_bytree (float): Column subsample ratio for away XGBoost model
            gb_away_n_estimators (int): Number of estimators for away Gradient Boosting model
            gb_away_learning_rate (float): Learning rate for away Gradient Boosting model
            gb_away_max_depth (int): Maximum depth for away Gradient Boosting model
            gb_away_subsample (float): Subsample ratio for away Gradient Boosting model
            ridge_away_alpha (float): Alpha for away Ridge model
            lasso_away_alpha (float): Alpha for away Lasso model
            final_away_alpha (float): Alpha for away final estimator
        """
        # Initialize base class with new parameters
        super().__init__(
            model_id=model_id, 
            random_state=random_state,
            ensemble_type=ensemble_type,
            use_advanced_validation=use_advanced_validation
        )
        
        # Store feature engineering parameters
        self.use_momentum_features = use_momentum_features
        self.use_streak_features = use_streak_features
        self.use_efficiency_features = use_efficiency_features
        self.momentum_window = momentum_window
        self.streak_window = streak_window
        
        # Update feature engineer configuration
        self.feature_engineer.config.update({
            'use_momentum_features': use_momentum_features,
            'use_streak_features': use_streak_features,
            'use_efficiency_features': use_efficiency_features,
            'momentum_window': momentum_window,
            'streak_window': streak_window
        })

        # Store hyperparameters
        self.xgb_home_n_estimators = xgb_home_n_estimators
        self.xgb_home_learning_rate = xgb_home_learning_rate
        self.xgb_home_max_depth = xgb_home_max_depth
        self.xgb_home_subsample = xgb_home_subsample
        self.xgb_home_colsample_bytree = xgb_home_colsample_bytree
        self.gb_home_n_estimators = gb_home_n_estimators
        self.gb_home_learning_rate = gb_home_learning_rate
        self.gb_home_max_depth = gb_home_max_depth
        self.gb_home_subsample = gb_home_subsample
        self.ridge_home_alpha = ridge_home_alpha
        self.lasso_home_alpha = lasso_home_alpha
        self.final_home_alpha = final_home_alpha
        self.xgb_away_n_estimators = xgb_away_n_estimators
        self.xgb_away_learning_rate = xgb_away_learning_rate
        self.xgb_away_max_depth = xgb_away_max_depth
        self.xgb_away_subsample = xgb_away_subsample
        self.xgb_away_colsample_bytree = xgb_away_colsample_bytree
        self.gb_away_n_estimators = gb_away_n_estimators
        self.gb_away_learning_rate = gb_away_learning_rate
        self.gb_away_max_depth = gb_away_max_depth
        self.gb_away_subsample = gb_away_subsample
        self.ridge_away_alpha = ridge_away_alpha
        self.lasso_away_alpha = lasso_away_alpha
        self.final_away_alpha = final_away_alpha

        # Models will be created by parent class with ensemble_type
        # Update model info
        self.model_info["parameters"] = {
            "random_state": random_state,
            "ensemble_type": ensemble_type,
            "use_advanced_validation": use_advanced_validation,
            "use_momentum_features": use_momentum_features,
            "use_streak_features": use_streak_features,
            "use_efficiency_features": use_efficiency_features,
            "momentum_window": momentum_window,
            "streak_window": streak_window,
            "xgb_home_n_estimators": xgb_home_n_estimators,
            "xgb_home_learning_rate": xgb_home_learning_rate,
            "xgb_home_max_depth": xgb_home_max_depth,
            "xgb_home_subsample": xgb_home_subsample,
            "xgb_home_colsample_bytree": xgb_home_colsample_bytree,
            "gb_home_n_estimators": gb_home_n_estimators,
            "gb_home_learning_rate": gb_home_learning_rate,
            "gb_home_max_depth": gb_home_max_depth,
            "gb_home_subsample": gb_home_subsample,
            "ridge_home_alpha": ridge_home_alpha,
            "lasso_home_alpha": lasso_home_alpha,
            "final_home_alpha": final_home_alpha,
            "xgb_away_n_estimators": xgb_away_n_estimators,
            "xgb_away_learning_rate": xgb_away_learning_rate,
            "xgb_away_max_depth": xgb_away_max_depth,
            "xgb_away_subsample": xgb_away_subsample,
            "xgb_away_colsample_bytree": xgb_away_colsample_bytree,
            "gb_away_n_estimators": gb_away_n_estimators,
            "gb_away_learning_rate": gb_away_learning_rate,
            "gb_away_max_depth": gb_away_max_depth,
            "gb_away_subsample": gb_away_subsample,
            "ridge_away_alpha": ridge_away_alpha,
            "lasso_away_alpha": lasso_away_alpha,
            "final_away_alpha": final_away_alpha
        }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimize score prediction model")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for testing")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random state for reproducibility")
    args = parser.parse_args()

    # Create models directory if it doesn't exist
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    # Run optimization
    best_params, best_score, best_model = optimize_score_model(
        n_trials=args.n_trials,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Print results
    print(f"Best parameters: {best_params}")
    print(f"Best score (negative MAE): {best_score}")
    print(f"Best model ID: {best_model.model_id}")
    print(f"Best model saved to: {os.path.join(MODELS_DIR, f'optimized_score_model_{best_model.model_id}.pkl')}")
    print(f"Best model info saved to: {os.path.join(MODELS_DIR, f'optimized_score_model_info_{best_model.model_id}.json')}")
