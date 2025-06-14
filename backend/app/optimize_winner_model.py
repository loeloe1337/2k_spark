"""
Script to optimize the winner prediction model using Bayesian optimization.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from config.settings import (
    PLAYER_STATS_FILE, MATCH_HISTORY_FILE, MODELS_DIR, DEFAULT_RANDOM_STATE
)
from config.logging_config import get_model_tuning_logger
from utils.logging import log_execution_time, log_exceptions
from core.models.winner_prediction import WinnerPredictionModel
from core.models.registry import ModelRegistry
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
def optimize_winner_model(n_trials=20, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, 
                         quick_mode=True, sample_size=0.5):
    """
    Optimize the winner prediction model using Bayesian optimization.
    
    Args:
        n_trials (int): Number of optimization trials (reduced default for faster training)
        test_size (float): Proportion of data to use for testing
        random_state (int): Random state for reproducibility
        quick_mode (bool): Use faster optimization settings
        sample_size (float): Fraction of data to use for optimization (speeds up training)
        
    Returns:
        tuple: (best_params, best_score, best_model)
    """
    # Load data
    player_stats, matches = load_data()
    
    # Define parameter space for winner prediction model
    # Optimized ranges for faster training while maintaining effectiveness
    if quick_mode:
        param_space = {
            # Reduced Random Forest parameters for faster training
            'n_estimators': {
                'type': 'integer',
                'low': 50,
                'high': 200  # Reduced from 500
            },
            'max_depth': {
                'type': 'integer',
                'low': 5,
                'high': 15  # Narrowed range
            },
            'min_samples_split': {
                'type': 'integer',
                'low': 2,
                'high': 10  # Reduced from 20
            },
            'min_samples_leaf': {
                'type': 'integer',
                'low': 1,
                'high': 5  # Reduced from 10
            },
            'max_features': {
                'type': 'categorical',
                'categories': ['sqrt', 'log2']  # Removed None for faster training
            },
            'bootstrap': {
                'type': 'categorical',
                'categories': [True]  # Fixed to True for faster training
            },
            'class_weight': {
                'type': 'categorical',
                'categories': ['balanced', None]  # Reduced options
            }
        }
    else:
        # Full parameter space for thorough optimization
        param_space = {
            'n_estimators': {
                'type': 'integer',
                'low': 50,
                'high': 500
            },
            'max_depth': {
                'type': 'integer',
                'low': 3,
                'high': 20
            },
            'min_samples_split': {
                'type': 'integer',
                'low': 2,
                'high': 20
            },
            'min_samples_leaf': {
                'type': 'integer',
                'low': 1,
                'high': 10
            },
            'max_features': {
                'type': 'categorical',
                'categories': ['sqrt', 'log2', None]
            },
            'bootstrap': {
                'type': 'categorical',
                'categories': [True, False]
            },
            'class_weight': {
                'type': 'categorical',
                'categories': ['balanced', 'balanced_subsample', None]
            }
        }
    
    # Sample data for faster optimization if requested
    if sample_size < 1.0 and len(matches) > 100:
        import random
        random.seed(random_state)
        sample_matches = random.sample(matches, int(len(matches) * sample_size))
        logger.info(f"Using {len(sample_matches)} matches for optimization (sample_size={sample_size})")
    else:
        sample_matches = matches
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        model_class=OptimizedWinnerPredictionModel,
        param_space=param_space,
        random_state=random_state
    )
    
    # Run optimization with sampled data
    logger.info(f"Starting optimization with {n_trials} trials")
    logger.info(f"Quick mode: {quick_mode}")
    logger.info(f"Sample size: {sample_size if sample_size < 1.0 else 'Full dataset'}")
    
    # Set early stopping parameters based on mode
    early_stopping_rounds = 5 if quick_mode else 10
    min_improvement = 0.005 if quick_mode else 0.001
    
    best_params, best_score, best_model = optimizer.optimize(
        player_stats=player_stats,
        matches=sample_matches,
        n_trials=n_trials,
        test_size=test_size,
        scoring='accuracy',
        early_stopping_rounds=early_stopping_rounds,
        min_improvement=min_improvement
    )
    
    # If we used sampled data, retrain the best model on full data
    if sample_size < 1.0 and len(matches) > 100:
        logger.info("Retraining best model on full dataset...")
        final_model = OptimizedWinnerPredictionModel(**best_params)
        final_model.train(player_stats, matches, test_size=test_size)
        best_model = final_model
        # Update score with full data performance
        best_score = best_model.model_info.get("metrics", {}).get('accuracy', best_score)
    
    # Save the best model
    model_path = os.path.join(MODELS_DIR, f"optimized_winner_model_{best_model.model_id}.pkl")
    info_path = os.path.join(MODELS_DIR, f"optimized_winner_model_info_{best_model.model_id}.json")
    best_model.save(model_path, info_path)
    
    # Update model registry
    registry = ModelRegistry(MODELS_DIR)
    registry.add_model(
        model_id=best_model.model_id,
        model_path=model_path,
        info_path=info_path,
        accuracy=best_score
    )
    
    return best_params, best_score, best_model


class OptimizedWinnerPredictionModel(WinnerPredictionModel):
    """
    Winner prediction model with optimized hyperparameters.
    """
    
    def __init__(
        self,
        model_id=None,
        random_state=DEFAULT_RANDOM_STATE,
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight=None,
        feature_config=None
    ):
        """
        Initialize the optimized winner prediction model.
        
        Args:
            model_id (str): Model ID
            random_state (int): Random state for reproducibility
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
            min_samples_split (int): Minimum number of samples required to split an internal node
            min_samples_leaf (int): Minimum number of samples required to be at a leaf node
            max_features (str): Number of features to consider when looking for the best split
            bootstrap (bool): Whether bootstrap samples are used when building trees
            class_weight (str): Weights associated with classes
            feature_config (dict): Feature configuration dictionary
        """
        # Initialize base class with basic parameters
        super().__init__(
            model_id=model_id,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            feature_config=feature_config
        )
        
        # Store additional hyperparameters
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        
        # Update model info
        self.model_info["parameters"].update({
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "class_weight": class_weight
        })
        
        # Initialize model with all hyperparameters
        # Add n_jobs=-1 for parallel processing to speed up training
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1  # Use all available cores for faster training
        )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimize winner prediction model")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of optimization trials (reduced default for faster training)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for testing")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random state for reproducibility")
    parser.add_argument("--quick-mode", action="store_true", default=True, help="Use faster optimization settings")
    parser.add_argument("--full-mode", action="store_true", help="Use full optimization settings (slower but more thorough)")
    parser.add_argument("--sample-size", type=float, default=0.7, help="Fraction of data to use for optimization (0.1-1.0)")
    args = parser.parse_args()
    
    # Determine mode
    quick_mode = not args.full_mode
    
    # Create models directory if it doesn't exist
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Run optimization
    best_params, best_score, best_model = optimize_winner_model(
        n_trials=args.n_trials,
        test_size=args.test_size,
        random_state=args.random_state,
        quick_mode=quick_mode,
        sample_size=args.sample_size
    )
    
    # Print mode information
    mode_str = "Quick Mode" if quick_mode else "Full Mode"
    print(f"\nOptimization completed in {mode_str}")
    print(f"Trials: {args.n_trials}, Sample size: {args.sample_size}")
    
    # Print results
    print(f"Best parameters: {best_params}")
    print(f"Best score (accuracy): {best_score}")
    print(f"Best model ID: {best_model.model_id}")
    print(f"Best model saved to: {os.path.join(MODELS_DIR, f'optimized_winner_model_{best_model.model_id}.pkl')}")
    print(f"Best model info saved to: {os.path.join(MODELS_DIR, f'optimized_winner_model_info_{best_model.model_id}.json')}")
