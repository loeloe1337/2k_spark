#!/usr/bin/env python3
"""
Script to train score prediction and winner prediction models.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from config.settings import (
    PLAYER_STATS_FILE, MATCH_HISTORY_FILE, MODELS_DIR, DEFAULT_RANDOM_STATE
)
from config.logging_config import get_score_model_training_logger
from utils.logging import log_execution_time, log_exceptions
from core.models.score_prediction import ScorePredictionModel
from core.models.winner_prediction import WinnerPredictionModel
from core.models.registry import ScoreModelRegistry, ModelRegistry

logger = get_score_model_training_logger()


@log_execution_time(logger)
@log_exceptions(logger)
def load_data(dev_mode=False, dev_sample_size=200):
    """
    Load player stats and match history data.

    Args:
        dev_mode (bool): If True, load only a subset of data for faster training
        dev_sample_size (int): Number of matches to use in development mode

    Returns:
        tuple: (player_stats, matches)
    """
    logger.info(f"Loading player stats from {PLAYER_STATS_FILE}")
    with open(PLAYER_STATS_FILE, 'r', encoding='utf-8') as f:
        player_stats = json.load(f)

    logger.info(f"Loading match history from {MATCH_HISTORY_FILE}")
    with open(MATCH_HISTORY_FILE, 'r', encoding='utf-8') as f:
        matches = json.load(f)

    if dev_mode:
        # Reduce dataset size for faster training during development
        original_match_count = len(matches)
        matches = matches[:dev_sample_size]
        logger.info(f"Development mode: Using {len(matches)} matches out of {original_match_count} for faster training")
        
        if dev_sample_size < 150:
            logger.warning(f"Development sample size ({dev_sample_size}) is quite small. Consider using at least 150 matches for reliable feature extraction.")
        
        # Also reduce player stats to only those involved in the sampled matches
        match_player_ids = set()
        for match in matches:
            if 'homePlayer' in match and 'id' in match['homePlayer']:
                match_player_ids.add(str(match['homePlayer']['id']))
            if 'awayPlayer' in match and 'id' in match['awayPlayer']:
                match_player_ids.add(str(match['awayPlayer']['id']))
        
        filtered_player_stats = {pid: stats for pid, stats in player_stats.items() 
                               if pid in match_player_ids}
        logger.info(f"Development mode: Using {len(filtered_player_stats)} players out of {len(player_stats)}")
        player_stats = filtered_player_stats

    return player_stats, matches


@log_execution_time(logger)
@log_exceptions(logger)
def train_score_model(
    player_stats, 
    matches, 
    ensemble_type='stacking',
    use_advanced_validation=True,
    use_momentum_features=True,
    use_streak_features=True,
    use_efficiency_features=True,
    momentum_window=5,
    streak_window=5,
    test_size=0.2,
    random_state=DEFAULT_RANDOM_STATE
):
    """
    Train a score prediction model.

    Args:
        player_stats (dict): Player statistics data
        matches (list): Match history data
        ensemble_type (str): Type of ensemble ('stacking' or 'voting')
        use_advanced_validation (bool): Whether to use advanced validation
        use_momentum_features (bool): Whether to use momentum features
        use_streak_features (bool): Whether to use streak features
        use_efficiency_features (bool): Whether to use efficiency features
        momentum_window (int): Window size for momentum features
        streak_window (int): Window size for streak features
        test_size (float): Proportion of data to use for testing
        random_state (int): Random state for reproducibility

    Returns:
        ScorePredictionModel: Trained model
    """
    logger.info(f"Training score prediction model with ensemble_type={ensemble_type}")
    
    # Create model with enhanced features
    model = ScorePredictionModel(
        ensemble_type=ensemble_type,
        use_advanced_validation=use_advanced_validation,
        random_state=random_state
    )
    
    # Update feature engineering configuration
    model.feature_engineer.feature_config.update({
        'use_momentum_features': use_momentum_features,
        'use_streak_features': use_streak_features,
        'use_efficiency_features': use_efficiency_features,
        'momentum_window': momentum_window,
        'streak_window': streak_window
    })
    
    # Train the model
    model.train(player_stats, matches, test_size=test_size)
    
    # Evaluate the model
    metrics = model.evaluate(player_stats, matches)
    logger.info(f"Score model metrics: {metrics}")
    
    return model


@log_execution_time(logger)
@log_exceptions(logger)
def train_winner_model(
    player_stats, 
    matches, 
    ensemble_type='voting',
    use_advanced_validation=True,
    use_momentum_features=True,
    use_streak_features=True,
    use_efficiency_features=True,
    momentum_window=5,
    streak_window=5,
    test_size=0.2,
    random_state=DEFAULT_RANDOM_STATE
):
    """
    Train a winner prediction model.

    Args:
        player_stats (dict): Player statistics data
        matches (list): Match history data
        ensemble_type (str): Type of ensemble ('voting' or 'stacking')
        use_advanced_validation (bool): Whether to use advanced validation
        use_momentum_features (bool): Whether to use momentum features
        use_streak_features (bool): Whether to use streak features
        use_efficiency_features (bool): Whether to use efficiency features
        momentum_window (int): Window size for momentum features
        streak_window (int): Window size for streak features
        test_size (float): Proportion of data to use for testing
        random_state (int): Random state for reproducibility

    Returns:
        WinnerPredictionModel: Trained model
    """
    logger.info(f"Training winner prediction model with ensemble_type={ensemble_type}")
    
    # Create feature configuration
    feature_config = {
        'use_momentum_features': use_momentum_features,
        'use_streak_features': use_streak_features,
        'use_efficiency_features': use_efficiency_features,
        'momentum_window': momentum_window,
        'streak_window': streak_window
    }
    
    # Create model with enhanced features
    model = WinnerPredictionModel(
        ensemble_type=ensemble_type,
        use_advanced_validation=use_advanced_validation,
        feature_config=feature_config,
        random_state=random_state
    )
    
    # Train the model
    model.train(player_stats, matches, test_size=test_size)
    
    # Evaluate the model
    metrics = model.evaluate(player_stats, matches)
    logger.info(f"Winner model metrics: {metrics}")
    
    return model


@log_execution_time(logger)
@log_exceptions(logger)
def save_model(model, model_type, registry_class, player_stats=None, matches=None):
    """
    Save a trained model and update the registry.

    Args:
        model: Trained model instance
        model_type (str): Type of model ('score' or 'winner')
        registry_class: Registry class to use
        player_stats (dict): Player statistics dictionary (required for evaluation)
        matches (list): List of match data dictionaries (required for evaluation)

    Returns:
        tuple: (model_path, info_path)
    """
    # Generate file paths
    model_path = os.path.join(MODELS_DIR, f"{model_type}_model_{model.model_id}.pkl")
    info_path = os.path.join(MODELS_DIR, f"{model_type}_model_info_{model.model_id}.json")
    
    # Save the model
    model.save(model_path, info_path)
    logger.info(f"Saved {model_type} model to {model_path}")
    
    # Update registry
    registry = registry_class(MODELS_DIR)
    
    if model_type == 'score':
        # For score models, use total score MAE as the metric
        metrics = model.evaluate(player_stats, matches)
        metric_value = metrics.get('total_score_mae', float('inf'))
        registry.add_model(
            model_id=model.model_id,
            model_path=model_path,
            info_path=info_path,
            total_score_mae=metric_value
        )
    else:
        # For winner models, use accuracy as the metric
        metrics = model.evaluate(player_stats, matches)
        metric_value = metrics.get('accuracy', 0.0)
        registry.add_model(
            model_id=model.model_id,
            model_path=model_path,
            info_path=info_path,
            accuracy=metric_value
        )
    
    logger.info(f"Updated {model_type} model registry")
    return model_path, info_path


@log_execution_time(logger)
@log_exceptions(logger)
def train_all_models(
    ensemble_configs=None,
    feature_configs=None,
    test_size=0.2,
    random_state=DEFAULT_RANDOM_STATE,
    dev_mode=False,
    dev_sample_size=200
):
    """
    Train multiple model configurations.

    Args:
        ensemble_configs (list): List of ensemble configurations to try
        feature_configs (list): List of feature configurations to try
        test_size (float): Proportion of data to use for testing
        random_state (int): Random state for reproducibility
        dev_mode (bool): If True, use smaller dataset for faster training
        dev_sample_size (int): Number of matches to use in development mode

    Returns:
        dict: Dictionary containing trained models and their metrics
    """
    # Default configurations
    if ensemble_configs is None:
        ensemble_configs = [
            {'type': 'stacking', 'validation': True},
            {'type': 'voting', 'validation': True},
            {'type': 'stacking', 'validation': False}
        ]
    
    if feature_configs is None:
        feature_configs = [
            {
                'momentum': True, 'streak': True, 'efficiency': True,
                'momentum_window': 5, 'streak_window': 5
            },
            {
                'momentum': True, 'streak': False, 'efficiency': True,
                'momentum_window': 7, 'streak_window': 5
            },
            {
                'momentum': False, 'streak': True, 'efficiency': True,
                'momentum_window': 5, 'streak_window': 3
            }
        ]
    
    # Load data
    player_stats, matches = load_data(
        dev_mode=dev_mode,
        dev_sample_size=dev_sample_size
    )
    
    results = {
        'score_models': [],
        'winner_models': [],
        'training_timestamp': datetime.now().isoformat()
    }
    
    # Train different configurations
    for i, ensemble_config in enumerate(ensemble_configs):
        for j, feature_config in enumerate(feature_configs):
            config_name = f"config_{i}_{j}"
            logger.info(f"Training models with {config_name}")
            
            try:
                # Train score model
                score_model = train_score_model(
                    player_stats=player_stats,
                    matches=matches,
                    ensemble_type=ensemble_config['type'],
                    use_advanced_validation=ensemble_config['validation'],
                    use_momentum_features=feature_config['momentum'],
                    use_streak_features=feature_config['streak'],
                    use_efficiency_features=feature_config['efficiency'],
                    momentum_window=feature_config['momentum_window'],
                    streak_window=feature_config['streak_window'],
                    test_size=test_size,
                    random_state=random_state
                )
                
                # Save score model
                score_model_path, score_info_path = save_model(
                    score_model, 'score', ScoreModelRegistry, player_stats, matches
                )
                
                results['score_models'].append({
                    'config_name': config_name,
                    'model_id': score_model.model_id,
                    'model_path': score_model_path,
                    'info_path': score_info_path,
                    'metrics': score_model.evaluate(),
                    'ensemble_config': ensemble_config,
                    'feature_config': feature_config
                })
                
                # Train winner model
                winner_model = train_winner_model(
                    player_stats=player_stats,
                    matches=matches,
                    ensemble_type=ensemble_config['type'],
                    use_advanced_validation=ensemble_config['validation'],
                    use_momentum_features=feature_config['momentum'],
                    use_streak_features=feature_config['streak'],
                    use_efficiency_features=feature_config['efficiency'],
                    momentum_window=feature_config['momentum_window'],
                    streak_window=feature_config['streak_window'],
                    test_size=test_size,
                    random_state=random_state
                )
                
                # Save winner model
                winner_model_path, winner_info_path = save_model(
                    winner_model, 'winner', ModelRegistry, player_stats, matches
                )
                
                results['winner_models'].append({
                    'config_name': config_name,
                    'model_id': winner_model.model_id,
                    'model_path': winner_model_path,
                    'info_path': winner_info_path,
                    'metrics': winner_model.evaluate(),
                    'ensemble_config': ensemble_config,
                    'feature_config': feature_config
                })
                
                logger.info(f"Successfully trained models for {config_name}")
                
            except Exception as e:
                logger.error(f"Failed to train models for {config_name}: {str(e)}")
                continue
    
    # Save training results
    results_path = os.path.join(MODELS_DIR, f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training results saved to {results_path}")
    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train prediction models")
    parser.add_argument("--model-type", choices=['score', 'winner', 'all'], default='all',
                       help="Type of model to train")
    parser.add_argument("--ensemble-type", choices=['stacking', 'voting'], default='stacking',
                       help="Type of ensemble to use")
    parser.add_argument("--advanced-validation", action='store_true',
                       help="Use advanced validation techniques")
    parser.add_argument("--momentum-features", action='store_true', default=True,
                       help="Use momentum features")
    parser.add_argument("--streak-features", action='store_true', default=True,
                       help="Use streak features")
    parser.add_argument("--efficiency-features", action='store_true', default=True,
                       help="Use efficiency features")
    parser.add_argument("--momentum-window", type=int, default=5,
                       help="Window size for momentum features")
    parser.add_argument("--streak-window", type=int, default=5,
                       help="Window size for streak features")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proportion of data to use for testing")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE,
                       help="Random state for reproducibility")
    parser.add_argument("--train-multiple", action='store_true',
                       help="Train multiple model configurations")
    parser.add_argument("--dev-mode", action='store_true',
                       help="Development mode: use smaller dataset for faster training/debugging")
    parser.add_argument("--dev-sample-size", type=int, default=200,
                       help="Number of matches to use in development mode (default: 200, minimum recommended: 150)")
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    
    if args.train_multiple:
        # Train multiple configurations
        results = train_all_models(
            test_size=args.test_size,
            random_state=args.random_state,
            dev_mode=args.dev_mode,
            dev_sample_size=args.dev_sample_size
        )
        print(f"Trained {len(results['score_models'])} score models and {len(results['winner_models'])} winner models")
    else:
        # Load data
        player_stats, matches = load_data(
            dev_mode=args.dev_mode,
            dev_sample_size=args.dev_sample_size
        )
        
        if args.model_type in ['score', 'all']:
            # Train score model
            score_model = train_score_model(
                player_stats=player_stats,
                matches=matches,
                ensemble_type=args.ensemble_type,
                use_advanced_validation=args.advanced_validation,
                use_momentum_features=args.momentum_features,
                use_streak_features=args.streak_features,
                use_efficiency_features=args.efficiency_features,
                momentum_window=args.momentum_window,
                streak_window=args.streak_window,
                test_size=args.test_size,
                random_state=args.random_state
            )
            
            # Save score model
            score_model_path, score_info_path = save_model(
                score_model, 'score', ScoreModelRegistry, player_stats, matches
            )
            print(f"Score model saved to: {score_model_path}")
            print(f"Score model metrics: {score_model.evaluate(player_stats, matches)}")
        
        if args.model_type in ['winner', 'all']:
            # Train winner model
            winner_model = train_winner_model(
                player_stats=player_stats,
                matches=matches,
                ensemble_type=args.ensemble_type,
                use_advanced_validation=args.advanced_validation,
                use_momentum_features=args.momentum_features,
                use_streak_features=args.streak_features,
                use_efficiency_features=args.efficiency_features,
                momentum_window=args.momentum_window,
                streak_window=args.streak_window,
                test_size=args.test_size,
                random_state=args.random_state
            )
            
            # Save winner model
            winner_model_path, winner_info_path = save_model(
                winner_model, 'winner', ModelRegistry, player_stats, matches
            )
            print(f"Winner model saved to: {winner_model_path}")
            print(f"Winner model metrics: {winner_model.evaluate(player_stats, matches)}")