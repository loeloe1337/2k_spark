#!/usr/bin/env python3
"""
Hyperparameter optimization for score prediction model.
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from scipy.stats import randint, uniform
import json
import os
from datetime import datetime

from config.settings import DEFAULT_RANDOM_STATE
from config.logging_config import get_score_model_training_logger
from utils.logging import log_execution_time, log_exceptions
from core.models.score_prediction import ScorePredictionModel
from core.models.feature_engineering import FeatureEngineer

logger = get_score_model_training_logger()


class ScoreModelOptimizer:
    """
    Hyperparameter optimizer for score prediction models.
    """

    def __init__(self, random_state=DEFAULT_RANDOM_STATE):
        """
        Initialize the optimizer.

        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.best_params = {}
        self.best_scores = {}
        self.optimization_history = []

    @log_execution_time(logger)
    @log_exceptions(logger)
    def optimize_ensemble_type(self, player_stats, matches, cv_folds=5):
        """
        Find the best ensemble type for the score prediction model.

        Args:
            player_stats (dict): Player statistics dictionary
            matches (list): List of match data dictionaries
            cv_folds (int): Number of cross-validation folds

        Returns:
            dict: Optimization results
        """
        logger.info("Starting ensemble type optimization")
        
        ensemble_types = ['stacking', 'voting', 'single']
        results = {}
        
        # Enhanced feature configuration
        feature_config = {
            'use_basic_features': True,
            'use_team_features': True,
            'use_h2h_features': True,
            'use_recent_form_features': True,
            'use_advanced_features': True,
            'use_temporal_features': True,
            'use_momentum_features': True,
            'use_streak_features': True,
            'use_efficiency_features': True,
            'recent_matches_window': 10,
            'momentum_window': 5,
            'streak_window': 8
        }
        
        for ensemble_type in ensemble_types:
            logger.info(f"Testing ensemble type: {ensemble_type}")
            
            try:
                # Create model with current ensemble type
                model = ScorePredictionModel(
                    random_state=self.random_state,
                    feature_config=feature_config,
                    ensemble_type=ensemble_type,
                    use_advanced_validation=True
                )
                
                # Train and evaluate
                model.train(player_stats, matches, test_size=0.2)
                
                # Store results
                results[ensemble_type] = {
                    'total_score_mae': model.model_info['metrics']['total_score_mae'],
                    'home_score_mae': model.model_info['metrics']['home_score_mae'],
                    'away_score_mae': model.model_info['metrics']['away_score_mae'],
                    'home_cv_mae_mean': model.model_info['metrics'].get('home_cv_mae_mean', None),
                    'away_cv_mae_mean': model.model_info['metrics'].get('away_cv_mae_mean', None)
                }
                
                logger.info(f"Ensemble {ensemble_type} - Total MAE: {results[ensemble_type]['total_score_mae']:.4f}")
                
            except Exception as e:
                logger.error(f"Error testing ensemble type {ensemble_type}: {str(e)}")
                results[ensemble_type] = {'error': str(e)}
        
        # Find best ensemble type
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_ensemble = min(valid_results.keys(), 
                              key=lambda x: valid_results[x]['total_score_mae'])
            self.best_params['ensemble_type'] = best_ensemble
            self.best_scores['ensemble_optimization'] = valid_results[best_ensemble]
            
            logger.info(f"Best ensemble type: {best_ensemble} with MAE: {valid_results[best_ensemble]['total_score_mae']:.4f}")
        
        return results

    @log_execution_time(logger)
    @log_exceptions(logger)
    def optimize_feature_config(self, player_stats, matches, ensemble_type='stacking'):
        """
        Optimize feature configuration parameters.

        Args:
            player_stats (dict): Player statistics dictionary
            matches (list): List of match data dictionaries
            ensemble_type (str): Ensemble type to use

        Returns:
            dict: Optimization results
        """
        logger.info("Starting feature configuration optimization")
        
        # Parameter grid for feature configuration
        param_grid = {
            'recent_matches_window': [5, 8, 10, 15],
            'momentum_window': [3, 5, 7],
            'streak_window': [5, 8, 10]
        }
        
        best_config = None
        best_score = float('inf')
        results = []
        
        # Grid search over feature parameters
        for recent_window in param_grid['recent_matches_window']:
            for momentum_window in param_grid['momentum_window']:
                for streak_window in param_grid['streak_window']:
                    
                    feature_config = {
                        'use_basic_features': True,
                        'use_team_features': True,
                        'use_h2h_features': True,
                        'use_recent_form_features': True,
                        'use_advanced_features': True,
                        'use_temporal_features': True,
                        'use_momentum_features': True,
                        'use_streak_features': True,
                        'use_efficiency_features': True,
                        'recent_matches_window': recent_window,
                        'momentum_window': momentum_window,
                        'streak_window': streak_window
                    }
                    
                    try:
                        # Create and train model
                        model = ScorePredictionModel(
                            random_state=self.random_state,
                            feature_config=feature_config,
                            ensemble_type=ensemble_type,
                            use_advanced_validation=True
                        )
                        
                        model.train(player_stats, matches, test_size=0.2)
                        
                        score = model.model_info['metrics']['total_score_mae']
                        
                        result = {
                            'config': feature_config.copy(),
                            'total_score_mae': score,
                            'home_score_mae': model.model_info['metrics']['home_score_mae'],
                            'away_score_mae': model.model_info['metrics']['away_score_mae']
                        }
                        
                        results.append(result)
                        
                        if score < best_score:
                            best_score = score
                            best_config = feature_config.copy()
                        
                        logger.info(f"Config {recent_window}-{momentum_window}-{streak_window}: MAE = {score:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error with config {recent_window}-{momentum_window}-{streak_window}: {str(e)}")
        
        self.best_params['feature_config'] = best_config
        self.best_scores['feature_optimization'] = best_score
        
        logger.info(f"Best feature config found with MAE: {best_score:.4f}")
        
        return {
            'best_config': best_config,
            'best_score': best_score,
            'all_results': results
        }

    @log_execution_time(logger)
    @log_exceptions(logger)
    def run_full_optimization(self, player_stats, matches, save_results=True):
        """
        Run complete optimization pipeline.

        Args:
            player_stats (dict): Player statistics dictionary
            matches (list): List of match data dictionaries
            save_results (bool): Whether to save results to file

        Returns:
            dict: Complete optimization results
        """
        logger.info("Starting full optimization pipeline")
        
        optimization_results = {
            'timestamp': datetime.now().isoformat(),
            'num_matches': len(matches),
            'random_state': self.random_state
        }
        
        # Step 1: Optimize ensemble type
        ensemble_results = self.optimize_ensemble_type(player_stats, matches)
        optimization_results['ensemble_optimization'] = ensemble_results
        
        # Step 2: Optimize feature configuration with best ensemble
        best_ensemble = self.best_params.get('ensemble_type', 'stacking')
        feature_results = self.optimize_feature_config(player_stats, matches, best_ensemble)
        optimization_results['feature_optimization'] = feature_results
        
        # Step 3: Final model training with best parameters
        logger.info("Training final model with optimized parameters")
        final_model = ScorePredictionModel(
            random_state=self.random_state,
            feature_config=self.best_params['feature_config'],
            ensemble_type=self.best_params['ensemble_type'],
            use_advanced_validation=True
        )
        
        final_model.train(player_stats, matches, test_size=0.2)
        
        optimization_results['final_model'] = {
            'best_ensemble_type': self.best_params['ensemble_type'],
            'best_feature_config': self.best_params['feature_config'],
            'final_metrics': final_model.model_info['metrics']
        }
        
        # Save results if requested
        if save_results:
            self._save_optimization_results(optimization_results)
        
        logger.info(f"Optimization complete. Final MAE: {final_model.model_info['metrics']['total_score_mae']:.4f}")
        
        return optimization_results

    def _save_optimization_results(self, results):
        """
        Save optimization results to file.

        Args:
            results (dict): Optimization results
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = "optimization_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"score_model_optimization_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            # Save results
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Optimization results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving optimization results: {str(e)}")


def main():
    """
    Main function for running score model optimization.
    """
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    sys.path.insert(0, project_root)
    
    from data.data_loader import DataLoader
    
    try:
        # Load data
        data_loader = DataLoader()
        player_stats = data_loader.load_player_stats()
        matches = data_loader.load_match_history()
        
        if not matches:
            logger.error("No match data available for optimization")
            return
        
        # Run optimization
        optimizer = ScoreModelOptimizer()
        results = optimizer.run_full_optimization(player_stats, matches)
        
        print("\nOptimization Results:")
        print(f"Best Ensemble Type: {results['final_model']['best_ensemble_type']}")
        print(f"Final Total MAE: {results['final_model']['final_metrics']['total_score_mae']:.4f}")
        print(f"Final Home MAE: {results['final_model']['final_metrics']['home_score_mae']:.4f}")
        print(f"Final Away MAE: {results['final_model']['final_metrics']['away_score_mae']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in optimization: {str(e)}")
        raise


if __name__ == "__main__":
    main()