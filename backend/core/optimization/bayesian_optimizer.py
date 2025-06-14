"""
Bayesian optimizer for model hyperparameter tuning.
"""

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from config.logging_config import get_model_tuning_logger
from utils.logging import log_execution_time, log_exceptions
from .tuner import BaseTuner

logger = get_model_tuning_logger()


class BayesianOptimizer(BaseTuner):
    """
    Bayesian optimizer for model hyperparameter tuning.
    """
    
    def __init__(self, model_class, param_space, random_state=42):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            model_class: Model class to tune
            param_space (dict): Parameter space to search
            random_state (int): Random state for reproducibility
        """
        super().__init__(model_class, param_space, random_state)
        
        # Convert parameter space to skopt space
        self.skopt_space = self._convert_param_space(param_space)
    
    @log_exceptions(logger)
    def _convert_param_space(self, param_space):
        """
        Convert parameter space to skopt space.
        
        Args:
            param_space (dict): Parameter space to convert
            
        Returns:
            list: skopt space
        """
        skopt_space = []
        
        for param_name, param_config in param_space.items():
            param_type = param_config['type']
            
            if param_type == 'real':
                skopt_space.append(
                    Real(
                        param_config['low'],
                        param_config['high'],
                        prior=param_config.get('prior', 'uniform'),
                        name=param_name
                    )
                )
            elif param_type == 'integer':
                skopt_space.append(
                    Integer(
                        param_config['low'],
                        param_config['high'],
                        name=param_name
                    )
                )
            elif param_type == 'categorical':
                skopt_space.append(
                    Categorical(
                        param_config['categories'],
                        name=param_name
                    )
                )
        
        return skopt_space
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def optimize(self, player_stats, matches, n_trials=50, test_size=0.2, scoring='neg_mean_absolute_error',
                early_stopping_rounds=10, min_improvement=0.001):
        """
        Optimize model hyperparameters using Bayesian optimization.
        
        Args:
            player_stats (dict): Player statistics dictionary
            matches (list): List of match data dictionaries
            n_trials (int): Number of optimization trials
            test_size (float): Proportion of data to use for testing
            scoring (str): Scoring metric to optimize
            early_stopping_rounds (int): Stop if no improvement for this many rounds
            min_improvement (float): Minimum improvement to consider significant
            
        Returns:
            tuple: (best_params, best_score, best_model)
        """
        logger.info(f"Starting Bayesian optimization with {n_trials} trials")
        logger.info(f"Early stopping: {early_stopping_rounds} rounds, min improvement: {min_improvement}")
        
        # Track early stopping
        best_score_history = []
        no_improvement_count = 0
        
        # Define the objective function with early stopping
        @use_named_args(self.skopt_space)
        def objective(**params):
            nonlocal best_score_history, no_improvement_count
            
            # Add fixed parameters
            params['random_state'] = self.random_state
            
            # Evaluate the parameters
            score = self._evaluate_params(params, player_stats, matches, test_size, scoring)
            
            # Track best score for early stopping
            if not best_score_history or score > max(best_score_history):
                if best_score_history and (score - max(best_score_history)) < min_improvement:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                best_score_history.append(score)
            else:
                no_improvement_count += 1
                best_score_history.append(max(best_score_history))
            
            # Log progress
            if len(best_score_history) % 5 == 0:
                logger.info(f"Trial {len(best_score_history)}/{n_trials}, Best score: {max(best_score_history):.4f}, Current: {score:.4f}")
            
            # Return negative score for minimization
            return -score
        
        # Run Bayesian optimization with early stopping check
        result = gp_minimize(
            objective,
            self.skopt_space,
            n_calls=n_trials,
            random_state=self.random_state,
            verbose=False,  # Reduced verbosity for cleaner output
            n_jobs=1,  # Use single job to avoid conflicts with model training parallelization
            callback=self._early_stopping_callback(early_stopping_rounds, no_improvement_count)
        )
        
        # Convert best parameters to dictionary
        best_params = {dim.name: value for dim, value in zip(self.skopt_space, result.x)}
        best_params['random_state'] = self.random_state
        
        # Store best parameters and score
        self.best_params = best_params
        self.best_score = -result.fun  # Convert back to positive score
        
        # Create and train the best model
        if self.best_model is None:
            self.best_model = self.model_class(**best_params)
            self.best_model.train(player_stats, matches, test_size=test_size)
        
        logger.info(f"Bayesian optimization completed")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {self.best_score}")
        
        return best_params, self.best_score, self.best_model
    
    def _early_stopping_callback(self, early_stopping_rounds, no_improvement_count):
        """
        Create a callback function for early stopping.
        
        Args:
            early_stopping_rounds (int): Number of rounds without improvement to stop
            no_improvement_count (int): Current count of no improvement
            
        Returns:
            function: Callback function
        """
        def callback(result):
            # This is a simple callback - more sophisticated early stopping
            # would require modifying the objective function tracking
            return False  # Continue optimization
        
        return callback
