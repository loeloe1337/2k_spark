"""
Score prediction model for NBA 2K25 eSports matches.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_regression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

from config.settings import DEFAULT_RANDOM_STATE
from config.logging_config import get_score_model_training_logger
from utils.logging import log_execution_time, log_exceptions
from core.models.base import BaseModel
from core.models.feature_engineering import FeatureEngineer

logger = get_score_model_training_logger()


class ScorePredictionModel(BaseModel):
    """
    Model for predicting match scores.
    """

    def __init__(self, model_id=None, random_state=DEFAULT_RANDOM_STATE, feature_config=None, 
                 ensemble_type='stacking', use_advanced_validation=True):
        """
        Initialize the score prediction model.

        Args:
            model_id (str): Model ID
            random_state (int): Random state for reproducibility
            feature_config (dict): Feature configuration dictionary
            ensemble_type (str): Type of ensemble ('stacking', 'voting', 'single')
            use_advanced_validation (bool): Whether to use advanced validation techniques
        """
        super().__init__(model_id, random_state)

        # Store configuration
        self.ensemble_type = ensemble_type
        self.use_advanced_validation = use_advanced_validation
        
        # Create feature engineer
        self.feature_engineer = FeatureEngineer(feature_config)

        # Create home and away score models
        self.home_model, self.away_model = self._create_models(random_state)

        # Initialize feature selectors
        self.home_selector = None
        self.away_selector = None

        # Update model info
        self.model_info["parameters"] = {
            "random_state": random_state,
            "feature_config": feature_config,
            "ensemble_type": ensemble_type,
            "use_advanced_validation": use_advanced_validation
        }

        # Store both models
        self.model = {
            "home_model": self.home_model,
            "away_model": self.away_model
        }

    @log_exceptions(logger)
    def _create_models(self, random_state):
        """
        Create home and away score prediction models.

        Args:
            random_state (int): Random state for reproducibility

        Returns:
            tuple: (home_model, away_model)
        """
        # Create base models
        base_models = {
            'xgb': XGBRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbosity=0
            ),
            'lgb': LGBMRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbosity=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=random_state
            ),
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            ),
            'ridge': Ridge(alpha=1.0, random_state=random_state),
            'lasso': Lasso(alpha=0.1, random_state=random_state),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }

        # Create ensemble models based on type
        if self.ensemble_type == 'stacking':
            home_ensemble = self._create_stacking_ensemble(base_models, random_state)
            away_ensemble = self._create_stacking_ensemble(base_models, random_state)
        elif self.ensemble_type == 'voting':
            home_ensemble = self._create_voting_ensemble(base_models)
            away_ensemble = self._create_voting_ensemble(base_models)
        else:  # single model
            home_ensemble = base_models['xgb']
            away_ensemble = base_models['xgb']

        # Create feature scaling and model pipeline
        home_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', home_ensemble)
        ])

        away_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', away_ensemble)
        ])

        return home_pipeline, away_pipeline

    def _create_stacking_ensemble(self, base_models, random_state):
        """Create a stacking ensemble."""
        estimators = [
            ('xgb', base_models['xgb']),
            ('lgb', base_models['lgb']),
            ('gb', base_models['gb']),
            ('rf', base_models['rf']),
            ('ridge', base_models['ridge'])
        ]
        
        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=0.5, random_state=random_state),
            cv=5,
            n_jobs=-1
        )

    def _create_voting_ensemble(self, base_models):
        """Create a voting ensemble."""
        estimators = [
            ('xgb', base_models['xgb']),
            ('lgb', base_models['lgb']),
            ('gb', base_models['gb']),
            ('rf', base_models['rf'])
        ]
        
        return VotingRegressor(
            estimators=estimators,
            n_jobs=-1
        )

    @log_execution_time(logger)
    @log_exceptions(logger)
    def train(self, player_stats, matches, test_size=0.2):
        """
        Train the model on match data.

        Args:
            player_stats (dict): Player statistics dictionary
            matches (list): List of match data dictionaries
            test_size (float): Proportion of data to use for testing

        Returns:
            self: The trained model
        """
        logger.info(f"Training score prediction model with {len(matches)} matches")

        # Extract features and labels using the feature engineer
        X, y_home, y_away = self.feature_engineer.extract_features(
            player_stats, matches, for_score_prediction=True
        )

        if len(X) == 0:
            logger.error("No valid features extracted from matches")
            raise ValueError("No valid features extracted from matches")

        logger.info(f"Extracted {len(X)} samples with {X.shape[1]} features")

        # Split data into training and testing sets
        X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
            X, y_home, y_away, test_size=test_size, random_state=self.random_state
        )

        # Advanced feature selection
        logger.info("Performing advanced feature selection")
        X_train_home, X_test_home, home_selector = self._perform_feature_selection(
            X_train, X_test, y_home_train, 'home'
        )
        X_train_away, X_test_away, away_selector = self._perform_feature_selection(
            X_train, X_test, y_away_train, 'away'
        )
        
        # Store feature selectors
        self.home_selector = home_selector
        self.away_selector = away_selector

        # Train home score model
        logger.info(f"Training home score model with {len(X_train_home)} samples")
        self.home_model.fit(X_train_home, y_home_train)

        # Train away score model
        logger.info(f"Training away score model with {len(X_train_away)} samples")
        self.away_model.fit(X_train_away, y_away_train)

        # Perform cross-validation if advanced validation is enabled
        cv_metrics = {}
        if self.use_advanced_validation:
            logger.info("Performing cross-validation")
            cv_metrics = self._perform_cross_validation(X_train_home, X_train_away, y_home_train, y_away_train)

        # Evaluate models
        logger.info("Evaluating models")
        metrics = self._evaluate_models(
            X_test_home, X_test_away, y_home_test, y_away_test
        )
        
        # Combine metrics with cross-validation results
        if cv_metrics:
            metrics.update(cv_metrics)

        # Update model info
        self.model_info["metrics"] = metrics
        self.model_info["home_score_mae"] = metrics["home_score_mae"]
        self.model_info["away_score_mae"] = metrics["away_score_mae"]
        self.model_info["total_score_mae"] = metrics["total_score_mae"]
        self.model_info["data_files"] = {
            "player_stats": "player_stats.json",
            "match_history": "match_history.json"
        }
        self.model_info["num_samples"] = len(X)
        self.model_info["num_features"] = {
            "original": X.shape[1],
            "home_selected": X_train_home.shape[1],
            "away_selected": X_train_away.shape[1]
        }

        logger.info(f"Models trained with total score MAE: {metrics['total_score_mae']:.4f}")
        return self

    @log_exceptions(logger)
    def evaluate(self, player_stats, matches):
        """
        Evaluate the model on match data.

        Args:
            player_stats (dict): Player statistics dictionary
            matches (list): List of match data dictionaries

        Returns:
            dict: Evaluation metrics
        """
        # Extract features and labels using the feature engineer
        X, y_home, y_away = self.feature_engineer.extract_features(
            player_stats, matches, for_score_prediction=True
        )

        if len(X) == 0:
            logger.error("No valid features extracted from matches")
            raise ValueError("No valid features extracted from matches")

        # Apply feature selection
        X_home = self.home_selector.transform(X)
        X_away = self.away_selector.transform(X)

        # Evaluate models
        return self._evaluate_models(X_home, X_away, y_home, y_away)

    @log_exceptions(logger)
    def _evaluate_models(self, X_test_home, X_test_away, y_home_test, y_away_test):
        """
        Evaluate the models on test data.

        Args:
            X_test_home (numpy.ndarray): Test features for home model
            X_test_away (numpy.ndarray): Test features for away model
            y_home_test (numpy.ndarray): Test home scores
            y_away_test (numpy.ndarray): Test away scores

        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_home_pred = self.home_model.predict(X_test_home)
        y_away_pred = self.away_model.predict(X_test_away)

        # Calculate metrics for home score
        home_mae = mean_absolute_error(y_home_test, y_home_pred)
        home_mse = mean_squared_error(y_home_test, y_home_pred)
        home_rmse = np.sqrt(home_mse)
        home_r2 = r2_score(y_home_test, y_home_pred)

        # Calculate metrics for away score
        away_mae = mean_absolute_error(y_away_test, y_away_pred)
        away_mse = mean_squared_error(y_away_test, y_away_pred)
        away_rmse = np.sqrt(away_mse)
        away_r2 = r2_score(y_away_test, y_away_pred)

        # Calculate metrics for total score
        total_score_test = y_home_test + y_away_test
        total_score_pred = y_home_pred + y_away_pred
        total_mae = mean_absolute_error(total_score_test, total_score_pred)
        total_mse = mean_squared_error(total_score_test, total_score_pred)
        total_rmse = np.sqrt(total_mse)
        total_r2 = r2_score(total_score_test, total_score_pred)

        # Calculate metrics for score difference
        diff_test = y_home_test - y_away_test
        diff_pred = y_home_pred - y_away_pred
        diff_mae = mean_absolute_error(diff_test, diff_pred)
        diff_mse = mean_squared_error(diff_test, diff_pred)
        diff_rmse = np.sqrt(diff_mse)
        diff_r2 = r2_score(diff_test, diff_pred)

        return {
            "home_score_mae": float(home_mae),
            "home_score_mse": float(home_mse),
            "home_score_rmse": float(home_rmse),
            "home_score_r2": float(home_r2),

            "away_score_mae": float(away_mae),
            "away_score_mse": float(away_mse),
            "away_score_rmse": float(away_rmse),
            "away_score_r2": float(away_r2),

            "total_score_mae": float(total_mae),
            "total_score_mse": float(total_mse),
            "total_score_rmse": float(total_rmse),
            "total_score_r2": float(total_r2),

            "score_diff_mae": float(diff_mae),
            "score_diff_mse": float(diff_mse),
            "score_diff_rmse": float(diff_rmse),
            "score_diff_r2": float(diff_r2)
        }

    @log_exceptions(logger)
    def predict(self, player_stats, match):
        """
        Predict the score of a match.

        Args:
            player_stats (dict): Player statistics dictionary
            match (dict): Match data dictionary

        Returns:
            dict: Prediction results
        """
        home_player_id = str(match['homePlayer']['id'])
        away_player_id = str(match['awayPlayer']['id'])

        # Check if player stats are available
        if home_player_id not in player_stats or away_player_id not in player_stats:
            logger.warning(f"Player stats not available for {home_player_id} or {away_player_id}")
            return {
                "home_score": 60,
                "away_score": 60,
                "total_score": 120,
                "score_diff": 0
            }

        # Create a single-match list for feature extraction
        match_list = [match]

        # Extract features using the feature engineer
        try:
            X, _, _ = self.feature_engineer.extract_features(
                player_stats, match_list, for_score_prediction=True
            )

            if len(X) == 0:
                raise ValueError("No features extracted")

            # Apply feature selection
            X_home = self.home_selector.transform(X)
            X_away = self.away_selector.transform(X)

            # Make prediction
            home_score = self.home_model.predict(X_home)[0]
            away_score = self.away_model.predict(X_away)[0]

        except Exception as e:
            logger.error(f"Error predicting score: {str(e)}")
            # Fallback to a simpler prediction method
            home_player = player_stats[home_player_id]
            away_player = player_stats[away_player_id]

            # Use average scores as fallback
            home_score = home_player.get('avg_score', 60)
            away_score = away_player.get('avg_score', 60)

        # Round scores to integers
        home_score = round(home_score)
        away_score = round(away_score)

        # Ensure scores are positive
        home_score = max(0, home_score)
        away_score = max(0, away_score)

        # Calculate total score and difference
        total_score = home_score + away_score
        score_diff = home_score - away_score

        return {
            "home_score": int(home_score),
            "away_score": int(away_score),
            "total_score": int(total_score),
            "score_diff": int(score_diff)
        }

    def _perform_feature_selection(self, X_train, X_test, y_train, model_type):
        """
        Perform advanced feature selection combining multiple methods.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            model_type: 'home' or 'away'
            
        Returns:
            tuple: (X_train_selected, X_test_selected, selector)
        """
        logger.info(f"Performing feature selection for {model_type} model")
        
        # Method 1: Tree-based feature importance
        tree_selector = SelectFromModel(
            XGBRegressor(n_estimators=50, random_state=self.random_state, verbosity=0),
            threshold="median"
        )
        
        # Method 2: Statistical feature selection
        k_best_selector = SelectKBest(score_func=f_regression, k=min(50, X_train.shape[1]))
        
        # Method 3: Recursive Feature Elimination
        rfe_selector = RFE(
            estimator=Ridge(random_state=self.random_state),
            n_features_to_select=min(30, X_train.shape[1]),
            step=1
        )
        
        # Apply tree-based selection first
        X_train_tree = tree_selector.fit_transform(X_train, y_train)
        X_test_tree = tree_selector.transform(X_test)
        
        # Apply statistical selection on tree-selected features
        if X_train_tree.shape[1] > 10:
            k_best_selector.set_params(k=min(X_train_tree.shape[1], 25))
            X_train_final = k_best_selector.fit_transform(X_train_tree, y_train)
            X_test_final = k_best_selector.transform(X_test_tree)
        else:
            X_train_final = X_train_tree
            X_test_final = X_test_tree
            
        logger.info(f"Selected {X_train_final.shape[1]} features for {model_type} model")
        
        # Create combined selector
        class CombinedSelector:
            def __init__(self, tree_sel, stat_sel=None):
                self.tree_selector = tree_sel
                self.stat_selector = stat_sel
                
            def transform(self, X):
                X_tree = self.tree_selector.transform(X)
                if self.stat_selector is not None:
                    return self.stat_selector.transform(X_tree)
                return X_tree
        
        combined_selector = CombinedSelector(
            tree_selector, 
            k_best_selector if X_train_tree.shape[1] > 10 else None
        )
        
        return X_train_final, X_test_final, combined_selector
    
    def _perform_cross_validation(self, X_train_home, X_train_away, y_home_train, y_away_train):
        """
        Perform cross-validation to get robust performance estimates.
        
        Args:
            X_train_home: Home model training features
            X_train_away: Away model training features
            y_home_train: Home training labels
            y_away_train: Away training labels
            
        Returns:
            dict: Cross-validation metrics
        """
        cv_folds = 5
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Cross-validation for home model
        home_cv_scores = cross_val_score(
            self.home_model, X_train_home, y_home_train,
            cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        # Cross-validation for away model
        away_cv_scores = cross_val_score(
            self.away_model, X_train_away, y_away_train,
            cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        return {
            'home_cv_mae_mean': float(-home_cv_scores.mean()),
            'home_cv_mae_std': float(home_cv_scores.std()),
            'away_cv_mae_mean': float(-away_cv_scores.mean()),
            'away_cv_mae_std': float(away_cv_scores.std()),
            'cv_folds': cv_folds
        }
