"""Enhanced winner prediction model for NBA 2K25 eSports matches."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

from config.settings import DEFAULT_RANDOM_STATE
from config.logging_config import get_model_tuning_logger
from utils.logging import log_execution_time, log_exceptions
from core.models.base import BaseModel
from core.models.feature_engineering import FeatureEngineer

logger = get_model_tuning_logger()


class WinnerPredictionModel(BaseModel):
    """
    Enhanced model for predicting match winners using ensemble methods.
    """

    def __init__(self, model_id=None, random_state=DEFAULT_RANDOM_STATE, 
                 ensemble_type='voting', feature_config=None, use_advanced_validation=True):
        """
        Initialize the enhanced winner prediction model.

        Args:
            model_id (str): Model ID
            random_state (int): Random state for reproducibility
            ensemble_type (str): Type of ensemble ('voting', 'stacking', 'single')
            feature_config (dict): Feature configuration dictionary
            use_advanced_validation (bool): Whether to use advanced validation techniques
        """
        super().__init__(model_id, random_state)
        self.ensemble_type = ensemble_type
        self.use_advanced_validation = use_advanced_validation

        # Create feature engineer
        self.feature_engineer = FeatureEngineer(feature_config)

        # Update model info
        self.model_info["parameters"] = {
            "ensemble_type": ensemble_type,
            "random_state": random_state,
            "feature_config": feature_config,
            "use_advanced_validation": use_advanced_validation
        }

        # Initialize models based on ensemble type
        self._initialize_models()

        # Initialize feature selector and scaler
        self.feature_selector = None
        self.scaler = StandardScaler()
        
        # Store validation results
        self.validation_results = {}

    def _initialize_models(self):
        """
        Initialize models based on ensemble type.
        """
        if self.ensemble_type == 'single':
            # Single XGBoost model
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        
        elif self.ensemble_type == 'voting':
            # Voting ensemble
            base_models = [
                ('rf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    class_weight='balanced',
                    random_state=self.random_state
                )),
                ('xgb', XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    eval_metric='logloss'
                )),
                ('lgb', lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    verbose=-1
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=self.random_state
                )),
                ('lr', LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ))
            ]
            
            self.model = VotingClassifier(
                estimators=base_models,
                voting='soft'
            )
        
        elif self.ensemble_type == 'stacking':
            # Stacking ensemble (simplified - using voting for now)
            # In a full implementation, this would use StackingClassifier
            self._initialize_models_voting()
            self.ensemble_type = 'voting'  # Fallback to voting
    
    def _initialize_models_voting(self):
        """Helper method to initialize voting ensemble."""
        base_models = [
            ('rf', RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                class_weight='balanced',
                random_state=self.random_state
            )),
            ('xgb', XGBClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss'
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=-1
            ))
        ]
        
        self.model = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )

    @log_execution_time(logger)
    @log_exceptions(logger)
    def train(self, player_stats, matches, test_size=0.2, min_samples=100, cv_folds=5):
        """
        Train the enhanced model on match data.

        Args:
            player_stats (dict): Player statistics dictionary
            matches (list): List of match data dictionaries
            test_size (float): Proportion of data to use for testing
            min_samples (int): Minimum number of samples required for training
            cv_folds (int): Number of cross-validation folds

        Returns:
            self: The trained model
        """
        logger.info(f"Training enhanced winner prediction model with {len(matches)} matches")
        logger.info(f"Ensemble type: {self.ensemble_type}")
        logger.info(f"Feature config: {self.feature_engineer.feature_config}")

        # Extract features and labels using the feature engineer
        X, y = self.feature_engineer.extract_features(
            player_stats, matches, for_score_prediction=False
        )
        
        logger.info(f"Feature extraction result: X shape = {X.shape if len(X) > 0 else 'empty'}, y shape = {y.shape if len(y) > 0 else 'empty'}")

        if len(X) == 0:
            logger.error("No valid features extracted from matches")
            raise ValueError("No valid features extracted from matches")

        # Check if we have enough samples
        if len(X) < min_samples:
            logger.error(f"Insufficient samples for training: {len(X)} < {min_samples}")
            raise ValueError(f"Insufficient samples for training: {len(X)} < {min_samples}")

        logger.info(f"Extracted {len(X)} samples with {X.shape[1]} features")

        # Advanced feature selection
        logger.info("Performing advanced feature selection")
        X_selected = self._advanced_feature_selection(X, y)
        
        # Scale features
        logger.info("Scaling features")
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Advanced validation if enabled
        if self.use_advanced_validation:
            logger.info("Performing advanced validation")
            self._advanced_validation(X_scaled, y, cv_folds)
        
        # Train the model
        logger.info("Training the ensemble model")
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        logger.info("Evaluating model performance")
        self._evaluate_model(X_train, X_test, y_train, y_test)
        
        # Store feature importance if available
        self._store_feature_importance(X_selected.shape[1])
        
        # Mark as trained
        self.is_trained = True
        self.model_info["training_samples"] = len(X_train)
        self.model_info["test_samples"] = len(X_test)
        self.model_info["features_count"] = X_selected.shape[1]
        
        logger.info("Model training completed successfully")
        return self
    
    def _advanced_feature_selection(self, X, y):
        """
        Perform advanced feature selection using multiple methods.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
            
        Returns:
            np.array: Selected features
        """
        # Method 1: Tree-based feature importance
        selector1 = SelectFromModel(
            XGBClassifier(n_estimators=100, random_state=self.random_state),
            threshold='median'
        )
        X_selected1 = selector1.fit_transform(X, y)
        
        # Method 2: Recursive Feature Elimination
        if X.shape[1] > 50:  # Only use RFE if we have many features
            selector2 = RFE(
                estimator=RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                n_features_to_select=min(30, X.shape[1] // 2)
            )
            X_selected2 = selector2.fit_transform(X, y)
            
            # Use the intersection of both methods
            selected_features1 = selector1.get_support()
            selected_features2 = selector2.get_support()
            
            # Combine selections (union for more features)
            combined_selection = selected_features1 | selected_features2
            X_selected = X[:, combined_selection]
            
            # Store the combined selector
            self.feature_selector = {
                'tree_based': selector1,
                'rfe': selector2,
                'combined_mask': combined_selection
            }
        else:
            X_selected = X_selected1
            self.feature_selector = {'tree_based': selector1}
        
        logger.info(f"Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")
        return X_selected
    
    def _advanced_validation(self, X, y, cv_folds):
        """
        Perform advanced cross-validation with multiple metrics.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
            cv_folds (int): Number of CV folds
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Multiple scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric in scoring_metrics:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric)
            self.validation_results[f'cv_{metric}'] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            logger.info(f"CV {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Store in model info
        self.model_info["cross_validation"] = self.validation_results
    
    def _evaluate_model(self, X_train, X_test, y_train, y_test):
        """
        Evaluate model performance on train and test sets.
        
        Args:
            X_train (np.array): Training features
            X_test (np.array): Test features
            y_train (np.array): Training labels
            y_test (np.array): Test labels
        """
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average='weighted'),
            'recall': recall_score(y_train, y_train_pred, average='weighted'),
            'f1': f1_score(y_train, y_train_pred, average='weighted')
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, average='weighted'),
            'recall': recall_score(y_test, y_test_pred, average='weighted'),
            'f1': f1_score(y_test, y_test_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }
        
        # Store metrics
        self.model_info["metrics"] = {
            'train': train_metrics,
            'test': test_metrics
        }
        
        # Log results
        logger.info(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {test_metrics['f1']:.4f}")
        logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        
        # Return combined metrics
        return {
            'train': train_metrics,
            'test': test_metrics
        }
    
    def _store_feature_importance(self, n_features):
        """
        Store feature importance if available.
        
        Args:
            n_features (int): Number of features
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Single model with feature importance
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'estimators_'):
                # Ensemble model - average importance across estimators
                importances = []
                for estimator in self.model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                
                if importances:
                    importance = np.mean(importances, axis=0)
                else:
                    importance = np.ones(n_features) / n_features
            else:
                # No feature importance available
                importance = np.ones(n_features) / n_features
            
            # Store top features
            top_indices = np.argsort(importance)[::-1][:20]
            self.model_info["feature_importance"] = {
                'importance': importance.tolist(),
                'top_features': top_indices.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            self.model_info["feature_importance"] = None

    @log_exceptions(logger)
    def predict(self, player_stats, match):
        """
        Predict the winner of a match.

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
                "home_win_probability": 0.5,
                "away_win_probability": 0.5,
                "predicted_winner": "home" if np.random.random() > 0.5 else "away",
                "confidence": 0.5,
                "prediction_method": "fallback_random"
            }

        # Create a single-match list for feature extraction
        match_list = [match]

        # Extract features using the feature engineer
        try:
            X, _ = self.feature_engineer.extract_features(
                player_stats, match_list, for_score_prediction=False
            )

            if len(X) == 0:
                raise ValueError("No features extracted")

            # Apply feature selection if available
            if self.feature_selector is not None:
                X_selected = self.feature_selector.transform(X)
                prediction_method = "model_with_feature_selection"
            else:
                logger.warning("Feature selector not available, using raw features")
                X_selected = X
                prediction_method = "model_without_feature_selection"

            # Make prediction
            probabilities = self.model.predict_proba(X_selected)[0]

        except Exception as e:
            logger.error(f"Error predicting winner: {str(e)}")
            # Fallback to a simpler prediction method
            home_player = player_stats[home_player_id]
            away_player = player_stats[away_player_id]

            # Use win rates as fallback
            home_win_rate = home_player.get('win_rate', 0.5)
            away_win_rate = away_player.get('win_rate', 0.5)

            # Normalize win rates to probabilities
            total = home_win_rate + away_win_rate
            if total > 0:
                home_prob = home_win_rate / total
                away_prob = away_win_rate / total
            else:
                home_prob = 0.5
                away_prob = 0.5

            probabilities = [away_prob, home_prob]
            prediction_method = "fallback_win_rates"

        home_win_probability = probabilities[1]
        away_win_probability = probabilities[0]

        predicted_winner = "home" if home_win_probability > away_win_probability else "away"
        confidence = max(home_win_probability, away_win_probability)

        return {
            "home_win_probability": float(home_win_probability),
            "away_win_probability": float(away_win_probability),
            "predicted_winner": predicted_winner,
            "confidence": float(confidence),
            "prediction_method": prediction_method
        }

    @log_exceptions(logger)
    def evaluate(self, player_stats, matches, min_samples=100, cv_folds=5):
        """
        Evaluate the model on match data.

        Args:
            player_stats (dict): Player statistics dictionary
            matches (list): List of match data dictionaries
            min_samples (int): Minimum number of samples required for evaluation
            cv_folds (int): Number of cross-validation folds

        Returns:
            dict: Evaluation metrics
        """
        # Extract features and labels using the feature engineer
        X, y = self.feature_engineer.extract_features(
            player_stats, matches, for_score_prediction=False
        )

        if len(X) == 0:
            logger.error("No valid features extracted from matches")
            raise ValueError("No valid features extracted from matches")

        # Check if we have enough samples
        if len(X) < min_samples:
            logger.warning(f"Insufficient samples for reliable evaluation: {len(X)} < {min_samples}")
            logger.warning("Evaluation results may not be statistically significant")

        # Apply feature selection if available
        if self.feature_selector is not None:
            if isinstance(self.feature_selector, dict):
                if 'combined_mask' in self.feature_selector:
                    # Use combined mask for feature selection
                    X_selected = X[:, self.feature_selector['combined_mask']]
                elif 'tree_based' in self.feature_selector:
                    # Use tree-based selector
                    X_selected = self.feature_selector['tree_based'].transform(X)
                else:
                    logger.warning("Unknown feature selector format, using raw features")
                    X_selected = X
            else:
                # Legacy single selector
                X_selected = self.feature_selector.transform(X)
        else:
            logger.warning("Feature selector not available, using raw features")
            X_selected = X

        # Perform cross-validation if we have enough samples
        if len(X) >= cv_folds * 2:  # Ensure at least 2 samples per fold
            logger.info(f"Performing {cv_folds}-fold cross-validation")
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(self.model, X_selected, y, cv=cv, scoring='accuracy')

            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # Split data for standard metrics evaluation
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            # Get standard metrics
            metrics = self._evaluate_model(X_train, X_test, y_train, y_test)

            # Add cross-validation results
            metrics["cv_scores"] = cv_scores.tolist()
            metrics["cv_mean_accuracy"] = float(cv_scores.mean())
            metrics["cv_std_accuracy"] = float(cv_scores.std())
            metrics["validation_method"] = f"{cv_folds}-fold cross-validation"

            return metrics
        else:
            logger.warning(f"Insufficient samples for {cv_folds}-fold cross-validation, using standard evaluation")
            # Split data for standard evaluation
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            return self._evaluate_model(X_train, X_test, y_train, y_test)
