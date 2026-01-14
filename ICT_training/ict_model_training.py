"""
ICT Model Training Pipeline
============================
Trains ML models to filter ICT trading signals for quality.

Features:
- Multiple model types (XGBoost, LightGBM, Random Forest, ensemble)
- Time series cross-validation
- Hyperparameter tuning
- Feature importance analysis
- Comprehensive evaluation metrics
- Profit simulation
- Model persistence

Input: Prepared train/val/test datasets from data preparation
Output: Trained model, evaluation reports, feature importance

Author: ICT ML System
Version: 1.0
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import lightgbm as lgb
from typing import Tuple, List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from datetime import datetime
import os

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # Model Selection
    model_type: str = "xgboost"      # 'xgboost', 'lightgbm', 'random_forest', 'ensemble'
    
    # XGBoost Parameters
    xgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 2.0,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc'
    })
    
    # LightGBM Parameters
    lgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 2.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    })
    
    # Random Forest Parameters
    rf_params: Dict = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    })
    
    # Ensemble Weights
    ensemble_weights: List[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])
    
    # Cross-Validation
    cv_folds: int = 5
    use_time_series_cv: bool = True
    
    # Hyperparameter Tuning
    tune_hyperparams: bool = False
    tuning_method: str = "random"     # 'grid', 'random'
    tuning_iterations: int = 50
    
    # Early Stopping (for boosting models)
    early_stopping_rounds: int = 50
    
    # Calibration
    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"  # 'isotonic', 'sigmoid'
    
    # Evaluation
    probability_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8])
    
    # Profit Simulation
    run_profit_simulation: bool = True
    risk_reward_ratio: float = 1.5
    
    # Output
    output_dir: str = "models"
    save_model: bool = True
    save_reports: bool = True
    plot_results: bool = True


# =============================================================================
# MODEL FACTORY
# =============================================================================

class ModelFactory:
    """Factory for creating ML models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def create_model(self, model_type: Optional[str] = None):
        """Create model based on type"""
        
        model_type = model_type or self.config.model_type
        
        if model_type == 'xgboost':
            return xgb.XGBClassifier(**self.config.xgb_params)
        
        elif model_type == 'lightgbm':
            return lgb.LGBMClassifier(**self.config.lgb_params)
        
        elif model_type == 'random_forest':
            return RandomForestClassifier(**self.config.rf_params)
        
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        
        elif model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        
        elif model_type == 'ensemble':
            return self._create_ensemble()
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_ensemble(self):
        """Create ensemble of multiple models"""
        
        estimators = [
            ('xgb', xgb.XGBClassifier(**self.config.xgb_params)),
            ('lgb', lgb.LGBMClassifier(**self.config.lgb_params)),
            ('rf', RandomForestClassifier(**self.config.rf_params))
        ]
        
        return VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=self.config.ensemble_weights
        )


# =============================================================================
# TIME SERIES CROSS-VALIDATION
# =============================================================================

class TimeSeriesCrossValidator:
    """Custom time series cross-validation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.cv_results = []
    
    def cross_validate(
        self, 
        model, 
        X: pd.DataFrame, 
        y: pd.Series,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation.
        
        Uses expanding window: train on past, validate on future.
        """
        logger.info(f"Running {self.config.cv_folds}-fold time series CV...")
        
        if self.config.use_time_series_cv:
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            # Fall back to stratified k-fold (not recommended for time series)
            from sklearn.model_selection import StratifiedKFold
            tscv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=False)
        
        fold_metrics = []
        feature_importances = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X, y)):
            logger.info(f"  Fold {fold + 1}/{self.config.cv_folds}")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Clone model for this fold
            fold_model = self._clone_model(model)
            
            # Fit with early stopping for boosting models
            if hasattr(fold_model, 'fit') and self._supports_early_stopping(fold_model):
                fold_model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )
            else:
                fold_model.fit(X_train_fold, y_train_fold)
            
            # Predict
            y_pred = fold_model.predict(X_val_fold)
            y_pred_proba = fold_model.predict_proba(X_val_fold)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_fold_metrics(y_val_fold, y_pred, y_pred_proba)
            metrics['fold'] = fold + 1
            metrics['train_size'] = len(X_train_fold)
            metrics['val_size'] = len(X_val_fold)
            fold_metrics.append(metrics)
            
            # Get feature importance
            if hasattr(fold_model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': fold_model.feature_importances_
                })
                feature_importances.append(importance)
            
            logger.info(f"    AUC: {metrics['auc']:.4f}, AP: {metrics['avg_precision']:.4f}")
        
        # Aggregate results
        cv_results = self._aggregate_cv_results(fold_metrics, feature_importances)
        self.cv_results = fold_metrics
        
        return cv_results
    
    def _clone_model(self, model):
        """Clone model for cross-validation"""
        from sklearn.base import clone
        return clone(model)
    
    def _supports_early_stopping(self, model) -> bool:
        """Check if model supports early stopping"""
        return isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier))
    
    def _calculate_fold_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics for a single fold"""
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'avg_precision': average_precision_score(y_true, y_pred_proba)
        }
    
    def _aggregate_cv_results(
        self, 
        fold_metrics: List[Dict], 
        feature_importances: List[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Aggregate cross-validation results"""
        
        metrics_df = pd.DataFrame(fold_metrics)
        
        # Mean and std of metrics
        result = {}
        for col in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'avg_precision']:
            result[f'{col}_mean'] = metrics_df[col].mean()
            result[f'{col}_std'] = metrics_df[col].std()
        
        # Aggregate feature importance
        if feature_importances:
            importance_df = pd.concat(feature_importances)
            avg_importance = importance_df.groupby('feature')['importance'].mean()
            result['feature_importance'] = avg_importance.sort_values(ascending=False)
        
        result['fold_metrics'] = fold_metrics
        
        return result


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

class HyperparameterTuner:
    """Hyperparameter tuning for models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.best_params = {}
    
    def tune(
        self, 
        model, 
        X: pd.DataFrame, 
        y: pd.Series,
        model_type: str
    ) -> Dict[str, Any]:
        """Tune hyperparameters using CV"""
        
        logger.info(f"Tuning hyperparameters for {model_type}...")
        
        param_grid = self._get_param_grid(model_type)
        
        if self.config.use_time_series_cv:
            cv = TimeSeriesSplit(n_splits=3)  # Fewer folds for tuning
        else:
            cv = 3
        
        if self.config.tuning_method == 'grid':
            search = GridSearchCV(
                model, param_grid, 
                cv=cv, 
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                model, param_grid,
                n_iter=self.config.tuning_iterations,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        
        search.fit(X, y)
        
        self.best_params = search.best_params_
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
    
    def _get_param_grid(self, model_type: str) -> Dict:
        """Get parameter grid for model type"""
        
        if model_type == 'xgboost':
            return {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1.0, 2.0]
            }
        
        elif model_type == 'lightgbm':
            return {
                'max_depth': [4, 6, 8, -1],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'num_leaves': [20, 31, 50],
                'min_child_samples': [10, 20, 30],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1.0, 2.0]
            }
        
        elif model_type == 'random_forest':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 5, 10],
                'max_features': ['sqrt', 'log2', 0.5]
            }
        
        return {}


# =============================================================================
# MODEL EVALUATOR
# =============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.evaluation_results = {}
    
    def evaluate(
        self, 
        model, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation on test set"""
        
        logger.info("Evaluating model on test set...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(y_test, y_pred, y_pred_proba)
        
        # Threshold analysis
        threshold_metrics = self._threshold_analysis(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance
        importance = self._get_feature_importance(model, feature_names)
        
        # Calibration
        calibration_data = self._calculate_calibration(y_test, y_pred_proba)
        
        # ROC and PR curves
        curves = self._calculate_curves(y_test, y_pred_proba)
        
        self.evaluation_results = {
            'basic_metrics': basic_metrics,
            'threshold_metrics': threshold_metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'feature_importance': importance,
            'calibration': calibration_data,
            'curves': curves,
            'predictions': {
                'y_test': y_test.values,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }
        
        # Log key metrics
        logger.info(f"Test Results:")
        logger.info(f"  Accuracy:  {basic_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {basic_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {basic_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {basic_metrics['f1']:.4f}")
        logger.info(f"  ROC AUC:   {basic_metrics['auc']:.4f}")
        logger.info(f"  Avg Prec:  {basic_metrics['avg_precision']:.4f}")
        
        return self.evaluation_results
    
    def _calculate_basic_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'avg_precision': average_precision_score(y_true, y_pred_proba)
        }
    
    def _threshold_analysis(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray
    ) -> Dict[float, Dict]:
        """Analyze performance at different probability thresholds"""
        
        results = {}
        
        for threshold in self.config.probability_thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            n_positive = y_pred_thresh.sum()
            
            if n_positive > 0:
                precision = precision_score(y_true, y_pred_thresh, zero_division=0)
                recall = recall_score(y_true, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            else:
                precision = recall = f1 = 0
            
            results[threshold] = {
                'n_predictions': int(n_positive),
                'pct_filtered': (1 - n_positive / len(y_pred_proba)) * 100,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            logger.info(f"  Threshold {threshold}: {n_positive} signals, "
                       f"Precision={precision:.3f}, Recall={recall:.3f}")
        
        return results
    
    def _get_feature_importance(
        self, 
        model, 
        feature_names: List[str]
    ) -> Optional[pd.DataFrame]:
        """Extract feature importance from model"""
        
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        
        elif hasattr(model, 'coef_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
            return importance
        
        return None
    
    def _calculate_calibration(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray
    ) -> Dict:
        """Calculate probability calibration metrics"""
        
        fraction_positives, mean_predicted = calibration_curve(
            y_true, y_pred_proba, n_bins=10, strategy='uniform'
        )
        
        # Brier score (lower is better)
        brier = np.mean((y_pred_proba - y_true) ** 2)
        
        return {
            'fraction_positives': fraction_positives,
            'mean_predicted': mean_predicted,
            'brier_score': brier
        }
    
    def _calculate_curves(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray
    ) -> Dict:
        """Calculate ROC and PR curves"""
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        
        # PR curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        return {
            'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
            'pr': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
        }


# =============================================================================
# PROFIT SIMULATOR
# =============================================================================

class ProfitSimulator:
    """Simulate trading profits based on model predictions"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def simulate(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Simulate profit/loss based on model predictions.
        
        Assumptions:
        - Win = risk_reward_ratio R
        - Loss = -1 R
        """
        logger.info("Running profit simulation...")
        
        thresholds = thresholds or self.config.probability_thresholds
        rr = self.config.risk_reward_ratio
        
        results = {}
        
        # Baseline: take all signals
        baseline = self._calculate_profit(y_true, np.ones(len(y_true)), rr)
        results['baseline'] = baseline
        logger.info(f"Baseline (all signals): {baseline['total_r']:.2f}R over {baseline['n_trades']} trades")
        
        # Different thresholds
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            profit = self._calculate_profit(y_true, y_pred, rr)
            results[f'threshold_{threshold}'] = profit
            
            logger.info(f"Threshold {threshold}: {profit['total_r']:.2f}R over {profit['n_trades']} trades, "
                       f"WR={profit['win_rate']*100:.1f}%, PF={profit['profit_factor']:.2f}")
        
        # Find optimal threshold
        optimal = self._find_optimal_threshold(y_true, y_pred_proba, rr)
        results['optimal'] = optimal
        
        return results
    
    def _calculate_profit(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        rr: float
    ) -> Dict:
        """Calculate profit metrics for given predictions"""
        
        # Filter to only trades taken
        mask = y_pred == 1
        trades_taken = y_true[mask]
        
        n_trades = len(trades_taken)
        
        if n_trades == 0:
            return {
                'n_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_r': 0,
                'avg_r': 0,
                'profit_factor': 0,
                'max_drawdown_r': 0
            }
        
        wins = trades_taken.sum()
        losses = n_trades - wins
        win_rate = wins / n_trades
        
        gross_profit = wins * rr
        gross_loss = losses * 1.0
        total_r = gross_profit - gross_loss
        avg_r = total_r / n_trades
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        cumulative = []
        running = 0
        for outcome in trades_taken:
            if outcome == 1:
                running += rr
            else:
                running -= 1
            cumulative.append(running)
        
        cumulative = np.array(cumulative)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
        
        return {
            'n_trades': n_trades,
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': win_rate,
            'total_r': total_r,
            'avg_r': avg_r,
            'profit_factor': profit_factor,
            'max_drawdown_r': max_drawdown
        }
    
    def _find_optimal_threshold(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray, 
        rr: float
    ) -> Dict:
        """Find threshold that maximizes profit"""
        
        best_threshold = 0.5
        best_profit = -float('inf')
        
        for threshold in np.arange(0.3, 0.9, 0.05):
            y_pred = (y_pred_proba >= threshold).astype(int)
            profit = self._calculate_profit(y_true, y_pred, rr)
            
            # Require minimum number of trades
            if profit['n_trades'] >= 10 and profit['total_r'] > best_profit:
                best_profit = profit['total_r']
                best_threshold = threshold
        
        # Return metrics at optimal threshold
        y_pred_opt = (y_pred_proba >= best_threshold).astype(int)
        optimal_metrics = self._calculate_profit(y_true, y_pred_opt, rr)
        optimal_metrics['optimal_threshold'] = best_threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.2f} with {optimal_metrics['total_r']:.2f}R profit")
        
        return optimal_metrics


# =============================================================================
# VISUALIZATION
# =============================================================================

class TrainingVisualizer:
    """Visualize training results"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
    
    def plot_all(self, evaluation_results: Dict, profit_results: Dict):
        """Create all visualization plots"""
        
        if not self.config.plot_results:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. ROC Curve
        self._plot_roc_curve(axes[0, 0], evaluation_results)
        
        # 2. Precision-Recall Curve
        self._plot_pr_curve(axes[0, 1], evaluation_results)
        
        # 3. Feature Importance
        self._plot_feature_importance(axes[0, 2], evaluation_results)
        
        # 4. Confusion Matrix
        self._plot_confusion_matrix(axes[1, 0], evaluation_results)
        
        # 5. Calibration Plot
        self._plot_calibration(axes[1, 1], evaluation_results)
        
        # 6. Profit by Threshold
        self._plot_profit_by_threshold(axes[1, 2], profit_results)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_results.png', dpi=150)
        plt.close()
        
        logger.info(f"Plots saved to {self.output_dir / 'training_results.png'}")
    
    def _plot_roc_curve(self, ax, results: Dict):
        """Plot ROC curve"""
        curves = results.get('curves', {})
        roc = curves.get('roc', {})
        
        if roc:
            ax.plot(roc['fpr'], roc['tpr'], 'b-', lw=2, 
                   label=f"AUC = {results['basic_metrics']['auc']:.3f}")
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
    
    def _plot_pr_curve(self, ax, results: Dict):
        """Plot Precision-Recall curve"""
        curves = results.get('curves', {})
        pr = curves.get('pr', {})
        
        if pr:
            ax.plot(pr['recall'], pr['precision'], 'g-', lw=2,
                   label=f"AP = {results['basic_metrics']['avg_precision']:.3f}")
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
    
    def _plot_feature_importance(self, ax, results: Dict):
        """Plot top feature importances"""
        importance = results.get('feature_importance')
        
        if importance is not None:
            top_n = min(15, len(importance))
            top_features = importance.head(top_n)
            
            ax.barh(range(top_n), top_features['importance'].values, color='steelblue')
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_features['feature'].values, fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title('Top Feature Importance')
            ax.invert_yaxis()
    
    def _plot_confusion_matrix(self, ax, results: Dict):
        """Plot confusion matrix"""
        cm = results.get('confusion_matrix')
        
        if cm is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
    
    def _plot_calibration(self, ax, results: Dict):
        """Plot calibration curve"""
        calibration = results.get('calibration', {})
        
        if calibration:
            ax.plot(calibration['mean_predicted'], calibration['fraction_positives'],
                   's-', label='Model')
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f"Calibration Curve (Brier={calibration['brier_score']:.3f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_profit_by_threshold(self, ax, profit_results: Dict):
        """Plot profit by threshold"""
        thresholds = []
        total_rs = []
        win_rates = []
        
        for key, value in profit_results.items():
            if key.startswith('threshold_'):
                threshold = float(key.split('_')[1])
                thresholds.append(threshold)
                total_rs.append(value['total_r'])
                win_rates.append(value['win_rate'] * 100)
        
        if thresholds:
            ax2 = ax.twinx()
            
            bars = ax.bar(thresholds, total_rs, width=0.08, alpha=0.7, 
                         color='steelblue', label='Total R')
            line = ax2.plot(thresholds, win_rates, 'ro-', lw=2, 
                           label='Win Rate %')
            
            ax.set_xlabel('Probability Threshold')
            ax.set_ylabel('Total R Profit', color='steelblue')
            ax2.set_ylabel('Win Rate %', color='red')
            ax.set_title('Profit by Threshold')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best')


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

class ModelTrainingPipeline:
    """Main pipeline for model training"""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        
        # Initialize components
        self.model_factory = ModelFactory(self.config)
        self.cross_validator = TimeSeriesCrossValidator(self.config)
        self.tuner = HyperparameterTuner(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.profit_simulator = ProfitSimulator(self.config)
        self.visualizer = TrainingVisualizer(self.config)
        
        # Store results
        self.model = None
        self.training_results = {}
    
    def run(
        self, 
        data_dir: str,
        train_file: str = 'train.csv',
        val_file: str = 'val.csv',
        test_file: str = 'test.csv',
        features_file: str = 'features.json'
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            data_dir: Directory containing prepared data
            train_file: Training data filename
            val_file: Validation data filename
            test_file: Test data filename
            features_file: Feature names JSON filename
        
        Returns:
            Dictionary containing trained model and all results
        """
        logger.info("="*60)
        logger.info("MODEL TRAINING PIPELINE")
        logger.info("="*60)
        
        data_dir = Path(data_dir)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info("Loading prepared data...")
        train_df = pd.read_csv(data_dir / train_file)
        val_df = pd.read_csv(data_dir / val_file)
        test_df = pd.read_csv(data_dir / test_file)
        
        with open(data_dir / features_file, 'r') as f:
            features_info = json.load(f)
        
        feature_names = features_info['features']
        
        # Separate features and targets
        X_train = train_df[feature_names]
        y_train = train_df['target']
        
        X_val = val_df[feature_names]
        y_val = val_df['target']
        
        X_test = test_df[feature_names]
        y_test = test_df['target']
        
        logger.info(f"Train: {len(X_train)} samples, {len(feature_names)} features")
        logger.info(f"Val:   {len(X_val)} samples")
        logger.info(f"Test:  {len(X_test)} samples")
        
        # Create model
        logger.info(f"Creating {self.config.model_type} model...")
        self.model = self.model_factory.create_model()
        
        # Hyperparameter tuning (optional)
        if self.config.tune_hyperparams:
            tuning_results = self.tuner.tune(
                self.model, X_train, y_train, self.config.model_type
            )
            # Update model with best params
            self.model = self.model_factory.create_model()
            self.model.set_params(**self.tuner.best_params)
            self.training_results['tuning'] = tuning_results
        
        # Cross-validation
        X_train_val = pd.concat([X_train, X_val], ignore_index=True)
        y_train_val = pd.concat([y_train, y_val], ignore_index=True)
        
        cv_results = self.cross_validator.cross_validate(
            self.model, X_train_val, y_train_val, feature_names
        )
        self.training_results['cv'] = cv_results
        
        logger.info(f"\nCV Results (mean ± std):")
        logger.info(f"  AUC:       {cv_results['auc_mean']:.4f} ± {cv_results['auc_std']:.4f}")
        logger.info(f"  Precision: {cv_results['precision_mean']:.4f} ± {cv_results['precision_std']:.4f}")
        logger.info(f"  Recall:    {cv_results['recall_mean']:.4f} ± {cv_results['recall_std']:.4f}")
        
        # Train final model on full train+val data
        logger.info("\nTraining final model...")
        
        if hasattr(self.model, 'fit') and isinstance(self.model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
            # Use early stopping with validation set
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_val, y_train_val)
        
        # Calibrate probabilities (optional)
        if self.config.calibrate_probabilities:
            logger.info("Calibrating probabilities...")
            self.model = CalibratedClassifierCV(
                self.model,
                method=self.config.calibration_method,
                cv='prefit'
            )
            self.model.fit(X_val, y_val)
        
        # Evaluate on test set
        evaluation_results = self.evaluator.evaluate(
            self.model, X_test, y_test, feature_names
        )
        self.training_results['evaluation'] = evaluation_results
        
        # Profit simulation
        if self.config.run_profit_simulation:
            profit_results = self.profit_simulator.simulate(
                y_test.values,
                evaluation_results['predictions']['y_pred_proba']
            )
            self.training_results['profit'] = profit_results
        else:
            profit_results = {}
        
        # Create visualizations
        self.visualizer.plot_all(evaluation_results, profit_results)
        
        # Save model and results
        if self.config.save_model:
            self._save_model(output_dir)
        
        if self.config.save_reports:
            self._save_reports(output_dir)
        
        # Final summary
        self._print_summary()
        
        return {
            'model': self.model,
            'cv_results': cv_results,
            'evaluation': evaluation_results,
            'profit': profit_results,
            'feature_importance': evaluation_results.get('feature_importance'),
            'training_results': self.training_results
        }
    
    def _save_model(self, output_dir: Path):
        """Save trained model"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = output_dir / f'ict_signal_filter_{timestamp}.joblib'
        
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Also save as 'latest'
        latest_path = output_dir / 'ict_signal_filter_latest.joblib'
        joblib.dump(self.model, latest_path)
    
    def _save_reports(self, output_dir: Path):
        """Save training reports"""
        
        # Save evaluation metrics
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.config.model_type,
            'cv_results': {
                k: float(v) if isinstance(v, (np.float64, np.float32)) else v
                for k, v in self.training_results['cv'].items()
                if k != 'feature_importance' and k != 'fold_metrics'
            },
            'test_metrics': self.training_results['evaluation']['basic_metrics'],
            'threshold_analysis': self.training_results['evaluation']['threshold_metrics']
        }
        
        # Add profit results if available
        if 'profit' in self.training_results:
            profit = self.training_results['profit']
            report['profit_simulation'] = {
                k: {kk: float(vv) if isinstance(vv, (np.float64, np.float32)) else vv 
                    for kk, vv in v.items()}
                for k, v in profit.items()
            }
        
        # Save JSON report
        with open(output_dir / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save feature importance as CSV
        importance = self.training_results['evaluation'].get('feature_importance')
        if importance is not None:
            importance.to_csv(output_dir / 'feature_importance.csv', index=False)
        
        logger.info(f"Reports saved to {output_dir}")
    
    def _print_summary(self):
        """Print final training summary"""
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        
        eval_metrics = self.training_results['evaluation']['basic_metrics']
        
        logger.info(f"\nModel: {self.config.model_type}")
        logger.info(f"\nTest Set Performance:")
        logger.info(f"  ROC AUC:          {eval_metrics['auc']:.4f}")
        logger.info(f"  Average Precision: {eval_metrics['avg_precision']:.4f}")
        logger.info(f"  Precision:         {eval_metrics['precision']:.4f}")
        logger.info(f"  Recall:            {eval_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:          {eval_metrics['f1']:.4f}")
        
        if 'profit' in self.training_results:
            profit = self.training_results['profit']
            baseline = profit.get('baseline', {})
            optimal = profit.get('optimal', {})
            
            logger.info(f"\nProfit Simulation:")
            logger.info(f"  Baseline (all signals):")
            logger.info(f"    Trades: {baseline.get('n_trades', 0)}")
            logger.info(f"    Win Rate: {baseline.get('win_rate', 0)*100:.1f}%")
            logger.info(f"    Total R: {baseline.get('total_r', 0):.2f}")
            
            if optimal:
                logger.info(f"  Optimal Threshold ({optimal.get('optimal_threshold', 0.5):.2f}):")
                logger.info(f"    Trades: {optimal.get('n_trades', 0)}")
                logger.info(f"    Win Rate: {optimal.get('win_rate', 0)*100:.1f}%")
                logger.info(f"    Total R: {optimal.get('total_r', 0):.2f}")
                logger.info(f"    Profit Factor: {optimal.get('profit_factor', 0):.2f}")
        
        # Top features
        importance = self.training_results['evaluation'].get('feature_importance')
        if importance is not None:
            logger.info(f"\nTop 10 Features:")
            for i, row in importance.head(10).iterrows():
                logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT Model Training Pipeline')
    parser.add_argument('data_dir', help='Directory containing prepared data')
    parser.add_argument('--output-dir', default='models',
                        help='Output directory for models (default: models)')
    parser.add_argument('--model', default='xgboost',
                        choices=['xgboost', 'lightgbm', 'random_forest', 'ensemble'],
                        help='Model type (default: xgboost)')
    parser.add_argument('--tune', action='store_true',
                        help='Enable hyperparameter tuning')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--no-calibration', action='store_true',
                        help='Disable probability calibration')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable visualization plots')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_type=args.model,
        output_dir=args.output_dir,
        tune_hyperparams=args.tune,
        cv_folds=args.cv_folds,
        calibrate_probabilities=not args.no_calibration,
        plot_results=not args.no_plots
    )
    
    # Run pipeline
    pipeline = ModelTrainingPipeline(config)
    pipeline.run(args.data_dir)


if __name__ == "__main__":
    main()
