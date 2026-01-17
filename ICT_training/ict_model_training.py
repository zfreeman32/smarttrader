"""
ICT Model Training Pipeline v2.0
================================
Optimized ML training pipeline for ICT trading signal filtering.

Key Improvements over v1:
- Proper handling of pre-SMOTE'd data (no double class weighting)
- Walk-forward validation with temporal gaps
- Trading-specific metrics (profit factor, Sharpe, expected value)
- Feature importance by ICT category
- Threshold optimization based on trading performance
- Better hyperparameters for ~6K sample dataset
- Calibration improvements for probability estimation

Author: ICT ML System
Version: 2.0
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
    brier_score_loss, log_loss
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
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
from collections import defaultdict
from scipy import stats

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE CATEGORIES FOR ICT ANALYSIS
# =============================================================================

FEATURE_CATEGORIES = {
    'time_session': [
        'hour', 'minute', 'day_of_week', 'day_of_month', 'month',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'in_london_kz', 'in_nyam_kz', 'in_nypm_kz', 'in_any_kz', 'in_sb_window',
        'is_monday', 'is_friday', 'is_midweek', 'london_open_hour', 'ny_open_hour'
    ],
    'volatility_volume': [
        'atr', 'atr_ratio', 'atr_pct', 'range_atr',
        'volume_ratio', 'volume_spike', 'volume_trend', 'rvol',
        'volatility', 'vol_sma', 'high_volatility', 'low_volatility'
    ],
    'trend_momentum': [
        'close_vs_ema8', 'close_vs_ema21', 'close_vs_ema50',
        'ema_bullish_stack', 'ema_bearish_stack', 'trend_ema',
        'roc_5', 'roc_10', 'roc_20', 'rsi', 'rsi_oversold', 'rsi_overbought'
    ],
    'market_structure': [
        'is_swing_high', 'is_swing_low', 'dist_to_swing_high', 'dist_to_swing_low',
        'higher_high', 'lower_high', 'higher_low', 'lower_low',
        'hh_count', 'hl_count', 'lh_count', 'll_count', 'structure_bias',
        'bos_bullish', 'bos_bearish', 'bars_since_bullish_bos', 'bars_since_bearish_bos',
        'recent_bullish_bos', 'recent_bearish_bos'
    ],
    'premium_discount': [
        'price_position', 'in_discount', 'in_premium', 'in_equilibrium'
    ],
    'ict_concepts': [
        'bullish_fvg', 'bearish_fvg', 'fvg_size', 'in_bullish_fvg', 'in_bearish_fvg',
        'recent_bullish_fvg', 'recent_bearish_fvg',
        'is_displacement', 'displacement_size', 'bullish_displacement', 'bearish_displacement',
        'recent_bullish_disp', 'recent_bearish_disp',
        'bullish_ob', 'bearish_ob', 'recent_bullish_ob', 'recent_bearish_ob',
        'sweep_high', 'sweep_low', 'recent_sweep_high', 'recent_sweep_low'
    ],
    'key_levels': [
        'dist_to_pdh', 'dist_to_pdl', 'near_pdh', 'near_pdl', 'above_pdh', 'below_pdl'
    ],
    'candle_patterns': [
        'body_pct', 'is_bullish', 'is_bearish',
        'upper_wick_ratio', 'lower_wick_ratio',
        'is_doji', 'is_hammer', 'is_shooting_star', 'is_marubozu',
        'bullish_engulf', 'bearish_engulf', 'bullish_rejection', 'bearish_rejection'
    ],
    'confluence': [
        'trend_zone_aligned', 'fvg_trend_aligned',
        'bullish_confluence', 'bearish_confluence', 'max_confluence', 'net_confluence'
    ]
}


# =============================================================================
# CONFIGURATION - OPTIMIZED FOR YOUR DATA
# =============================================================================

@dataclass
class TrainingConfigV2:
    """
    Optimized configuration for ICT signal filtering.
    
    Key changes from v1:
    - No scale_pos_weight (data already SMOTE'd)
    - Reduced n_estimators for ~6K samples
    - Higher regularization to prevent overfitting
    - Optimized for precision (fewer false positives)
    """
    
    # Model Selection
    model_type: str = "xgboost"
    
    # XGBoost Parameters - OPTIMIZED for ~6K samples, 112 features
    xgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 200,           # Reduced from 300 (smaller dataset)
        'max_depth': 5,                 # Reduced to prevent overfitting
        'learning_rate': 0.03,          # Lower for better generalization
        'subsample': 0.8,
        'colsample_bytree': 0.7,        # More aggressive feature sampling
        'colsample_bylevel': 0.7,
        'min_child_weight': 5,          # Increased for regularization
        'gamma': 0.2,                   # Increased for pruning
        'reg_alpha': 0.5,               # L1 regularization
        'reg_lambda': 2.0,              # L2 regularization (increased)
        # NO scale_pos_weight - data is already SMOTE balanced
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc',
        'tree_method': 'hist',          # Faster training
        'max_bin': 256
    })
    
    # LightGBM Parameters - OPTIMIZED
    lgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.03,
        'num_leaves': 20,               # Conservative for small dataset
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_samples': 30,        # Increased
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        # NO scale_pos_weight
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'force_row_wise': True
    })
    
    # Random Forest Parameters - OPTIMIZED
    rf_params: Dict = field(default_factory=lambda: {
        'n_estimators': 300,            # RF benefits from more trees
        'max_depth': 8,
        'min_samples_split': 20,        # Increased
        'min_samples_leaf': 10,         # Increased
        'max_features': 0.3,            # ~33% of features per tree
        'class_weight': None,           # Data already balanced
        'random_state': 42,
        'n_jobs': -1,
        'oob_score': True               # Use OOB for additional validation
    })
    
    # Ensemble Weights (XGB, LGB, RF)
    ensemble_weights: List[float] = field(default_factory=lambda: [0.45, 0.35, 0.20])
    
    # Cross-Validation
    cv_folds: int = 5
    use_time_series_cv: bool = True
    cv_gap: int = 50                    # Gap between train and val to prevent leakage
    
    # Hyperparameter Tuning
    tune_hyperparams: bool = True       # Enable by default
    tuning_iterations: int = 30         # Reduced for faster iteration
    
    # Early Stopping
    early_stopping_rounds: int = 30     # Reduced for smaller dataset
    
    # Calibration
    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"
    
    # Trading-Specific Evaluation
    probability_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    
    # Profit Simulation
    run_profit_simulation: bool = True
    risk_reward_ratio: float = 1.5      # TP1 target
    spread_pips: float = 0.8            # Typical EUR/USD spread
    commission_per_lot: float = 0.0     # Set if applicable
    
    # Output
    output_dir: str = "models"
    save_model: bool = True
    save_reports: bool = True
    plot_results: bool = True


# =============================================================================
# WALK-FORWARD CROSS-VALIDATION WITH GAPS
# =============================================================================

class PurgedTimeSeriesSplit:
    """
    Time series split with gap between train and validation.
    Prevents data leakage from overlapping windows.
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 50, test_size: float = 0.2):
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size
    
    def split(self, X, y=None, groups=None):
        """Generate train/val indices with temporal gap"""
        n_samples = len(X)
        test_size = int(n_samples * self.test_size / self.n_splits)
        
        for i in range(self.n_splits):
            # Calculate test boundaries
            test_end = n_samples - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size
            
            # Training ends before gap
            train_end = test_start - self.gap
            train_start = 0
            
            if train_end <= train_start:
                continue
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# =============================================================================
# MODEL FACTORY
# =============================================================================

class ModelFactoryV2:
    """Factory for creating ML models with optimized parameters"""
    
    def __init__(self, config: TrainingConfigV2):
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
        
        elif model_type == 'ensemble':
            return self._create_ensemble()
        
        elif model_type == 'stacking':
            return self._create_stacking()
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_ensemble(self):
        """Create voting ensemble"""
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
    
    def _create_stacking(self):
        """Create stacking ensemble with meta-learner"""
        from sklearn.ensemble import StackingClassifier
        
        estimators = [
            ('xgb', xgb.XGBClassifier(**self.config.xgb_params)),
            ('lgb', lgb.LGBMClassifier(**self.config.lgb_params)),
            ('rf', RandomForestClassifier(**self.config.rf_params))
        ]
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3,
            passthrough=False
        )


# =============================================================================
# HYPERPARAMETER TUNING - OPTIMIZED SEARCH SPACE
# =============================================================================

class HyperparameterTunerV2:
    """Hyperparameter tuning with trading-optimized search space"""
    
    def __init__(self, config: TrainingConfigV2):
        self.config = config
        self.best_params = {}
    
    def tune(
        self, 
        model_type: str,
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Any]:
        """Tune hyperparameters using RandomizedSearchCV"""
        
        logger.info(f"Tuning hyperparameters for {model_type}...")
        
        model = ModelFactoryV2(self.config).create_model(model_type)
        param_grid = self._get_param_grid(model_type)
        
        # Use purged time series split
        cv = PurgedTimeSeriesSplit(n_splits=3, gap=self.config.cv_gap)
        
        search = RandomizedSearchCV(
            model, param_grid,
            n_iter=self.config.tuning_iterations,
            cv=cv,
            scoring='average_precision',  # Better for imbalanced data
            n_jobs=-1,
            random_state=42,
            verbose=1,
            refit=True
        )
        
        search.fit(X, y)
        
        self.best_params = search.best_params_
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV AP score: {search.best_score_:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_
        }
    
    def _get_param_grid(self, model_type: str) -> Dict:
        """Optimized parameter grids for ~6K samples"""
        
        if model_type == 'xgboost':
            return {
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.02, 0.03, 0.05],
                'n_estimators': [150, 200, 250, 300],
                'min_child_weight': [3, 5, 7, 10],
                'subsample': [0.7, 0.8, 0.85],
                'colsample_bytree': [0.6, 0.7, 0.8],
                'gamma': [0.1, 0.2, 0.3],
                'reg_alpha': [0.1, 0.3, 0.5, 1.0],
                'reg_lambda': [1.0, 2.0, 3.0]
            }
        
        elif model_type == 'lightgbm':
            return {
                'max_depth': [3, 4, 5, 6, -1],
                'learning_rate': [0.01, 0.02, 0.03, 0.05],
                'n_estimators': [150, 200, 250, 300],
                'num_leaves': [15, 20, 25, 31],
                'min_child_samples': [20, 30, 50],
                'subsample': [0.7, 0.8, 0.85],
                'colsample_bytree': [0.6, 0.7, 0.8],
                'reg_alpha': [0.1, 0.3, 0.5],
                'reg_lambda': [1.0, 2.0, 3.0]
            }
        
        elif model_type == 'random_forest':
            return {
                'n_estimators': [200, 300, 400],
                'max_depth': [6, 8, 10, None],
                'min_samples_split': [15, 20, 30],
                'min_samples_leaf': [8, 10, 15],
                'max_features': [0.2, 0.3, 0.4, 'sqrt']
            }
        
        return {}


# =============================================================================
# TRADING-SPECIFIC EVALUATOR
# =============================================================================

class TradingEvaluator:
    """Comprehensive evaluation with trading-specific metrics"""
    
    def __init__(self, config: TrainingConfigV2):
        self.config = config
        self.results = {}
    
    def evaluate(
        self, 
        model, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Full evaluation with trading metrics"""
        
        logger.info("Evaluating model on test set...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Classification metrics
        class_metrics = self._classification_metrics(y_test, y_pred, y_pred_proba)
        
        # Threshold analysis (KEY for trading)
        threshold_analysis = self._threshold_analysis(y_test, y_pred_proba)
        
        # Find optimal threshold for trading
        optimal_threshold = self._find_optimal_trading_threshold(y_test, y_pred_proba)
        
        # Calibration analysis
        calibration = self._calibration_analysis(y_test, y_pred_proba)
        
        # Feature importance by category
        importance = self._feature_importance_by_category(model, feature_names)
        
        # Probability distribution analysis
        prob_analysis = self._probability_distribution_analysis(y_test, y_pred_proba)
        
        self.results = {
            'classification_metrics': class_metrics,
            'threshold_analysis': threshold_analysis,
            'optimal_threshold': optimal_threshold,
            'calibration': calibration,
            'feature_importance': importance,
            'probability_analysis': prob_analysis,
            'predictions': {
                'y_test': y_test.values,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }
        
        # Log key results
        self._log_results(class_metrics, optimal_threshold)
        
        return self.results
    
    def _classification_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict:
        """Calculate standard classification metrics"""
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'avg_precision': average_precision_score(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def _threshold_analysis(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray
    ) -> Dict[float, Dict]:
        """Analyze performance at different thresholds"""
        
        results = {}
        rr = self.config.risk_reward_ratio
        
        for threshold in self.config.probability_thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            n_signals = y_pred_thresh.sum()
            
            if n_signals == 0:
                results[threshold] = {
                    'n_signals': 0,
                    'precision': 0,
                    'recall': 0,
                    'expected_value_per_trade': 0,
                    'profit_factor': 0
                }
                continue
            
            # Trades taken
            mask = y_pred_thresh == 1
            wins = y_true[mask].sum()
            losses = n_signals - wins
            
            precision = wins / n_signals if n_signals > 0 else 0
            recall = wins / y_true.sum() if y_true.sum() > 0 else 0
            
            # Trading metrics
            gross_profit = wins * rr
            gross_loss = losses * 1.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Expected value per trade (in R)
            ev = (precision * rr) - ((1 - precision) * 1.0)
            
            results[threshold] = {
                'n_signals': int(n_signals),
                'pct_of_total': n_signals / len(y_pred_proba) * 100,
                'wins': int(wins),
                'losses': int(losses),
                'precision': precision,
                'recall': recall,
                'profit_factor': profit_factor,
                'expected_value_per_trade': ev,
                'total_r': gross_profit - gross_loss
            }
        
        return results
    
    def _find_optimal_trading_threshold(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray
    ) -> Dict:
        """Find threshold that maximizes trading performance"""
        
        rr = self.config.risk_reward_ratio
        breakeven_precision = 1 / (1 + rr)  # Precision needed to break even
        
        best_threshold = 0.5
        best_score = -float('inf')
        best_metrics = {}
        
        for threshold in np.arange(0.40, 0.90, 0.02):
            y_pred = (y_pred_proba >= threshold).astype(int)
            n_trades = y_pred.sum()
            
            if n_trades < 20:  # Minimum trades for statistical significance
                continue
            
            mask = y_pred == 1
            wins = y_true[mask].sum()
            losses = n_trades - wins
            
            precision = wins / n_trades
            
            # Calculate combined score
            # Balances: precision, number of trades, and profitability
            gross_profit = wins * rr
            gross_loss = losses
            total_r = gross_profit - gross_loss
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            ev = (precision * rr) - ((1 - precision) * 1.0)
            
            # Score: EV * sqrt(n_trades) - penalize if too few trades
            score = ev * np.sqrt(n_trades) if ev > 0 else ev * n_trades
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'n_trades': int(n_trades),
                    'wins': int(wins),
                    'losses': int(losses),
                    'precision': precision,
                    'profit_factor': profit_factor if profit_factor != float('inf') else 999,
                    'expected_value': ev,
                    'total_r': total_r,
                    'breakeven_precision': breakeven_precision
                }
        
        logger.info(f"Optimal trading threshold: {best_threshold:.2f}")
        logger.info(f"  Trades: {best_metrics.get('n_trades', 0)}, "
                   f"Precision: {best_metrics.get('precision', 0):.1%}, "
                   f"EV: {best_metrics.get('expected_value', 0):.3f}R")
        
        return best_metrics
    
    def _calibration_analysis(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray
    ) -> Dict:
        """Analyze probability calibration"""
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        # Expected calibration error
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        # Win rate by probability bucket
        buckets = pd.cut(y_pred_proba, bins=10, labels=False)
        bucket_analysis = []
        
        for bucket in range(10):
            mask = buckets == bucket
            if mask.sum() > 0:
                actual_wr = y_true[mask].mean()
                predicted_wr = y_pred_proba[mask].mean()
                n_samples = mask.sum()
                bucket_analysis.append({
                    'bucket': bucket,
                    'predicted_prob': predicted_wr,
                    'actual_win_rate': actual_wr,
                    'n_samples': int(n_samples),
                    'calibration_error': abs(actual_wr - predicted_wr)
                })
        
        return {
            'fraction_positives': prob_true.tolist(),
            'mean_predicted': prob_pred.tolist(),
            'expected_calibration_error': ece,
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'bucket_analysis': bucket_analysis
        }
    
    def _feature_importance_by_category(
        self, 
        model, 
        feature_names: List[str]
    ) -> Dict:
        """Get feature importance grouped by ICT category"""
        
        if not hasattr(model, 'feature_importances_'):
            # Try to get from underlying estimator (for calibrated models)
            if hasattr(model, 'estimator') and hasattr(model.estimator, 'feature_importances_'):
                importances = model.estimator.feature_importances_
            else:
                return {}
        else:
            importances = model.feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Group by category
        category_importance = defaultdict(float)
        feature_to_category = {}
        
        for category, features in FEATURE_CATEGORIES.items():
            for feature in features:
                feature_to_category[feature] = category
        
        for _, row in importance_df.iterrows():
            feature = row['feature']
            imp = row['importance']
            category = feature_to_category.get(feature, 'other')
            category_importance[category] += imp
        
        # Normalize
        total = sum(category_importance.values())
        if total > 0:
            category_importance = {k: v/total for k, v in category_importance.items()}
        
        return {
            'individual': importance_df.to_dict('records'),
            'by_category': dict(sorted(category_importance.items(), 
                                       key=lambda x: x[1], reverse=True)),
            'top_10': importance_df.head(10).to_dict('records')
        }
    
    def _probability_distribution_analysis(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray
    ) -> Dict:
        """Analyze probability distributions for wins vs losses"""
        
        wins_proba = y_pred_proba[y_true == 1]
        losses_proba = y_pred_proba[y_true == 0]
        
        return {
            'wins': {
                'mean': float(wins_proba.mean()),
                'std': float(wins_proba.std()),
                'median': float(np.median(wins_proba)),
                'q25': float(np.percentile(wins_proba, 25)),
                'q75': float(np.percentile(wins_proba, 75))
            },
            'losses': {
                'mean': float(losses_proba.mean()),
                'std': float(losses_proba.std()),
                'median': float(np.median(losses_proba)),
                'q25': float(np.percentile(losses_proba, 25)),
                'q75': float(np.percentile(losses_proba, 75))
            },
            'separation': {
                'ks_statistic': float(stats.ks_2samp(wins_proba, losses_proba)[0]),
                'mean_difference': float(wins_proba.mean() - losses_proba.mean())
            }
        }
    
    def _log_results(self, class_metrics: Dict, optimal: Dict):
        """Log key evaluation results"""
        
        logger.info("\n" + "="*50)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(f"ROC AUC:          {class_metrics['roc_auc']:.4f}")
        logger.info(f"Avg Precision:    {class_metrics['avg_precision']:.4f}")
        logger.info(f"Brier Score:      {class_metrics['brier_score']:.4f}")
        
        if optimal:
            logger.info(f"\nOptimal Trading Threshold: {optimal.get('threshold', 0.5):.2f}")
            logger.info(f"  Expected trades: {optimal.get('n_trades', 0)}")
            logger.info(f"  Win rate:        {optimal.get('precision', 0):.1%}")
            logger.info(f"  Profit factor:   {optimal.get('profit_factor', 0):.2f}")
            logger.info(f"  Expected value:  {optimal.get('expected_value', 0):.3f}R per trade")


# =============================================================================
# PROFIT SIMULATOR - ENHANCED
# =============================================================================

class TradingProfitSimulator:
    """Enhanced profit simulation with transaction costs"""
    
    def __init__(self, config: TrainingConfigV2):
        self.config = config
    
    def simulate(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Run full profit simulation"""
        
        logger.info("Running profit simulation...")
        
        rr = self.config.risk_reward_ratio
        results = {}
        
        # Baseline (all signals)
        baseline = self._simulate_strategy(y_true, np.ones(len(y_true)), rr)
        results['baseline'] = baseline
        
        # Different thresholds
        for threshold in self.config.probability_thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            strategy = self._simulate_strategy(y_true, y_pred, rr)
            strategy['threshold'] = threshold
            results[f'threshold_{threshold}'] = strategy
        
        # Optimal threshold
        optimal = self._find_optimal_threshold(y_true, y_pred_proba, rr)
        results['optimal'] = optimal
        
        # Calculate improvement metrics
        if baseline['total_r'] != 0:
            improvement = (optimal['total_r'] - baseline['total_r']) / abs(baseline['total_r']) * 100
        else:
            improvement = 0
        
        results['improvement'] = {
            'r_improvement': optimal['total_r'] - baseline['total_r'],
            'pct_improvement': improvement,
            'trade_reduction': (1 - optimal['n_trades'] / baseline['n_trades']) * 100 if baseline['n_trades'] > 0 else 0
        }
        
        self._log_simulation_results(results)
        
        return results
    
    def _simulate_strategy(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        rr: float
    ) -> Dict:
        """Simulate a trading strategy"""
        
        mask = y_pred == 1
        trades = y_true[mask]
        n_trades = len(trades)
        
        if n_trades == 0:
            return {
                'n_trades': 0, 'wins': 0, 'losses': 0,
                'win_rate': 0, 'total_r': 0, 'avg_r': 0,
                'profit_factor': 0, 'sharpe_ratio': 0,
                'max_drawdown_r': 0, 'max_consecutive_losses': 0
            }
        
        wins = trades.sum()
        losses = n_trades - wins
        win_rate = wins / n_trades
        
        gross_profit = wins * rr
        gross_loss = losses * 1.0
        total_r = gross_profit - gross_loss
        avg_r = total_r / n_trades
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Generate trade returns
        trade_returns = np.where(trades == 1, rr, -1.0)
        
        # Sharpe ratio (annualized assuming 250 trading days)
        if trade_returns.std() > 0:
            daily_sharpe = trade_returns.mean() / trade_returns.std()
            annual_sharpe = daily_sharpe * np.sqrt(250 / max(1, len(trade_returns)))
        else:
            annual_sharpe = 0
        
        # Max drawdown
        cumulative = np.cumsum(trade_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_drawdown = drawdown.max()
        
        # Max consecutive losses
        max_consec_loss = self._max_consecutive(trades, target=0)
        
        return {
            'n_trades': int(n_trades),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': win_rate,
            'total_r': float(total_r),
            'avg_r': float(avg_r),
            'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999,
            'sharpe_ratio': float(annual_sharpe),
            'max_drawdown_r': float(max_drawdown),
            'max_consecutive_losses': max_consec_loss
        }
    
    def _max_consecutive(self, arr: np.ndarray, target: int) -> int:
        """Find maximum consecutive occurrences of target"""
        max_count = 0
        current_count = 0
        
        for val in arr:
            if val == target:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _find_optimal_threshold(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray, 
        rr: float
    ) -> Dict:
        """Find threshold maximizing risk-adjusted returns"""
        
        best_threshold = 0.5
        best_score = -float('inf')
        
        for threshold in np.arange(0.4, 0.85, 0.02):
            y_pred = (y_pred_proba >= threshold).astype(int)
            result = self._simulate_strategy(y_true, y_pred, rr)
            
            # Require minimum trades
            if result['n_trades'] < 15:
                continue
            
            # Score combining profit and risk adjustment
            # Sharpe-like: total_r / sqrt(max_drawdown)
            if result['max_drawdown_r'] > 0:
                score = result['total_r'] / np.sqrt(result['max_drawdown_r'])
            else:
                score = result['total_r']
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        y_pred_opt = (y_pred_proba >= best_threshold).astype(int)
        optimal_result = self._simulate_strategy(y_true, y_pred_opt, rr)
        optimal_result['optimal_threshold'] = best_threshold
        
        return optimal_result
    
    def _log_simulation_results(self, results: Dict):
        """Log simulation summary"""
        
        baseline = results.get('baseline', {})
        optimal = results.get('optimal', {})
        
        logger.info("\n" + "="*50)
        logger.info("PROFIT SIMULATION RESULTS")
        logger.info("="*50)
        
        logger.info(f"\nBaseline (all signals):")
        logger.info(f"  Trades: {baseline.get('n_trades', 0)}")
        logger.info(f"  Win Rate: {baseline.get('win_rate', 0):.1%}")
        logger.info(f"  Total R: {baseline.get('total_r', 0):.2f}")
        logger.info(f"  Profit Factor: {baseline.get('profit_factor', 0):.2f}")
        
        if optimal:
            logger.info(f"\nOptimal Threshold ({optimal.get('optimal_threshold', 0.5):.2f}):")
            logger.info(f"  Trades: {optimal.get('n_trades', 0)}")
            logger.info(f"  Win Rate: {optimal.get('win_rate', 0):.1%}")
            logger.info(f"  Total R: {optimal.get('total_r', 0):.2f}")
            logger.info(f"  Profit Factor: {optimal.get('profit_factor', 0):.2f}")
            logger.info(f"  Sharpe Ratio: {optimal.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Max Drawdown: {optimal.get('max_drawdown_r', 0):.2f}R")


# =============================================================================
# VISUALIZATION
# =============================================================================

class TrainingVisualizerV2:
    """Enhanced visualization for trading ML"""
    
    def __init__(self, config: TrainingConfigV2):
        self.config = config
        self.output_dir = Path(config.output_dir)
    
    def plot_all(
        self, 
        evaluation_results: Dict, 
        profit_results: Dict,
        cv_results: Optional[Dict] = None
    ):
        """Create comprehensive visualizations"""
        
        if not self.config.plot_results:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Main results figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        self._plot_roc_pr_curves(axes[0, 0], axes[0, 1], evaluation_results)
        self._plot_calibration(axes[0, 2], evaluation_results)
        self._plot_feature_importance(axes[1, 0], evaluation_results)
        self._plot_threshold_analysis(axes[1, 1], evaluation_results)
        self._plot_profit_by_threshold(axes[1, 2], profit_results)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_results_v2.png', dpi=150)
        plt.close()
        
        # Additional plots
        self._plot_probability_distributions(evaluation_results)
        self._plot_category_importance(evaluation_results)
        
        logger.info(f"Plots saved to {self.output_dir}")
    
    def _plot_roc_pr_curves(self, ax_roc, ax_pr, results: Dict):
        """Plot ROC and PR curves"""
        
        preds = results.get('predictions', {})
        y_test = preds.get('y_test', [])
        y_proba = preds.get('y_pred_proba', [])
        metrics = results.get('classification_metrics', {})
        
        if len(y_test) == 0:
            return
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax_roc.plot(fpr, tpr, 'b-', lw=2, 
                    label=f"AUC = {metrics.get('roc_auc', 0):.3f}")
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, alpha=0.3)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ax_pr.plot(recall, precision, 'g-', lw=2,
                   label=f"AP = {metrics.get('avg_precision', 0):.3f}")
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve')
        ax_pr.legend(loc='lower left')
        ax_pr.grid(True, alpha=0.3)
        
        # Add breakeven line
        breakeven = 1 / (1 + self.config.risk_reward_ratio)
        ax_pr.axhline(y=breakeven, color='red', linestyle='--', 
                      label=f'Breakeven ({breakeven:.1%})')
        ax_pr.legend()
    
    def _plot_calibration(self, ax, results: Dict):
        """Plot calibration curve"""
        
        calibration = results.get('calibration', {})
        
        if not calibration:
            return
        
        prob_pred = calibration.get('mean_predicted', [])
        prob_true = calibration.get('fraction_positives', [])
        
        ax.plot(prob_pred, prob_true, 's-', markersize=8, label='Model')
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
        
        ece = calibration.get('expected_calibration_error', 0)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Actual Win Rate')
        ax.set_title(f'Calibration Curve (ECE={ece:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_feature_importance(self, ax, results: Dict):
        """Plot top feature importances"""
        
        importance = results.get('feature_importance', {})
        top_10 = importance.get('top_10', [])
        
        if not top_10:
            return
        
        features = [f['feature'][:20] for f in top_10]
        importances = [f['importance'] for f in top_10]
        
        ax.barh(range(len(features)), importances, color='steelblue')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importance')
        ax.invert_yaxis()
    
    def _plot_threshold_analysis(self, ax, results: Dict):
        """Plot metrics by threshold"""
        
        threshold_data = results.get('threshold_analysis', {})
        
        thresholds = []
        precisions = []
        evs = []
        
        for thresh, data in threshold_data.items():
            thresholds.append(thresh)
            precisions.append(data.get('precision', 0) * 100)
            evs.append(data.get('expected_value_per_trade', 0))
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(thresholds, precisions, 'b-o', label='Win Rate %')
        line2 = ax2.plot(thresholds, evs, 'r-s', label='Expected Value (R)')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Win Rate %', color='blue')
        ax2.set_ylabel('Expected Value (R)', color='red')
        ax.set_title('Threshold Analysis')
        
        ax.axhline(y=100/(1+self.config.risk_reward_ratio), color='blue', 
                   linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
        ax.grid(True, alpha=0.3)
    
    def _plot_profit_by_threshold(self, ax, profit_results: Dict):
        """Plot profit metrics by threshold"""
        
        thresholds = []
        total_rs = []
        n_trades = []
        
        for key, value in profit_results.items():
            if key.startswith('threshold_'):
                threshold = float(key.split('_')[1])
                thresholds.append(threshold)
                total_rs.append(value.get('total_r', 0))
                n_trades.append(value.get('n_trades', 0))
        
        if not thresholds:
            return
        
        ax2 = ax.twinx()
        
        bars = ax.bar(thresholds, total_rs, width=0.04, alpha=0.7, 
                      color='steelblue', label='Total R')
        line = ax2.plot(thresholds, n_trades, 'ro-', lw=2, 
                        label='# Trades')
        
        ax.set_xlabel('Probability Threshold')
        ax.set_ylabel('Total R Profit', color='steelblue')
        ax2.set_ylabel('Number of Trades', color='red')
        ax.set_title('Profit by Threshold')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    def _plot_probability_distributions(self, results: Dict):
        """Plot probability distributions for wins vs losses"""
        
        preds = results.get('predictions', {})
        y_test = preds.get('y_test', [])
        y_proba = preds.get('y_pred_proba', [])
        
        if len(y_test) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        wins_proba = y_proba[y_test == 1]
        losses_proba = y_proba[y_test == 0]
        
        ax.hist(losses_proba, bins=30, alpha=0.6, label='Losses', color='red', density=True)
        ax.hist(wins_proba, bins=30, alpha=0.6, label='Wins', color='green', density=True)
        
        ax.axvline(x=np.median(losses_proba), color='darkred', linestyle='--', 
                   label=f'Loss Median: {np.median(losses_proba):.2f}')
        ax.axvline(x=np.median(wins_proba), color='darkgreen', linestyle='--',
                   label=f'Win Median: {np.median(wins_proba):.2f}')
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title('Probability Distribution: Wins vs Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'probability_distributions.png', dpi=150)
        plt.close()
    
    def _plot_category_importance(self, results: Dict):
        """Plot feature importance by category"""
        
        importance = results.get('feature_importance', {})
        by_category = importance.get('by_category', {})
        
        if not by_category:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = list(by_category.keys())
        importances = [by_category[c] * 100 for c in categories]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
        bars = ax.bar(categories, importances, color=colors)
        
        ax.set_xlabel('Feature Category')
        ax.set_ylabel('Importance (%)')
        ax.set_title('Feature Importance by ICT Category')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, importances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_importance.png', dpi=150)
        plt.close()


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

class ModelTrainingPipelineV2:
    """Enhanced training pipeline for ICT signal filtering"""
    
    def __init__(self, config: Optional[TrainingConfigV2] = None):
        self.config = config or TrainingConfigV2()
        
        # Initialize components
        self.model_factory = ModelFactoryV2(self.config)
        self.tuner = HyperparameterTunerV2(self.config)
        self.evaluator = TradingEvaluator(self.config)
        self.profit_simulator = TradingProfitSimulator(self.config)
        self.visualizer = TrainingVisualizerV2(self.config)
        
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
        """Run complete training pipeline"""
        
        logger.info("="*60)
        logger.info("ICT MODEL TRAINING PIPELINE v2.0")
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
        
        # Log data info
        logger.info(f"Train: {len(X_train)} samples ({y_train.mean():.1%} positive)")
        logger.info(f"Val:   {len(X_val)} samples ({y_val.mean():.1%} positive)")
        logger.info(f"Test:  {len(X_test)} samples ({y_test.mean():.1%} positive)")
        logger.info(f"Features: {len(feature_names)}")
        
        # Combine train+val for CV
        X_train_val = pd.concat([X_train, X_val], ignore_index=True)
        y_train_val = pd.concat([y_train, y_val], ignore_index=True)
        
        # Hyperparameter tuning
        if self.config.tune_hyperparams:
            tuning_results = self.tuner.tune(
                self.config.model_type, X_train_val, y_train_val
            )
            self.model = tuning_results['best_estimator']
            self.training_results['tuning'] = tuning_results
        else:
            logger.info(f"Creating {self.config.model_type} model with default params...")
            self.model = self.model_factory.create_model()
        
        # Train final model
        logger.info("Training final model...")
        
        if isinstance(self.model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_val, y_train_val)
        
        # Calibrate probabilities
        if self.config.calibrate_probabilities:
            logger.info("Calibrating probabilities...")
            calibrated = CalibratedClassifierCV(
                self.model,
                method=self.config.calibration_method,
                cv='prefit'
            )
            calibrated.fit(X_val, y_val)
            self.model = calibrated
        
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
        
        # Visualizations
        self.visualizer.plot_all(evaluation_results, profit_results)
        
        # Save outputs
        if self.config.save_model:
            self._save_model(output_dir)
        
        if self.config.save_reports:
            self._save_reports(output_dir)
        
        # Final summary
        self._print_summary()
        
        return {
            'model': self.model,
            'evaluation': evaluation_results,
            'profit': profit_results,
            'feature_importance': evaluation_results.get('feature_importance'),
            'training_results': self.training_results
        }
    
    def _save_model(self, output_dir: Path):
        """Save trained model"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = output_dir / f'ict_signal_filter_v2_{timestamp}.joblib'
        joblib.dump(self.model, model_path)
        
        latest_path = output_dir / 'ict_signal_filter_v2_latest.joblib'
        joblib.dump(self.model, latest_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def _save_reports(self, output_dir: Path):
        """Save comprehensive reports"""
        
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.config.model_type,
            'config': {
                'risk_reward_ratio': self.config.risk_reward_ratio,
                'calibration': self.config.calibrate_probabilities,
                'tuning': self.config.tune_hyperparams
            },
            'evaluation': convert_to_serializable(self.training_results.get('evaluation', {})),
            'profit_simulation': convert_to_serializable(self.training_results.get('profit', {}))
        }
        
        with open(output_dir / 'training_report_v2.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save feature importance
        importance = self.training_results.get('evaluation', {}).get('feature_importance', {})
        if importance.get('individual'):
            pd.DataFrame(importance['individual']).to_csv(
                output_dir / 'feature_importance_v2.csv', index=False
            )
        
        logger.info(f"Reports saved to {output_dir}")
    
    def _print_summary(self):
        """Print final summary"""
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE - SUMMARY")
        logger.info("="*60)
        
        eval_metrics = self.training_results.get('evaluation', {}).get('classification_metrics', {})
        optimal = self.training_results.get('evaluation', {}).get('optimal_threshold', {})
        profit = self.training_results.get('profit', {})
        
        logger.info(f"\nModel Performance:")
        logger.info(f"  ROC AUC:          {eval_metrics.get('roc_auc', 0):.4f}")
        logger.info(f"  Avg Precision:    {eval_metrics.get('avg_precision', 0):.4f}")
        
        if optimal:
            logger.info(f"\nRecommended Trading Configuration:")
            logger.info(f"  Threshold:        {optimal.get('threshold', 0.5):.2f}")
            logger.info(f"  Expected Trades:  {optimal.get('n_trades', 0)}")
            logger.info(f"  Win Rate:         {optimal.get('precision', 0):.1%}")
            logger.info(f"  Expected Value:   {optimal.get('expected_value', 0):.3f}R per trade")
        
        if profit:
            opt_profit = profit.get('optimal', {})
            improvement = profit.get('improvement', {})
            
            if opt_profit:
                logger.info(f"\nProfit Simulation (Optimal):")
                logger.info(f"  Total R:          {opt_profit.get('total_r', 0):.2f}")
                logger.info(f"  Profit Factor:    {opt_profit.get('profit_factor', 0):.2f}")
                logger.info(f"  Sharpe Ratio:     {opt_profit.get('sharpe_ratio', 0):.2f}")
            
            if improvement:
                logger.info(f"\nImprovement vs Baseline:")
                logger.info(f"  R Improvement:    {improvement.get('r_improvement', 0):+.2f}")
                logger.info(f"  Trade Reduction:  {improvement.get('trade_reduction', 0):.1f}%")
        
        # Feature insights
        importance = self.training_results.get('evaluation', {}).get('feature_importance', {})
        by_category = importance.get('by_category', {})
        
        if by_category:
            logger.info(f"\nTop Feature Categories:")
            for i, (cat, imp) in enumerate(list(by_category.items())[:5]):
                logger.info(f"  {i+1}. {cat}: {imp*100:.1f}%")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT Model Training Pipeline v2')
    parser.add_argument('data_dir', help='Directory containing prepared data')
    parser.add_argument('--output-dir', default='models',
                        help='Output directory (default: models)')
    parser.add_argument('--model', default='xgboost',
                        choices=['xgboost', 'lightgbm', 'random_forest', 'ensemble', 'stacking'],
                        help='Model type (default: xgboost)')
    parser.add_argument('--tune', action='store_true',
                        help='Enable hyperparameter tuning')
    parser.add_argument('--no-calibration', action='store_true',
                        help='Disable probability calibration')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--rr', type=float, default=1.5,
                        help='Risk:Reward ratio (default: 1.5)')
    
    args = parser.parse_args()
    
    config = TrainingConfigV2(
        model_type=args.model,
        output_dir=args.output_dir,
        tune_hyperparams=args.tune,
        calibrate_probabilities=not args.no_calibration,
        plot_results=not args.no_plots,
        risk_reward_ratio=args.rr
    )
    
    pipeline = ModelTrainingPipelineV2(config)
    pipeline.run(args.data_dir)


if __name__ == "__main__":
    main()