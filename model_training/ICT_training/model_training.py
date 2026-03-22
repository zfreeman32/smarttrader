"""
ICT Unified Training Pipeline V5
================================
Updated to support multiple target types:
- Classification: binary, multiclass
- Regression: mfe_regression, expected_r, time_based

Changes from V4:
- Dual model support (classifier/regressor)
- Regression metrics (RMSE, MAE, R², directional accuracy)
- Trading evaluation for continuous predictions
- MFE-based position sizing analysis

Usage:
    # Classification (legacy)
    python model_training.py data/prepared_v5 --target-type binary
    
    # Regression (new)
    python model_training.py data/prepared_v5 --target-type mfe_regression

Author: ICT ML System
Version: 5.0
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    # Classification metrics
    roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, brier_score_loss, log_loss,
    precision_recall_curve, roc_curve,
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, median_absolute_error
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import logging
import warnings
import gc
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# TARGET TYPE DEFINITIONS
# =============================================================================

CLASSIFICATION_TARGETS = ['binary', 'binary_fixed_rr', 'time_based_close', 
                          'profitable_close', 'threshold_profit', 'dynamic_rr']
REGRESSION_TARGETS = ['mfe_regression', 'expected_r', 'multi_horizon']


def is_regression_target(target_type: str) -> bool:
    """Check if target type requires regression model"""
    return target_type in REGRESSION_TARGETS


# =============================================================================
# FEATURE CATEGORIES
# =============================================================================

FEATURE_CATEGORIES = {
    'time_session': [
        'hour', 'minute', 'day_of_week', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'in_asian', 'in_london_kz', 'in_nyam_kz', 'in_nypm_kz', 'in_any_kz',
        'in_london_sb', 'in_nyam_sb', 'in_nypm_sb', 'in_sb_window', 'in_overlap',
        'is_monday', 'is_friday', 'is_midweek'
    ],
    'volatility_volume': [
        'atr', 'atr_ratio', 'atr_pct', 'range_atr', 'volume_ratio', 'volume_spike',
        'volume_trend', 'rvol', 'volatility', 'vol_sma', 'high_volatility', 'low_volatility'
    ],
    'trend_momentum': [
        'close_vs_ema8', 'close_vs_ema21', 'close_vs_ema50', 'ema_bullish_stack',
        'ema_bearish_stack', 'trend_ema', 'roc_5', 'roc_10', 'roc_20', 'rsi',
        'rsi_oversold', 'rsi_overbought'
    ],
    'market_structure': [
        'is_swing_high', 'is_swing_low', 'dist_to_swing_high', 'dist_to_swing_low',
        'higher_high', 'lower_high', 'higher_low', 'lower_low', 'hh_count', 'hl_count',
        'lh_count', 'll_count', 'structure_bias', 'bos_bullish', 'bos_bearish',
        'bars_since_bullish_bos', 'bars_since_bearish_bos', 'recent_bullish_bos', 'recent_bearish_bos'
    ],
    'premium_discount': [
        'price_position', 'in_discount', 'in_premium', 'in_equilibrium', 'dist_to_eq', 'price_zone'
    ],
    'ict_concepts': [
        'bullish_fvg', 'bearish_fvg', 'fvg_size', 'recent_bullish_fvg', 'recent_bearish_fvg',
        'is_displacement', 'displacement_size', 'bullish_displacement', 'bearish_displacement',
        'recent_bullish_disp', 'recent_bearish_disp', 'bullish_ob', 'bearish_ob', 'ob_low',
        'recent_bullish_ob', 'recent_bearish_ob', 'sweep_high', 'sweep_low',
        'recent_sweep_high', 'recent_sweep_low'
    ],
    'key_levels': [
        'dist_to_pdh', 'dist_to_pdl', 'near_pdh', 'near_pdl', 'above_pdh', 'below_pdl'
    ],
    'candle_patterns': [
        'body_pct', 'is_bullish', 'is_bearish', 'upper_wick_ratio', 'lower_wick_ratio',
        'is_doji', 'is_hammer', 'is_shooting_star', 'is_marubozu',
        'bullish_engulf', 'bearish_engulf', 'bullish_rejection', 'bearish_rejection'
    ],
    'confluence': [
        'trend_zone_aligned', 'fvg_trend_aligned', 'bullish_confluence', 'bearish_confluence',
        'max_confluence', 'net_confluence', 'signal_quality'
    ]
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class UnifiedConfigV5:
    """Configuration for V5 training with regression support"""
    
    # === Target Type ===
    target_type: str = "mfe_regression"  # See CLASSIFICATION_TARGETS and REGRESSION_TARGETS
    
    # === Walk-Forward Settings ===
    method: str = "rolling"
    train_months: int = 24
    test_months: int = 6
    step_months: int = 3
    purge_gap_days: int = 5
    min_train_samples: int = 1000
    min_test_samples: int = 200
    
    # === Model Settings ===
    model_type: str = "xgboost"  # 'xgboost' or 'lightgbm'
    
    # XGBoost Classification params
    xgb_clf_params: Dict = field(default_factory=lambda: {
        'n_estimators': 600,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 1.5,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc',
        'tree_method': 'hist',
    })
    
    # XGBoost Regression params
    xgb_reg_params: Dict = field(default_factory=lambda: {
        'n_estimators': 600,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
    })
    
    # LightGBM Classification params
    lgb_clf_params: Dict = field(default_factory=lambda: {
        'n_estimators': 600,
        'max_depth': 12,
        'learning_rate': 0.05,
        'num_leaves': 127,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 1.5,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    })
    
    # LightGBM Regression params
    lgb_reg_params: Dict = field(default_factory=lambda: {
        'n_estimators': 600,
        'max_depth': 10,
        'learning_rate': 0.05,
        'num_leaves': 127,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'objective': 'regression',
    })
    
    # === Hyperparameter Tuning ===
    tune_hyperparams: bool = True
    tuning_iterations: int = 30
    tuning_sample_size: int = 200000
    
    # === Calibration (classification only) ===
    calibrate: bool = True
    calibration_method: str = "isotonic"
    
    # === Trading Evaluation ===
    risk_reward_ratio: float = 2.0  # UPDATED to match v5 default
    
    # Classification thresholds
    probability_thresholds: List[float] = field(default_factory=lambda: 
        [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    
    # Regression thresholds (predicted MFE thresholds)
    mfe_thresholds: List[float] = field(default_factory=lambda:
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # Trade if predicted MFE > threshold
    
    # === Output ===
    output_dir: str = "unified_results_v5"
    save_production_model: bool = True
    plot_results: bool = True


# =============================================================================
# WALK-FORWARD SPLITTER
# =============================================================================

class WalkForwardSplitter:
    """Generate temporal train/test splits"""
    
    def __init__(self, config: UnifiedConfigV5):
        self.config = config
        
    def split(self, n_samples: int, is_entry_point_data: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits"""
        logger.info(f"Generating splits for {n_samples:,} samples...")
        
        if is_entry_point_data or n_samples < 50000:
            logger.info(f"Using ratio-based splits (entry-point/small dataset mode)")
            return self._ratio_based_splits(n_samples)
        
        logger.info(f"Using month-based walk-forward (raw data mode)")
        return self._month_based_splits(n_samples)
    
    def _ratio_based_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create walk-forward splits using ratios"""
        splits = []
        
        min_train = max(300, n_samples // 15)
        min_test = max(100, n_samples // 30)
        test_pct = 0.125
        
        fold_configs = [
            (0.50, 0.50 + test_pct),
            (0.625, 0.625 + test_pct),
            (0.75, 0.75 + test_pct),
            (0.875, 1.00),
        ]
        
        for train_end_pct, test_end_pct in fold_configs:
            train_end = int(n_samples * train_end_pct)
            test_start = train_end
            test_end = min(int(n_samples * test_end_pct), n_samples)
            
            if train_end >= min_train and (test_end - test_start) >= min_test:
                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, test_end)
                splits.append((train_idx, test_idx))
                
                logger.info(f"  Fold {len(splits)}: train=0-{train_end:,} ({len(train_idx):,}), "
                           f"test={test_start:,}-{test_end:,} ({len(test_idx):,})")
        
        if len(splits) == 0:
            logger.warning("Could not create proper folds - using simple 70/30 split")
            train_end = int(n_samples * 0.7)
            splits = [(np.arange(0, train_end), np.arange(train_end, n_samples))]
        
        logger.info(f"Created {len(splits)} walk-forward folds")
        return splits
    
    def _month_based_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create month-based walk-forward splits"""
        if n_samples > 2000000:
            samples_per_day = 1440 * 0.7
        elif n_samples > 500000:
            samples_per_day = 288 * 0.7
        elif n_samples > 100000:
            samples_per_day = 96 * 0.7
        else:
            samples_per_day = 24 * 0.7
        samples_per_month = samples_per_day * 21
        
        train_samples = int(self.config.train_months * samples_per_month)
        test_samples = int(self.config.test_months * samples_per_month)
        step_samples = int(self.config.step_months * samples_per_month)
        purge_samples = int(self.config.purge_gap_days * samples_per_day)
        
        splits = []
        start_idx = 0
        
        while True:
            train_start = start_idx
            train_end = train_start + train_samples
            test_start = train_end + purge_samples
            test_end = test_start + test_samples
            
            if test_end > n_samples:
                break
            
            if (train_end - train_start) >= self.config.min_train_samples and \
               (test_end - test_start) >= self.config.min_test_samples:
                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)
                splits.append((train_idx, test_idx))
            
            start_idx += step_samples
        
        if len(splits) == 0:
            return self._ratio_based_splits(n_samples)
        
        return splits


# =============================================================================
# HYPERPARAMETER TUNER (Updated for Regression)
# =============================================================================

class HyperparameterTunerV5:
    """Hyperparameter tuning for both classification and regression"""
    
    def __init__(self, config: UnifiedConfigV5):
        self.config = config
        self.is_regression = is_regression_target(config.target_type)
        
    def tune(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Dict[str, Any]:
        """Tune hyperparameters"""
        
        # Sample for speed
        if len(X) > self.config.tuning_sample_size:
            idx = np.linspace(0, len(X)-1, self.config.tuning_sample_size, dtype=int)
            X_sample = X.iloc[idx]
            y_sample = y.iloc[idx]
        else:
            X_sample = X
            y_sample = y
        
        logger.info(f"Tuning on {len(X_sample):,} samples (regression={self.is_regression})...")
        
        # Get model and param grid based on type
        if self.is_regression:
            model, param_grid = self._get_regression_config(model_type)
            scoring = 'neg_mean_squared_error'
        else:
            model, param_grid = self._get_classification_config(model_type)
            scoring = 'average_precision'
        
        cv = TimeSeriesSplit(n_splits=3)
        
        search = RandomizedSearchCV(
            model, param_grid,
            n_iter=self.config.tuning_iterations,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        search.fit(X_sample, y_sample)
        
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        logger.info(f"Best params: {search.best_params_}")
        
        return search.best_params_
    
    def _get_classification_config(self, model_type: str):
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(**self.config.xgb_clf_params)
            param_grid = {
                'max_depth': [6, 8, 10, 12],
                'learning_rate': [0.03, 0.05, 0.07],
                'n_estimators': [400, 600, 800],
                'min_child_weight': [5, 10, 20],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'scale_pos_weight': [1.0, 1.3, 1.5, 1.7]
            }
        else:
            model = lgb.LGBMClassifier(**self.config.lgb_clf_params)
            param_grid = {
                'max_depth': [8, 10, 12, -1],
                'learning_rate': [0.03, 0.05, 0.07],
                'n_estimators': [400, 600, 800],
                'num_leaves': [63, 127, 255],
                'min_child_samples': [50, 100, 200],
                'subsample': [0.7, 0.8, 0.9],
            }
        return model, param_grid
    
    def _get_regression_config(self, model_type: str):
        if model_type == 'xgboost':
            model = xgb.XGBRegressor(**self.config.xgb_reg_params)
            param_grid = {
                'max_depth': [5, 6, 8, 10],
                'learning_rate': [0.03, 0.05, 0.07],
                'n_estimators': [400, 600, 800],
                'min_child_weight': [5, 10, 20],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
            }
        else:
            model = lgb.LGBMRegressor(**self.config.lgb_reg_params)
            param_grid = {
                'max_depth': [6, 8, 10, -1],
                'learning_rate': [0.03, 0.05, 0.07],
                'n_estimators': [400, 600, 800],
                'num_leaves': [31, 63, 127],
                'min_child_samples': [50, 100, 200],
                'subsample': [0.7, 0.8, 0.9],
            }
        return model, param_grid


# =============================================================================
# FOLD TRAINER (Updated for Regression)
# =============================================================================

class EnhancedFoldTrainerV5:
    """Train and evaluate a single fold - supports classification and regression"""
    
    def __init__(self, config: UnifiedConfigV5, tuned_params: Optional[Dict] = None):
        self.config = config
        self.tuned_params = tuned_params
        self.is_regression = is_regression_target(config.target_type)
        
    def train_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        fold_num: int,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train and evaluate fold"""
        
        # Get appropriate params
        params = self._get_model_params()
        if self.tuned_params:
            params.update(self.tuned_params)
        
        # Create model
        model = self._create_model(params)
        
        # Split for early stopping
        val_size = min(50000, int(len(X_train) * 0.1))
        X_train_fit = X_train.iloc[:-val_size]
        y_train_fit = y_train.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        
        # Train
        start_time = datetime.now()
        model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        train_time = (datetime.now() - start_time).total_seconds()
        
        # Predict and evaluate based on task type
        if self.is_regression:
            y_pred = model.predict(X_test)
            metrics = self._calculate_regression_metrics(y_test.values, y_pred)
            trading = self._calculate_regression_trading_metrics(y_test.values, y_pred)
            separation = self._calculate_regression_separation(y_test.values, y_pred)
            calibration = {}  # Not applicable for regression
        else:
            # Calibrate for classification
            if self.config.calibrate:
                calibrated = CalibratedClassifierCV(
                    model, method=self.config.calibration_method, cv='prefit'
                )
                calibrated.fit(X_val, y_val)
                model = calibrated
            
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            metrics = self._calculate_classification_metrics(y_test.values, y_pred, y_pred_proba)
            trading = self._calculate_classification_trading_metrics(y_test.values, y_pred_proba)
            calibration = self._calculate_calibration(y_test.values, y_pred_proba)
            separation = self._calculate_classification_separation(y_test.values, y_pred_proba)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'estimator') and hasattr(model.estimator, 'feature_importances_'):
            importance = model.estimator.feature_importances_
        else:
            importance = np.zeros(len(feature_names))
        
        result = {
            'fold': fold_num,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_time': train_time,
            'metrics': metrics,
            'trading_metrics': trading,
            'calibration': calibration,
            'separation': separation,
            'feature_importance': dict(zip(feature_names, importance.tolist())),
            'model': model if fold_num == 0 else None
        }
        
        # Add task-specific fields
        if self.is_regression:
            result['train_mean'] = float(y_train.mean())
            result['test_mean'] = float(y_test.mean())
            result['predictions'] = {'y_true': y_test.values, 'y_pred': y_pred}
        else:
            result['train_positive_rate'] = float(y_train.mean())
            result['test_positive_rate'] = float(y_test.mean())
            result['predictions'] = {'y_true': y_test.values, 'y_pred_proba': y_pred_proba}
        
        return result
    
    def _get_model_params(self) -> Dict:
        if self.is_regression:
            if self.config.model_type == 'xgboost':
                return self.config.xgb_reg_params.copy()
            return self.config.lgb_reg_params.copy()
        else:
            if self.config.model_type == 'xgboost':
                return self.config.xgb_clf_params.copy()
            return self.config.lgb_clf_params.copy()
    
    def _create_model(self, params: Dict):
        if self.is_regression:
            if self.config.model_type == 'xgboost':
                return xgb.XGBRegressor(**params)
            return lgb.LGBMRegressor(**params)
        else:
            if self.config.model_type == 'xgboost':
                return xgb.XGBClassifier(**params)
            return lgb.LGBMClassifier(**params)
    
    # =========================================================================
    # REGRESSION METRICS
    # =========================================================================
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)
        median_ae = median_absolute_error(y_true, y_pred)
        
        # Directional accuracy (did we predict direction correctly?)
        # For MFE, this means: did we predict high MFE for high actual MFE?
        y_true_high = y_true > np.median(y_true)
        y_pred_high = y_pred > np.median(y_pred)
        directional_acc = (y_true_high == y_pred_high).mean()
        
        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Mean Absolute Percentage Error (for interpretability)
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'explained_variance': float(explained_var),
            'median_ae': float(median_ae),
            'directional_accuracy': float(directional_acc),
            'correlation': float(correlation) if not np.isnan(correlation) else 0,
            'mape': float(mape),
        }
    
    def _calculate_regression_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate trading metrics for regression (MFE prediction).
        
        Strategy: Trade when predicted MFE > threshold
        Actual outcome: Use actual MFE (or could simulate with R:R)
        """
        results = {}
        
        for threshold in self.config.mfe_thresholds:
            # Trade signal: predicted MFE above threshold
            trade_mask = y_pred >= threshold
            n_trades = trade_mask.sum()
            
            if n_trades == 0:
                results[threshold] = {
                    'n_trades': 0, 'mean_mfe': 0, 'total_r': 0,
                    'hit_rate': 0, 'avg_pred': 0
                }
                continue
            
            # Actual outcomes for trades taken
            actual_mfe = y_true[trade_mask]
            predicted_mfe = y_pred[trade_mask]
            
            # Hit rate: how many achieved at least the threshold MFE?
            hit_rate = (actual_mfe >= threshold).mean()
            
            # Mean actual MFE for trades taken
            mean_mfe = actual_mfe.mean()
            
            # Total R (simplified: sum of actual MFE - assumes no stop loss)
            # For more realistic: total_r = sum(min(actual_mfe, threshold) - 1)  # if SL at -1R
            total_r = actual_mfe.sum()
            
            # Profit factor (total profit / total loss)
            profits = actual_mfe[actual_mfe > 0].sum()
            losses = abs(actual_mfe[actual_mfe < 0].sum())
            pf = profits / losses if losses > 0 else 999.0
            
            results[threshold] = {
                'n_trades': int(n_trades),
                'hit_rate': float(hit_rate),
                'mean_actual_mfe': float(mean_mfe),
                'mean_predicted_mfe': float(predicted_mfe.mean()),
                'prediction_error': float(predicted_mfe.mean() - mean_mfe),
                'total_r': float(total_r),
                'profit_factor': float(pf),
                'pct_of_trades': float(n_trades / len(y_true) * 100),
            }
        
        return results
    
    def _calculate_regression_separation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate prediction separation for regression"""
        from scipy import stats
        
        # Split into high/low actual MFE
        median_mfe = np.median(y_true)
        high_mfe_mask = y_true > median_mfe
        low_mfe_mask = y_true <= median_mfe
        
        pred_for_high = y_pred[high_mfe_mask]
        pred_for_low = y_pred[low_mfe_mask]
        
        # KS test on predictions
        if len(pred_for_high) > 10 and len(pred_for_low) > 10:
            ks_stat = stats.ks_2samp(pred_for_high, pred_for_low)[0]
        else:
            ks_stat = 0
        
        return {
            'ks_statistic': float(ks_stat),
            'high_mfe_pred_mean': float(pred_for_high.mean()) if len(pred_for_high) > 0 else 0,
            'low_mfe_pred_mean': float(pred_for_low.mean()) if len(pred_for_low) > 0 else 0,
            'mean_difference': float(pred_for_high.mean() - pred_for_low.mean()) if len(pred_for_high) > 0 and len(pred_for_low) > 0 else 0,
        }
    
    # =========================================================================
    # CLASSIFICATION METRICS
    # =========================================================================
    
    def _calculate_classification_metrics(self, y_true, y_pred, y_proba):
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            auc = 0.5
        return {
            'roc_auc': float(auc),
            'avg_precision': float(average_precision_score(y_true, y_proba)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'brier_score': float(brier_score_loss(y_true, y_proba)),
        }
    
    def _calculate_classification_trading_metrics(self, y_true, y_proba):
        rr = self.config.risk_reward_ratio
        results = {}
        
        for threshold in self.config.probability_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            n_trades = y_pred.sum()
            
            if n_trades == 0:
                results[threshold] = {'n_trades': 0, 'win_rate': 0, 'ev': 0, 'total_r': 0}
                continue
            
            mask = y_pred == 1
            wins = y_true[mask].sum()
            losses = n_trades - wins
            
            win_rate = wins / n_trades
            ev = (win_rate * rr) - ((1 - win_rate) * 1.0)
            total_r = wins * rr - losses
            pf = (wins * rr) / losses if losses > 0 else 999.0
            
            results[threshold] = {
                'n_trades': int(n_trades),
                'wins': int(wins),
                'losses': int(losses),
                'win_rate': float(win_rate),
                'profit_factor': float(pf),
                'ev': float(ev),
                'total_r': float(total_r)
            }
        
        return results
    
    def _calculate_calibration(self, y_true, y_proba):
        if len(y_true) > 50000:
            idx = np.random.choice(len(y_true), 50000, replace=False)
            y_true = y_true[idx]
            y_proba = y_proba[idx]
        
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        return {
            'ece': float(ece),
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }
    
    def _calculate_classification_separation(self, y_true, y_proba):
        from scipy import stats
        wins_proba = y_proba[y_true == 1]
        losses_proba = y_proba[y_true == 0]
        
        if len(wins_proba) > 0 and len(losses_proba) > 0:
            ks_stat = stats.ks_2samp(wins_proba, losses_proba)[0]
            mean_diff = wins_proba.mean() - losses_proba.mean()
        else:
            ks_stat = 0
            mean_diff = 0
        
        return {
            'ks_statistic': float(ks_stat),
            'wins_mean': float(wins_proba.mean()) if len(wins_proba) > 0 else 0,
            'losses_mean': float(losses_proba.mean()) if len(losses_proba) > 0 else 0,
            'mean_difference': float(mean_diff),
        }


# =============================================================================
# STABILITY ANALYZER (Updated for Regression)
# =============================================================================

class StabilityAnalyzerV5:
    """Analyze stability across folds - supports classification and regression"""
    
    def __init__(self, config: UnifiedConfigV5):
        self.config = config
        self.is_regression = is_regression_target(config.target_type)
    
    def analyze(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Comprehensive stability analysis"""
        
        if self.is_regression:
            return self._analyze_regression(fold_results)
        return self._analyze_classification(fold_results)
    
    def _analyze_regression(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Analyze regression model stability"""
        
        r2_scores = [r['metrics']['r2'] for r in fold_results]
        rmse_scores = [r['metrics']['rmse'] for r in fold_results]
        correlations = [r['metrics']['correlation'] for r in fold_results]
        dir_acc = [r['metrics']['directional_accuracy'] for r in fold_results]
        
        # R² stability
        r2_stats = {
            'mean': float(np.mean(r2_scores)),
            'std': float(np.std(r2_scores)),
            'min': float(np.min(r2_scores)),
            'max': float(np.max(r2_scores)),
            'above_0': sum(r > 0 for r in r2_scores),
        }
        
        # RMSE stability
        rmse_stats = {
            'mean': float(np.mean(rmse_scores)),
            'std': float(np.std(rmse_scores)),
        }
        
        # Correlation stability
        corr_stats = {
            'mean': float(np.mean(correlations)),
            'std': float(np.std(correlations)),
            'above_03': sum(c > 0.3 for c in correlations),
        }
        
        # Separation
        ks_stats = [r['separation']['ks_statistic'] for r in fold_results]
        separation_stats = {
            'ks_mean': float(np.mean(ks_stats)),
            'ks_std': float(np.std(ks_stats)),
        }
        
        # Trading stability by MFE threshold
        trading_stability = {}
        for thresh in self.config.mfe_thresholds:
            total_rs = []
            hit_rates = []
            for r in fold_results:
                tm = r['trading_metrics'].get(thresh, {})
                if tm.get('n_trades', 0) > 20:
                    total_rs.append(tm['total_r'])
                    hit_rates.append(tm['hit_rate'])
            
            if total_rs:
                trading_stability[thresh] = {
                    'hit_rate_mean': float(np.mean(hit_rates)),
                    'hit_rate_std': float(np.std(hit_rates)),
                    'total_r_mean': float(np.mean(total_rs)),
                    'profitable_folds': sum(r > 0 for r in total_rs),
                }
        
        # Feature stability
        feature_stability = self._analyze_feature_stability(fold_results)
        
        # Verdict
        verdict = self._generate_regression_verdict(r2_stats, corr_stats, separation_stats, trading_stability, len(fold_results))
        
        return {
            'r2_stats': r2_stats,
            'rmse_stats': rmse_stats,
            'correlation_stats': corr_stats,
            'directional_accuracy_mean': float(np.mean(dir_acc)),
            'separation_stats': separation_stats,
            'trading_stability': trading_stability,
            'feature_stability': feature_stability,
            'verdict': verdict,
        }
    
    def _analyze_classification(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Analyze classification model stability (original logic)"""
        
        aucs = [r['metrics']['roc_auc'] for r in fold_results]
        ks_stats = [r['separation']['ks_statistic'] for r in fold_results]
        mean_diffs = [r['separation']['mean_difference'] for r in fold_results]
        
        auc_stats = {
            'mean': float(np.mean(aucs)),
            'std': float(np.std(aucs)),
            'min': float(np.min(aucs)),
            'max': float(np.max(aucs)),
            'above_055': sum(a > 0.55 for a in aucs),
        }
        
        separation_stats = {
            'ks_mean': float(np.mean(ks_stats)),
            'ks_std': float(np.std(ks_stats)),
            'mean_diff_mean': float(np.mean(mean_diffs)),
            'mean_diff_std': float(np.std(mean_diffs)),
        }
        
        # Trading stability
        trading_stability = {}
        for thresh in self.config.probability_thresholds:
            wrs = []
            total_rs = []
            for r in fold_results:
                tm = r['trading_metrics'].get(thresh, {})
                if tm.get('n_trades', 0) > 50:
                    wrs.append(tm['win_rate'])
                    total_rs.append(tm['total_r'])
            
            if wrs:
                trading_stability[thresh] = {
                    'wr_mean': float(np.mean(wrs)),
                    'wr_std': float(np.std(wrs)),
                    'total_r_mean': float(np.mean(total_rs)),
                    'profitable_folds': sum(r > 0 for r in total_rs),
                }
        
        feature_stability = self._analyze_feature_stability(fold_results)
        verdict = self._generate_classification_verdict(auc_stats, separation_stats, trading_stability, len(fold_results))
        
        return {
            'auc_stats': auc_stats,
            'separation_stats': separation_stats,
            'trading_stability': trading_stability,
            'feature_stability': feature_stability,
            'verdict': verdict,
        }
    
    def _analyze_feature_stability(self, fold_results: List[Dict]) -> Dict:
        importance_by_feature = defaultdict(list)
        for r in fold_results:
            for feat, imp in r['feature_importance'].items():
                importance_by_feature[feat].append(imp)
        
        feature_stability = {}
        for feat, imps in importance_by_feature.items():
            feature_stability[feat] = {
                'mean': float(np.mean(imps)),
                'std': float(np.std(imps)),
            }
        
        # Return top 20
        return dict(sorted(feature_stability.items(), key=lambda x: x[1]['mean'], reverse=True)[:20])
    
    def _generate_regression_verdict(self, r2_stats, corr_stats, sep_stats, trading, n_folds):
        score = 0
        issues = []
        strengths = []
        
        # R² score
        mean_r2 = r2_stats['mean']
        if mean_r2 >= 0.3:
            score += 30
            strengths.append(f"Good R² ({mean_r2:.3f})")
        elif mean_r2 >= 0.15:
            score += 15
            strengths.append(f"Moderate R² ({mean_r2:.3f})")
        elif mean_r2 >= 0:
            score += 5
        else:
            issues.append(f"Negative R² ({mean_r2:.3f}) - worse than mean prediction")
        
        # Correlation
        mean_corr = corr_stats['mean']
        if mean_corr >= 0.5:
            score += 25
            strengths.append(f"Strong correlation ({mean_corr:.3f})")
        elif mean_corr >= 0.3:
            score += 15
            strengths.append(f"Moderate correlation ({mean_corr:.3f})")
        else:
            issues.append(f"Weak correlation ({mean_corr:.3f})")
        
        # Separation
        ks = sep_stats.get('ks_mean', 0)
        if ks >= 0.2:
            score += 20
            strengths.append(f"Good prediction separation (KS={ks:.3f})")
        elif ks >= 0.1:
            score += 10
        else:
            issues.append(f"Poor prediction separation (KS={ks:.3f})")
        
        # Stability
        if r2_stats['std'] < 0.05:
            score += 10
            strengths.append("Stable R² across folds")
        elif r2_stats['std'] > 0.1:
            issues.append(f"Unstable R² across folds (std={r2_stats['std']:.3f})")
        
        # Grade
        if score >= 70:
            grade = 'A'
            rec = "Model shows strong MFE prediction ability. Consider deployment with proper risk management."
        elif score >= 50:
            grade = 'B'
            rec = "Model shows useful MFE prediction. May benefit from feature engineering or more data."
        elif score >= 30:
            grade = 'C'
            rec = "Model shows weak prediction. Review features and target definition."
        else:
            grade = 'D'
            rec = "Model not viable. Consider fundamental changes to approach."
        
        # Best threshold
        best_thresh = None
        best_r = -999
        for thresh, stats in trading.items():
            if stats.get('total_r_mean', -999) > best_r:
                best_r = stats['total_r_mean']
                best_thresh = thresh
        
        return {
            'grade': grade,
            'score': score,
            'recommendation': rec,
            'strengths': strengths,
            'issues': issues,
            'best_mfe_threshold': best_thresh,
        }
    
    def _generate_classification_verdict(self, auc_stats, sep_stats, trading, n_folds):
        score = 0
        issues = []
        strengths = []
        
        mean_auc = auc_stats['mean']
        if mean_auc >= 0.55:
            score += 30
            strengths.append(f"Good AUC ({mean_auc:.3f})")
        elif mean_auc >= 0.52:
            score += 15
        else:
            issues.append(f"Weak AUC ({mean_auc:.3f})")
        
        ks = sep_stats.get('ks_mean', 0)
        if ks >= 0.15:
            score += 25
            strengths.append(f"Good separation (KS={ks:.3f})")
        elif ks >= 0.08:
            score += 10
        else:
            issues.append(f"Poor separation (KS={ks:.3f})")
        
        if auc_stats['std'] < 0.02:
            score += 15
            strengths.append("Stable AUC")
        elif auc_stats['std'] > 0.04:
            issues.append(f"Unstable AUC (std={auc_stats['std']:.3f})")
        
        if score >= 60:
            grade = 'A'
            rec = "Model viable for trading with appropriate threshold selection."
        elif score >= 40:
            grade = 'B'
            rec = "Model shows promise but needs refinement."
        elif score >= 20:
            grade = 'C'
            rec = "Model weak. Review features and data."
        else:
            grade = 'D'
            rec = "Model not viable."
        
        return {
            'grade': grade,
            'score': score,
            'recommendation': rec,
            'strengths': strengths,
            'issues': issues,
        }


# =============================================================================
# UNIFIED TRAINING PIPELINE
# =============================================================================

class UnifiedTrainingPipelineV5:
    """Main pipeline - updated for V5 targets"""
    
    def __init__(self, config: UnifiedConfigV5):
        self.config = config
        self.is_regression = is_regression_target(config.target_type)
        
        self.splitter = WalkForwardSplitter(config)
        self.tuner = HyperparameterTunerV5(config)
        self.analyzer = StabilityAnalyzerV5(config)
        
        self.fold_results = []
        self.tuned_params = None
        self.production_model = None
        self.aggregated = {}
        self.analysis = {}
    
    def run(self, data_dir: str) -> Dict[str, Any]:
        """Run complete training pipeline"""
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info(f"ICT UNIFIED TRAINING PIPELINE V5")
        logger.info(f"Target Type: {self.config.target_type} ({'REGRESSION' if self.is_regression else 'CLASSIFICATION'})")
        logger.info("=" * 60)
        
        # Load data
        df = self._load_data(data_dir)
        feature_names = [c for c in df.columns if c != 'target']
        
        logger.info(f"Data: {len(df):,} samples, {len(feature_names)} features")
        logger.info(f"Target stats: mean={df['target'].mean():.4f}, std={df['target'].std():.4f}")
        
        # Generate splits
        splits = self.splitter.split(len(df), is_entry_point_data=True)
        
        # Hyperparameter tuning
        if self.config.tune_hyperparams:
            logger.info("\n" + "=" * 50)
            logger.info("HYPERPARAMETER TUNING")
            logger.info("=" * 50)
            
            X = df[feature_names]
            y = df['target']
            self.tuned_params = self.tuner.tune(X, y, self.config.model_type)
        
        # Walk-forward validation
        logger.info("\n" + "=" * 50)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("=" * 50)
        
        trainer = EnhancedFoldTrainerV5(self.config, self.tuned_params)
        
        for fold_num, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"\n--- Fold {fold_num + 1}/{len(splits)} ---")
            
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            X_train = train_df[feature_names]
            y_train = train_df['target']
            X_test = test_df[feature_names]
            y_test = test_df['target']
            
            result = trainer.train_fold(X_train, y_train, X_test, y_test, fold_num, feature_names)
            self.fold_results.append(result)
            
            # Log key metrics
            if self.is_regression:
                logger.info(f"  R²: {result['metrics']['r2']:.4f}, "
                           f"RMSE: {result['metrics']['rmse']:.4f}, "
                           f"Correlation: {result['metrics']['correlation']:.4f}")
            else:
                logger.info(f"  AUC: {result['metrics']['roc_auc']:.4f}, "
                           f"Separation: {result['separation']['ks_statistic']:.4f}")
            
            gc.collect()
        
        # Aggregate results
        self.aggregated = self._aggregate_results()
        
        # Stability analysis
        logger.info("\n" + "=" * 50)
        logger.info("STABILITY ANALYSIS")
        logger.info("=" * 50)
        self.analysis = self.analyzer.analyze(self.fold_results)
        
        # Train production model
        if self.config.save_production_model:
            self._train_production_model(df, feature_names)
        
        # Print summary
        self._print_summary()
        
        # Save reports
        self._save_reports(output_dir)
        
        # Create plots
        if self.config.plot_results:
            self._create_plots(output_dir)
        
        return {
            'fold_results': self.fold_results,
            'aggregated': self.aggregated,
            'analysis': self.analysis,
            'production_model': self.production_model,
        }
    
    def _load_data(self, data_dir: str) -> pd.DataFrame:
        """Load prepared data"""
        data_path = Path(data_dir)
        
        if data_path.is_file():
            return pd.read_csv(data_path)
        
        # Load train/val/test and combine
        dfs = []
        for split in ['train', 'val', 'test']:
            path = data_path / f'{split}.csv'
            if path.exists():
                dfs.append(pd.read_csv(path))
        
        if not dfs:
            raise FileNotFoundError(f"No data found in {data_dir}")
        
        return pd.concat(dfs, ignore_index=True)
    
    def _aggregate_results(self) -> Dict:
        """Aggregate fold results"""
        if self.is_regression:
            return self._aggregate_regression_results()
        return self._aggregate_classification_results()
    
    def _aggregate_regression_results(self) -> Dict:
        """Aggregate regression results"""
        all_y_true = np.concatenate([r['predictions']['y_true'] for r in self.fold_results])
        all_y_pred = np.concatenate([r['predictions']['y_pred'] for r in self.fold_results])
        
        overall_metrics = {
            'r2': float(r2_score(all_y_true, all_y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(all_y_true, all_y_pred))),
            'mae': float(mean_absolute_error(all_y_true, all_y_pred)),
            'correlation': float(np.corrcoef(all_y_true, all_y_pred)[0, 1]),
            'n_samples': len(all_y_true),
        }
        
        # Aggregate trading metrics
        trading = {}
        for thresh in self.config.mfe_thresholds:
            n_trades = 0
            total_r = 0
            hit_count = 0
            for r in self.fold_results:
                tm = r['trading_metrics'].get(thresh, {})
                n = tm.get('n_trades', 0)
                n_trades += n
                total_r += tm.get('total_r', 0)
                hit_count += int(tm.get('hit_rate', 0) * n)
            
            if n_trades > 0:
                trading[thresh] = {
                    'n_trades': n_trades,
                    'hit_rate': float(hit_count / n_trades),
                    'total_r': float(total_r),
                }
            else:
                trading[thresh] = {'n_trades': 0, 'hit_rate': 0, 'total_r': 0}
        
        return {
            'overall_metrics': overall_metrics,
            'trading_metrics': trading,
            'n_folds': len(self.fold_results),
        }
    
    def _aggregate_classification_results(self) -> Dict:
        """Aggregate classification results"""
        all_y_true = np.concatenate([r['predictions']['y_true'] for r in self.fold_results])
        all_y_proba = np.concatenate([r['predictions']['y_pred_proba'] for r in self.fold_results])
        
        overall_metrics = {
            'roc_auc': float(roc_auc_score(all_y_true, all_y_proba)),
            'avg_precision': float(average_precision_score(all_y_true, all_y_proba)),
            'brier_score': float(brier_score_loss(all_y_true, all_y_proba)),
            'n_samples': len(all_y_true),
        }
        
        # Trading metrics
        rr = self.config.risk_reward_ratio
        trading = {}
        for thresh in self.config.probability_thresholds:
            y_pred = (all_y_proba >= thresh).astype(int)
            n_trades = y_pred.sum()
            if n_trades > 0:
                wins = all_y_true[y_pred == 1].sum()
                losses = n_trades - wins
                trading[thresh] = {
                    'n_trades': int(n_trades),
                    'win_rate': float(wins / n_trades),
                    'total_r': float(wins * rr - losses),
                }
            else:
                trading[thresh] = {'n_trades': 0, 'win_rate': 0, 'total_r': 0}
        
        return {
            'overall_metrics': overall_metrics,
            'trading_metrics': trading,
            'n_folds': len(self.fold_results),
        }
    
    def _train_production_model(self, df: pd.DataFrame, feature_names: List[str]):
        """Train final production model"""
        
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING PRODUCTION MODEL")
        logger.info("=" * 50)
        
        X = df[feature_names]
        y = df['target']
        
        # Get params
        if self.is_regression:
            if self.config.model_type == 'xgboost':
                params = self.config.xgb_reg_params.copy()
            else:
                params = self.config.lgb_reg_params.copy()
        else:
            if self.config.model_type == 'xgboost':
                params = self.config.xgb_clf_params.copy()
            else:
                params = self.config.lgb_clf_params.copy()
        
        if self.tuned_params:
            params.update(self.tuned_params)
        
        # Create model
        if self.is_regression:
            if self.config.model_type == 'xgboost':
                model = xgb.XGBRegressor(**params)
            else:
                model = lgb.LGBMRegressor(**params)
        else:
            if self.config.model_type == 'xgboost':
                model = xgb.XGBClassifier(**params)
            else:
                model = lgb.LGBMClassifier(**params)
        
        # Train with validation
        val_size = int(len(X) * 0.1)
        X_train = X.iloc[:-val_size]
        y_train = y.iloc[:-val_size]
        X_val = X.iloc[-val_size:]
        y_val = y.iloc[-val_size:]
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        
        # Calibrate (classification only)
        if not self.is_regression and self.config.calibrate:
            calibrated = CalibratedClassifierCV(model, method=self.config.calibration_method, cv='prefit')
            calibrated.fit(X_val, y_val)
            model = calibrated
        
        self.production_model = model
        
        # Save
        output_dir = Path(self.config.output_dir)
        model_path = output_dir / 'production_model.joblib'
        joblib.dump(model, model_path)
        
        # Save feature names
        with open(output_dir / 'feature_names.json', 'w') as f:
            json.dump(feature_names, f)
        
        logger.info(f"Production model saved to {model_path}")
    
    def _print_summary(self):
        """Print summary"""
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 70)
        
        agg = self.aggregated
        analysis = self.analysis
        
        overall = agg.get('overall_metrics', {})
        
        if self.is_regression:
            logger.info(f"\nOverall Performance ({agg.get('n_folds', 0)} folds, {overall.get('n_samples', 0):,} samples):")
            logger.info(f"  R²:          {overall.get('r2', 0):.4f}")
            logger.info(f"  RMSE:        {overall.get('rmse', 0):.4f}")
            logger.info(f"  MAE:         {overall.get('mae', 0):.4f}")
            logger.info(f"  Correlation: {overall.get('correlation', 0):.4f}")
            
            r2_stats = analysis.get('r2_stats', {})
            logger.info(f"\nR² Stability:")
            logger.info(f"  Mean ± Std:  {r2_stats.get('mean', 0):.4f} ± {r2_stats.get('std', 0):.4f}")
            logger.info(f"  Range:       [{r2_stats.get('min', 0):.4f}, {r2_stats.get('max', 0):.4f}]")
        else:
            logger.info(f"\nOverall Performance ({agg.get('n_folds', 0)} folds, {overall.get('n_samples', 0):,} samples):")
            logger.info(f"  ROC AUC:       {overall.get('roc_auc', 0):.4f}")
            logger.info(f"  Avg Precision: {overall.get('avg_precision', 0):.4f}")
            logger.info(f"  Brier Score:   {overall.get('brier_score', 0):.4f}")
        
        # Verdict
        verdict = analysis.get('verdict', {})
        logger.info(f"\n{'='*50}")
        logger.info(f"VERDICT: {verdict.get('grade', '?')} ({verdict.get('score', 0)}/100)")
        logger.info(f"{'='*50}")
        logger.info(f"{verdict.get('recommendation', '')}")
        
        if verdict.get('strengths'):
            logger.info(f"\nStrengths:")
            for s in verdict['strengths']:
                logger.info(f"  ✓ {s}")
        
        if verdict.get('issues'):
            logger.info(f"\nIssues:")
            for i in verdict['issues']:
                logger.info(f"  ✗ {i}")
    
    def _save_reports(self, output_dir: Path):
        """Save reports"""
        
        def convert(obj):
            if isinstance(obj, (np.int64, np.int32, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'target_type': self.config.target_type,
                'is_regression': self.is_regression,
                'model_type': self.config.model_type,
                'risk_reward': self.config.risk_reward_ratio,
            },
            'tuned_params': convert(self.tuned_params) if self.tuned_params else None,
            'aggregated': convert(self.aggregated),
            'analysis': convert(self.analysis),
        }
        
        with open(output_dir / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\nReport saved to {output_dir / 'training_report.json'}")
    
    def _create_plots(self, output_dir: Path):
        """Create visualization plots"""
        
        try:
            if self.is_regression:
                self._create_regression_plots(output_dir)
            else:
                self._create_classification_plots(output_dir)
        except Exception as e:
            logger.warning(f"Could not create plots: {e}")
    
    def _create_regression_plots(self, output_dir: Path):
        """Create regression-specific plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Predicted vs Actual scatter
        ax = axes[0, 0]
        all_y_true = np.concatenate([r['predictions']['y_true'] for r in self.fold_results])
        all_y_pred = np.concatenate([r['predictions']['y_pred'] for r in self.fold_results])
        
        ax.scatter(all_y_true, all_y_pred, alpha=0.3, s=10)
        ax.plot([all_y_true.min(), all_y_true.max()], 
                [all_y_true.min(), all_y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual MFE')
        ax.set_ylabel('Predicted MFE')
        ax.set_title(f"Predicted vs Actual (R²={self.aggregated['overall_metrics']['r2']:.3f})")
        
        # 2. R² by fold
        ax = axes[0, 1]
        r2_by_fold = [r['metrics']['r2'] for r in self.fold_results]
        ax.bar(range(len(r2_by_fold)), r2_by_fold)
        ax.axhline(y=np.mean(r2_by_fold), color='r', linestyle='--', label=f'Mean: {np.mean(r2_by_fold):.3f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel('R²')
        ax.set_title('R² by Fold')
        ax.legend()
        
        # 3. Trading performance by threshold
        ax = axes[1, 0]
        thresholds = list(self.aggregated['trading_metrics'].keys())
        total_rs = [self.aggregated['trading_metrics'][t]['total_r'] for t in thresholds]
        n_trades = [self.aggregated['trading_metrics'][t]['n_trades'] for t in thresholds]
        
        ax.bar(range(len(thresholds)), total_rs)
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f'{t}R' for t in thresholds], rotation=45)
        ax.set_xlabel('MFE Threshold')
        ax.set_ylabel('Total R')
        ax.set_title('Total R by MFE Threshold')
        ax.axhline(y=0, color='r', linestyle='-')
        
        # 4. Residual distribution
        ax = axes[1, 1]
        residuals = all_y_pred - all_y_true
        ax.hist(residuals, bins=50, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel('Residual (Predicted - Actual)')
        ax.set_ylabel('Count')
        ax.set_title(f'Residual Distribution (MAE={np.mean(np.abs(residuals)):.3f})')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'regression_analysis.png', dpi=150)
        plt.close()
        
        logger.info(f"Plots saved to {output_dir}")
    
    def _create_classification_plots(self, output_dir: Path):
        """Create classification-specific plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. AUC by fold
        ax = axes[0, 0]
        aucs = [r['metrics']['roc_auc'] for r in self.fold_results]
        ax.bar(range(len(aucs)), aucs)
        ax.axhline(y=np.mean(aucs), color='r', linestyle='--', label=f'Mean: {np.mean(aucs):.3f}')
        ax.axhline(y=0.5, color='gray', linestyle=':', label='Random')
        ax.set_xlabel('Fold')
        ax.set_ylabel('ROC AUC')
        ax.set_title('AUC by Fold')
        ax.legend()
        
        # 2. Probability distributions
        ax = axes[0, 1]
        all_y_true = np.concatenate([r['predictions']['y_true'] for r in self.fold_results])
        all_y_proba = np.concatenate([r['predictions']['y_pred_proba'] for r in self.fold_results])
        
        ax.hist(all_y_proba[all_y_true == 1], bins=50, alpha=0.5, label='Wins', density=True)
        ax.hist(all_y_proba[all_y_true == 0], bins=50, alpha=0.5, label='Losses', density=True)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title('Probability Distribution by Outcome')
        ax.legend()
        
        # 3. Trading performance
        ax = axes[1, 0]
        thresholds = list(self.aggregated['trading_metrics'].keys())
        total_rs = [self.aggregated['trading_metrics'][t]['total_r'] for t in thresholds]
        
        ax.bar(range(len(thresholds)), total_rs)
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f'{t:.0%}' for t in thresholds], rotation=45)
        ax.set_xlabel('Probability Threshold')
        ax.set_ylabel('Total R')
        ax.set_title('Total R by Threshold')
        ax.axhline(y=0, color='r', linestyle='-')
        
        # 4. Calibration
        ax = axes[1, 1]
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(all_y_true, all_y_proba, n_bins=10)
        ax.plot(prob_pred, prob_true, 's-', label='Model')
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'classification_analysis.png', dpi=150)
        plt.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT Unified Training Pipeline V5')
    parser.add_argument('data_dir', help='Directory containing prepared data')
    parser.add_argument('--output-dir', default='unified_results_v5')
    parser.add_argument('--target-type', default='mfe_regression',
                       choices=CLASSIFICATION_TARGETS + REGRESSION_TARGETS,
                       help='Target type (determines if regression or classification)')
    parser.add_argument('--model', default='xgboost', choices=['xgboost', 'lightgbm'])
    parser.add_argument('--rr', type=float, default=2.0, help='Risk:Reward ratio')
    parser.add_argument('--no-tune', action='store_true')
    parser.add_argument('--no-calibrate', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    
    args = parser.parse_args()
    
    config = UnifiedConfigV5(
        target_type=args.target_type,
        model_type=args.model,
        risk_reward_ratio=args.rr,
        tune_hyperparams=not args.no_tune,
        calibrate=not args.no_calibrate,
        plot_results=not args.no_plots,
        output_dir=args.output_dir,
    )
    
    pipeline = UnifiedTrainingPipelineV5(config)
    pipeline.run(args.data_dir)


if __name__ == "__main__":
    main()