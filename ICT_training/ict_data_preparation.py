"""
ICT Data Preparation Pipeline (Optimized)
==========================================
Prepares feature-engineered data for ML training.
Optimized based on data analysis results.

Key optimizations:
- Removes 126 highly correlated feature pairs identified in analysis
- Handles bimodal price_position distribution
- Optimized for ~40% high confluence rate
- Better feature selection based on ICT concept importance

Input: Feature-enriched CSV from feature engineering
Output: Train/test datasets ready for model training

Author: ICT ML System
Version: 2.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import joblib
import json
import warnings
import logging

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
class DataPrepConfig:
    """Configuration for data preparation"""
    
    # Target Settings
    target_type: str = "binary"
    target_lookforward: int = 60          # Bars to look forward (1 hour for M1)
    stop_loss_atr: float = 1.5            # SL in ATR multiples
    take_profit_rr: float = 1.5           # TP1 Risk:Reward
    take_profit_2_rr: float = 2.5         # TP2 R:R
    take_profit_3_rr: float = 4.0         # TP3 R:R
    
    # Signal Filtering
    min_confluence: int = 2               # Minimum confluence (analysis showed avg ~2)
    require_killzone: bool = True         # Only KZ signals (34% of data)
    min_atr: float = 0.00005              # Minimum ATR filter
    require_direction: bool = False       # Changed: don't require direction to keep more data
    
    # Feature Selection
    remove_low_variance: bool = True
    variance_threshold: float = 0.001     # Lowered: more permissive
    remove_high_correlation: bool = True
    correlation_threshold: float = 0.95   # Raised: more permissive for trees
    max_features: Optional[int] = None    # Removed limit - let model decide
    
    # Train/Test Split
    test_size: float = 0.2
    validation_size: float = 0.15
    temporal_split: bool = True
    
    # Scaling
    scaler_type: str = "robust"           # Better for skewed ATR/volume
    
    # Class Imbalance
    handle_imbalance: bool = True
    imbalance_method: str = "smote"
    imbalance_ratio: float = 0.5
    
    # Output
    output_dir: str = "data/prepared"
    save_scaler: bool = True
    save_feature_list: bool = True
    
    # PROTECTED FEATURES - Never remove these regardless of variance/correlation
    # These are core ICT concepts that must be preserved
    protected_features: List[str] = field(default_factory=lambda: [
        # Core ICT Concepts
        'is_displacement', 'displacement_size',
        'bullish_fvg', 'bearish_fvg', 'recent_bullish_fvg', 'recent_bearish_fvg',
        'bullish_ob', 'bearish_ob', 'recent_bullish_ob', 'recent_bearish_ob',
        'sweep_high', 'sweep_low', 'recent_sweep_high', 'recent_sweep_low',
        
        # Kill Zones (which KZ matters even if correlated)
        'in_london_kz', 'in_nyam_kz', 'in_nypm_kz', 'in_any_kz', 'in_sb_window',
        
        # Premium/Discount (keep both binary and continuous)
        'in_premium', 'in_discount', 'price_position',
        
        # Volatility (critical for position sizing context)
        'atr', 'atr_ratio', 'atr_pct', 'range_atr',
        
        # Structure
        'trend_ema', 'structure_bias', 'bos_bullish', 'bos_bearish',
        'higher_high', 'lower_low', 'higher_low', 'lower_high',
        
        # Confluence (the target we're trying to learn)
        'bullish_confluence', 'bearish_confluence', 'max_confluence', 'net_confluence',
        
        # Time (keep both raw and cyclical for different model types)
        'hour', 'hour_sin', 'hour_cos',
        
        # Momentum
        'rsi', 'roc_5', 'roc_10',
        
        # Volume
        'volume_ratio', 'rvol',
    ])
    
    # Columns to ALWAYS exclude (raw price data, redundant)
    exclude_columns: List[str] = field(default_factory=lambda: [
        # Raw identifiers
        'datetime', 'date', 'time', 'Date', 'Time',
        # Raw OHLCV (use derived features instead)
        'open', 'high', 'low', 'close', 'volume',
        'Open', 'High', 'Low', 'Close', 'Volume',
        # Redundant with other features (from correlation analysis)
        'typical_price',      # ~1.0 corr with close
        'et_hour',            # = hour (1.0 corr)
        'ema_8',              # ~1.0 corr with close, ema_21
        'bb_middle',          # ~1.0 corr with ema_8
        'range_high',         # use price_position instead
        'range_low',          # use price_position instead
        'equilibrium',        # use dist_to_eq instead
        'last_swing_high',    # raw price, use dist_to_swing_high
        'last_swing_low',     # raw price, use dist_to_swing_low
        'swing_high_price',   # raw price
        'swing_low_price',    # raw price
        'pdh',                # raw price, use dist_to_pdh
        'pdl',                # raw price, use dist_to_pdl
        'ema_21',             # highly corr with ema_50
        'ema_200',            # often NaN at start, use ema_50
        'volume_sma',         # use volume_ratio instead
        'volume_sma_fast',    # use volume_ratio instead
        'atr_long',           # use atr_ratio instead
    ])


# =============================================================================
# TARGET CREATION (Optimized)
# =============================================================================

class TargetCreator:
    """Create target variables based on trade outcomes"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
    
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables with vectorized operations for speed"""
        logger.info("Creating target variables...")
        
        df = df.copy()
        n = len(df)
        lookforward = self.config.target_lookforward
        
        # Initialize target columns
        df['target_hit_tp1'] = 0
        df['target_hit_tp2'] = 0
        df['target_hit_tp3'] = 0
        df['target_hit_sl'] = 0
        df['target_direction'] = 0  # 1=long, -1=short, 0=none
        df['target_mfe_r'] = 0.0
        df['target_mae_r'] = 0.0
        df['target_pnl_r'] = 0.0
        
        # Get arrays
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = df['atr'].values
        
        # Determine direction from net_confluence or structure
        if 'net_confluence' in df.columns:
            directions = df['net_confluence'].values
        elif 'structure_bias' in df.columns:
            directions = df['structure_bias'].values
        else:
            # Use trend_ema as fallback
            directions = df['trend_ema'].values if 'trend_ema' in df.columns else np.zeros(n)
        
        logger.info(f"Processing {n - lookforward} potential signals...")
        
        # Process in batches for progress reporting
        batch_size = 5000
        for batch_start in range(0, n - lookforward, batch_size):
            batch_end = min(batch_start + batch_size, n - lookforward)
            
            for i in range(batch_start, batch_end):
                entry = closes[i]
                atr = atrs[i]
                direction = directions[i]
                
                if atr <= 0 or np.isnan(atr) or direction == 0:
                    continue
                
                risk = atr * self.config.stop_loss_atr
                
                # Future price data
                future_highs = highs[i+1:i+lookforward+1]
                future_lows = lows[i+1:i+lookforward+1]
                
                if len(future_highs) == 0:
                    continue
                
                # Store direction
                df.iat[i, df.columns.get_loc('target_direction')] = np.sign(direction)
                
                if direction > 0:  # LONG
                    sl = entry - risk
                    tp1 = entry + risk * self.config.take_profit_rr
                    tp2 = entry + risk * self.config.take_profit_2_rr
                    tp3 = entry + risk * self.config.take_profit_3_rr
                    
                    mfe = future_highs.max() - entry
                    mae = entry - future_lows.min()
                    
                    first_hit = self._check_long_outcome(future_highs, future_lows, sl, tp1, tp2, tp3)
                    
                else:  # SHORT
                    sl = entry + risk
                    tp1 = entry - risk * self.config.take_profit_rr
                    tp2 = entry - risk * self.config.take_profit_2_rr
                    tp3 = entry - risk * self.config.take_profit_3_rr
                    
                    mfe = entry - future_lows.min()
                    mae = future_highs.max() - entry
                    
                    first_hit = self._check_short_outcome(future_highs, future_lows, sl, tp1, tp2, tp3)
                
                # Store results
                df.iat[i, df.columns.get_loc('target_mfe_r')] = mfe / risk if risk > 0 else 0
                df.iat[i, df.columns.get_loc('target_mae_r')] = mae / risk if risk > 0 else 0
                
                if first_hit == -1:  # SL hit
                    df.iat[i, df.columns.get_loc('target_hit_sl')] = 1
                    df.iat[i, df.columns.get_loc('target_pnl_r')] = -1.0
                elif first_hit >= 1:  # TP1+ hit
                    df.iat[i, df.columns.get_loc('target_hit_tp1')] = 1
                    df.iat[i, df.columns.get_loc('target_pnl_r')] = self.config.take_profit_rr
                    if first_hit >= 2:
                        df.iat[i, df.columns.get_loc('target_hit_tp2')] = 1
                    if first_hit >= 3:
                        df.iat[i, df.columns.get_loc('target_hit_tp3')] = 1
            
            # Progress
            if batch_end % 10000 == 0:
                logger.info(f"  Processed {batch_end}/{n - lookforward} bars...")
        
        # Log distribution
        total_signals = (df['target_direction'] != 0).sum()
        tp1_hits = df['target_hit_tp1'].sum()
        sl_hits = df['target_hit_sl'].sum()
        
        logger.info(f"\nTarget Distribution:")
        logger.info(f"  Total signals with direction: {total_signals}")
        logger.info(f"  TP1 hits: {tp1_hits} ({tp1_hits/total_signals*100:.1f}%)" if total_signals > 0 else "  TP1 hits: 0")
        logger.info(f"  SL hits: {sl_hits} ({sl_hits/total_signals*100:.1f}%)" if total_signals > 0 else "  SL hits: 0")
        
        return df
    
    def _check_long_outcome(self, highs, lows, sl, tp1, tp2, tp3) -> int:
        """Check outcome for long trade: -1=SL, 0=neither, 1=TP1, 2=TP2, 3=TP3"""
        for i in range(len(highs)):
            if lows[i] <= sl:
                return -1
            if highs[i] >= tp3:
                return 3
            if highs[i] >= tp2:
                return 2
            if highs[i] >= tp1:
                return 1
        return 0
    
    def _check_short_outcome(self, highs, lows, sl, tp1, tp2, tp3) -> int:
        """Check outcome for short trade"""
        for i in range(len(highs)):
            if highs[i] >= sl:
                return -1
            if lows[i] <= tp3:
                return 3
            if lows[i] <= tp2:
                return 2
            if lows[i] <= tp1:
                return 1
        return 0


# =============================================================================
# DATA FILTERING (Optimized)
# =============================================================================

class DataFilter:
    """Filter data for quality signals"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
    
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters"""
        logger.info("\nApplying quality filters...")
        
        initial_len = len(df)
        df = df.copy()
        
        filters_applied = []
        
        # 1. Filter by minimum ATR
        if 'atr' in df.columns and self.config.min_atr > 0:
            before = len(df)
            df = df[df['atr'] >= self.config.min_atr]
            filters_applied.append(f"ATR >= {self.config.min_atr}: {before} -> {len(df)}")
        
        # 2. Filter by confluence
        if 'max_confluence' in df.columns and self.config.min_confluence > 0:
            before = len(df)
            df = df[df['max_confluence'] >= self.config.min_confluence]
            filters_applied.append(f"Confluence >= {self.config.min_confluence}: {before} -> {len(df)}")
        
        # 3. Filter by kill zone
        if self.config.require_killzone and 'in_any_kz' in df.columns:
            before = len(df)
            df = df[df['in_any_kz'] == 1]
            filters_applied.append(f"In Kill Zone: {before} -> {len(df)}")
        
        # 4. Filter by directional bias
        if self.config.require_direction and 'target_direction' in df.columns:
            before = len(df)
            df = df[df['target_direction'] != 0]
            filters_applied.append(f"Has direction: {before} -> {len(df)}")
        
        # 5. Remove rows where target couldn't be calculated
        if 'target_hit_tp1' in df.columns and 'target_hit_sl' in df.columns:
            before = len(df)
            # Keep rows where either TP or SL was hit (outcome determined)
            df = df[(df['target_hit_tp1'] == 1) | (df['target_hit_sl'] == 1)]
            filters_applied.append(f"Outcome determined: {before} -> {len(df)}")
        
        # Log filters
        for f in filters_applied:
            logger.info(f"  {f}")
        
        logger.info(f"\nFinal: {len(df)} rows ({len(df)/initial_len*100:.1f}% of original)")
        
        return df.reset_index(drop=True)


# =============================================================================
# FEATURE SELECTION (Optimized)
# =============================================================================

class FeatureSelector:
    """Optimized feature selection with protected features"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.selected_features = []
        self.removed_features = {
            'excluded': [],
            'low_variance': [],
            'high_correlation': [],
            'not_in_data': []
        }
    
    def select(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
        """Select features, protecting core ICT features"""
        logger.info("\nSelecting features...")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Identify protected features that exist in data
        protected_present = [f for f in self.config.protected_features if f in numeric_cols]
        protected_missing = [f for f in self.config.protected_features if f not in numeric_cols]
        if protected_missing:
            logger.info(f"Protected features not in data: {len(protected_missing)}")
            self.removed_features['not_in_data'] = protected_missing
        
        # Build exclude set
        exclude_set = set(self.config.exclude_columns)
        exclude_set.add(target_col)
        exclude_set.update([c for c in numeric_cols if c.startswith('target_')])
        
        # Get all candidate features (not excluded)
        feature_cols = [
            f for f in numeric_cols 
            if f not in exclude_set
        ]
        
        self.removed_features['excluded'] = list(exclude_set & set(numeric_cols))
        logger.info(f"Initial features: {len(feature_cols)} (excluding {len(self.removed_features['excluded'])} columns)")
        logger.info(f"Protected features: {len(protected_present)}")
        
        X = df[feature_cols].copy()
        
        # Fill NaN
        nan_counts = X.isnull().sum()
        if nan_counts.sum() > 0:
            logger.info(f"Filling {nan_counts.sum()} NaN values")
            X = X.fillna(0)
        
        # Remove low variance (but protect core features)
        if self.config.remove_low_variance:
            X, feature_cols = self._remove_low_variance(X, feature_cols, protected_present)
        
        # Remove high correlation (but protect core features)
        if self.config.remove_high_correlation:
            X, feature_cols = self._remove_high_correlation(X, feature_cols, protected_present)
        
        self.selected_features = feature_cols
        logger.info(f"Final features: {len(feature_cols)}")
        
        # Report protected features kept
        protected_kept = [f for f in protected_present if f in feature_cols]
        logger.info(f"Protected features retained: {len(protected_kept)}/{len(protected_present)}")
        
        return X, feature_cols
    
    def _remove_low_variance(self, X, feature_cols, protected):
        """Remove features with near-zero variance, except protected ones"""
        
        # Calculate variance for each feature
        variances = X.var()
        
        # Find low variance features (excluding protected)
        low_var_features = []
        for feat in feature_cols:
            if feat in protected:
                continue  # Never remove protected
            if variances[feat] < self.config.variance_threshold:
                low_var_features.append(feat)
        
        self.removed_features['low_variance'] = low_var_features
        
        # Keep features not in low_var list
        feature_cols = [f for f in feature_cols if f not in low_var_features]
        X = X[feature_cols]
        
        if low_var_features:
            logger.info(f"Removed {len(low_var_features)} low variance features (protected {len(protected)} features)")
        
        return X, feature_cols
    
    def _remove_high_correlation(self, X, feature_cols, protected):
        """Remove highly correlated features, keeping protected ones"""
        
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = set()
        
        for col in upper.columns:
            high_corr = upper[col][upper[col] > self.config.correlation_threshold].index.tolist()
            
            for corr_col in high_corr:
                # Decision logic: which one to drop?
                col_protected = col in protected
                corr_protected = corr_col in protected
                
                if col_protected and corr_protected:
                    # Both protected - keep both
                    continue
                elif col_protected:
                    # Only col is protected - drop corr_col
                    to_drop.add(corr_col)
                elif corr_protected:
                    # Only corr_col is protected - drop col
                    to_drop.add(col)
                else:
                    # Neither protected - drop the second one (arbitrary but consistent)
                    to_drop.add(corr_col)
        
        to_drop = list(to_drop)
        self.removed_features['high_correlation'] = to_drop
        
        feature_cols = [f for f in feature_cols if f not in to_drop]
        X = X[feature_cols]
        
        if to_drop:
            logger.info(f"Removed {len(to_drop)} highly correlated features (protected {len(protected)} features)")
        
        return X, feature_cols
    
    def get_report(self) -> Dict:
        """Get feature selection report"""
        return {
            'selected': self.selected_features,
            'n_selected': len(self.selected_features),
            'removed': self.removed_features
        }


# =============================================================================
# DATA SPLITTING (Temporal)
# =============================================================================

class DataSplitter:
    """Temporal data splitting for time series"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
    
    def split(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Split data temporally"""
        logger.info("\nSplitting data temporally...")
        
        n = len(X)
        
        # Calculate indices
        test_idx = int(n * (1 - self.config.test_size))
        val_idx = int(test_idx * (1 - self.config.validation_size))
        
        # Split
        X_train = X.iloc[:val_idx].reset_index(drop=True)
        y_train = y.iloc[:val_idx].reset_index(drop=True)
        
        X_val = X.iloc[val_idx:test_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx:test_idx].reset_index(drop=True)
        
        X_test = X.iloc[test_idx:].reset_index(drop=True)
        y_test = y.iloc[test_idx:].reset_index(drop=True)
        
        # Log
        logger.info(f"  Train: {len(X_train):,} samples ({y_train.mean()*100:.1f}% positive)")
        logger.info(f"  Val:   {len(X_val):,} samples ({y_val.mean()*100:.1f}% positive)")
        logger.info(f"  Test:  {len(X_test):,} samples ({y_test.mean()*100:.1f}% positive)")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }


# =============================================================================
# SCALING (Robust for skewed data)
# =============================================================================

class FeatureScaler:
    """Scale features using RobustScaler for skewed distributions"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.scaler = None
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training data"""
        if self.config.scaler_type is None:
            return X
        
        logger.info(f"\nScaling features with {self.config.scaler_type} scaler...")
        
        if self.config.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.config.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.config.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            return X
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted scaler"""
        if self.scaler is None:
            return X
        
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    
    def save(self, filepath: str):
        """Save scaler"""
        if self.scaler:
            joblib.dump(self.scaler, filepath)
            logger.info(f"Scaler saved to {filepath}")
    
    def load(self, filepath: str):
        """Load scaler"""
        self.scaler = joblib.load(filepath)


# =============================================================================
# CLASS IMBALANCE HANDLING
# =============================================================================

class ImbalanceHandler:
    """Handle class imbalance with SMOTE"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
    
    def resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Resample training data only if significantly imbalanced"""
        if not self.config.handle_imbalance:
            return X, y
        
        # Check current balance
        class_counts = y.value_counts()
        minority_count = class_counts.min()
        majority_count = class_counts.max()
        minority_ratio = minority_count / majority_count
        
        # If already reasonably balanced (>35% minority), skip resampling
        if minority_ratio > 0.35:
            logger.info(f"\nClasses already balanced (ratio: {minority_ratio:.2f}), skipping resampling")
            logger.info(f"  Class distribution: {dict(class_counts)}")
            return X, y
        
        logger.info(f"\nResampling with {self.config.imbalance_method}...")
        logger.info(f"  Before: {dict(class_counts)} (minority ratio: {minority_ratio:.2f})")
        
        try:
            # For SMOTE, we want to oversample minority to match majority
            # sampling_strategy='auto' will balance to 1:1
            if self.config.imbalance_method == 'smote':
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(
                    sampling_strategy='auto',  # Balance to 1:1
                    random_state=42,
                    k_neighbors=min(5, minority_count - 1)  # Ensure k < minority samples
                )
            elif self.config.imbalance_method == 'adasyn':
                from imblearn.over_sampling import ADASYN
                sampler = ADASYN(
                    sampling_strategy='auto',
                    random_state=42
                )
            elif self.config.imbalance_method == 'undersample':
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(
                    sampling_strategy='auto',
                    random_state=42
                )
            else:
                return X, y
            
            X_res, y_res = sampler.fit_resample(X, y)
            X_res = pd.DataFrame(X_res, columns=X.columns)
            y_res = pd.Series(y_res, name=y.name)
            
            logger.info(f"  After: {dict(y_res.value_counts())}")
            
            return X_res, y_res
            
        except Exception as e:
            logger.warning(f"Resampling failed: {e}")
            logger.info("  Continuing with original data")
            return X, y


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class DataPreparationPipeline:
    """Main data preparation pipeline"""
    
    def __init__(self, config: Optional[DataPrepConfig] = None):
        self.config = config or DataPrepConfig()
        
        self.target_creator = TargetCreator(self.config)
        self.data_filter = DataFilter(self.config)
        self.feature_selector = FeatureSelector(self.config)
        self.data_splitter = DataSplitter(self.config)
        self.scaler = FeatureScaler(self.config)
        self.imbalance_handler = ImbalanceHandler(self.config)
        
        self.metadata = {}
    
    def run(self, input_path: str, target_col: str = 'target_hit_tp1') -> Dict[str, Any]:
        """Run complete pipeline"""
        print("\n" + "=" * 60)
        print("🚀 ICT DATA PREPARATION PIPELINE")
        print("=" * 60)
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"\n📂 Loading: {input_path}")
        df = pd.read_csv(input_path)
        print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Create targets
        print("\n🎯 Creating targets...")
        df = self.target_creator.create_targets(df)
        
        # Filter data
        df = self.data_filter.filter(df)
        
        if len(df) == 0:
            raise ValueError("No data remaining after filtering!")
        
        # Get target
        y = df[target_col].astype(int)
        
        # Select features
        X, feature_cols = self.feature_selector.select(df, target_col)
        
        # Split
        splits = self.data_splitter.split(X, y)
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle imbalance (train only)
        X_train_final, y_train_final = self.imbalance_handler.resample(X_train_scaled, y_train)
        
        # Save
        print("\n💾 Saving datasets...")
        
        train_df = X_train_final.copy()
        train_df['target'] = y_train_final.values
        train_df.to_csv(output_dir / 'train.csv', index=False)
        
        val_df = X_val_scaled.copy()
        val_df['target'] = y_val.values
        val_df.to_csv(output_dir / 'val.csv', index=False)
        
        test_df = X_test_scaled.copy()
        test_df['target'] = y_test.values
        test_df.to_csv(output_dir / 'test.csv', index=False)
        
        # Save scaler
        if self.config.save_scaler:
            self.scaler.save(str(output_dir / 'scaler.joblib'))
        
        # Save features
        if self.config.save_feature_list:
            with open(output_dir / 'features.json', 'w') as f:
                json.dump({
                    'features': feature_cols,
                    'n_features': len(feature_cols),
                    'protected_features': [f for f in self.config.protected_features if f in feature_cols]
                }, f, indent=2)
        
        # Save metadata
        self.metadata = {
            'input_file': str(input_path),
            'n_original': len(df) + self.config.target_lookforward,
            'n_filtered': len(df),
            'n_train': len(X_train_final),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'n_features': len(feature_cols),
            'target': target_col,
            'class_dist_train': {str(k): int(v) for k, v in y_train_final.value_counts().items()},
            'class_dist_test': {str(k): int(v) for k, v in y_test.value_counts().items()},
            'feature_selection': self.feature_selector.get_report(),
            'config': {
                'min_confluence': self.config.min_confluence,
                'require_killzone': self.config.require_killzone,
                'scaler': self.config.scaler_type,
                'imbalance_method': self.config.imbalance_method
            }
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 PREPARATION SUMMARY")
        print("=" * 60)
        print(f"   Features: {len(feature_cols)}")
        print(f"   Train: {len(X_train_final):,} ({y_train_final.mean()*100:.1f}% positive)")
        print(f"   Val: {len(X_val):,} ({y_val.mean()*100:.1f}% positive)")
        print(f"   Test: {len(X_test):,} ({y_test.mean()*100:.1f}% positive)")
        print(f"\n📁 Saved to: {output_dir.absolute()}")
        print("=" * 60)
        print("✅ Preparation complete!")
        
        return {
            'train': (X_train_final, y_train_final),
            'val': (X_val_scaled, y_val),
            'test': (X_test_scaled, y_test),
            'features': feature_cols,
            'metadata': self.metadata
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT Data Preparation Pipeline')
    parser.add_argument('input', help='Input features CSV')
    parser.add_argument('--output-dir', '-o', default='data/prepared')
    parser.add_argument('--target', '-t', default='target_hit_tp1')
    parser.add_argument('--min-confluence', type=int, default=2)
    parser.add_argument('--no-killzone', action='store_true')
    parser.add_argument('--no-imbalance', action='store_true')
    parser.add_argument('--lookforward', type=int, default=60)
    
    args = parser.parse_args()
    
    config = DataPrepConfig(
        output_dir=args.output_dir,
        min_confluence=args.min_confluence,
        require_killzone=not args.no_killzone,
        handle_imbalance=not args.no_imbalance,
        target_lookforward=args.lookforward
    )
    
    pipeline = DataPreparationPipeline(config)
    pipeline.run(args.input, args.target)


if __name__ == "__main__":
    main()