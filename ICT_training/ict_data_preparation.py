"""
ICT Data Preparation Pipeline
==============================
Prepares feature-engineered data for ML training.

Responsibilities:
- Create target variables (trade outcomes)
- Feature selection and validation
- Train/validation/test split (temporal)
- Handle missing values
- Feature scaling
- Class imbalance handling
- Save train-ready datasets

Input: Feature-enriched CSV from feature engineering
Output: Train/test datasets ready for model training

Author: ICT ML System
Version: 1.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
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
    target_type: str = "binary"          # 'binary', 'multiclass', 'regression'
    target_lookforward: int = 50         # Bars to look forward for outcome
    stop_loss_atr: float = 1.5           # SL in ATR multiples
    take_profit_rr: float = 1.5          # TP in Risk:Reward ratio
    take_profit_2_rr: float = 2.5        # TP2 R:R
    take_profit_3_rr: float = 4.0        # TP3 R:R
    
    # Signal Filtering (quality thresholds)
    min_confluence: int = 1              # Minimum confluence score
    require_killzone: bool = True        # Only include killzone signals
    min_atr: float = 0.0001              # Minimum ATR (filter low volatility)
    
    # Feature Selection
    remove_low_variance: bool = True
    variance_threshold: float = 0.01
    remove_high_correlation: bool = True
    correlation_threshold: float = 0.95
    max_features: Optional[int] = None   # Limit features (None = no limit)
    
    # Train/Test Split
    test_size: float = 0.2               # 20% test
    validation_size: float = 0.1         # 10% validation (from train)
    temporal_split: bool = True          # Use temporal split (recommended)
    
    # Scaling
    scaler_type: str = "robust"          # 'standard', 'robust', 'minmax', None
    
    # Class Imbalance
    handle_imbalance: bool = True
    imbalance_method: str = "smote"      # 'smote', 'adasyn', 'undersample', 'smote_tomek'
    imbalance_ratio: float = 0.4         # Target ratio for minority class
    
    # Output
    output_dir: str = "data/prepared"
    save_scaler: bool = True
    save_feature_list: bool = True
    
    # Columns to exclude from features
    exclude_columns: List[str] = field(default_factory=lambda: [
        'datetime', 'date', 'time', 'open', 'high', 'low', 'close', 'volume',
        'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'
    ])


# =============================================================================
# TARGET CREATION
# =============================================================================

class TargetCreator:
    """Create target variables based on trade outcomes"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
    
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables by looking forward from each bar.
        
        For each potential entry:
        - Calculate SL and TP levels
        - Track forward to see which is hit first
        - Create binary and continuous targets
        """
        logger.info("Creating target variables...")
        
        df = df.copy()
        
        # Ensure required columns exist
        required = ['close', 'high', 'low', 'atr']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Initialize target columns
        df['target_hit_tp1'] = 0
        df['target_hit_tp2'] = 0
        df['target_hit_tp3'] = 0
        df['target_hit_sl'] = 0
        df['target_first_hit'] = 0  # -1=SL, 0=neither, 1=TP1, 2=TP2, 3=TP3
        df['target_mfe'] = 0.0      # Max Favorable Excursion (pips)
        df['target_mae'] = 0.0      # Max Adverse Excursion (pips)
        df['target_mfe_r'] = 0.0    # MFE in R-multiples
        df['target_mae_r'] = 0.0    # MAE in R-multiples
        df['target_pnl_r'] = 0.0    # Final P&L in R-multiples
        
        # Get numpy arrays for faster computation
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = df['atr'].values
        
        n = len(df)
        lookforward = self.config.target_lookforward
        
        logger.info(f"Processing {n} bars with {lookforward} bar lookforward...")
        
        # Process each bar as potential entry
        for i in range(n - lookforward):
            entry = closes[i]
            atr = atrs[i]
            
            if atr <= 0 or np.isnan(atr):
                continue
            
            # Calculate risk (SL distance)
            risk = atr * self.config.stop_loss_atr
            
            # For simplicity, we'll compute for LONG trades
            # (In production, you'd determine direction from confluence)
            # Here we compute both and use the one that matches the signal
            
            # Long trade levels
            long_sl = entry - risk
            long_tp1 = entry + risk * self.config.take_profit_rr
            long_tp2 = entry + risk * self.config.take_profit_2_rr
            long_tp3 = entry + risk * self.config.take_profit_3_rr
            
            # Short trade levels
            short_sl = entry + risk
            short_tp1 = entry - risk * self.config.take_profit_rr
            short_tp2 = entry - risk * self.config.take_profit_2_rr
            short_tp3 = entry - risk * self.config.take_profit_3_rr
            
            # Look forward
            future_highs = highs[i+1:i+lookforward+1]
            future_lows = lows[i+1:i+lookforward+1]
            
            if len(future_highs) == 0:
                continue
            
            # Calculate MFE/MAE for both directions
            long_mfe = (future_highs.max() - entry)
            long_mae = (entry - future_lows.min())
            short_mfe = (entry - future_lows.min())
            short_mae = (future_highs.max() - entry)
            
            # Determine which direction based on confluence
            # (Use net_confluence if available, otherwise use structure_bias)
            if 'net_confluence' in df.columns:
                direction = df.iloc[i]['net_confluence']
            elif 'structure_bias' in df.columns:
                direction = df.iloc[i]['structure_bias']
            else:
                direction = 0
            
            # Determine trade outcome based on direction
            if direction > 0:  # Bullish bias - evaluate long
                mfe = long_mfe
                mae = long_mae
                sl = long_sl
                tp1 = long_tp1
                tp2 = long_tp2
                tp3 = long_tp3
                
                hit_sl = (future_lows <= sl).any()
                hit_tp1 = (future_highs >= tp1).any()
                hit_tp2 = (future_highs >= tp2).any()
                hit_tp3 = (future_highs >= tp3).any()
                
                # Determine which hit first
                first_hit = self._determine_first_hit_long(
                    future_highs, future_lows, sl, tp1, tp2, tp3
                )
                
            elif direction < 0:  # Bearish bias - evaluate short
                mfe = short_mfe
                mae = short_mae
                sl = short_sl
                tp1 = short_tp1
                tp2 = short_tp2
                tp3 = short_tp3
                
                hit_sl = (future_highs >= sl).any()
                hit_tp1 = (future_lows <= tp1).any()
                hit_tp2 = (future_lows <= tp2).any()
                hit_tp3 = (future_lows <= tp3).any()
                
                first_hit = self._determine_first_hit_short(
                    future_highs, future_lows, sl, tp1, tp2, tp3
                )
            else:
                # No clear direction - evaluate as long by default
                mfe = long_mfe
                mae = long_mae
                hit_sl = (future_lows <= long_sl).any()
                hit_tp1 = (future_highs >= long_tp1).any()
                hit_tp2 = (future_highs >= long_tp2).any()
                hit_tp3 = (future_highs >= long_tp3).any()
                first_hit = self._determine_first_hit_long(
                    future_highs, future_lows, long_sl, long_tp1, long_tp2, long_tp3
                )
            
            # Store results
            df.iloc[i, df.columns.get_loc('target_hit_tp1')] = int(hit_tp1 and first_hit >= 1)
            df.iloc[i, df.columns.get_loc('target_hit_tp2')] = int(hit_tp2 and first_hit >= 2)
            df.iloc[i, df.columns.get_loc('target_hit_tp3')] = int(hit_tp3 and first_hit >= 3)
            df.iloc[i, df.columns.get_loc('target_hit_sl')] = int(hit_sl and first_hit == -1)
            df.iloc[i, df.columns.get_loc('target_first_hit')] = first_hit
            df.iloc[i, df.columns.get_loc('target_mfe')] = mfe
            df.iloc[i, df.columns.get_loc('target_mae')] = mae
            df.iloc[i, df.columns.get_loc('target_mfe_r')] = mfe / risk if risk > 0 else 0
            df.iloc[i, df.columns.get_loc('target_mae_r')] = mae / risk if risk > 0 else 0
            
            # P&L in R
            if first_hit == -1:
                df.iloc[i, df.columns.get_loc('target_pnl_r')] = -1.0
            elif first_hit == 1:
                df.iloc[i, df.columns.get_loc('target_pnl_r')] = self.config.take_profit_rr
            elif first_hit == 2:
                df.iloc[i, df.columns.get_loc('target_pnl_r')] = self.config.take_profit_2_rr
            elif first_hit == 3:
                df.iloc[i, df.columns.get_loc('target_pnl_r')] = self.config.take_profit_3_rr
        
        # Log target distribution
        logger.info(f"Target distribution:")
        logger.info(f"  hit_tp1: {df['target_hit_tp1'].sum()} ({df['target_hit_tp1'].mean()*100:.1f}%)")
        logger.info(f"  hit_sl:  {df['target_hit_sl'].sum()} ({df['target_hit_sl'].mean()*100:.1f}%)")
        
        return df
    
    def _determine_first_hit_long(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray,
        sl: float, 
        tp1: float, 
        tp2: float, 
        tp3: float
    ) -> int:
        """Determine which level was hit first for a long trade"""
        
        for i in range(len(highs)):
            # Check SL first (worst case)
            if lows[i] <= sl:
                return -1
            # Check TPs
            if highs[i] >= tp3:
                return 3
            if highs[i] >= tp2:
                return 2
            if highs[i] >= tp1:
                return 1
        
        return 0  # Neither hit
    
    def _determine_first_hit_short(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray,
        sl: float, 
        tp1: float, 
        tp2: float, 
        tp3: float
    ) -> int:
        """Determine which level was hit first for a short trade"""
        
        for i in range(len(highs)):
            # Check SL first
            if highs[i] >= sl:
                return -1
            # Check TPs
            if lows[i] <= tp3:
                return 3
            if lows[i] <= tp2:
                return 2
            if lows[i] <= tp1:
                return 1
        
        return 0


# =============================================================================
# DATA FILTERING
# =============================================================================

class DataFilter:
    """Filter data for quality signals only"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
    
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to data"""
        logger.info("Applying quality filters...")
        
        initial_len = len(df)
        df = df.copy()
        
        # Filter by minimum ATR (avoid low volatility)
        if 'atr' in df.columns and self.config.min_atr > 0:
            df = df[df['atr'] >= self.config.min_atr]
            logger.info(f"After ATR filter: {len(df)} ({len(df)/initial_len*100:.1f}%)")
        
        # Filter by confluence
        if 'max_confluence' in df.columns and self.config.min_confluence > 0:
            df = df[df['max_confluence'] >= self.config.min_confluence]
            logger.info(f"After confluence filter: {len(df)} ({len(df)/initial_len*100:.1f}%)")
        
        # Filter by kill zone
        if self.config.require_killzone and 'in_any_kz' in df.columns:
            df = df[df['in_any_kz'] == 1]
            logger.info(f"After killzone filter: {len(df)} ({len(df)/initial_len*100:.1f}%)")
        
        # Filter out rows where we couldn't calculate targets
        if 'target_first_hit' in df.columns:
            # Keep only rows where outcome is determined (not 0 unless no trade)
            # Actually, keep all - the target_first_hit=0 means outcome not reached
            pass
        
        # Remove last N rows (can't calculate targets for these)
        df = df.iloc[:-self.config.target_lookforward]
        
        logger.info(f"Final filtered dataset: {len(df)} rows ({len(df)/initial_len*100:.1f}%)")
        
        return df.reset_index(drop=True)


# =============================================================================
# FEATURE SELECTION
# =============================================================================

class FeatureSelector:
    """Select and validate features for training"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.selected_features = []
        self.removed_features = {
            'low_variance': [],
            'high_correlation': [],
            'excluded': []
        }
    
    def select(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features for training.
        
        Args:
            df: Input dataframe
            target_col: Name of target column
        
        Returns:
            Tuple of (feature dataframe, list of feature names)
        """
        logger.info("Selecting features...")
        
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns
        exclude_patterns = self.config.exclude_columns + [
            'target_', 'Target_', 'label', 'Label'
        ]
        
        feature_cols = []
        for col in numeric_cols:
            exclude = False
            for pattern in exclude_patterns:
                if pattern in col or col == target_col:
                    exclude = True
                    self.removed_features['excluded'].append(col)
                    break
            if not exclude:
                feature_cols.append(col)
        
        logger.info(f"Initial features: {len(feature_cols)}")
        
        X = df[feature_cols].copy()
        
        # Remove low variance features
        if self.config.remove_low_variance:
            X, feature_cols = self._remove_low_variance(X, feature_cols)
        
        # Remove highly correlated features
        if self.config.remove_high_correlation:
            X, feature_cols = self._remove_high_correlation(X, feature_cols)
        
        # Limit number of features if specified
        if self.config.max_features and len(feature_cols) > self.config.max_features:
            X, feature_cols = self._select_top_features(
                X, df[target_col], feature_cols
            )
        
        self.selected_features = feature_cols
        logger.info(f"Final features: {len(feature_cols)}")
        
        return X, feature_cols
    
    def _remove_low_variance(
        self, 
        X: pd.DataFrame, 
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with low variance"""
        
        selector = VarianceThreshold(threshold=self.config.variance_threshold)
        
        try:
            selector.fit(X)
            mask = selector.get_support()
            
            removed = [f for f, keep in zip(feature_cols, mask) if not keep]
            self.removed_features['low_variance'] = removed
            
            feature_cols = [f for f, keep in zip(feature_cols, mask) if keep]
            X = X[feature_cols]
            
            logger.info(f"Removed {len(removed)} low variance features")
        except Exception as e:
            logger.warning(f"Low variance removal failed: {e}")
        
        return X, feature_cols
    
    def _remove_high_correlation(
        self, 
        X: pd.DataFrame, 
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features"""
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs with high correlation
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = []
        for col in upper.columns:
            high_corr = upper[col][upper[col] > self.config.correlation_threshold].index.tolist()
            to_drop.extend(high_corr)
        
        to_drop = list(set(to_drop))
        self.removed_features['high_correlation'] = to_drop
        
        feature_cols = [f for f in feature_cols if f not in to_drop]
        X = X[feature_cols]
        
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        
        return X, feature_cols
    
    def _select_top_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select top N features by mutual information"""
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create feature importance dataframe
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        # Select top features
        top_features = importance.head(self.config.max_features)['feature'].tolist()
        
        X = X[top_features]
        feature_cols = top_features
        
        logger.info(f"Selected top {len(feature_cols)} features by mutual information")
        
        return X, feature_cols
    
    def get_feature_report(self) -> Dict:
        """Get report of feature selection"""
        return {
            'selected_features': self.selected_features,
            'n_selected': len(self.selected_features),
            'removed': self.removed_features
        }


# =============================================================================
# DATA SPLITTING
# =============================================================================

class DataSplitter:
    """Split data into train/validation/test sets"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
    
    def split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data temporally or randomly.
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        logger.info("Splitting data...")
        
        if self.config.temporal_split:
            return self._temporal_split(X, y)
        else:
            return self._random_split(X, y)
    
    def _temporal_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data temporally (chronological).
        
        Train on past, test on future.
        """
        n = len(X)
        
        # Calculate split indices
        test_idx = int(n * (1 - self.config.test_size))
        val_idx = int(test_idx * (1 - self.config.validation_size))
        
        # Split
        X_train = X.iloc[:val_idx].reset_index(drop=True)
        y_train = y.iloc[:val_idx].reset_index(drop=True)
        
        X_val = X.iloc[val_idx:test_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx:test_idx].reset_index(drop=True)
        
        X_test = X.iloc[test_idx:].reset_index(drop=True)
        y_test = y.iloc[test_idx:].reset_index(drop=True)
        
        logger.info(f"Temporal split:")
        logger.info(f"  Train: {len(X_train)} ({y_train.mean()*100:.1f}% positive)")
        logger.info(f"  Val:   {len(X_val)} ({y_val.mean()*100:.1f}% positive)")
        logger.info(f"  Test:  {len(X_test)} ({y_test.mean()*100:.1f}% positive)")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def _random_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Random split (not recommended for time series)"""
        
        logger.warning("Using random split - not recommended for time series data!")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=42,
            stratify=y
        )
        
        # Second split: train vs val
        val_ratio = self.config.validation_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=42,
            stratify=y_temp
        )
        
        logger.info(f"Random split:")
        logger.info(f"  Train: {len(X_train)} ({y_train.mean()*100:.1f}% positive)")
        logger.info(f"  Val:   {len(X_val)} ({y_val.mean()*100:.1f}% positive)")
        logger.info(f"  Test:  {len(X_test)} ({y_test.mean()*100:.1f}% positive)")
        
        return {
            'train': (X_train.reset_index(drop=True), y_train.reset_index(drop=True)),
            'val': (X_val.reset_index(drop=True), y_val.reset_index(drop=True)),
            'test': (X_test.reset_index(drop=True), y_test.reset_index(drop=True))
        }


# =============================================================================
# SCALING
# =============================================================================

class FeatureScaler:
    """Scale features for training"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.scaler = None
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on training data and transform"""
        
        if self.config.scaler_type is None:
            return X
        
        logger.info(f"Scaling features with {self.config.scaler_type} scaler...")
        
        # Create scaler
        if self.config.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.config.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.config.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown scaler type: {self.config.scaler_type}")
            return X
        
        # Fit and transform
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
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled
    
    def save(self, filepath: str):
        """Save fitted scaler"""
        if self.scaler is not None:
            joblib.dump(self.scaler, filepath)
            logger.info(f"Scaler saved to {filepath}")
    
    def load(self, filepath: str):
        """Load fitted scaler"""
        self.scaler = joblib.load(filepath)
        logger.info(f"Scaler loaded from {filepath}")


# =============================================================================
# CLASS IMBALANCE HANDLING
# =============================================================================

class ImbalanceHandler:
    """Handle class imbalance in training data"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.sampler = None
    
    def resample(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Resample training data to handle class imbalance.
        
        Only apply to training data, not validation/test!
        """
        if not self.config.handle_imbalance:
            return X, y
        
        logger.info(f"Handling class imbalance with {self.config.imbalance_method}...")
        logger.info(f"Before: {dict(y.value_counts())}")
        
        # Create sampler
        if self.config.imbalance_method == 'smote':
            self.sampler = SMOTE(
                sampling_strategy=self.config.imbalance_ratio,
                random_state=42,
                k_neighbors=5
            )
        elif self.config.imbalance_method == 'adasyn':
            self.sampler = ADASYN(
                sampling_strategy=self.config.imbalance_ratio,
                random_state=42
            )
        elif self.config.imbalance_method == 'undersample':
            self.sampler = RandomUnderSampler(
                sampling_strategy=self.config.imbalance_ratio,
                random_state=42
            )
        elif self.config.imbalance_method == 'smote_tomek':
            self.sampler = SMOTETomek(
                sampling_strategy=self.config.imbalance_ratio,
                random_state=42
            )
        else:
            logger.warning(f"Unknown imbalance method: {self.config.imbalance_method}")
            return X, y
        
        # Resample
        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)
            
            logger.info(f"After: {dict(y_resampled.value_counts())}")
            
            return X_resampled, y_resampled
        
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            return X, y


# =============================================================================
# MAIN DATA PREPARATION PIPELINE
# =============================================================================

class DataPreparationPipeline:
    """Main pipeline for data preparation"""
    
    def __init__(self, config: Optional[DataPrepConfig] = None):
        self.config = config or DataPrepConfig()
        
        # Initialize components
        self.target_creator = TargetCreator(self.config)
        self.data_filter = DataFilter(self.config)
        self.feature_selector = FeatureSelector(self.config)
        self.data_splitter = DataSplitter(self.config)
        self.scaler = FeatureScaler(self.config)
        self.imbalance_handler = ImbalanceHandler(self.config)
        
        # Store metadata
        self.metadata = {}
    
    def run(
        self, 
        input_path: str,
        target_col: str = 'target_hit_tp1'
    ) -> Dict[str, Any]:
        """
        Run the complete data preparation pipeline.
        
        Args:
            input_path: Path to feature-engineered CSV
            target_col: Name of target column to use
        
        Returns:
            Dictionary containing prepared datasets and metadata
        """
        logger.info("="*60)
        logger.info("DATA PREPARATION PIPELINE")
        logger.info("="*60)
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Create targets
        df = self.target_creator.create_targets(df)
        
        # Filter data
        df = self.data_filter.filter(df)
        
        # Get target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        y = df[target_col].astype(int)
        
        # Select features
        X, feature_cols = self.feature_selector.select(df, target_col)
        
        # Split data
        splits = self.data_splitter.split(X, y)
        
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        # Scale features (fit on train only)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle class imbalance (train only)
        X_train_balanced, y_train_balanced = self.imbalance_handler.resample(
            X_train_scaled, y_train
        )
        
        # Store metadata
        self.metadata = {
            'n_samples_original': len(df),
            'n_samples_train': len(X_train_balanced),
            'n_samples_val': len(X_val),
            'n_samples_test': len(X_test),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'target_column': target_col,
            'class_distribution_train': dict(y_train_balanced.value_counts()),
            'class_distribution_test': dict(y_test.value_counts()),
            'scaler_type': self.config.scaler_type,
            'imbalance_method': self.config.imbalance_method,
            'feature_selection_report': self.feature_selector.get_feature_report()
        }
        
        # Save datasets
        logger.info("Saving prepared datasets...")
        
        # Train
        train_data = X_train_balanced.copy()
        train_data['target'] = y_train_balanced.values
        train_data.to_csv(output_dir / 'train.csv', index=False)
        
        # Validation
        val_data = X_val_scaled.copy()
        val_data['target'] = y_val.values
        val_data.to_csv(output_dir / 'val.csv', index=False)
        
        # Test
        test_data = X_test_scaled.copy()
        test_data['target'] = y_test.values
        test_data.to_csv(output_dir / 'test.csv', index=False)
        
        # Save scaler
        if self.config.save_scaler:
            self.scaler.save(str(output_dir / 'scaler.joblib'))
        
        # Save feature list
        if self.config.save_feature_list:
            with open(output_dir / 'features.json', 'w') as f:
                json.dump({
                    'features': feature_cols,
                    'n_features': len(feature_cols)
                }, f, indent=2)
        
        # Save metadata
        with open(output_dir / 'metadata.json', 'w') as f:
            # Convert any non-serializable types
            meta_serializable = {}
            for k, v in self.metadata.items():
                if isinstance(v, (np.int64, np.int32)):
                    meta_serializable[k] = int(v)
                elif isinstance(v, (np.float64, np.float32)):
                    meta_serializable[k] = float(v)
                elif isinstance(v, dict):
                    meta_serializable[k] = {
                        str(kk): int(vv) if isinstance(vv, (np.int64, np.int32)) else vv 
                        for kk, vv in v.items()
                    }
                else:
                    meta_serializable[k] = v
            json.dump(meta_serializable, f, indent=2)
        
        logger.info(f"Datasets saved to {output_dir}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("PREPARATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Train samples: {len(X_train_balanced)}")
        logger.info(f"Val samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Train class balance: {dict(y_train_balanced.value_counts())}")
        
        return {
            'train': (X_train_balanced, y_train_balanced),
            'val': (X_val_scaled, y_val),
            'test': (X_test_scaled, y_test),
            'feature_names': feature_cols,
            'metadata': self.metadata,
            'scaler': self.scaler
        }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT Data Preparation Pipeline')
    parser.add_argument('input', help='Input features CSV file')
    parser.add_argument('--output-dir', default='data/prepared',
                        help='Output directory (default: data/prepared)')
    parser.add_argument('--target', default='target_hit_tp1',
                        help='Target column name (default: target_hit_tp1)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--min-confluence', type=int, default=1,
                        help='Minimum confluence filter (default: 1)')
    parser.add_argument('--no-killzone-filter', action='store_true',
                        help='Disable killzone filter')
    parser.add_argument('--imbalance-method', default='smote',
                        choices=['smote', 'adasyn', 'undersample', 'smote_tomek', 'none'],
                        help='Class imbalance handling method')
    
    args = parser.parse_args()
    
    # Create config
    config = DataPrepConfig(
        output_dir=args.output_dir,
        test_size=args.test_size,
        min_confluence=args.min_confluence,
        require_killzone=not args.no_killzone_filter,
        imbalance_method=args.imbalance_method if args.imbalance_method != 'none' else None,
        handle_imbalance=args.imbalance_method != 'none'
    )
    
    # Run pipeline
    pipeline = DataPreparationPipeline(config)
    pipeline.run(args.input, args.target)


if __name__ == "__main__":
    main()
