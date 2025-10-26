# %%
import os
import sys
import argparse

def setup_environment():
    """Setup environment variables before TensorFlow import"""
    # Suppress TensorFlow warnings initially
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # CUDA configuration
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # GPU memory management
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # For reproducibility
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'  # Disable initially for stability
    
    # Memory optimization
    os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'] = 'false'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Setup environment before ANY TensorFlow imports
setup_environment()

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split, KFold
from sklearn.preprocessing import RobustScaler
import keras_tuner as kt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from numpy.lib.stride_tricks import sliding_window_view
import preprocess_data

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def parse_gpu_args():
    """Parse command line arguments for GPU selection"""
    parser = argparse.ArgumentParser(description='Multi-class Direction Training with GPU Selection')
    parser.add_argument('--gpus', type=str, default='0', 
                       help='Comma-separated list of GPU IDs to use (e.g., "0,1,2" or "0")')
    parser.add_argument('--memory-limit', type=int, default=None,
                       help='Memory limit per GPU in MB (optional)')
    
    # If running in notebook, provide default args
    try:
        args = parser.parse_args()
    except SystemExit:
        # Running in notebook, use defaults
        class DefaultArgs:
            gpus = '0'
            memory_limit = None
        args = DefaultArgs()
    
    return args

def configure_gpu_selection(gpu_ids, memory_limit=None):
    """
    Enhanced GPU configuration with specific GPU selection
    """
    print(f"🔧 Configuring GPUs: {gpu_ids}")
    
    try:
        # Get all available GPUs
        all_gpus = tf.config.list_physical_devices('GPU')
        
        if not all_gpus:
            print("❌ No GPUs detected")
            return False, 0
        
        print(f"✅ Detected {len(all_gpus)} total GPU(s)")
        for i, gpu in enumerate(all_gpus):
            print(f"  GPU {i}: {gpu}")
        
        # Parse requested GPU IDs
        if isinstance(gpu_ids, str):
            requested_ids = [int(x.strip()) for x in gpu_ids.split(',')]
        else:
            requested_ids = [gpu_ids] if isinstance(gpu_ids, int) else list(gpu_ids)
        
        # Validate requested GPU IDs
        valid_ids = [gpu_id for gpu_id in requested_ids if 0 <= gpu_id < len(all_gpus)]
        if not valid_ids:
            print(f"❌ No valid GPU IDs in {requested_ids}. Available: 0-{len(all_gpus)-1}")
            return False, 0
        
        if len(valid_ids) != len(requested_ids):
            invalid_ids = set(requested_ids) - set(valid_ids)
            print(f"⚠️  Invalid GPU IDs ignored: {invalid_ids}")
        
        # Select and configure GPUs
        selected_gpus = [all_gpus[i] for i in valid_ids]
        print(f"🎯 Using GPU(s): {valid_ids}")
        
        # Configure each selected GPU
        for i, gpu in enumerate(selected_gpus):
            print(f"Configuring GPU {valid_ids[i]}: {gpu}")
            
            try:
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory limit if specified
                if memory_limit:
                    tf.config.experimental.set_memory_limit(gpu, memory_limit)
                    print(f"  Set memory limit: {memory_limit} MB")
                
            except RuntimeError as e:
                if "memory growth" in str(e).lower() or "already initialized" in str(e).lower():
                    print(f"⚠️  GPU {valid_ids[i]} already configured")
                else:
                    print(f"❌ Error configuring GPU {valid_ids[i]}: {e}")
                    return False, 0
        
        # Set visible devices to only selected GPUs
        tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')
        
        # Test GPU with error handling
        try:
            with tf.device(f'/GPU:{valid_ids[0]}'):
                # Small test operation
                test_tensor = tf.constant([[1.0, 2.0]], dtype=tf.float32)
                result = tf.reduce_sum(test_tensor)
                _ = result.numpy()  # Force execution
                print(f"✅ GPU test successful on GPU {valid_ids[0]}")
                
            return True, len(selected_gpus)
            
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            print("Falling back to CPU...")
            return False, 0
            
    except Exception as e:
        print(f"❌ GPU configuration failed: {e}")
        return False, 0

def configure_cpu_optimal():
    """Configure TensorFlow for optimal CPU performance"""
    print("🔧 Configuring for CPU optimization...")
    
    try:
        # Set thread counts for CPU optimization
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
        
        print("✅ CPU optimization configured")
        return True
        
    except RuntimeError as e:
        if "cannot be modified after initialization" in str(e):
            print("⚠️  TensorFlow already initialized. Cannot modify threading settings.")
            print("Restart Python session for optimal CPU configuration.")
        else:
            print(f"❌ CPU configuration error: {e}")
        return False

def setup_tensorflow(gpu_ids='0', memory_limit=None):
    """Enhanced TensorFlow setup with GPU selection"""
    print("🚀 Setting up TensorFlow with GPU selection...")
    
    # Try GPU configuration first
    gpu_success, num_gpus = configure_gpu_selection(gpu_ids, memory_limit)

    if gpu_success:
        print(f"✅ Using GPU acceleration with {num_gpus} GPU(s)")
        
        # Configure for stability rather than maximum performance
        tf.config.optimizer.set_jit(False)  # Disable XLA initially for stability
        
        return 'GPU', num_gpus
    else:
        print("⚠️  GPU setup failed. Configuring for CPU...")
        cpu_success = configure_cpu_optimal()
        if cpu_success:
            print("✅ Using optimized CPU configuration")
            return 'CPU', 0
        else:
            print("⚠️  Using default TensorFlow configuration")
            return 'DEFAULT', 0

# Import all classification models
from classification_model_build import (
    build_LSTM_classifier,
    build_GRU_classifier,
    build_Conv1D_classifier,
    build_Conv1D_LSTM_classifier,
)

# ======================================
# MULTI-CLASS CONFIGURATION MANAGEMENT
# ======================================
class MultiClassConfigManager:
    """
    Configuration manager for multi-class direction_3class_5 prediction
    """
    def __init__(self):
        # Common settings
        self.lookback_window = 60
        self.test_size = 0.2
        self.random_seed = 42
        
        # Memory optimization settings
        self.max_samples = 1000000
        self.chunk_size = 10000
        self.use_streaming = True
        
        # Training parameters
        self.batch_size = 512
        self.max_epochs = 30
        self.steps_per_epoch = 100
        self.validation_steps = 50
        self.early_stopping_patience = 7
        self.initial_learning_rate = 1e-4
        
        # Model selection
        self.selected_models = ['LSTM', 'GRU', 'Conv1D', 'Conv1D_LSTM']
        
        # Data transformation
        self.use_lag_features = True
        self.lag_periods = [5, 10, 20, 30, 60]
        
        # Hyperparameter tuning
        self.max_trials = 150
        self.max_consecutive_failed_trials = 40
        
        # Feature selection settings
        self.use_important_features_only = True
        self.exclude_future_looking = True
        
        # Multi-class target configuration (only direction_3class_5)
        self.target_config = self._create_target_config()
    
    def _create_target_config(self):
        """Create configuration for direction_3class_5 target"""
        return {
            'data_file': 'direction_3class_5_EURUSD_1min_filtered.csv',
            'target_col': 'direction_3class_5',
            'num_classes': 3,  # 0, 1, 2
            'results_file': 'direction_3class_5_model_training_results.txt',
            'model_save_dir': 'models/direction_3class_5_models',
            'project_name_suffix': 'direction_3class_5_multiclass',
            'important_features': [
                # Core technical indicators
                'Date', 'Low', 'Volume', 'adx_trend_buy_signal', 'adx_trend_sell_signal', 
                'pzosx_buy_signal', 'current_candle_height', 'average_candle_height', 
                'bb_short_entry_signal', 'camarilla_buy_signal', 'camarilla_sell_signal', 
                'cci_bullish_signal', 'cci_bearish_signal', 'cmf_buy_signal', 'cmf_sell_signal',
                'dpo_overbought_signal', 'dpo_buy_signal', 'dpo_sell_signal', 
                'ehlers_stoch_buy_signal', 'eight_month_avg_buy_signal', 'eight_month_avg_sell_signal',
                'EMA_bullish_signal', 'eom_buy_signal', 'eom_sell_signal', 
                'gap_momentum_buy_signal', 'gap_momentum_sell_signal', 'golden_cross_buy_signal',
                'hacolt', 'ironbot_buy_signal', 'ironbot_sell_signal', 'kc_buy_signal', 
                'kc_sell_signal', 'moving_average_buy_signal', 'PPO', 'pzo_lx_sell_signal',
                'rocwb_buy_signal', 'spectrum_bars_buy_signal', 'stc_overbought_signal', 
                'stc_oversold_signal', 'stoch_buy_signal', 'stoch_sell_signal', 
                'stochastic_strat_buy_signal', 'stochrsi_overbought_signal', 'stochrsi_oversold_signal',
                '5_8_13_buy_signal', '5_8_13_sell_signal', 'w5_8_13_buy_signal', 'w5_8_13_sell_signal',
                'sve_zl_rb_perc_buy_signal', 'sve_zl_rb_perc_sell_signal', 'svesc_buy_signal', 
                'svesc_sell_signal', 'volatility_band_buy_signal', 'vols_switch_buy_signal', 
                'vols_switch_sell_signal', 'vpn_sell_signal', 'vwma_breakouts_buy_signal', 
                'williams_buy_signal', 'williams_sell_signal',
                # Past direction features (valid for direction_3class_5 prediction)
                'direction_1', 'direction_3class_1', 'returns_1', 'direction_3', 
                'direction_3class_3', 'returns_3', 'direction_5', 'returns_5', 
                'direction_10', 'direction_3class_10', 'returns_10', 'direction_14', 
                'direction_3class_14', 'returns_14'
            ],
            'exclude_columns': [
                # Trading signal columns
                'long_signal', 'short_signal', 'close_position',
                # Time and price columns (keep some for features)
                'Time', 'datetime', 'Open', 'High', 'Close',
                # No future-looking features to exclude for direction_3class_5 since it's our target
            ]
        }

# Dictionary of model builders
MODEL_BUILDERS = {
    "LSTM": build_LSTM_classifier,
    "GRU": build_GRU_classifier,
    "Conv1D": build_Conv1D_classifier,
    "Conv1D_LSTM": build_Conv1D_LSTM_classifier,
}

# ======================================
# DATA LOADING AND PREPROCESSING
# ======================================
def load_and_preprocess_data(config):
    """
    Load and preprocess the data for direction_3class_5 multi-class prediction
    """
    target_config = config.target_config
    
    print(f"Loading and preprocessing data for multi-class direction_3class_5 prediction...")
    data = pd.read_csv(target_config['data_file'], header=0)
    print(f"Original dataset size: {len(data)} rows")
    
    # Handle duplicate columns (mentioned Date appears twice)
    print("Checking for duplicate columns...")
    duplicate_cols = data.columns[data.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"Found duplicate columns: {duplicate_cols}")
        data = data.loc[:, ~data.columns.duplicated()]
        print("Removed duplicate columns")
    
    # Sample data if too large to prevent memory issues
    if len(data) > config.max_samples:
        print(f"Dataset too large ({len(data)} rows). Sampling {config.max_samples} rows...")
        # Use stratified sampling to maintain class balance
        target_col_temp = data[target_config['target_col']]
        
        _, data = train_test_split(
            data, 
            test_size=config.max_samples / len(data),
            stratify=target_col_temp,
            random_state=config.random_seed
        )
        print(f"Sampled dataset size: {len(data)} rows")
    
    # Basic data cleaning
    data = preprocess_data.clean_data(data)
    
    # Get important features and exclusions
    important_features = target_config['important_features']
    exclude_columns = target_config['exclude_columns']
    
    # Check which important features are actually present in the dataset
    available_important_features = [col for col in important_features if col in data.columns]
    missing_features = [col for col in important_features if col not in data.columns]
    
    if missing_features:
        print(f"Warning: Missing important features: {missing_features}")
    
    print(f"Using {len(available_important_features)} important features")
    
    # Use only the important features that are available
    feature_columns = available_important_features
    
    print(f"Selected features: {feature_columns[:10]}..." if len(feature_columns) > 10 else f"Selected features: {feature_columns}")
    print(f"Excluded columns: {[col for col in exclude_columns if col in data.columns]}")
    
    # Add Volume processing if available
    if 'Volume' in data.columns:
        print("Processing Volume column...")
        # Log transform (handles high skewness)
        data['Volume_log'] = np.log1p(data['Volume'])
        feature_columns.append('Volume_log')
        
        # Winsorize extreme values (cap at percentiles)
        q_low, q_high = data['Volume'].quantile(0.01), data['Volume'].quantile(0.99)
        data['Volume_winsor'] = data['Volume'].clip(q_low, q_high)
        feature_columns.append('Volume_winsor')
        
        # Rank transform (completely resistant to outliers)
        data['Volume_rank'] = data['Volume'].rank(pct=True)
        feature_columns.append('Volume_rank')
    
    # Add lag features if enabled - BUT ONLY FOR TECHNICAL INDICATORS
    if config.use_lag_features:
        print(f"Adding lag features for technical indicators...")
        lag_list = config.lag_periods
        
        # Only add lagged versions of safe technical indicators, NOT the target
        safe_indicators = ['RSI', 'adx_trend_buy_signal', 'cci_bullish_signal', 
                          'golden_cross_buy_signal', 'williams_buy_signal']
        
        for lag in lag_list:
            for indicator in safe_indicators:
                if indicator in data.columns and indicator in feature_columns:
                    lag_col = f'{indicator}_lag_{lag}'
                    data[lag_col] = data[indicator].shift(lag)
                    feature_columns.append(lag_col)
        
        # Add rolling stats on lagged technical indicators
        lag_indicator_cols = [col for col in feature_columns if '_lag_' in col]
        if lag_indicator_cols:
            data['indicators_lag_mean'] = data[lag_indicator_cols].mean(axis=1)
            data['indicators_lag_std'] = data[lag_indicator_cols].std(axis=1)
            feature_columns.extend(['indicators_lag_mean', 'indicators_lag_std'])
    
    # Fill missing values
    data = data.bfill().ffill()
    
    # Split features and target
    features = data[feature_columns]
    target = data[target_config['target_col']]
    
    # Ensure target is multi-class (0, 1, 2)
    print(f"Target distribution before conversion: {target.value_counts().sort_index()}")
    target = target.astype(int)
    
    # Validate target classes
    unique_classes = sorted(target.unique())
    expected_classes = [0, 1, 2]
    
    if set(unique_classes) != set(expected_classes):
        print(f"Warning: Expected classes {expected_classes}, found {unique_classes}")
        # Map any unexpected values to valid classes if needed
        target = target.clip(0, 2)
    
    print(f"Target distribution after conversion: {target.value_counts().sort_index()}")
    
    # Validate that all feature columns are numeric
    non_numeric_cols = []
    for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        print(f"Warning: Found non-numeric columns in features: {non_numeric_cols}")
        print("Attempting to convert to numeric or remove...")
        
        for col in non_numeric_cols:
            try:
                features[col] = pd.to_numeric(features[col], errors='coerce')
                print(f"Successfully converted {col} to numeric")
            except:
                print(f"Removing non-numeric column: {col}")
                features = features.drop(columns=[col])
    
    print(f"Final dataset shape - Features: {features.shape}, Target: {target.shape}")
    print(f"Memory usage: Features: {features.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    return features, target

# ======================================
# DATA SCALING
# ======================================
def scale_data_multiclass_chunked(train_X, test_X, n_features, chunk_size=10000):
    """
    Scale the data using RobustScaler for multi-class classification with chunked processing
    """
    print("Scaling features using chunked processing...")
    print(f"Input shapes - Train: {train_X.shape}, Test: {test_X.shape}")
    print(f"Estimated memory usage: {(train_X.nbytes + test_X.nbytes) / 1e9:.2f} GB")
    
    # Create and fit the scaler using chunks of training data
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    
    print("Fitting scaler on training data...")
    # Fit scaler on chunks to avoid memory issues
    for i in range(0, len(train_X), chunk_size):
        end_i = min(i + chunk_size, len(train_X))
        chunk = train_X[i:end_i]
        
        # Reshape chunk to 2D for scaling
        chunk_2d = chunk.reshape(-1, n_features)
        chunk_2d = np.nan_to_num(chunk_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Partial fit the scaler
        if i == 0:
            scaler.fit(chunk_2d)
        else:
            # For RobustScaler, we need to recompute on all data seen so far
            # This is a limitation, so we'll fit on the first large chunk only
            pass
    
    print("Transforming training data...")
    # Transform training data in chunks
    for i in range(0, len(train_X), chunk_size):
        end_i = min(i + chunk_size, len(train_X))
        chunk = train_X[i:end_i]
        
        # Reshape to 2D
        original_shape = chunk.shape
        chunk_2d = chunk.reshape(-1, n_features)
        chunk_2d = np.nan_to_num(chunk_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Transform
        chunk_2d = scaler.transform(chunk_2d)
        chunk_2d = np.clip(chunk_2d, -10, 10)  # Cap values to prevent instability
        
        # Reshape back and store in-place
        train_X[i:end_i] = chunk_2d.reshape(original_shape)
    
    print("Transforming test data...")
    # Transform test data in chunks
    for i in range(0, len(test_X), chunk_size):
        end_i = min(i + chunk_size, len(test_X))
        chunk = test_X[i:end_i]
        
        # Reshape to 2D
        original_shape = chunk.shape
        chunk_2d = chunk.reshape(-1, n_features)
        chunk_2d = np.nan_to_num(chunk_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Transform
        chunk_2d = scaler.transform(chunk_2d)
        chunk_2d = np.clip(chunk_2d, -10, 10)
        
        # Reshape back and store in-place
        test_X[i:end_i] = chunk_2d.reshape(original_shape)
    
    data_stats = {
        'min': np.min(train_X),
        'max': np.max(train_X),
        'mean': np.mean(train_X),
        'std': np.std(train_X)
    }
    print(f"Data statistics after scaling: {data_stats}")
    
    return train_X, test_X, scaler

# ======================================
# TIME SERIES WINDOW CREATION 
# ======================================
def create_sliding_windows_multiclass_chunked(features, target, n_in, chunk_size=10000):
    """
    Create sliding windows for time series multi-class classification
    """
    print(f"Creating sliding windows with lookback={n_in} using chunked processing")
    n_features = features.shape[1]
    total_samples = len(features)
    
    print(f"Processing {total_samples} samples in chunks of {chunk_size}")
    
    # Calculate total windows needed
    total_windows = total_samples - n_in + 1
    if total_windows <= 0:
        raise ValueError(f"Not enough data: need at least {n_in} samples, got {total_samples}")
    
    print(f"Will create {total_windows} windows")
    
    # Convert data to float32 for memory efficiency
    features_values = features.values.astype(np.float32)
    target_values = target.values.astype(np.int32)
    
    # Process in chunks to avoid memory overflow
    feature_chunks = []
    target_chunks = []
    
    # Process data in overlapping chunks to maintain sequence continuity
    for start_idx in range(0, total_samples - n_in + 1, chunk_size):
        # Ensure we have enough data for the window
        end_idx = min(start_idx + chunk_size + n_in - 1, total_samples)
        
        print(f"Processing chunk {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
        
        # Get chunk data
        chunk_features = features_values[start_idx:end_idx]
        chunk_targets = target_values[start_idx:end_idx]
        
        # Create windows for this chunk
        if len(chunk_features) >= n_in:
            chunk_windows = sliding_window_view(chunk_features, window_shape=n_in, axis=0)
            chunk_windows = np.transpose(chunk_windows, (0, 2, 1))
            
            # Corresponding targets (current time step prediction)
            chunk_target_windows = chunk_targets[n_in-1:n_in-1+len(chunk_windows)]
            
            # Only take the windows that don't overlap with next chunk
            if start_idx + chunk_size < total_samples - n_in + 1:
                take_windows = min(chunk_size, len(chunk_windows))
                chunk_windows = chunk_windows[:take_windows]
                chunk_target_windows = chunk_target_windows[:take_windows]
            
            feature_chunks.append(chunk_windows)
            target_chunks.append(chunk_target_windows)
    
    # Concatenate all chunks
    print("Concatenating chunks...")
    features_array = np.concatenate(feature_chunks, axis=0)
    target_array = np.concatenate(target_chunks, axis=0)
    
    # Handle NaNs in features
    nan_count = np.isnan(features_array).sum()
    if nan_count > 0:
        print(f"Replacing {nan_count} NaNs in features.")
        # Process in chunks to avoid memory issues
        for i in range(0, len(features_array), chunk_size):
            end_i = min(i + chunk_size, len(features_array))
            chunk = features_array[i:end_i]
            
            original_shape = chunk.shape
            chunk_2d = chunk.reshape(-1, n_features)
            
            # Replace NaNs with column means
            for j in range(n_features):
                col_data = chunk_2d[:, j]
                if np.isnan(col_data).any():
                    col_mean = np.nanmean(col_data)
                    chunk_2d[np.isnan(col_data), j] = col_mean
            
            features_array[i:end_i] = chunk_2d.reshape(original_shape)
    
    print(f"Final shapes — Features: {features_array.shape}, Target: {target_array.shape}")
    print(f"Memory usage: Features: {features_array.nbytes / 1e6:.1f} MB, Target: {target_array.nbytes / 1e6:.1f} MB")
    
    return features_array, target_array

# ======================================
# MULTI-CLASS FOCAL LOSS
# ======================================
def multiclass_focal_loss(gamma=2.0, alpha=None):
    """
    Multi-class focal loss for handling class imbalance
    """
    def focal_loss_fn(y_true, y_pred):
        # Use a larger epsilon to prevent numerical issues
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Convert to one-hot if needed
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred + epsilon)
        
        # Calculate p_t
        p_t = tf.reduce_sum(y_true * y_pred, axis=1)
        
        # Calculate alpha_t if alpha is provided
        if alpha is not None:
            if isinstance(alpha, (list, tuple, np.ndarray)):
                alpha_t = tf.reduce_sum(y_true * alpha, axis=1)
            else:
                alpha_t = alpha
        else:
            alpha_t = 1.0
        
        # Calculate focal weight
        focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
        
        # Calculate focal loss
        focal_loss = focal_weight[:, tf.newaxis] * ce
        
        # Replace any NaN or inf with zeros
        focal_loss = tf.where(tf.math.is_finite(focal_loss), focal_loss, tf.zeros_like(focal_loss))
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    
    return focal_loss_fn

# ======================================
# CALLBACKS AND LEARNING RATE SCHEDULERS
# ======================================
def cosine_annealing_warmup_schedule(epoch, lr, total_epochs=50, warmup_epochs=3, min_lr=1e-6):
    """
    Cosine annealing with warmup learning rate schedule
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return lr * ((epoch + 1) / warmup_epochs)
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

class NaNSafetyCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor='loss', patience=3):
        super(NaNSafetyCallback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.nan_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get(self.monitor)
        
        if loss is None:
            return
            
        if np.isnan(loss) or np.isinf(loss):
            self.nan_count += 1
            print(f"NaN/Inf detected in {self.monitor} at epoch {epoch+1}")
            
            if self.nan_count >= self.patience:
                print(f"Stopping training due to {self.nan_count} consecutive NaN/Inf values")
                self.model.stop_training = True
        else:
            # Reset counter if we get a valid loss
            self.nan_count = 0

class EarlyStopper(tf.keras.callbacks.Callback):
    def __init__(self, baseline=0.5, min_epoch=5):  # Lower baseline for 3-class
        super(EarlyStopper, self).__init__()
        self.baseline = baseline
        self.min_epoch = min_epoch
    
    def on_epoch_end(self, epoch, logs=None):
        # Only check after minimum epochs
        if epoch >= self.min_epoch:
            current = logs.get('val_accuracy')
            if current and current < self.baseline:
                print(f"\nStopping trial: val_accuracy {current} below threshold {self.baseline}")
                self.model.stop_training = True

def create_callbacks():
    """Create fresh callbacks for each trial"""
    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: cosine_annealing_warmup_schedule(epoch, lr)
        ),
        EarlyStopper(baseline=0.5),  # Lower baseline for 3-class
        NaNSafetyCallback()
    ]

# ======================================
# ROBUST PREPROCESSING
# ======================================
def robust_preprocessing(X_train, X_test, threshold=10.0):
    """
    Apply robust preprocessing to prevent NaNs during training for 3D time series data
    """
    # Get shapes
    n_samples_train, n_timesteps, n_features = X_train.shape
    
    # Replace NaNs with zeros
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=threshold, neginf=-threshold)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=threshold, neginf=-threshold)
    
    # Check if we have extreme values
    train_max = np.max(np.abs(X_train))
    if train_max > threshold:
        print(f"Warning: Extreme values detected ({train_max:.2f}). Clipping to ±{threshold}...")
        X_train = np.clip(X_train, -threshold, threshold)
        X_test = np.clip(X_test, -threshold, threshold)
    
    # Check for constant features across all timesteps
    X_train_reshaped = X_train.reshape(-1, n_features)
    std_per_feature = np.std(X_train_reshaped, axis=0)
    constant_features = np.where(std_per_feature < 1e-10)[0]
    
    if len(constant_features) > 0:
        print(f"Warning: {len(constant_features)} constant features detected. Adding small noise...")
        for idx in constant_features:
            noise = np.random.normal(0, 1e-6, size=(n_samples_train, n_timesteps))
            X_train[:, :, idx] += noise
    
    return X_train, X_test

# ======================================
# MODEL TRAINING UTILITIES
# ======================================
def get_custom_model_builder(model_name, model_builder, train_X, num_classes=3):
    def custom_model_builder(hp):
        try:
            # Build model for multi-class classification
            model = model_builder(hp, num_classes=num_classes)
                
            if model is None:
                print(f"Model builder for {model_name} returned None")
                return None

            optimizer = tf.keras.optimizers.Adam(
                hp.Float("learning_rate", min_value=1e-6, max_value=5e-4, sampling="log"),
                clipvalue=1.0  # Clip gradients to avoid exploding values
            )
            
            # Use multi-class focal loss or categorical crossentropy
            loss_type = hp.Choice("loss_type", ["focal", "categorical_crossentropy"])
            
            if loss_type == "focal":
                loss_fn = multiclass_focal_loss(
                    gamma=hp.Float("focal_gamma", min_value=1.0, max_value=3.0, step=0.5, default=2.0)
                )
            else:
                loss_fn = 'sparse_categorical_crossentropy'
            
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy', 
                        tf.keras.metrics.SparseCategoricalAccuracy(),
                        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)],
                jit_compile=True  # Enable XLA compilation
            )
            return model
        except Exception as e:
            print(f"Error in custom model builder for {model_name}: {e}")
            return None

    return custom_model_builder

# ======================================
# MAIN TRAINING PIPELINE
# ======================================
def train_multiclass_models(config, gpu_ids='0'):
    """
    Train multiple multi-class models for direction_3class_5 prediction
    """
    try:
        import psutil
    except ImportError:
        print("psutil not available, memory monitoring disabled")
        psutil = None
        
    import gc
    
    def print_memory_usage(step_name):
        """Print current memory usage"""
        if psutil:
            memory = psutil.virtual_memory()
            print(f"[{step_name}] Memory usage: {memory.percent:.1f}% ({memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB)")
        
        # Check GPU memory if available
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                gpu_used = mem_info['current'] / (1024**3)
                gpu_total = mem_info['peak'] / (1024**3)
                print(f"[{step_name}] GPU memory: {gpu_used:.1f}/{gpu_total:.1f} GB")
        except:
            pass
    
    target_config = config.target_config
    
    print(f"\n{'='*80}")
    print(f"🎯 TRAINING MULTI-CLASS MODELS FOR: direction_3class_5")
    print(f"{'='*80}")
    print_memory_usage("Start training")
    
    # Set up logging
    log_file_path = target_config['results_file']
    with open(log_file_path, "w") as log_file:
        log_file.write(f"DIRECTION_3CLASS_5 Multi-Class Models Training Results\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.write(f"Number of classes: {target_config['num_classes']}\n")
        log_file.write(f"Target column: {target_config['target_col']}\n\n")
    
    # Step 1: Load and preprocess data
    features, target = load_and_preprocess_data(config)
    print_memory_usage("After data loading")
    
    # Step 2: Create sliding windows using chunked processing
    features_array, target_array = create_sliding_windows_multiclass_chunked(
        features, target, config.lookback_window, chunk_size=config.chunk_size
    )
    print_memory_usage("After window creation")
    
    # Free original data to save memory
    del features, target
    gc.collect()
    print_memory_usage("After cleanup")
    
    # Step 3: Compute class weights for multi-class
    print("Computing class weights for multi-class...")
    classes = np.array([0, 1, 2]) 
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=target_array.flatten()
    )
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
    print(f"Computed class weights: {class_weight_dict}")
    
    # Step 4: Split into train/test sets
    print("Splitting data...")
    train_X, test_X, train_y, test_y = train_test_split(
        features_array, target_array, 
        test_size=config.test_size, 
        random_state=config.random_seed, 
        shuffle=False,
        stratify=target_array  # Stratify for multi-class
    )
    print_memory_usage("After train/test split")
    
    # Free original arrays to save memory
    del features_array, target_array
    gc.collect()
    print_memory_usage("After split cleanup")
    
    # Step 5: Scale the data using chunked processing
    n_features = train_X.shape[2]  # Last dimension is features
    train_X, test_X, scaler = scale_data_multiclass_chunked(
        train_X, test_X, n_features, chunk_size=config.chunk_size
    )
    print_memory_usage("After scaling")
    
    # Step 6: Apply robust preprocessing
    train_X, test_X = robust_preprocessing(train_X, test_X)
    print_memory_usage("After preprocessing")
    
    # Flatten target arrays
    train_y = train_y.flatten()
    test_y = test_y.flatten()
    
    # Step 7: Create streaming datasets for memory efficiency
    print("Creating streaming datasets...")
    
    def create_streaming_dataset(X, y, batch_size, shuffle=False):
        """Create memory-efficient streaming dataset"""
        with tf.device('/CPU:0'):
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            if shuffle:
                buffer_size = min(5000, len(X))
                dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    train_dataset = create_streaming_dataset(train_X, train_y, config.batch_size, shuffle=True)
    test_dataset = create_streaming_dataset(test_X, test_y, config.batch_size, shuffle=False)
    
    # Calculate actual steps per epoch
    full_train_steps = max(1, len(train_X) // config.batch_size)
    full_val_steps = max(1, len(test_X) // config.batch_size)
    
    train_steps_per_epoch = min(config.steps_per_epoch, full_train_steps)
    val_steps_per_epoch = min(config.validation_steps, full_val_steps)
    
    print(f"Dataset sizes - Train: {len(train_X):,}, Test: {len(test_X):,}")
    print(f"Batch size: {config.batch_size}")
    print(f"Training steps per epoch: {train_steps_per_epoch}")
    print(f"Validation steps per epoch: {val_steps_per_epoch}")
    
    print_memory_usage("After dataset creation")
    
    # Get list of models to train
    model_list = config.selected_models
    
    # Step 8: Iterate through models
    for model_name in model_list:
        print(f"\n\n{'='*50}")
        print(f"🚀 Training model: {model_name} for direction_3class_5 prediction")
        print(f"{'='*50}")
        print_memory_usage(f"Start {model_name}")
        
        try:
            # Clear any existing models from memory
            tf.keras.backend.clear_session()
            gc.collect()
            print_memory_usage(f"After session clear - {model_name}")
            
            # Log model training start
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n\n## Model: {model_name} (direction_3class_5 multi-class)\n")
                log_file.write("=" * 50 + "\n")
            
            # Get model builder and callbacks
            model_builder = MODEL_BUILDERS[model_name]
            callbacks = create_callbacks()
            custom_builder = get_custom_model_builder(model_name, model_builder, train_X, num_classes=3)
            
            # Step 9: Hyperparameter tuning
            print("\nStarting hyperparameter search...")
            tuner = kt.BayesianOptimization(
                hypermodel=custom_builder,
                objective='val_loss',
                max_trials=config.max_trials,  
                directory='models',
                project_name=f'{target_config["project_name_suffix"]}_trials_{model_name}',
                overwrite=True,
                executions_per_trial=1,
                max_consecutive_failed_trials=config.max_consecutive_failed_trials,
                seed=config.random_seed
            )

            tuner_success = True
            try:
                tuner.search(
                    train_dataset,
                    epochs=5,  # Reduced epochs for faster tuning
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=test_dataset,
                    validation_steps=val_steps_per_epoch,
                    callbacks=create_callbacks(),
                    class_weight=class_weight_dict,
                    verbose=1 
                )
                print_memory_usage(f"After hyperparameter search - {model_name}")
            except Exception as e:
                print(f"Warning: Hyperparameter search error: {e}")
                print(f"Skipping model {model_name} due to hyperparameter search failure")
                tuner_success = False
                
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"\n❌ Error in hyperparameter search for {model_name}: {str(e)}\n")
                
                continue
            
            # Step 10: Train final model
            if tuner_success:
                best_hps = tuner.get_best_hyperparameters(1)
                if best_hps:
                    best_hp = best_hps[0]
                else:
                    print(f"No valid hyperparameters found for {model_name}. Moving to next model.")
                    continue
                
                # Step 11: Train final model
                print("\nTraining final model...")
                
                # Build model with best hyperparameters
                final_model = custom_builder(best_hp)
                if final_model is None:
                    print(f"Failed to build final model for {model_name}")
                    continue
                
                # Train model
                history = final_model.fit(
                    train_dataset,
                    epochs=config.max_epochs,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=test_dataset,
                    validation_steps=val_steps_per_epoch,
                    callbacks=create_callbacks(),
                    class_weight=class_weight_dict,
                    verbose=1
                )
                
                print_memory_usage(f"After training - {model_name}")
                
                # Print best hyperparameters
                print("\nBest Hyperparameters:")
                for param in best_hp.values:
                    print(f"- {param}: {best_hp.values[param]}")
                
                # Step 12: Evaluate the model
                print(f"📊 Evaluating {model_name} model for direction_3class_5...")
                evaluation = final_model.evaluate(test_dataset, verbose=1)
                
                # Step 13: Save the model
                model_path = f'{target_config["model_save_dir"]}/{model_name}.h5'
                try:
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    final_model.save(model_path)
                    print(f"Model saved to {model_path}")
                except Exception as e:
                    print(f"Error saving model: {e}")
                
                # Log results
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"\n📌 Model: {model_name} (direction_3class_5 multi-class)\n")
                    log_file.write("=" * 40 + "\n")
                    log_file.write("Hyperparameters:\n")
                    for param in best_hp.values:
                        log_file.write(f"- {param}: {best_hp.values[param]}\n")

                    log_file.write("\nTest Metrics:\n")
                    for i, metric in enumerate(final_model.metrics_names):
                        log_file.write(f"{metric}: {evaluation[i]:.4f}\n")
                    
                    log_file.write("=" * 80 + "\n\n")
        
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n❌ Error training {model_name}: {str(e)}\n")
                log_file.write("=" * 80 + "\n")
        
        finally:
            # Clean up after each model to free memory
            tf.keras.backend.clear_session()
            gc.collect()
            print_memory_usage(f"After cleanup - {model_name}")
    
    print(f"\n✅ All direction_3class_5 models trained! Results saved to {log_file_path}")
    print_memory_usage("End training")

# ======================================
# MAIN EXECUTION
# ======================================
if __name__ == "__main__":
    # Parse command line arguments for GPU selection
    args = parse_gpu_args()
    
    print(f"🎯 MULTI-CLASS DIRECTION_3CLASS_5 TRAINING")
    print(f"Selected GPUs: {args.gpus}")
    if args.memory_limit:
        print(f"Memory limit per GPU: {args.memory_limit} MB")
    print("="*80)
    
    # Configure TensorFlow with selected GPUs
    device_type, device_count = setup_tensorflow(args.gpus, args.memory_limit)
    
    # Initialize config
    config = MultiClassConfigManager()
    n_in = config.lookback_window
    
    # Verify what device TensorFlow will use
    print("\nFinal device configuration check:")
    print(f"Available devices: {tf.config.list_physical_devices()}")
    
    # Start training
    print(f"\n🎯 Starting multi-class direction_3class_5 training...")
    print(f"Target: direction_3class_5 (3 classes: 0, 1, 2)")
    print(f"Dataset: {config.target_config['data_file']}")
    print(f"Models: {config.selected_models}")
    
    train_multiclass_models(config, args.gpus)