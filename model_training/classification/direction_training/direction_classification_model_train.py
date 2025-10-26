# %%
import os
import sys

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

# ======================================
# GPU SELECTION AND CONFIGURATION
# ======================================

def list_available_gpus():
    """
    List all available GPUs with detailed information
    """
    print("🔍 Detecting available GPUs...")
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            print("❌ No GPUs detected")
            return []
        
        print(f"✅ Detected {len(gpus)} GPU(s):")
        gpu_info = []
        
        for i, gpu in enumerate(gpus):
            try:
                # Get GPU details
                gpu_details = tf.config.experimental.get_device_details(gpu)
                gpu_name = gpu_details.get('device_name', 'Unknown GPU')
                compute_capability = gpu_details.get('compute_capability', 'Unknown')
                
                print(f"  GPU {i}: {gpu_name}")
                print(f"    Device: {gpu}")
                print(f"    Compute Capability: {compute_capability}")
                
                # Try to get memory info
                try:
                    # Enable memory growth first to get memory info
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Create a temporary context to get memory info
                    with tf.device(gpu.name):
                        # Force GPU initialization
                        _ = tf.constant([1.0])
                    
                    mem_info = tf.config.experimental.get_memory_info(gpu.name.split('/')[-1])
                    total_memory_gb = mem_info['current'] / (1024**3)
                    print(f"    Memory: {total_memory_gb:.1f} GB available")
                    
                except Exception as mem_e:
                    print(f"    Memory: Unable to query ({str(mem_e)[:50]}...)")
                
                gpu_info.append({
                    'id': i,
                    'name': gpu_name,
                    'device': gpu,
                    'compute_capability': compute_capability
                })
                
            except Exception as e:
                print(f"  GPU {i}: {gpu} (Error getting details: {e})")
                gpu_info.append({
                    'id': i,
                    'name': f'GPU {i}',
                    'device': gpu,
                    'compute_capability': 'Unknown'
                })
        
        return gpu_info
        
    except Exception as e:
        print(f"❌ Error detecting GPUs: {e}")
        return []

def get_gpu_selection(gpu_info):
    """
    Get user input for GPU selection
    """
    if not gpu_info:
        print("No GPUs available for selection")
        return [], 'cpu'
    
    print(f"\n🎯 GPU Selection Options:")
    print("="*50)
    print("Available options:")
    print("  0: Use CPU only")
    print("  1: Use single GPU (you'll select which one)")
    print("  2: Use multiple GPUs (you'll select which ones)")
    print("  3: Use all available GPUs")
    print("  4: Auto-select best GPU")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (0-4): ").strip()
            
            if choice == '0':
                return [], 'cpu'
            
            elif choice == '1':
                print(f"\nSelect a single GPU:")
                for gpu in gpu_info:
                    print(f"  {gpu['id']}: {gpu['name']}")
                
                while True:
                    try:
                        gpu_id = int(input(f"Enter GPU ID (0-{len(gpu_info)-1}): "))
                        if 0 <= gpu_id < len(gpu_info):
                            return [gpu_info[gpu_id]['device']], 'single'
                        else:
                            print(f"Invalid GPU ID. Please enter 0-{len(gpu_info)-1}")
                    except ValueError:
                        print("Please enter a valid number")
            
            elif choice == '2':
                print(f"\nSelect multiple GPUs (enter comma-separated IDs):")
                for gpu in gpu_info:
                    print(f"  {gpu['id']}: {gpu['name']}")
                
                while True:
                    try:
                        gpu_ids_str = input(f"Enter GPU IDs (e.g., 0,1,2): ").strip()
                        gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',')]
                        
                        if all(0 <= gpu_id < len(gpu_info) for gpu_id in gpu_ids):
                            if len(set(gpu_ids)) != len(gpu_ids):
                                print("Duplicate GPU IDs detected. Please enter unique IDs.")
                                continue
                            selected_gpus = [gpu_info[gpu_id]['device'] for gpu_id in gpu_ids]
                            return selected_gpus, 'multi'
                        else:
                            print(f"Invalid GPU ID(s). Please enter IDs between 0-{len(gpu_info)-1}")
                    except ValueError:
                        print("Please enter valid comma-separated numbers")
            
            elif choice == '3':
                selected_gpus = [gpu['device'] for gpu in gpu_info]
                return selected_gpus, 'all'
            
            elif choice == '4':
                # Auto-select the first GPU (or implement more sophisticated logic)
                print("Auto-selecting the first available GPU...")
                return [gpu_info[0]['device']], 'auto'
            
            else:
                print("Invalid choice. Please enter 0, 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\nSelection cancelled. Using CPU.")
            return [], 'cpu'
        except Exception as e:
            print(f"Error: {e}. Please try again.")

def configure_selected_gpus(selected_gpus, selection_mode):
    """
    Configure TensorFlow to use selected GPUs with enhanced error handling
    """
    print(f"\n🔧 Configuring TensorFlow for {selection_mode} GPU usage...")
    
    try:
        if not selected_gpus:
            # CPU configuration
            print("Configuring for CPU-only execution...")
            tf.config.set_visible_devices([], 'GPU')
            
            # Optimize CPU performance
            tf.config.threading.set_inter_op_parallelism_threads(0)
            tf.config.threading.set_intra_op_parallelism_threads(0)
            
            print("✅ CPU configuration complete")
            return 'CPU', 0
        
        # GPU configuration
        print(f"Configuring {len(selected_gpus)} GPU(s)...")
        
        # Set visible devices to only the selected GPUs
        tf.config.set_visible_devices(selected_gpus, 'GPU')
        
        # Configure each selected GPU
        for i, gpu in enumerate(selected_gpus):
            print(f"Configuring GPU {i}: {gpu}")
            
            try:
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Optional: Set memory limit (uncomment and adjust as needed)
                # tf.config.experimental.set_memory_limit(gpu, 40960)  # 40GB limit for H100
                
                print(f"  ✅ Memory growth enabled for GPU {i}")
                
            except RuntimeError as e:
                if "memory growth" in str(e).lower() or "already initialized" in str(e).lower():
                    print(f"  ⚠️  GPU {i} already configured")
                else:
                    print(f"  ❌ Error configuring GPU {i}: {e}")
                    return False, 0
        
        # Test GPU functionality
        print("Testing GPU functionality...")
        try:
            with tf.device(selected_gpus[0].name.replace('physical_device:', '')):
                test_tensor = tf.constant([[1.0, 2.0]], dtype=tf.float32)
                result = tf.reduce_sum(test_tensor)
                _ = result.numpy()
                print("✅ GPU test successful")
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            print("Falling back to CPU...")
            return configure_selected_gpus([], 'cpu')
        
        # Setup distributed strategy for multiple GPUs
        strategy = None
        if len(selected_gpus) > 1:
            print(f"Setting up distributed training strategy for {len(selected_gpus)} GPUs...")
            try:
                strategy = tf.distribute.MirroredStrategy()
                print(f"✅ MirroredStrategy created with {strategy.num_replicas_in_sync} replicas")
            except Exception as e:
                print(f"⚠️  Failed to create distributed strategy: {e}")
                print("Continuing with single GPU...")
                strategy = None
        
        return f'GPU_{len(selected_gpus)}', len(selected_gpus), strategy
        
    except Exception as e:
        print(f"❌ GPU configuration failed: {e}")
        print("Falling back to CPU...")
        return configure_selected_gpus([], 'cpu')

def setup_tensorflow_with_gpu_selection():
    """
    Enhanced TensorFlow setup with user GPU selection
    """
    print("🚀 Setting up TensorFlow with GPU selection...")
    print("="*60)
    
    # Step 1: List available GPUs
    gpu_info = list_available_gpus()
    
    # Step 2: Get user selection
    selected_gpus, selection_mode = get_gpu_selection(gpu_info)
    
    # Step 3: Configure TensorFlow
    if len(selected_gpus) > 1:
        device_type, device_count, strategy = configure_selected_gpus(selected_gpus, selection_mode)
        return device_type, device_count, strategy
    else:
        result = configure_selected_gpus(selected_gpus, selection_mode)
        if len(result) == 3:
            device_type, device_count, strategy = result
            return device_type, device_count, strategy
        else:
            device_type, device_count = result
            return device_type, device_count, None

# Configure TensorFlow with GPU selection
device_setup = setup_tensorflow_with_gpu_selection()
if len(device_setup) == 3:
    device_type, device_count, distributed_strategy = device_setup
else:
    device_type, device_count = device_setup
    distributed_strategy = None

# Import all classification models
from classification_model_build import (
    build_LSTM_classifier,
    build_GRU_classifier,
    build_Conv1D_classifier,
    build_Conv1D_LSTM_classifier,
    # build_BiLSTM_Attention_classifier,
    # build_Transformer_classifier,
    # build_MultiStream_classifier,
    # build_ResNet_classifier,
    # build_TCN_classifier,
    # build_CatBoostClassifier_model,
    # build_LightGBMClassifier_model,
    # build_XGBoostClassifier_model,
    # build_RandomForestClassifier_model
)

# ======================================
# MULTI-TARGET CONFIGURATION MANAGEMENT
# ======================================
class MultiTargetConfigManager:
    """
    Configuration manager for multiple direction classification targets
    """
    def __init__(self):
        # Common settings
        self.lookback_window = 60
        self.test_size = 0.2
        self.random_seed = 42
        
        # Memory optimization settings
        self.max_samples = 1000000
        self.chunk_size = 20000
        self.use_streaming = True
        
        # Training parameters - adjust based on available GPUs
        if device_count > 1:
            # Scale batch size with number of GPUs
            self.batch_size = 512 * device_count
            self.max_epochs = 50  # Can afford more epochs with distributed training
            self.steps_per_epoch = 200
            self.validation_steps = 100
        else:
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
        
        # Hyperparameter tuning - adjust based on GPU count
        if device_count > 1:
            self.max_trials = 200  # More trials with distributed training
            self.max_consecutive_failed_trials = 50
        else:
            self.max_trials = 150
            self.max_consecutive_failed_trials = 40
        
        # Feature selection settings
        self.use_important_features_only = True
        self.exclude_future_looking = True
        
        # Multi-target configurations
        self.target_configs = self._create_target_configs()
        
        # Store distributed strategy
        self.distributed_strategy = distributed_strategy
    
    def _create_target_configs(self):
        """Create configurations for each target"""
        return {
            'direction_1': {
                'data_file': 'direction_1_EURUSD_1min_filtered.csv',
                'target_col': 'direction_1',
                'results_file': 'direction_1_model_training_results.txt',
                'model_save_dir': 'models/direction_1_models',
                'project_name_suffix': 'direction_1',
                'important_features': [
                    # Core technical indicators - safe for direction_1
                    'adx_trend_buy_signal', 'adx_trend_sell_signal', 'pzosx_buy_signal', 
                    'current_candle_height', 'atr_se_sell_signal', 'bb_short_entry_signal',
                    'camarilla_buy_signal', 'camarilla_sell_signal', 'cci_bullish_signal', 
                    'cci_bearish_signal', 'cmf_buy_signal', 'cmf_sell_signal',
                    'dpo_overbought_signal', 'dpo_buy_signal', 'dpo_sell_signal',
                    'ehlers_stoch_buy_signal', 'eight_month_avg_buy_signal', 'eight_month_avg_sell_signal',
                    'EMA_bullish_signal', 'eom_buy_signal', 'eom_sell_signal',
                    'gap_momentum_buy_signal', 'gap_momentum_sell_signal', 'gap_up_le_buy_signal',
                    'golden_cross_buy_signal', 'ironbot_buy_signal', 'ironbot_sell_signal',
                    'kama_cross_sell_signal', 'kc_buy_signal', 'kc_sell_signal',
                    'moving_average_buy_signal', 'pzo_lx_sell_signal', 'rocwb_buy_signal',
                    'RSI', 'spectrum_bars_buy_signal', 'stc_overbought_signal', 'stc_oversold_signal',
                    'stoch_buy_signal', 'stoch_sell_signal', 'stochastic_strat_buy_signal',
                    'stochrsi_overbought_signal', 'stochrsi_oversold_signal',
                    '5_8_13_buy_signal', '5_8_13_sell_signal', 'w5_8_13_buy_signal', 'w5_8_13_sell_signal',
                    'sve_ha_typ_cross_buy_signal', 'sve_ha_typ_cross_sell_signal',
                    'sve_zl_rb_perc_buy_signal', 'sve_zl_rb_perc_sell_signal',
                    'svesc_buy_signal', 'svesc_sell_signal', 'volatility_band_buy_signal',
                    'vols_switch_buy_signal', 'vols_switch_sell_signal', 'vpn_sell_signal',
                    'vwma_breakouts_buy_signal', 'williams_buy_signal', 'williams_sell_signal'
                ],
                'exclude_columns': [
                    # Target columns (keep direction_1 as target, exclude others)
                    'long_signal', 'short_signal', 'close_position',
                    # Time and price columns  
                    'Date', 'Time', 'datetime', 'Open', 'High', 'Low', 'Close',
                    # Future-looking features for direction_1
                    'direction_3', 'direction_5', 'direction_10', 'direction_14',
                    'direction_3class_1', 'direction_3class_3', 'direction_3class_5', 
                    'direction_3class_10', 'direction_3class_14',
                    'returns_1', 'returns_3', 'returns_5', 'returns_10', 'returns_14'
                ]
            },
            
            'direction_5': {
                'data_file': 'direction_5_EURUSD_1min_filtered.csv',
                'target_col': 'direction_5',
                'results_file': 'direction_5_model_training_results.txt',
                'model_save_dir': 'models/direction_5_models',
                'project_name_suffix': 'direction_5',
                'important_features': [
                    # Technical indicators
                    'adx_trend_buy_signal', 'adx_trend_sell_signal', 'pzosx_buy_signal',
                    'acc_dist_sell_signal', 'current_candle_height', 'bb_short_entry_signal',
                    'camarilla_buy_signal', 'camarilla_sell_signal', 'cci_bullish_signal',
                    'cci_bearish_signal', 'cci_sell_signal', 'cmf_buy_signal', 'cmf_sell_signal',
                    'dpo_overbought_signal', 'dpo_buy_signal', 'dpo_sell_signal',
                    'ehlers_stoch_buy_signal', 'eight_month_avg_buy_signal', 'eight_month_avg_sell_signal',
                    'EMA_bullish_signal', 'eom_buy_signal', 'eom_sell_signal',
                    'gap_momentum_buy_signal', 'gap_momentum_sell_signal', 'golden_cross_buy_signal',
                    'ironbot_buy_signal', 'ironbot_sell_signal', 'kama_cross_buy_signal',
                    'kama_cross_sell_signal', 'kc_buy_signal', 'kc_sell_signal',
                    'moving_average_buy_signal', 'pzo_lx_sell_signal', 'rocwb_buy_signal',
                    'rsi_overbought_signal', 'RSI', 'spectrum_bars_buy_signal',
                    'stc_overbought_signal', 'stc_oversold_signal', 'stoch_buy_signal',
                    'stoch_sell_signal', 'stochastic_strat_buy_signal', 'stochrsi_overbought_signal',
                    'stochrsi_oversold_signal', '5_8_13_buy_signal', '5_8_13_sell_signal',
                    'w5_8_13_buy_signal', 'w5_8_13_sell_signal', 'sve_ha_typ_cross_sell_signal',
                    'sve_zl_rb_perc_buy_signal', 'sve_zl_rb_perc_sell_signal',
                    'svesc_buy_signal', 'svesc_sell_signal', 'volatility_band_buy_signal',
                    'vols_switch_buy_signal', 'vols_switch_sell_signal', 'vpn_sell_signal',
                    'vwma_breakouts_buy_signal', 'williams_buy_signal', 'williams_sell_signal',
                    # Past direction features (valid for direction_5 prediction)
                    'direction_1', 'direction_3',
                    # Past direction class features
                    'direction_3class_1', 'direction_3class_3',
                    # Past returns (valid for direction_5 prediction)
                    'returns_1', 'returns_3'
                ],
                'exclude_columns': [
                    # Target columns (keep direction_5 as target, exclude others)
                    'long_signal', 'short_signal', 'close_position',
                    # Time and price columns  
                    'Date', 'Time', 'datetime', 'Open', 'High', 'Low', 'Close',
                    # Future-looking features for direction_5
                    'direction_10', 'direction_14',
                    'direction_3class_5', 'direction_3class_10', 'direction_3class_14',
                    'returns_5', 'returns_10', 'returns_14'
                ]
            },
            
            'direction_14': {
                'data_file': 'direction_14_EURUSD_1min_filtered.csv',
                'target_col': 'direction_14',
                'results_file': 'direction_14_model_training_results.txt',
                'model_save_dir': 'models/direction_14_models',
                'project_name_suffix': 'direction_14',
                'important_features': [
                    # Price features (valid for direction_14)
                    'Low',  # Note: excluding duplicate 'Date' as it's not numeric
                    # Technical indicators
                    'adx_trend_buy_signal', 'adx_trend_sell_signal', 'pzosx_buy_signal',
                    'current_candle_height', 'bb_short_entry_signal', 'camarilla_buy_signal',
                    'camarilla_sell_signal', 'cci_bullish_signal', 'cci_bearish_signal',
                    'cmf_buy_signal', 'cmf_sell_signal', 'dpo_overbought_signal',
                    'dpo_buy_signal', 'dpo_sell_signal', 'ehlers_stoch_buy_signal',
                    'eight_month_avg_buy_signal', 'eight_month_avg_sell_signal',
                    'EMA_bullish_signal', 'eom_buy_signal', 'eom_sell_signal',
                    'gap_momentum_buy_signal', 'gap_momentum_sell_signal', 'golden_cross_buy_signal',
                    'ironbot_buy_signal', 'ironbot_sell_signal', 'kama_cross_sell_signal',
                    'kc_buy_signal', 'kc_sell_signal', 'moving_average_buy_signal',
                    'pzo_lx_sell_signal', 'rocwb_buy_signal', 'RSI', 'spectrum_bars_buy_signal',
                    'stc_overbought_signal', 'stc_oversold_signal', 'stoch_buy_signal',
                    'stoch_sell_signal', 'stochastic_strat_buy_signal', 'stochrsi_overbought_signal',
                    'stochrsi_oversold_signal', '5_8_13_buy_signal', '5_8_13_sell_signal',
                    'w5_8_13_buy_signal', 'w5_8_13_sell_signal', 'sve_ha_typ_cross_buy_signal',
                    'sve_ha_typ_cross_sell_signal', 'sve_zl_rb_perc_buy_signal',
                    'sve_zl_rb_perc_sell_signal', 'svesc_buy_signal', 'svesc_sell_signal',
                    'volatility_band_buy_signal', 'vols_switch_buy_signal', 'vols_switch_sell_signal',
                    'vpn_sell_signal', 'vwma_breakouts_buy_signal', 'williams_buy_signal',
                    'williams_sell_signal',
                    # Past direction features (valid for direction_14 prediction)
                    'direction_1', 'direction_3', 'direction_5', 'direction_10',
                    # Past direction class features
                    'direction_3class_1', 'direction_3class_3', 'direction_3class_5', 'direction_3class_10',
                    # Past returns (valid for direction_14 prediction)
                    'returns_1', 'returns_3', 'returns_5', 'returns_10'
                ],
                'exclude_columns': [
                    # Target columns (keep direction_14 as target, exclude others)
                    'long_signal', 'short_signal', 'close_position',
                    # Time and price columns (excluding Low which is a valid feature)
                    'Date', 'Time', 'datetime', 'Open', 'High', 'Close',
                    # Future-looking features for direction_14 (none for the longest horizon)
                    'direction_3class_14', 'returns_14'
                ]
            }
        }

# Dictionary of model builders
MODEL_BUILDERS = {
    "LSTM": build_LSTM_classifier,
    "GRU": build_GRU_classifier,
    "Conv1D": build_Conv1D_classifier,
    "Conv1D_LSTM": build_Conv1D_LSTM_classifier,
    # "BiLSTM_Attention": build_BiLSTM_Attention_classifier,
    # "Transformer": build_Transformer_classifier,
    # "MultiStream": build_MultiStream_classifier,
    # "ResNet": build_ResNet_classifier,
    # "TCN": build_TCN_classifier,
    # "CatBoost" : build_CatBoostClassifier_model,
    # "LightGBM" : build_LightGBMClassifier_model,
    # "XGBOOST" : build_XGBoostClassifier_model,
    # "RandomForest" : build_RandomForestClassifier_model
}

# Initialize config
config = MultiTargetConfigManager()

# Make variables from config available globally
n_in = config.lookback_window

# ======================================
# DATA LOADING AND PREPROCESSING
# ======================================
def load_and_preprocess_data(config, target_name):
    """
    Load and preprocess the data with feature engineering for specific target prediction
    Memory-optimized version for large datasets with improved column handling
    """
    target_config = config.target_configs[target_name]
    
    print(f"Loading and preprocessing data for {target_name} prediction...")
    data = pd.read_csv(target_config['data_file'], header=0)
    print(f"Original dataset size: {len(data)} rows")
    
    # Handle duplicate columns
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
    
    # Get important features and exclusions for this target
    important_features = target_config['important_features']
    exclude_columns = target_config['exclude_columns']
    
    # Check which important features are actually present in the dataset
    available_important_features = [col for col in important_features if col in data.columns]
    missing_features = [col for col in important_features if col not in data.columns]
    
    if missing_features:
        print(f"Warning: Missing important features for {target_name}: {missing_features}")
    
    print(f"Using {len(available_important_features)} important features for {target_name}")
    
    # Use only the important features that are available and safe
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
        print(f"Adding lag features for technical indicators for {target_name}...")
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
    
    # Ensure target is binary (0 or 1)
    print(f"Target distribution before conversion: {target.value_counts()}")
    target = target.astype(int)
    print(f"Target distribution after conversion: {target.value_counts()}")
    
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
    
    print(f"Final dataset shape for {target_name} - Features: {features.shape}, Target: {target.shape}")
    print(f"Memory usage: Features: {features.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    return features, target

# ======================================
# DATA SCALING
# ======================================
def scale_data_classification_chunked(train_X, test_X, n_features, chunk_size=10000):
    """
    Scale the data using RobustScaler for classification with chunked processing
    to avoid memory issues with large datasets
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
def create_sliding_windows_classification_chunked(features, target, n_in, chunk_size=10000):
    """
    Create sliding windows for time series classification using chunked processing
    to avoid memory issues with large datasets
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
# FOCAL LOSS FOR NEURAL NETWORKS
# ======================================
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        # Use a larger epsilon to prevent numerical issues
        epsilon = 1e-5
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Use a more stable implementation
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1 - alpha)
        
        # Take log and apply gamma factor
        loss = -alpha_t * tf.pow(1. - pt, gamma) * tf.math.log(pt + epsilon)
        
        # Replace any NaN or inf with zeros
        loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
        
        return tf.reduce_mean(loss)
    
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
    def __init__(self, baseline=0.65, min_epoch=5):
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

class GPUMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for i, gpu in enumerate(gpus):
                    try:
                        mem_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                        logs[f'gpu_{i}_used_memory'] = mem_info['current'] / (1024**3)  # Convert to GB
                        print(f"\nGPU {i} Memory Used: {logs[f'gpu_{i}_used_memory']:.2f} GB")
                    except:
                        pass
        except Exception as e:
            print(f"Error monitoring GPU memory: {e}")

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
        EarlyStopper(baseline=0.65),
        NaNSafetyCallback(),
        GPUMemoryCallback()
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
def get_custom_model_builder(model_name, model_builder, train_X, distributed_strategy=None):
    def custom_model_builder(hp):
        try:
            if distributed_strategy:
                # Create model within distributed strategy scope
                with distributed_strategy.scope():
                    # Special handling for models that need explicit input shape
                    if model_name in ["MultiStream", "ResNet", "TCN"]:
                        # These models need explicit input shape - (timesteps, features)
                        input_shape = (n_in, train_X.shape[2])
                        model = model_builder(hp, input_shape=input_shape, num_classes=2)
                    else:
                        # Other models work with default parameters
                        model = model_builder(hp, num_classes=2)
                        
                    if model is None:
                        print(f"Model builder for {model_name} returned None")
                        return None

                    optimizer = tf.keras.optimizers.Adam(
                        hp.Float("learning_rate", min_value=1e-6, max_value=5e-4, sampling="log"),
                        clipvalue=1.0  # Clip gradients to avoid exploding values
                    )
                    model.compile(
                        optimizer=optimizer,
                        loss=focal_loss(
                            gamma=hp.Float("focal_gamma", min_value=1.0, max_value=3.0, step=0.5, default=2.0),
                            alpha=hp.Float("focal_alpha", min_value=0.1, max_value=0.9, step=0.1, default=0.25)
                        ),
                        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
                        jit_compile=True  # Enable XLA compilation
                    )
            else:
                # Single GPU or CPU model creation
                # Special handling for models that need explicit input shape
                if model_name in ["MultiStream", "ResNet", "TCN"]:
                    # These models need explicit input shape - (timesteps, features)
                    input_shape = (n_in, train_X.shape[2])
                    model = model_builder(hp, input_shape=input_shape, num_classes=2)
                else:
                    # Other models work with default parameters
                    model = model_builder(hp, num_classes=2)
                    
                if model is None:
                    print(f"Model builder for {model_name} returned None")
                    return None

                optimizer = tf.keras.optimizers.Adam(
                    hp.Float("learning_rate", min_value=1e-6, max_value=5e-4, sampling="log"),
                    clipvalue=1.0  # Clip gradients to avoid exploding values
                )
                model.compile(
                    optimizer=optimizer,
                    loss=focal_loss(
                        gamma=hp.Float("focal_gamma", min_value=1.0, max_value=3.0, step=0.5, default=2.0),
                        alpha=hp.Float("focal_alpha", min_value=0.1, max_value=0.9, step=0.1, default=0.25)
                    ),
                    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
                    jit_compile=True  # Enable XLA compilation
                )
            
            return model
        except Exception as e:
            print(f"Error in custom model builder for {model_name}: {e}")
            return None

    return custom_model_builder

def cross_validate_best_models(tuner, train_X, train_y, test_X, test_y, model_name, class_weight_dict, n_folds=5):
    """
    Cross-validate the top models from tuning to select the most robust one for classification
    """
    print(f"Running {n_folds}-fold time series cross-validation...")
    
    # Get ALL hyperparameter configurations that completed successfully
    try:
        all_trials = tuner.oracle.get_best_trials(num_trials=100)
        
        # Filter to only include trials that didn't result in NaN losses
        successful_trials = [
            trial for trial in all_trials 
            if trial.metrics.get_last_value('val_loss') is not None and 
            not np.isnan(trial.metrics.get_last_value('val_loss'))
        ]
        
        print(f"Found {len(successful_trials)} successful trials out of {len(all_trials)} total")
        
        # Need at least 3 successful hyperparameter sets
        if len(successful_trials) < 3:
            print("Not enough successful trials for cross-validation. Using best trial directly.")
            best_hps = tuner.get_best_hyperparameters(1)
            return best_hps[0] if best_hps else None, []
        
        # Get top hyperparameter configurations from successful trials
        top_hps = [trial.hyperparameters for trial in successful_trials[:3]]
        
    except Exception as e:
        print(f"Error getting hyperparameters: {e}")
        return None, []
    
    # Setup TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_folds)
    
    cv_results = []
    
    # For each hyperparameter configuration
    for hp_idx, hp in enumerate(top_hps):
        print(f"\nValidating hyperparameter set {hp_idx+1}/{len(top_hps)}")
        
        fold_metrics = {
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_auc': []
        }
        
        nan_count = 0
        all_folds_successful = True
        
        # Run time series cross-validation
        for fold, (train_idx, val_idx) in enumerate(tscv.split(train_X)):
            print(f"  Fold {fold+1}/{n_folds}")
            
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Split data
                    X_train_fold, X_val_fold = train_X[train_idx], train_X[val_idx]
                    y_train_fold, y_val_fold = train_y[train_idx], train_y[val_idx]
                    
                    # Get custom model builder
                    custom_builder = get_custom_model_builder(model_name, MODEL_BUILDERS[model_name], train_X, config.distributed_strategy)
                    
                    # Build model with more stable options
                    model = custom_builder(hp)
                    if model is None:
                        print(f"  Model creation failed for fold {fold+1}")
                        all_folds_successful = False
                        break
                    
                    # Train with early stopping and NaN detection
                    history = model.fit(
                        X_train_fold, y_train_fold,
                        epochs=25,
                        batch_size=64,
                        validation_data=(X_val_fold, y_val_fold),
                        callbacks=create_callbacks(),
                        class_weight=class_weight_dict,
                        verbose=0
                    )
                    
                    # If we get here, training succeeded
                    if history.history.get('val_loss'):
                        best_epoch_idx = np.argmin(history.history['val_loss'])
                        for metric in fold_metrics.keys():
                            if metric in history.history:
                                fold_metrics[metric].append(history.history[metric][best_epoch_idx])
                    
                    break
                    
                except Exception as e:
                    print(f"  Retry {retry+1}/{max_retries}: Error in fold {fold+1}: {e}")
                    if retry == max_retries - 1:
                        all_folds_successful = False
                        nan_count += 1
                        break
                    tf.keras.backend.clear_session()
            
            if not all_folds_successful:
                break
        
        # Only consider hyperparameter set if at least half the folds were successful
        if nan_count < n_folds // 2:
            cv_summary = {}
            for metric, values in fold_metrics.items():
                if values:
                    cv_summary[f'{metric}_mean'] = np.mean(values)
                    cv_summary[f'{metric}_std'] = np.std(values)
            
            cv_summary['nan_folds'] = nan_count
            cv_summary['valid_folds'] = n_folds - nan_count
            cv_summary['hyperparameters'] = hp
            
            cv_results.append(cv_summary)
            
            val_loss_mean = cv_summary.get('val_loss_mean', 'N/A')
            val_acc_mean = cv_summary.get('val_accuracy_mean', 'N/A')
            print(f"  Results: val_loss_mean={val_loss_mean:.4f}, val_acc_mean={val_acc_mean:.4f}")
        else:
            print(f"  ❌ Too many failed folds for this hyperparameter set - skipping")
    
    # Select the best hyperparameters based on cross-validation results
    if cv_results:
        # Sort by mean validation loss (ascending)
        cv_results.sort(key=lambda x: x.get('val_loss_mean', float('inf')))
        best_cv_result = cv_results[0]
        best_hp = best_cv_result['hyperparameters']
        
        print(f"\nBest cross-validated model:")
        print(f"- val_loss: {best_cv_result.get('val_loss_mean', 'N/A'):.4f} ± {best_cv_result.get('val_loss_std', 'N/A'):.4f}")
        print(f"- val_accuracy: {best_cv_result.get('val_accuracy_mean', 'N/A'):.4f} ± {best_cv_result.get('val_accuracy_std', 'N/A'):.4f}")
        print(f"- val_auc: {best_cv_result.get('val_auc_mean', 'N/A'):.4f} ± {best_cv_result.get('val_auc_std', 'N/A'):.4f}")
        
        return best_hp, cv_results
    else:
        print("No valid hyperparameter sets found during cross-validation")
        return None, []

# ======================================
# SINGLE TARGET TRAINING PIPELINE
# ======================================
def train_single_target_models(config, target_name):
    """
    Train multiple classification models for a single target with memory optimization
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
        else:
            print(f"[{step_name}] Memory monitoring not available")
            
        # Check GPU memory if available
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for i, gpu in enumerate(gpus):
                    try:
                        mem_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                        gpu_used = mem_info['current'] / (1024**3)
                        gpu_total = mem_info['peak'] / (1024**3)
                        print(f"[{step_name}] GPU {i} memory: {gpu_used:.1f}/{gpu_total:.1f} GB")
                    except:
                        pass
        except:
            pass
    
    target_config = config.target_configs[target_name]
    
    print(f"\n{'='*80}")
    print(f"🎯 TRAINING MODELS FOR TARGET: {target_name.upper()}")
    print(f"🔧 Device: {device_type} with {device_count} device(s)")
    if config.distributed_strategy:
        print(f"🔄 Distributed strategy: {config.distributed_strategy.num_replicas_in_sync} replicas")
    print(f"{'='*80}")
    print_memory_usage(f"Start {target_name}")
    
    # Set up logging
    log_file_path = target_config['results_file']
    with open(log_file_path, "w") as log_file:
        log_file.write(f"{target_name.upper()} Classification Models Training Results\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Device: {device_type} with {device_count} device(s)\n")
        if config.distributed_strategy:
            log_file.write(f"Distributed strategy: {config.distributed_strategy.num_replicas_in_sync} replicas\n")
        log_file.write("\n")
    
    # Step 1: Load and preprocess data
    features, target = load_and_preprocess_data(config, target_name)
    print_memory_usage(f"After data loading - {target_name}")
    
    # Step 2: Create sliding windows using chunked processing
    features_array, target_array = create_sliding_windows_classification_chunked(
        features, target, config.lookback_window, chunk_size=config.chunk_size
    )
    print_memory_usage(f"After window creation - {target_name}")
    
    # Free original data to save memory
    del features, target
    gc.collect()
    print_memory_usage(f"After cleanup - {target_name}")
    
    # Step 3: Compute class weights
    print("Computing class weights...")
    classes = np.array([0, 1]) 
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=target_array.flatten()
    )
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
    print(f"Computed class weights for {target_name}: {class_weight_dict}")
    
    # Step 4: Split into train/test sets
    print("Splitting data...")
    train_X, test_X, train_y, test_y = train_test_split(
        features_array, target_array, 
        test_size=config.test_size, 
        random_state=config.random_seed, 
        shuffle=False
    )
    print_memory_usage(f"After train/test split - {target_name}")
    
    # Free original arrays to save memory
    del features_array, target_array
    gc.collect()
    print_memory_usage(f"After split cleanup - {target_name}")
    
    # Step 5: Scale the data using chunked processing
    n_features = train_X.shape[2]  # Last dimension is features
    train_X, test_X, scaler = scale_data_classification_chunked(
        train_X, test_X, n_features, chunk_size=config.chunk_size
    )
    print_memory_usage(f"After scaling - {target_name}")
    
    # Step 6: Apply robust preprocessing
    train_X, test_X = robust_preprocessing(train_X, test_X)
    print_memory_usage(f"After preprocessing - {target_name}")
    
    # Flatten target arrays
    train_y = train_y.flatten()
    test_y = test_y.flatten()
    
    # Step 7: Create streaming datasets for memory efficiency
    print("Creating streaming datasets...")
    
    def create_streaming_dataset(X, y, batch_size, shuffle=False):
        """Create memory-efficient streaming dataset"""
        with tf.device('/CPU:0'):  # 🔧 Force all ops like tf.equal to run on CPU
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
    print(f"Full training steps available: {full_train_steps}")
    print(f"Limited training steps per epoch: {train_steps_per_epoch}")
    print(f"Full validation steps available: {full_val_steps}")
    print(f"Limited validation steps per epoch: {val_steps_per_epoch}")
    print(f"Samples per training epoch: {train_steps_per_epoch * config.batch_size:,}")
    print(f"Samples per validation epoch: {val_steps_per_epoch * config.batch_size:,}")
    
    if train_steps_per_epoch < full_train_steps:
        reduction_pct = (1 - train_steps_per_epoch / full_train_steps) * 100
        print(f"🚀 Training time reduced by {reduction_pct:.1f}% due to step limiting")
    
    print_memory_usage(f"After dataset creation - {target_name}")
    
    # Get list of models to train
    if config.selected_models:
        model_list = config.selected_models
    else:
        model_list = list(MODEL_BUILDERS.keys())
    
    # Step 8: Iterate through models
    for model_name in model_list:
        print(f"\n\n{'='*50}")
        print(f"🚀 Training model: {model_name} for {target_name} prediction")
        print(f"{'='*50}")
        print_memory_usage(f"Start {model_name} - {target_name}")
        
        try:
            # Clear any existing models from memory
            tf.keras.backend.clear_session()
            gc.collect()
            print_memory_usage(f"After session clear - {model_name} - {target_name}")
            
            # Log model training start
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n\n## Model: {model_name} ({target_name} target)\n")
                log_file.write("=" * 50 + "\n")
            
            # Get model builder and callbacks
            model_builder = MODEL_BUILDERS[model_name]
            callbacks = create_callbacks()
            custom_builder = get_custom_model_builder(model_name, model_builder, train_X, config.distributed_strategy)
            
            # Step 9: Hyperparameter tuning with reduced trials for memory efficiency
            print("\nStarting hyperparameter search...")
            print(f"Using {train_steps_per_epoch} steps per epoch (instead of {full_train_steps} full steps)")
            if train_steps_per_epoch < full_train_steps:
                reduction_pct = (1 - train_steps_per_epoch / full_train_steps) * 100
                print(f"This reduces training time by {reduction_pct:.1f}%")
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
                    steps_per_epoch=train_steps_per_epoch,  # Limit steps per epoch
                    validation_data=test_dataset,
                    validation_steps=val_steps_per_epoch,   # Limit validation steps
                    callbacks=create_callbacks(),
                    class_weight=class_weight_dict,
                    verbose=1 
                )
                print_memory_usage(f"After hyperparameter search - {model_name} - {target_name}")
            except Exception as e:
                print(f"Warning: Hyperparameter search error: {e}")
                print(f"Skipping model {model_name} due to hyperparameter search failure")
                tuner_success = False
                
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"\n❌ Error in hyperparameter search for {model_name}: {str(e)}\n")
                
                continue
            
            # Step 10: Cross-validation only if hyperparameter tuning succeeded
            if tuner_success:
                # Simplified cross-validation for memory efficiency
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
                    steps_per_epoch=train_steps_per_epoch,  # Limit steps per epoch
                    validation_data=test_dataset,
                    validation_steps=val_steps_per_epoch,   # Limit validation steps
                    callbacks=create_callbacks(),
                    class_weight=class_weight_dict,
                    verbose=1
                )
                
                print_memory_usage(f"After training - {model_name} - {target_name}")
                
                # Print best hyperparameters
                print("\nBest Hyperparameters:")
                for param in best_hp.values:
                    print(f"- {param}: {best_hp.values[param]}")
                
                # Step 12: Evaluate the model
                print(f"📊 Evaluating {model_name} model for {target_name}...")
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
                    log_file.write(f"\n📌 Model: {model_name} ({target_name} target)\n")
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
            print_memory_usage(f"After cleanup - {model_name} - {target_name}")
    
    print(f"\n✅ All {target_name} models trained! Results saved to {log_file_path}")
    print_memory_usage(f"End {target_name}")

# ======================================
# MAIN MULTI-TARGET TRAINING PIPELINE
# ======================================
def train_all_target_models(config):
    """
    Main function to train models for all targets sequentially
    """
    print("🎯 STARTING MULTI-TARGET DIRECTION CLASSIFICATION TRAINING")
    print("="*80)
    print(f"🔧 Running on: {device_type} with {device_count} device(s)")
    if config.distributed_strategy:
        print(f"🔄 Distributed training with {config.distributed_strategy.num_replicas_in_sync} replicas")
    print("="*80)
    
    # List of targets to train
    targets = ['direction_1', 'direction_5', 'direction_14']
    
    # Train models for each target
    for i, target_name in enumerate(targets, 1):
        print(f"\n\n🔄 PROCESSING TARGET {i}/{len(targets)}: {target_name}")
        print("="*80)
        
        try:
            # Train models for this target
            train_single_target_models(config, target_name)
            
            # Clear memory between targets
            import gc
            tf.keras.backend.clear_session()
            gc.collect()
            
            print(f"✅ Completed training for {target_name}")
            
        except Exception as e:
            print(f"❌ Error training models for {target_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Log the error and continue with next target
            try:
                target_config = config.target_configs[target_name]
                with open(target_config['results_file'], "a") as log_file:
                    log_file.write(f"\n❌ CRITICAL ERROR for {target_name}: {str(e)}\n")
                    log_file.write("=" * 80 + "\n")
            except:
                pass
    
    print(f"\n\n🎉 MULTI-TARGET TRAINING COMPLETE!")
    print("="*80)
    print(f"🔧 Ran on: {device_type} with {device_count} device(s)")
    if config.distributed_strategy:
        print(f"🔄 Used distributed training with {config.distributed_strategy.num_replicas_in_sync} replicas")
    print("📁 Results files created:")
    for target_name in targets:
        results_file = config.target_configs[target_name]['results_file']
        print(f"  - {target_name}: {results_file}")

# ======================================
# MAIN EXECUTION
# ======================================
if __name__ == "__main__":
    print("🎯 Starting multi-target direction classification training...")
    print("Targets: direction_1, direction_5, direction_14")
    print("Features: Trading signals and technical indicators (target-specific exclusions)")
    print(f"🔧 Configured to run on: {device_type} with {device_count} device(s)")
    if distributed_strategy:
        print(f"🔄 Distributed training enabled with {distributed_strategy.num_replicas_in_sync} replicas")
    
    # Start training all targets
    train_all_target_models(config)