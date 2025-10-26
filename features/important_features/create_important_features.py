import pandas as pd
import numpy as np
import glob
import os

def log(message, level="INFO"):
    """Simple logging function"""
    print(f"[{level}] {message}")

def create_direction_targets(df, periods=[1, 3, 5, 10, 14]):
    """
    Create price direction targets for different time horizons
    """
    log("Creating price direction targets")
    df = df.copy()
    
    for period in periods:
        # Future close price after 'period' bars
        future_close = df['Close'].shift(-period)
        
        # Calculate returns
        returns = (future_close - df['Close']) / df['Close']
        
        # Binary direction (1 = up, 0 = down)
        # Handle NaN values by using fillna before astype
        direction_binary = (returns > 0).fillna(False).astype(int)
        df[f'direction_{period}'] = direction_binary
        
        # Multi-class direction with neutral zone
        # Use pd.cut and handle NaN values properly
        direction_3class = pd.cut(
            returns, 
            bins=[-np.inf, -0.0005, 0.0005, np.inf],  # 0.05% threshold
            labels=[0, 1, 2],  # 0=down, 1=neutral, 2=up
            include_lowest=True
        )
        
        # Convert to numeric, handling NaN values
        # Fill NaN with 1 (neutral) or drop rows later
        direction_3class_numeric = pd.to_numeric(direction_3class, errors='coerce')
        direction_3class_numeric = direction_3class_numeric.fillna(1).astype(int)
        df[f'direction_3class_{period}'] = direction_3class_numeric
        
        # Add the actual returns for analysis
        df[f'returns_{period}'] = returns
        
    # Remove rows with NaN values in the original returns columns
    # This is safer than trying to predict with incomplete data
    returns_cols = [col for col in df.columns if col.startswith('returns_')]
    direction_cols = [col for col in df.columns if col.startswith('direction_')]
    
    # Count NaN values before dropping
    nan_counts = df[returns_cols + direction_cols].isnull().sum()
    total_nans = nan_counts.sum()
    
    if total_nans > 0:
        log(f"Found {total_nans} NaN values in direction/returns columns")
        log(f"NaN counts by column: {dict(nan_counts[nan_counts > 0])}")
    
    # Drop rows where any of the key columns have NaN
    # Keep rows where we have valid direction targets
    initial_length = len(df)
    df = df.dropna(subset=direction_cols)
    final_length = len(df)
    
    log(f"Created direction targets for periods: {periods}")
    log(f"Rows after removing NaN: {final_length} (removed {initial_length - final_length} rows)")
    
    # Log the distribution of direction targets
    for period in periods:
        binary_col = f'direction_{period}'
        multiclass_col = f'direction_3class_{period}'
        
        if binary_col in df.columns:
            binary_dist = df[binary_col].value_counts().sort_index()
            log(f"{binary_col} distribution: {dict(binary_dist)}")
        
        if multiclass_col in df.columns:
            multi_dist = df[multiclass_col].value_counts().sort_index()
            log(f"{multiclass_col} distribution: {dict(multi_dist)}")
    
    return df

def main():
    # Define the directory
    directory = '.'  # or provide full path if not in the same dir

    # Columns that should always be included
    base_columns = [
        'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'datetime', 'long_signal', 'short_signal', 'close_position'
    ]

    # Direction target columns that will be created
    direction_target_columns = [
        # Binary direction targets
        'direction_1', 'direction_3', 'direction_5', 'direction_10', 'direction_14',
        
        # Multi-class direction targets (with neutral zone)
        'direction_3class_1', 'direction_3class_3', 'direction_3class_5', 
        'direction_3class_10', 'direction_3class_14',
        
        # Returns for regression
        'returns_1', 'returns_3', 'returns_5', 'returns_10', 'returns_14'
    ]

    # Find the main dataset (assumes only one CSV file in the directory)
    dataset_files = glob.glob(os.path.join(directory, '*.csv'))
    
    # Filter out any previously created filtered files
    dataset_files = [f for f in dataset_files if '_filtered.csv' not in f]
    
    if len(dataset_files) != 1:
        raise Exception(f"There should be exactly one main CSV file in the directory. Found: {dataset_files}")
    
    dataset_path = dataset_files[0]
    log(f"Loading main dataset: {dataset_path}")
    
    # Load the main dataset
    df = pd.read_csv(dataset_path)
    log(f"Original dataset shape: {df.shape}")
    
    # Ensure datetime column exists for proper time series handling
    if 'datetime' not in df.columns:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        elif 'Date' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'])
        else:
            log("Warning: No datetime information found. Creating sequential datetime.", "WARNING")
            df['datetime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='1min')
    
    # Sort by datetime to ensure proper order for direction targets
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Create direction targets ONCE and store them in memory
    log("Creating direction targets (once for all datasets)...")
    df_with_targets = create_direction_targets(df, periods=[1, 3, 5, 10, 14])
    log(f"Dataset shape after adding direction targets: {df_with_targets.shape}")
    
    # Extract just the direction target columns to reuse
    direction_data = df_with_targets[direction_target_columns].copy()
    log(f"Direction targets created and stored in memory: {direction_data.shape}")
    
    # Keep the original dataframe for feature selection (without direction targets to save memory)
    df_original = df.copy()
    
    # Add direction target columns to base columns
    all_base_columns = base_columns + direction_target_columns
    
    # Process each important feature file
    feature_files = glob.glob(os.path.join(directory, '*_important_features.txt'))
    
    if not feature_files:
        log("No important features files found. Creating dataset with base columns + direction targets only.", "WARNING")
        feature_files = ['base_features_only']  # Dummy entry to create at least one output
    
    for feature_file in feature_files:
        if feature_file == 'base_features_only':
            # Special case: create dataset with just base columns + direction targets
            features = []
            output_name = 'base_with_direction_targets_EURUSD_1min_filtered.csv'
        else:
            # Read features from file
            with open(feature_file, 'r') as f:
                features = [line.strip() for line in f.readlines() if line.strip()]
            
            # Create output file name
            base_name = os.path.basename(feature_file).replace('_important_features.txt', '_EURUSD_1min_filtered.csv')
            output_name = base_name

        # Combine all desired columns (excluding direction targets from selection since we'll add them separately)
        base_and_feature_columns = base_columns + features
        
        # Filter to only include columns that actually exist in the original dataframe
        available_base_feature_columns = [col for col in base_and_feature_columns if col in df_original.columns]
        missing_columns = [col for col in base_and_feature_columns if col not in df_original.columns]
        
        if missing_columns:
            log(f"Warning: Missing columns in dataset: {missing_columns}", "WARNING")
        
        log(f"Selected {len(available_base_feature_columns)} base+feature columns for {output_name}")
        log(f"Base+Feature columns: {available_base_feature_columns[:10]}..." if len(available_base_feature_columns) > 10 else f"Base+Feature columns: {available_base_feature_columns}")
        
        # Create filtered dataset by combining base+features with pre-computed direction targets
        df_base_features = df_original[available_base_feature_columns].copy()
        
        # Add the pre-computed direction targets
        df_filtered = pd.concat([df_base_features, direction_data], axis=1)
        
        # Ensure we only keep rows that have valid direction targets (same filtering as original)
        df_filtered = df_filtered.dropna(subset=direction_target_columns)
        
        log(f"Final dataset shape after adding direction targets: {df_filtered.shape}")
        
        # Create output path
        output_path = os.path.join(directory, output_name)
        
        # Save filtered dataset
        df_filtered.to_csv(output_path, index=False)
        log(f"Saved filtered dataset: {output_path} (shape: {df_filtered.shape})")
        
        # Log some statistics about the direction targets
        direction_stats = {}
        for col in direction_target_columns:
            if col.startswith('direction_') and not col.startswith('direction_3class_'):
                if col in df_filtered.columns:
                    up_pct = (df_filtered[col] == 1).mean() * 100
                    direction_stats[col] = f"{up_pct:.1f}% up"
        
        if direction_stats:
            log(f"Direction target statistics for {output_name}:")
            for col, stat in direction_stats.items():
                log(f"  {col}: {stat}")

    log("Filtered datasets with direction targets created successfully!")

if __name__ == "__main__":
    main()