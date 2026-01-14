"""
Diagnostic Script for Pattern Validation Model Failure
========================================================
This script helps identify why the model is failing.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def diagnose_training_data(data_dir):
    """
    Comprehensive diagnostics for training data
    """
    print("="*70)
    print("PATTERN VALIDATION MODEL - DATA DIAGNOSTICS")
    print("="*70)
    
    # Load data
    print("\n1. LOADING DATA...")
    X_chart = np.load(f'{data_dir}/X_chart.npy')
    X_volume = np.load(f'{data_dir}/X_volume.npy')
    X_pattern = np.load(f'{data_dir}/X_pattern.npy')
    X_indicators = np.load(f'{data_dir}/X_indicators.npy', allow_pickle=True)
    y = np.load(f'{data_dir}/y.npy')
    
    print(f"✓ Loaded {len(y)} samples")
    
    # Check shapes
    print("\n2. DATA SHAPES:")
    print(f"   X_chart: {X_chart.shape}")
    print(f"   X_volume: {X_volume.shape}")
    print(f"   X_pattern: {X_pattern.shape}")
    print(f"   X_indicators: {X_indicators.shape}")
    print(f"   y (target): {y.shape}")
    
    # Check target distribution
    print("\n3. TARGET VARIABLE ANALYSIS:")
    print(f"   Min: {y.min():.4f}")
    print(f"   Max: {y.max():.4f}")
    print(f"   Mean: {y.mean():.4f}")
    print(f"   Median: {np.median(y):.4f}")
    print(f"   Std: {y.std():.4f}")
    print(f"   Unique values: {len(np.unique(y))}")
    
    # Check for constant targets
    if y.std() < 0.01:
        print("   ⚠️  WARNING: Target has very low variance - almost constant!")
        print("   → Your labels may be wrong or all patterns scored similarly")
    
    # Check target distribution
    print("\n   Target Distribution:")
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    for p in percentiles:
        val = np.percentile(y, p)
        print(f"   {p}th percentile: {val:.4f}")
    
    # Check for imbalance (if binary or near-binary)
    unique_vals = np.unique(y)
    if len(unique_vals) <= 10:
        print("\n   Value Counts:")
        for val in unique_vals:
            count = np.sum(y == val)
            pct = 100 * count / len(y)
            print(f"   {val:.4f}: {count} ({pct:.1f}%)")
    
    # Check features
    print("\n4. FEATURE ANALYSIS:")
    
    print("\n   X_chart (candlestick data):")
    print(f"   Min: {X_chart.min():.4f}")
    print(f"   Max: {X_chart.max():.4f}")
    print(f"   Mean: {X_chart.mean():.4f}")
    print(f"   Std: {X_chart.std():.4f}")
    print(f"   Any NaN: {np.isnan(X_chart).any()}")
    print(f"   Any Inf: {np.isinf(X_chart).any()}")
    
    # Check if all zeros
    if np.abs(X_chart).max() < 0.001:
        print("   ⚠️  WARNING: X_chart is all zeros or near-zero!")
    
    print("\n   X_volume:")
    print(f"   Min: {X_volume.min():.4f}")
    print(f"   Max: {X_volume.max():.4f}")
    print(f"   Mean: {X_volume.mean():.4f}")
    print(f"   Any NaN: {np.isnan(X_volume).any()}")
    
    print("\n   X_pattern:")
    print(f"   Min: {X_pattern.min():.4f}")
    print(f"   Max: {X_pattern.max():.4f}")
    print(f"   Mean: {X_pattern.mean():.4f}")
    print(f"   Any NaN: {np.isnan(X_pattern).any()}")
    
    print("\n   X_indicators:")
    print(f"   Type: {X_indicators.dtype}")
    print(f"   Min: {X_indicators.min():.4f}")
    print(f"   Max: {X_indicators.max():.4f}")
    print(f"   Mean: {X_indicators.mean():.4f}")
    print(f"   Any NaN: {np.isnan(X_indicators).any()}")
    
    # Check feature-target correlation
    print("\n5. FEATURE-TARGET RELATIONSHIP:")
    print("   Checking if features contain any signal...")
    
    # Flatten features and compute correlation with target
    X_chart_flat = X_chart.reshape(len(X_chart), -1)
    correlations = []
    for i in range(min(50, X_chart_flat.shape[1])):  # Check first 50 features
        corr = np.corrcoef(X_chart_flat[:, i], y)[0, 1]
        if not np.isnan(corr):
            correlations.append(abs(corr))
    
    if correlations:
        max_corr = max(correlations)
        print(f"   Max absolute correlation (chart features): {max_corr:.4f}")
        if max_corr < 0.05:
            print("   ⚠️  WARNING: Very weak correlation - features may not predict target!")
    
    # Check pattern features correlation
    pattern_corrs = []
    for i in range(X_pattern.shape[1]):
        corr = np.corrcoef(X_pattern[:, i], y)[0, 1]
        if not np.isnan(corr):
            pattern_corrs.append(corr)
    
    if pattern_corrs:
        print(f"\n   Pattern Feature Correlations with Target:")
        for i, corr in enumerate(pattern_corrs):
            print(f"   Feature {i}: {corr:.4f}")
    
    # Visualization
    print("\n6. CREATING DIAGNOSTIC PLOTS...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Target distribution
    axes[0, 0].hist(y, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Target Distribution')
    axes[0, 0].set_xlabel('Pattern Quality Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(y.mean(), color='r', linestyle='--', label=f'Mean: {y.mean():.3f}')
    axes[0, 0].legend()
    
    # Plot 2: Target boxplot
    axes[0, 1].boxplot(y)
    axes[0, 1].set_title('Target Boxplot')
    axes[0, 1].set_ylabel('Pattern Quality Score')
    
    # Plot 3: Sample chart data
    if X_chart.shape[0] > 0:
        sample_idx = 0
        axes[0, 2].plot(X_chart[sample_idx])
        axes[0, 2].set_title(f'Sample Chart Data (y={y[sample_idx]:.3f})')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].legend(['Open', 'High', 'Low', 'Close', 'Volume'], fontsize=8)
    
    # Plot 4: Feature means across samples
    chart_means = X_chart.mean(axis=1).mean(axis=1)  # Mean per sample
    axes[1, 0].scatter(chart_means, y, alpha=0.5)
    axes[1, 0].set_title('Chart Mean vs Target')
    axes[1, 0].set_xlabel('Mean Chart Value')
    axes[1, 0].set_ylabel('Target')
    
    # Plot 5: Pattern features
    if X_pattern.shape[1] > 0:
        axes[1, 1].scatter(X_pattern[:, 0], y, alpha=0.5)
        axes[1, 1].set_title('Pattern Feature 0 vs Target')
        axes[1, 1].set_xlabel('Pattern Feature 0')
        axes[1, 1].set_ylabel('Target')
    
    # Plot 6: Correlation heatmap (pattern features)
    if len(pattern_corrs) > 0:
        axes[1, 2].bar(range(len(pattern_corrs)), pattern_corrs)
        axes[1, 2].set_title('Pattern Feature Correlations')
        axes[1, 2].set_xlabel('Feature Index')
        axes[1, 2].set_ylabel('Correlation with Target')
        axes[1, 2].axhline(0, color='red', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('data_diagnostics.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: data_diagnostics.png")
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    issues = []
    
    if y.std() < 0.01:
        issues.append("❌ Target has almost no variance")
    
    if len(unique_vals) <= 3:
        issues.append("❌ Target has very few unique values")
    
    if np.abs(X_chart).max() < 0.001:
        issues.append("❌ Chart features are all zeros")
    
    if max_corr < 0.05:
        issues.append("❌ Features have no correlation with target")
    
    if np.isnan(X_chart).any() or np.isnan(y).any():
        issues.append("❌ Data contains NaN values")
    
    if len(issues) == 0:
        print("✓ No obvious data quality issues found")
        print("→ Problem may be with model architecture or hyperparameters")
    else:
        print("CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    
    print("\n" + "="*70)
    
    return {
        'target_stats': {
            'min': y.min(),
            'max': y.max(),
            'mean': y.mean(),
            'std': y.std(),
            'unique_values': len(unique_vals)
        },
        'max_correlation': max_corr if correlations else None,
        'issues': issues
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_training_data.py <data_directory>")
        print("Example: python diagnose_training_data.py ./training_data")
    else:
        data_dir = sys.argv[1]
        results = diagnose_training_data(data_dir)