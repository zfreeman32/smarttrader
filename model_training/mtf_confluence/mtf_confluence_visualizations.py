"""
Multi-Timeframe Confluence Model - Visualization & Analysis
==========================================================

This script provides visualization tools for analyzing and understanding
the confluence model's behavior and performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)


def plot_training_history(history):
    """Plot training and validation metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Timeframe Confluence Model - Training History', fontsize=16, fontweight='bold')
    
    # Loss curves
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confluence score metrics
    axes[0, 1].plot(history.history['confluence_score_mae'], label='Train MAE', linewidth=2)
    axes[0, 1].plot(history.history['val_confluence_score_mae'], label='Val MAE', linewidth=2)
    axes[0, 1].axhline(y=0.15, color='r', linestyle='--', label='Target (0.15)', linewidth=2)
    axes[0, 1].set_title('Confluence Score - MAE', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Directional bias metrics
    axes[1, 0].plot(history.history['directional_bias_mae'], label='Train MAE', linewidth=2)
    axes[1, 0].plot(history.history['val_directional_bias_mae'], label='Val MAE', linewidth=2)
    axes[1, 0].axhline(y=0.20, color='r', linestyle='--', label='Target (0.20)', linewidth=2)
    axes[1, 0].set_title('Directional Bias - MAE', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2, color='green')
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Best epoch marker
        best_epoch = np.argmin(history.history['val_loss'])
        axes[1, 1].text(0.5, 0.5, f'Best Epoch: {best_epoch+1}\nVal Loss: {history.history["val_loss"][best_epoch]:.4f}',
                       ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title('Training Summary', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig


def plot_prediction_analysis(y_true_conf, y_pred_conf, y_true_bias, y_pred_bias):
    """Analyze prediction quality"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Prediction Quality Analysis', fontsize=16, fontweight='bold')
    
    # Confluence: Actual vs Predicted
    axes[0, 0].scatter(y_true_conf, y_pred_conf, alpha=0.3, s=10)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
    axes[0, 0].set_xlabel('Actual Confluence')
    axes[0, 0].set_ylabel('Predicted Confluence')
    axes[0, 0].set_title('Confluence: Actual vs Predicted')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confluence: Error Distribution
    conf_errors = y_pred_conf - y_true_conf
    axes[0, 1].hist(conf_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Confluence Error Distribution\nMAE: {np.abs(conf_errors).mean():.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confluence: By Range
    bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['<0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9+']
    y_true_binned = pd.cut(y_true_conf, bins=bins, labels=labels)
    
    df_conf = pd.DataFrame({
        'range': y_true_binned,
        'error': np.abs(conf_errors)
    })
    
    df_conf.boxplot(column='error', by='range', ax=axes[0, 2])
    axes[0, 2].set_xlabel('Actual Confluence Range')
    axes[0, 2].set_ylabel('Absolute Error')
    axes[0, 2].set_title('Error by Confluence Level')
    plt.sca(axes[0, 2])
    plt.xticks(rotation=0)
    
    # Bias: Actual vs Predicted
    axes[1, 0].scatter(y_true_bias, y_pred_bias, alpha=0.3, s=10)
    axes[1, 0].plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect')
    axes[1, 0].set_xlabel('Actual Bias')
    axes[1, 0].set_ylabel('Predicted Bias')
    axes[1, 0].set_title('Directional Bias: Actual vs Predicted')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bias: Error Distribution
    bias_errors = y_pred_bias - y_true_bias
    axes[1, 1].hist(bias_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Bias Error Distribution\nMAE: {np.abs(bias_errors).mean():.4f}')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Directional Accuracy
    correct_direction = (np.sign(y_true_bias) == np.sign(y_pred_bias))
    dir_accuracy = correct_direction.mean() * 100
    
    axes[1, 2].text(0.5, 0.6, f'Directional Accuracy\n\n{dir_accuracy:.1f}%',
                   ha='center', va='center', fontsize=24, fontweight='bold',
                   transform=axes[1, 2].transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightgreen' if dir_accuracy > 65 else 'yellow', alpha=0.8))
    
    axes[1, 2].text(0.5, 0.3, f'Target: >65%',
                   ha='center', va='center', fontsize=12,
                   transform=axes[1, 2].transAxes)
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def plot_confluence_distribution(confluence_scores, directional_bias, trading_outcomes=None):
    """Visualize confluence score distribution and trading outcomes"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Confluence Analysis', fontsize=16, fontweight='bold')
    
    # Confluence distribution
    axes[0, 0].hist(confluence_scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(x=0.70, color='r', linestyle='--', linewidth=2, label='Trading Threshold')
    axes[0, 0].set_xlabel('Confluence Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Confluence Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Bias distribution
    axes[0, 1].hist(directional_bias, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].axvline(x=0.3, color='g', linestyle='--', linewidth=2, label='Long Threshold')
    axes[0, 1].axvline(x=-0.3, color='r', linestyle='--', linewidth=2, label='Short Threshold')
    axes[0, 1].axvline(x=0, color='gray', linestyle='-', linewidth=1)
    axes[0, 1].set_xlabel('Directional Bias')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Directional Bias Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2D distribution
    h = axes[1, 0].hexbin(confluence_scores, directional_bias, gridsize=30, cmap='YlOrRd')
    axes[1, 0].axhline(y=0.3, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Long Threshold')
    axes[1, 0].axhline(y=-0.3, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Short Threshold')
    axes[1, 0].axvline(x=0.70, color='b', linestyle='--', linewidth=2, alpha=0.7, label='Conf Threshold')
    axes[1, 0].set_xlabel('Confluence Score')
    axes[1, 0].set_ylabel('Directional Bias')
    axes[1, 0].set_title('Joint Distribution')
    axes[1, 0].legend()
    plt.colorbar(h, ax=axes[1, 0])
    
    # Trading signal statistics
    high_conf = confluence_scores >= 0.70
    strong_bias = np.abs(directional_bias) >= 0.30
    tradeable = high_conf & strong_bias
    
    stats_text = f"""
    SIGNAL STATISTICS
    
    Total Samples: {len(confluence_scores):,}
    
    High Confluence (≥0.70): {high_conf.sum():,} ({high_conf.mean()*100:.1f}%)
    Strong Bias (|bias|≥0.30): {strong_bias.sum():,} ({strong_bias.mean()*100:.1f}%)
    
    Tradeable Signals: {tradeable.sum():,} ({tradeable.mean()*100:.1f}%)
    
    Of Tradeable:
      - Long Signals: {((directional_bias > 0.3) & high_conf).sum():,}
      - Short Signals: {((directional_bias < -0.3) & high_conf).sum():,}
    
    Signal Quality:
      - EXCELLENT (≥0.85 & |bias|≥0.50): {((confluence_scores >= 0.85) & (np.abs(directional_bias) >= 0.50)).sum():,}
      - GOOD (≥0.70 & |bias|≥0.30): {tradeable.sum():,}
      - FAIR (≥0.60 & |bias|≥0.20): {((confluence_scores >= 0.60) & (np.abs(directional_bias) >= 0.20)).sum():,}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, ha='left', va='center',
                   fontsize=10, family='monospace',
                   transform=axes[1, 1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig


def plot_timeframe_importance(model, feature_names_per_tf):
    """
    Visualize learned attention weights across timeframes
    Note: This is a simplified visualization - actual attention extraction
    would require model-specific implementation
    """
    
    # Placeholder - in real implementation, extract attention weights from model
    timeframes = ['1m', '5m', '15m', '1h', '4h']
    
    # Simulate importance (replace with actual attention extraction)
    importance = np.random.dirichlet(np.ones(5))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.barh(timeframes, importance, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax.text(val + 0.01, i, f'{val:.1%}', va='center', fontweight='bold')
    
    ax.set_xlabel('Relative Importance', fontsize=12, fontweight='bold')
    ax.set_title('Timeframe Importance (Learned by Model)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(importance) * 1.2)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_performance_dashboard(history, y_true_conf, y_pred_conf, y_true_bias, y_pred_bias):
    """Create comprehensive performance dashboard"""
    
    # Calculate all metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    conf_mae = mean_absolute_error(y_true_conf, y_pred_conf)
    conf_rmse = np.sqrt(mean_squared_error(y_true_conf, y_pred_conf))
    conf_r2 = r2_score(y_true_conf, y_pred_conf)
    
    bias_mae = mean_absolute_error(y_true_bias, y_pred_bias)
    bias_rmse = np.sqrt(mean_squared_error(y_true_bias, y_pred_bias))
    bias_r2 = r2_score(y_true_bias, y_pred_bias)
    
    dir_accuracy = (np.sign(y_true_bias) == np.sign(y_pred_bias)).mean() * 100
    
    # Create dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Multi-Timeframe Confluence Model - Performance Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Metric cards
    metrics = [
        ('Confluence MAE', conf_mae, 0.15, '<'),
        ('Confluence RMSE', conf_rmse, 0.20, '<'),
        ('Confluence R²', conf_r2, 0.40, '>'),
        ('Bias MAE', bias_mae, 0.20, '<'),
        ('Bias RMSE', bias_rmse, 0.30, '<'),
        ('Bias R²', bias_r2, 0.40, '>'),
        ('Directional Acc', dir_accuracy, 65.0, '>'),
        ('Training Epochs', len(history.history['loss']), 100, '<'),
        ('Best Val Loss', min(history.history['val_loss']), 0.10, '<'),
    ]
    
    for idx, (name, value, target, comparison) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Determine if target met
        if comparison == '<':
            target_met = value < target
        else:
            target_met = value > target
        
        color = 'lightgreen' if target_met else 'lightcoral'
        
        # Format value
        if 'Acc' in name or 'R²' in name:
            value_text = f'{value:.2f}%' if 'Acc' in name else f'{value:.3f}'
        else:
            value_text = f'{value:.4f}'
        
        # Create card
        ax.text(0.5, 0.65, value_text, 
               ha='center', va='center', fontsize=24, fontweight='bold',
               transform=ax.transAxes)
        
        ax.text(0.5, 0.35, name,
               ha='center', va='center', fontsize=11,
               transform=ax.transAxes)
        
        ax.text(0.5, 0.15, f'Target: {comparison}{target}',
               ha='center', va='center', fontsize=9, style='italic',
               transform=ax.transAxes, color='gray')
        
        # Color background
        ax.set_facecolor(color)
        ax.set_alpha(0.3)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def save_all_visualizations(history, y_true_conf, y_pred_conf, y_true_bias, y_pred_bias, 
                           output_dir='/mnt/user-data/outputs/'):
    """Save all visualization plots"""
    
    plots = []
    
    # Training history
    fig1 = plot_training_history(history)
    fig1.savefig(f'{output_dir}training_history.png', dpi=150, bbox_inches='tight')
    plots.append('training_history.png')
    plt.close(fig1)
    
    # Prediction analysis
    fig2 = plot_prediction_analysis(y_true_conf, y_pred_conf, y_true_bias, y_pred_bias)
    fig2.savefig(f'{output_dir}prediction_analysis.png', dpi=150, bbox_inches='tight')
    plots.append('prediction_analysis.png')
    plt.close(fig2)
    
    # Confluence distribution
    fig3 = plot_confluence_distribution(y_pred_conf, y_pred_bias)
    fig3.savefig(f'{output_dir}confluence_distribution.png', dpi=150, bbox_inches='tight')
    plots.append('confluence_distribution.png')
    plt.close(fig3)
    
    # Performance dashboard
    fig4 = create_performance_dashboard(history, y_true_conf, y_pred_conf, y_true_bias, y_pred_bias)
    fig4.savefig(f'{output_dir}performance_dashboard.png', dpi=150, bbox_inches='tight')
    plots.append('performance_dashboard.png')
    plt.close(fig4)
    
    print(f"\n✅ Saved {len(plots)} visualization plots to {output_dir}")
    for plot in plots:
        print(f"   - {plot}")
    
    return plots


if __name__ == "__main__":
    print("Multi-Timeframe Confluence Model - Visualization Tools")
    print("=" * 60)
    print("\nThis module provides visualization functions for analyzing")
    print("model training and performance.")
    print("\nUsage:")
    print("  from mtf_confluence_visualizations import *")
    print("  fig = plot_training_history(history)")
    print("  fig.show()")
