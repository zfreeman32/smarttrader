"""
Multi-Timeframe Confluence Model - Visualizations
================================================

Generates comprehensive visualizations for training analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os


def plot_training_history(history, save_path: str):
    """Plot training and validation metrics over epochs"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Total loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confluence metrics
    axes[0, 1].plot(history.history['confluence_score_loss'], label='Train Conf Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_confluence_score_loss'], label='Val Conf Loss', linewidth=2)
    axes[0, 1].set_title('Confluence Score Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bias metrics
    axes[1, 0].plot(history.history['directional_bias_loss'], label='Train Bias Loss', linewidth=2)
    axes[1, 0].plot(history.history['val_directional_bias_loss'], label='Val Bias Loss', linewidth=2)
    axes[1, 0].set_title('Directional Bias Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAE metrics
    if 'confluence_score_mae' in history.history:
        axes[1, 1].plot(history.history['confluence_score_mae'], label='Train Conf MAE', linewidth=2)
        axes[1, 1].plot(history.history['val_confluence_score_mae'], label='Val Conf MAE', linewidth=2)
        axes[1, 1].set_title('Confluence Score MAE', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training history plot saved: {save_path}")


def plot_prediction_analysis(y_true_conf: np.ndarray,
                            y_pred_conf: np.ndarray,
                            y_true_bias: np.ndarray,
                            y_pred_bias: np.ndarray,
                            save_path: str):
    """Plot prediction vs actual analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Prediction Analysis', fontsize=16, fontweight='bold')
    
    # Confluence: Scatter
    axes[0, 0].scatter(y_true_conf, y_pred_conf, alpha=0.5, s=10)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_title('Confluence Score: Predicted vs Actual', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Actual Confluence')
    axes[0, 0].set_ylabel('Predicted Confluence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confluence: Error distribution
    conf_errors = y_pred_conf - y_true_conf
    axes[0, 1].hist(conf_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Confluence Prediction Errors', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bias: Scatter
    axes[1, 0].scatter(y_true_bias, y_pred_bias, alpha=0.5, s=10)
    axes[1, 0].plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_title('Directional Bias: Predicted vs Actual', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Actual Bias')
    axes[1, 0].set_ylabel('Predicted Bias')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bias: Error distribution
    bias_errors = y_pred_bias - y_true_bias
    axes[1, 1].hist(bias_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Bias Prediction Errors', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Prediction analysis plot saved: {save_path}")


def plot_confluence_distribution(y_pred_conf: np.ndarray,
                                 y_pred_bias: np.ndarray,
                                 save_path: str):
    """Plot distribution of predictions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Signal Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Confluence distribution
    axes[0, 0].hist(y_pred_conf, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 0].axvline(0.70, color='r', linestyle='--', linewidth=2, label='Min Tradeable (0.70)')
    axes[0, 0].axvline(0.85, color='g', linestyle='--', linewidth=2, label='Excellent (0.85)')
    axes[0, 0].set_title('Confluence Score Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Confluence Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Bias distribution
    axes[0, 1].hist(y_pred_bias, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(0, color='k', linestyle='-', linewidth=2)
    axes[0, 1].axvline(0.30, color='r', linestyle='--', linewidth=2, label='Strong Bullish')
    axes[0, 1].axvline(-0.30, color='r', linestyle='--', linewidth=2, label='Strong Bearish')
    axes[0, 1].set_title('Directional Bias Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Directional Bias')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2D distribution
    axes[1, 0].hexbin(y_pred_bias, y_pred_conf, gridsize=50, cmap='YlOrRd')
    axes[1, 0].axhline(0.70, color='b', linestyle='--', linewidth=2, label='Min Confluence')
    axes[1, 0].axvline(0, color='k', linestyle='-', linewidth=2)
    axes[1, 0].set_title('Confluence vs Bias (2D Distribution)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Directional Bias')
    axes[1, 0].set_ylabel('Confluence Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Signal quality pie chart
    excellent = np.sum((y_pred_conf >= 0.85) & (np.abs(y_pred_bias) >= 0.50))
    good = np.sum((y_pred_conf >= 0.70) & (np.abs(y_pred_bias) >= 0.30) & ~((y_pred_conf >= 0.85) & (np.abs(y_pred_bias) >= 0.50)))
    fair = np.sum((y_pred_conf >= 0.60) & (np.abs(y_pred_bias) >= 0.20) & ~((y_pred_conf >= 0.70) & (np.abs(y_pred_bias) >= 0.30)))
    poor = len(y_pred_conf) - excellent - good - fair
    
    sizes = [excellent, good, fair, poor]
    labels = [f'Excellent\n{excellent:,}', f'Good\n{good:,}', f'Fair\n{fair:,}', f'Poor\n{poor:,}']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    explode = (0.1, 0.05, 0, 0)
    
    axes[1, 1].pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
    axes[1, 1].set_title('Signal Quality Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confluence distribution plot saved: {save_path}")


def plot_performance_dashboard(metrics: Dict, save_path: str):
    """Create comprehensive performance dashboard"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Performance Dashboard', fontsize=18, fontweight='bold')
    
    # Metric cards
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.7, 'Confluence R²', ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.3, f"{metrics['conf_r2']:.4f}", ha='center', va='center', fontsize=24, color='blue')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.7, 'Directional Accuracy', ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.3, f"{metrics['directional_accuracy']:.1f}%", ha='center', va='center', fontsize=24, color='green')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.7, 'Tradeable Signals', ha='center', va='center', fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.3, f"{metrics['pct_tradeable_signals']:.1f}%", ha='center', va='center', fontsize=24, color='purple')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Confluence metrics bar chart
    ax4 = fig.add_subplot(gs[1, :2])
    conf_metrics = ['MAE', 'RMSE', 'Correlation']
    conf_values = [metrics['conf_mae'], metrics['conf_rmse'], metrics['conf_correlation']]
    bars = ax4.bar(conf_metrics, conf_values, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax4.set_title('Confluence Metrics', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Value')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Directional accuracy by confluence
    ax5 = fig.add_subplot(gs[1, 2])
    conf_thresholds = []
    dir_accs = []
    for threshold in [60, 70, 75, 80, 85]:
        key = f'dir_acc_conf_{threshold}'
        if key in metrics:
            conf_thresholds.append(f'{threshold}%')
            dir_accs.append(metrics[key])
    
    bars = ax5.barh(conf_thresholds, dir_accs, color='#9b59b6')
    ax5.set_title('Dir Acc by Confluence', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Accuracy (%)')
    ax5.grid(True, alpha=0.3, axis='x')
    
    for bar in bars:
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%', ha='left', va='center', fontsize=9)
    
    # Signal quality comparison
    ax6 = fig.add_subplot(gs[2, :])
    quality_labels = ['Excellent\n(≥0.85, |bias|≥0.50)', 'Good\n(≥0.70, |bias|≥0.30)', 'Fair\n(≥0.60, |bias|≥0.20)']
    quality_counts = [metrics['excellent_signals'], metrics['good_signals'], metrics['fair_signals']]
    colors = ['#2ecc71', '#3498db', '#f39c12']
    
    bars = ax6.bar(quality_labels, quality_counts, color=colors, edgecolor='black', linewidth=2)
    ax6.set_title('Signal Quality Distribution', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Count')
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Performance dashboard saved: {save_path}")


def save_all_visualizations(history,
                           y_true_conf: np.ndarray,
                           y_pred_conf: np.ndarray,
                           y_true_bias: np.ndarray,
                           y_pred_bias: np.ndarray,
                           metrics: Dict,
                           output_dir: str):
    """
    Generate and save all visualizations
    
    Args:
        history: Training history
        y_true_conf: True confluence scores
        y_pred_conf: Predicted confluence scores
        y_true_bias: True directional bias
        y_pred_bias: Predicted directional bias
        metrics: Performance metrics dictionary
        output_dir: Directory to save plots
    """
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Generate plots
    plot_training_history(
        history,
        os.path.join(plots_dir, '1_training_history.png')
    )
    
    plot_prediction_analysis(
        y_true_conf, y_pred_conf,
        y_true_bias, y_pred_bias,
        os.path.join(plots_dir, '2_prediction_analysis.png')
    )
    
    plot_confluence_distribution(
        y_pred_conf, y_pred_bias,
        os.path.join(plots_dir, '3_confluence_distribution.png')
    )
    
    plot_performance_dashboard(
        metrics,
        os.path.join(plots_dir, '4_performance_dashboard.png')
    )
    
    print("\n✓ All visualizations generated successfully!")
    print(f"✓ Plots saved to: {plots_dir}")


__all__ = [
    'plot_training_history',
    'plot_prediction_analysis',
    'plot_confluence_distribution',
    'plot_performance_dashboard',
    'save_all_visualizations'
]