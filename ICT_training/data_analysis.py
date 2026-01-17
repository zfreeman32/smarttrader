"""
ICT Features Data Analysis Script
=================================
Analyzes the feature-engineered dataset from ict_feature_engineering.py

Usage:
    python data_analysis.py features.csv
    python data_analysis.py features.csv --output-dir analysis_results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
import warnings
import argparse

warnings.filterwarnings('ignore')

# Set plotting style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass


def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """Analyze data quality metrics"""
    print("   • Data quality...", end=" ", flush=True)
    
    results = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    
    # Date range
    if 'datetime' in df.columns:
        results['start_date'] = str(df['datetime'].min())
        results['end_date'] = str(df['datetime'].max())
        results['days_covered'] = (df['datetime'].max() - df['datetime'].min()).days
    
    # Missing values
    missing = df.isnull().sum()
    results['total_missing'] = int(missing.sum())
    results['missing_pct'] = float(missing.sum() / df.size * 100)
    results['cols_with_missing'] = int((missing > 0).sum())
    
    # Duplicates
    results['duplicate_rows'] = int(df.duplicated().sum())
    
    print("✓")
    return results


def analyze_ict_features(df: pd.DataFrame) -> Dict:
    """Analyze ICT-specific features"""
    print("   • ICT features...", end=" ", flush=True)
    
    results = {}
    
    # FVG Analysis
    if 'bullish_fvg' in df.columns:
        results['bullish_fvg_count'] = int(df['bullish_fvg'].sum())
        results['bullish_fvg_rate'] = float(df['bullish_fvg'].mean() * 100)
    
    if 'bearish_fvg' in df.columns:
        results['bearish_fvg_count'] = int(df['bearish_fvg'].sum())
        results['bearish_fvg_rate'] = float(df['bearish_fvg'].mean() * 100)
    
    # Session Analysis
    session_cols = ['in_asian', 'in_london_kz', 'in_nyam_kz', 'in_nypm_kz', 'in_any_kz']
    results['sessions'] = {}
    for col in session_cols:
        if col in df.columns:
            results['sessions'][col] = {
                'count': int(df[col].sum()),
                'pct': float(df[col].mean() * 100)
            }
    
    # Confluence Analysis
    if 'bullish_confluence' in df.columns:
        results['bullish_confluence_avg'] = float(df['bullish_confluence'].mean())
        results['bullish_confluence_max'] = int(df['bullish_confluence'].max())
    
    if 'bearish_confluence' in df.columns:
        results['bearish_confluence_avg'] = float(df['bearish_confluence'].mean())
        results['bearish_confluence_max'] = int(df['bearish_confluence'].max())
    
    if 'max_confluence' in df.columns:
        results['high_confluence_rate'] = float((df['max_confluence'] >= 3).mean() * 100)
        results['confluence_distribution'] = df['max_confluence'].value_counts().to_dict()
    
    # Liquidity sweeps
    if 'sweep_high' in df.columns:
        results['sweep_high_count'] = int(df['sweep_high'].sum())
    if 'sweep_low' in df.columns:
        results['sweep_low_count'] = int(df['sweep_low'].sum())
    
    # Premium/Discount
    if 'in_premium' in df.columns:
        results['in_premium_rate'] = float(df['in_premium'].mean() * 100)
    if 'in_discount' in df.columns:
        results['in_discount_rate'] = float(df['in_discount'].mean() * 100)
    
    # Displacement
    if 'is_displacement' in df.columns:
        results['displacement_count'] = int(df['is_displacement'].sum())
        results['displacement_rate'] = float(df['is_displacement'].mean() * 100)
    
    # Order Blocks
    if 'bullish_ob' in df.columns:
        results['bullish_ob_count'] = int(df['bullish_ob'].sum())
    if 'bearish_ob' in df.columns:
        results['bearish_ob_count'] = int(df['bearish_ob'].sum())
    
    print("✓")
    return results


def analyze_correlations(df: pd.DataFrame) -> Dict:
    """Find highly correlated features"""
    print("   • Correlations...", end=" ", flush=True)
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Limit columns for performance
    if len(numeric_df.columns) > 80:
        numeric_df = numeric_df.iloc[:, :80]
    
    corr_matrix = numeric_df.corr()
    
    # Find high correlations
    high_corr_pairs = []
    threshold = 0.8
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': round(float(corr_val), 3)
                })
    
    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    print("✓")
    return {
        'threshold': threshold,
        'high_corr_count': len(high_corr_pairs),
        'top_pairs': high_corr_pairs[:20]
    }


def analyze_time_patterns(df: pd.DataFrame) -> Dict:
    """Analyze time-based patterns"""
    print("   • Time patterns...", end=" ", flush=True)
    
    results = {}
    
    if 'hour' in df.columns:
        results['bars_by_hour'] = df['hour'].value_counts().sort_index().to_dict()
        
        if 'atr' in df.columns:
            hourly_atr = df.groupby('hour')['atr'].mean()
            results['avg_atr_by_hour'] = {int(k): round(v, 6) for k, v in hourly_atr.items()}
        
        if 'volume' in df.columns:
            hourly_vol = df.groupby('hour')['volume'].mean()
            results['avg_volume_by_hour'] = {int(k): round(v, 2) for k, v in hourly_vol.items()}
    
    if 'day_of_week' in df.columns:
        day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
        day_counts = df['day_of_week'].value_counts().sort_index()
        results['bars_by_day'] = {day_names.get(k, str(k)): int(v) for k, v in day_counts.items()}
    
    print("✓")
    return results


def generate_visualizations(df: pd.DataFrame, output_dir: Path) -> List[str]:
    """Generate analysis plots"""
    print("\n📊 Generating visualizations...")
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    generated = []
    
    # 1. Session distribution
    try:
        session_cols = ['in_london_kz', 'in_nyam_kz', 'in_nypm_kz']
        present = [c for c in session_cols if c in df.columns]
        
        if present:
            fig, ax = plt.subplots(figsize=(10, 6))
            rates = [df[col].mean() * 100 for col in present]
            labels = ['London KZ', 'NY AM KZ', 'NY PM KZ'][:len(present)]
            colors = ['#3498db', '#e74c3c', '#f39c12'][:len(present)]
            
            bars = ax.bar(labels, rates, color=colors, edgecolor='black')
            ax.set_ylabel('% of Total Bars')
            ax.set_title('Kill Zone Coverage')
            
            for bar, rate in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{rate:.1f}%', ha='center', fontsize=11)
            
            filepath = plots_dir / 'session_coverage.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            generated.append(str(filepath))
            print(f"   ✓ session_coverage.png")
    except Exception as e:
        print(f"   ✗ session_coverage.png failed: {e}")
    
    # 2. Confluence distribution
    try:
        if 'max_confluence' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            counts = df['max_confluence'].value_counts().sort_index()
            ax.bar(counts.index, counts.values, color='steelblue', edgecolor='black')
            ax.set_xlabel('Confluence Score')
            ax.set_ylabel('Count')
            ax.set_title('Max Confluence Distribution')
            
            filepath = plots_dir / 'confluence_distribution.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            generated.append(str(filepath))
            print(f"   ✓ confluence_distribution.png")
    except Exception as e:
        print(f"   ✗ confluence_distribution.png failed: {e}")
    
    # 3. Hourly patterns
    try:
        if 'hour' in df.columns and 'atr' in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Bar count by hour
            hour_counts = df['hour'].value_counts().sort_index()
            axes[0].bar(hour_counts.index, hour_counts.values, color='steelblue', edgecolor='black')
            axes[0].set_xlabel('Hour')
            axes[0].set_ylabel('Bar Count')
            axes[0].set_title('Data Coverage by Hour')
            axes[0].set_xticks(range(0, 24))
            
            # Shade kill zones
            for ax in axes:
                ax.axvspan(2, 5, alpha=0.2, color='blue', label='London KZ')
                ax.axvspan(9, 12, alpha=0.2, color='red', label='NY AM KZ')
                ax.axvspan(14, 16, alpha=0.2, color='orange', label='NY PM KZ')
            
            # ATR by hour
            hourly_atr = df.groupby('hour')['atr'].mean()
            axes[1].bar(hourly_atr.index, hourly_atr.values, color='orange', edgecolor='black')
            axes[1].set_xlabel('Hour')
            axes[1].set_ylabel('Average ATR')
            axes[1].set_title('Volatility by Hour')
            axes[1].set_xticks(range(0, 24))
            
            plt.tight_layout()
            filepath = plots_dir / 'hourly_patterns.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            generated.append(str(filepath))
            print(f"   ✓ hourly_patterns.png")
    except Exception as e:
        print(f"   ✗ hourly_patterns.png failed: {e}")
    
    # 4. ICT features occurrence rates
    try:
        ict_features = [
            'bullish_fvg', 'bearish_fvg', 'bullish_ob', 'bearish_ob',
            'is_displacement', 'sweep_high', 'sweep_low',
            'in_premium', 'in_discount'
        ]
        present = [f for f in ict_features if f in df.columns]
        
        if len(present) >= 3:
            fig, ax = plt.subplots(figsize=(12, 6))
            rates = [df[f].mean() * 100 for f in present]
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(present)))
            bars = ax.barh(present, rates, color=colors, edgecolor='black')
            ax.set_xlabel('Occurrence Rate (%)')
            ax.set_title('ICT Feature Occurrence Rates')
            
            for bar, rate in zip(bars, rates):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                       f'{rate:.2f}%', va='center', fontsize=9)
            
            plt.tight_layout()
            filepath = plots_dir / 'ict_features.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            generated.append(str(filepath))
            print(f"   ✓ ict_features.png")
    except Exception as e:
        print(f"   ✗ ict_features.png failed: {e}")
    
    # 5. Correlation heatmap (key features)
    try:
        key_features = [
            'atr', 'volume_ratio', 'rsi', 'trend_ema',
            'in_any_kz', 'in_premium', 'in_discount',
            'bullish_confluence', 'bearish_confluence'
        ]
        present = [f for f in key_features if f in df.columns]
        
        if len(present) >= 5:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[present].corr()
            
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                       cmap='RdBu_r', center=0, square=True,
                       linewidths=0.5, ax=ax, vmin=-1, vmax=1)
            
            plt.title('Key Feature Correlations')
            plt.tight_layout()
            filepath = plots_dir / 'correlations.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            generated.append(str(filepath))
            print(f"   ✓ correlations.png")
    except Exception as e:
        print(f"   ✗ correlations.png failed: {e}")
    
    # 6. Feature distributions
    try:
        features_to_plot = ['atr', 'rsi', 'volume_ratio', 'price_position']
        present = [f for f in features_to_plot if f in df.columns]
        
        if present:
            n = len(present)
            fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
            if n == 1:
                axes = [axes]
            
            for ax, feat in zip(axes, present):
                data = df[feat].dropna()
                q1, q99 = data.quantile([0.01, 0.99])
                data_clipped = data.clip(q1, q99)
                ax.hist(data_clipped, bins=50, edgecolor='black', alpha=0.7)
                ax.set_title(feat)
                ax.set_xlabel('')
            
            plt.suptitle('Feature Distributions', fontweight='bold')
            plt.tight_layout()
            filepath = plots_dir / 'distributions.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            generated.append(str(filepath))
            print(f"   ✓ distributions.png")
    except Exception as e:
        print(f"   ✗ distributions.png failed: {e}")
    
    return generated


def generate_reports(results: Dict, output_dir: Path) -> Dict:
    """Generate JSON and Markdown reports"""
    print("\n📝 Generating reports...")
    
    # JSON report
    json_path = output_dir / 'analysis_report.json'
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(json_path, 'w') as f:
        json.dump(convert(results), f, indent=2, default=str)
    print(f"   ✓ {json_path.name}")
    
    # Markdown report
    md_path = output_dir / 'analysis_report.md'
    
    lines = [
        "# ICT Features Data Analysis Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n---\n",
        "## Data Quality\n",
    ]
    
    dq = results.get('data_quality', {})
    lines.append(f"- **Rows:** {dq.get('num_rows', 'N/A'):,}")
    lines.append(f"- **Columns:** {dq.get('num_columns', 'N/A')}")
    lines.append(f"- **Memory:** {dq.get('memory_mb', 0):.2f} MB")
    if 'start_date' in dq:
        lines.append(f"- **Date Range:** {dq.get('start_date')} to {dq.get('end_date')}")
    lines.append(f"- **Missing Values:** {dq.get('missing_pct', 0):.2f}%")
    lines.append("")
    
    ict = results.get('ict_features', {})
    lines.append("## ICT Features\n")
    
    if 'bullish_fvg_count' in ict:
        lines.append(f"- **Bullish FVGs:** {ict['bullish_fvg_count']:,} ({ict.get('bullish_fvg_rate', 0):.3f}%)")
    if 'bearish_fvg_count' in ict:
        lines.append(f"- **Bearish FVGs:** {ict['bearish_fvg_count']:,} ({ict.get('bearish_fvg_rate', 0):.3f}%)")
    if 'high_confluence_rate' in ict:
        lines.append(f"- **High Confluence (≥3):** {ict['high_confluence_rate']:.2f}%")
    if 'displacement_count' in ict:
        lines.append(f"- **Displacements:** {ict['displacement_count']:,} ({ict.get('displacement_rate', 0):.3f}%)")
    lines.append("")
    
    if 'sessions' in ict:
        lines.append("### Sessions\n")
        lines.append("| Session | Bars | % |")
        lines.append("|---------|------|---|")
        for sess, data in ict['sessions'].items():
            clean_name = sess.replace('in_', '').replace('_', ' ').upper()
            lines.append(f"| {clean_name} | {data['count']:,} | {data['pct']:.1f}% |")
        lines.append("")
    
    corr = results.get('correlations', {})
    lines.append("## Correlations\n")
    lines.append(f"- **Highly correlated pairs (≥{corr.get('threshold', 0.8)}):** {corr.get('high_corr_count', 0)}")
    lines.append("")
    
    if 'visualizations' in results:
        lines.append("## Visualizations\n")
        for viz in results['visualizations']:
            name = Path(viz).name
            lines.append(f"- [{name}](plots/{name})")
        lines.append("")
    
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"   ✓ {md_path.name}")
    
    return {'json': str(json_path), 'markdown': str(md_path)}


def print_summary(results: Dict):
    """Print analysis summary"""
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    
    dq = results.get('data_quality', {})
    print(f"\n📁 DATASET:")
    print(f"   Rows: {dq.get('num_rows', 0):,}")
    print(f"   Columns: {dq.get('num_columns', 0)}")
    if 'start_date' in dq:
        print(f"   Date Range: {dq['start_date']} to {dq['end_date']}")
    print(f"   Missing: {dq.get('missing_pct', 0):.2f}%")
    
    ict = results.get('ict_features', {})
    print(f"\n📈 ICT FEATURES:")
    if 'bullish_fvg_count' in ict:
        print(f"   Bullish FVGs: {ict['bullish_fvg_count']:,} ({ict.get('bullish_fvg_rate', 0):.3f}%)")
    if 'bearish_fvg_count' in ict:
        print(f"   Bearish FVGs: {ict['bearish_fvg_count']:,} ({ict.get('bearish_fvg_rate', 0):.3f}%)")
    if 'displacement_count' in ict:
        print(f"   Displacements: {ict['displacement_count']:,}")
    if 'high_confluence_rate' in ict:
        print(f"   High Confluence (≥3): {ict['high_confluence_rate']:.2f}%")
    
    if 'sessions' in ict:
        print(f"\n⏰ KILL ZONES:")
        for sess, data in ict['sessions'].items():
            if 'kz' in sess:
                clean_name = sess.replace('in_', '').replace('_', ' ').upper()
                print(f"   {clean_name}: {data['count']:,} bars ({data['pct']:.1f}%)")
    
    corr = results.get('correlations', {})
    print(f"\n🔗 CORRELATIONS:")
    print(f"   Highly correlated pairs: {corr.get('high_corr_count', 0)}")
    
    print(f"\n📊 VISUALIZATIONS: {len(results.get('visualizations', []))} plots generated")
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='ICT Features Data Analysis')
    parser.add_argument('input', help='Input features CSV file')
    parser.add_argument('--output-dir', '-o', default='analysis_output',
                       help='Output directory (default: analysis_output)')
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: File not found: {args.input}")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🚀 ICT DATA ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"\n📂 Input: {input_path.absolute()}")
    print(f"📁 Output: {output_dir.absolute()}")
    
    # Load data
    print(f"\n📥 Loading data...")
    try:
        df = pd.read_csv(args.input)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"   ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    # Run analysis
    print("\n🔍 Running analysis...")
    results = {}
    
    try:
        results['data_quality'] = analyze_data_quality(df)
        results['ict_features'] = analyze_ict_features(df)
        results['correlations'] = analyze_correlations(df)
        results['time_patterns'] = analyze_time_patterns(df)
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate visualizations
    try:
        results['visualizations'] = generate_visualizations(df, output_dir)
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        results['visualizations'] = []
    
    # Generate reports
    try:
        results['reports'] = generate_reports(results, output_dir)
    except Exception as e:
        print(f"⚠️ Report error: {e}")
    
    # Print summary
    print_summary(results)
    
    print(f"\n📁 Results saved to: {output_dir.absolute()}")
    
    return 0


if __name__ == "__main__":
    exit(main())