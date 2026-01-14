"""
Enhanced Best Model Selection for EURUSD Trading System
Handles multiple direction timeframes (1, 5, 14) and architectures
"""
import os
import json
import shutil
import pandas as pd
from glob import glob
from datetime import datetime
import re

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS AS NEEDED
# ============================================================================

BASE_DIR = r'C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot\models'

# Model directories to scan
MODEL_DIRS = {
    'sell_signal': os.path.join(BASE_DIR, 'sell_signal_models_6_26'),
    'long_signal': os.path.join(BASE_DIR, 'long_signal_models_6_26'),
    'direction': os.path.join(BASE_DIR, 'direction_models_6_25')
}

# Output directory
OUTPUT_DIR = r'C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot\src\models'

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def extract_timeframe_from_path(path):
    """
    Extract timeframe from direction model path
    e.g., 'direction_5_trials_LSTM' -> 5
    """
    match = re.search(r'direction_(\d+)_trials', path)
    if match:
        return int(match.group(1))
    return None


def get_trial_info(trial_dir, base_model_type):
    """Extract information from a single trial directory"""
    
    trial_json = os.path.join(trial_dir, 'trial.json')
    checkpoint = os.path.join(trial_dir, 'checkpoint.weights.h5')
    
    # Check files exist
    if not os.path.exists(trial_json) or not os.path.exists(checkpoint):
        return None
    
    # Load trial data
    try:
        with open(trial_json, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"    ⚠️  Error reading {trial_json}: {e}")
        return None
    
    # Extract architecture from path
    # e.g., .../direction_5_trials_LSTM/trial_001 -> LSTM
    path_parts = trial_dir.split(os.sep)
    architecture = 'unknown'
    for part in path_parts:
        if '_trials_' in part:
            architecture = part.split('_trials_')[-1]
            break
    
    # For direction models, extract timeframe and create specific model type
    timeframe = extract_timeframe_from_path(trial_dir)
    if base_model_type == 'direction' and timeframe:
        model_type = f'direction_{timeframe}'
    else:
        model_type = base_model_type
    
    # Extract metrics safely
    metrics = data.get('metrics', {}).get('metrics', {})
    
    def get_metric_value(metric_name):
        obs = metrics.get(metric_name, {}).get('observations', [{}])
        if obs:
            value = obs[0].get('value', [None])
            if isinstance(value, list) and value:
                return value[0]
        return None
    
    val_loss = get_metric_value('val_loss') or 1e6
    val_accuracy = get_metric_value('val_accuracy') or 0.0
    val_auc = get_metric_value('val_auc') or 0.0
    accuracy = get_metric_value('accuracy') or 0.0
    
    # Calculate composite score
    # Formula: val_accuracy + val_auc - (val_loss / 10)
    composite = val_accuracy + val_auc - (val_loss / 10)
    
    return {
        'trial_dir': trial_dir,
        'trial_id': data.get('trial_id', 'unknown'),
        'model_type': model_type,
        'timeframe': timeframe,  # Will be None for non-direction models
        'architecture': architecture,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_auc': val_auc,
        'accuracy': accuracy,
        'composite_score': composite,
        'hyperparameters': data.get('hyperparameters', {}).get('values', {}),
        'checkpoint': checkpoint
    }


def scan_model_directory(model_dir, base_model_type):
    """
    Scan a model directory for all trials
    Handles both regular and direction models with timeframes
    """
    all_trials = []
    
    # Find all trial directories
    # Pattern: model_dir/*_trials_*/trial_*
    trial_pattern = os.path.join(model_dir, '*_trials_*', 'trial_*')
    trial_dirs = glob(trial_pattern)
    
    if not trial_dirs:
        print(f"    ⚠️  No trials found with pattern: {trial_pattern}")
        return all_trials
    
    print(f"    Found {len(trial_dirs)} trial directories")
    
    # Group trials by type for summary
    trial_summary = {}
    
    for trial_dir in trial_dirs:
        # Get trial info
        info = get_trial_info(trial_dir, base_model_type)
        
        if info:
            all_trials.append(info)
            
            # Track for summary
            key = f"{info['model_type']}_{info['architecture']}"
            if key not in trial_summary:
                trial_summary[key] = []
            trial_summary[key].append(info['composite_score'])
            
        else:
            # Silently skip invalid trials
            pass
    
    # Print summary
    for key, scores in sorted(trial_summary.items()):
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        print(f"    ✅ {key}: {len(scores)} trials, avg={avg_score:.4f}, max={max_score:.4f}")
    
    return all_trials


def scan_all_models():
    """Scan all model directories and collect trial information"""
    
    all_trials = []
    
    for base_type, model_dir in MODEL_DIRS.items():
        print(f"\n{'='*80}")
        print(f"Scanning: {base_type}")
        print(f"Directory: {model_dir}")
        print(f"{'='*80}")
        
        if not os.path.exists(model_dir):
            print(f"⚠️  Directory not found, skipping...")
            continue
        
        # Scan the directory
        trials = scan_model_directory(model_dir, base_type)
        all_trials.extend(trials)
        
        print(f"\n    Total valid trials: {len(trials)}")
    
    return all_trials


def find_best_models(all_trials):
    """Find the best model for each model type (including direction timeframes)"""
    
    best_models = {}
    
    # Group by model type
    by_type = {}
    for trial in all_trials:
        model_type = trial['model_type']
        if model_type not in by_type:
            by_type[model_type] = []
        by_type[model_type].append(trial)
    
    # Find best in each group
    print(f"\n{'='*80}")
    print(f"BEST MODELS SELECTED")
    print(f"{'='*80}")
    
    for model_type, trials in sorted(by_type.items()):
        # Sort by composite score (descending)
        sorted_trials = sorted(trials, key=lambda x: x['composite_score'], reverse=True)
        best_models[model_type] = sorted_trials[0]
        
        print(f"\n🏆 {model_type.upper()}:")
        print(f"   Architecture: {best_models[model_type]['architecture']}")
        print(f"   Trial: {best_models[model_type]['trial_id']}")
        print(f"   Composite Score: {best_models[model_type]['composite_score']:.6f}")
        print(f"   Val Accuracy: {best_models[model_type]['val_accuracy']:.6f}")
        print(f"   Val AUC: {best_models[model_type]['val_auc']:.6f}")
        print(f"   Val Loss: {best_models[model_type]['val_loss']:.6f}")
        print(f"   Candidates evaluated: {len(trials)}")
    
    return best_models


def save_model(trial_info, output_dir):
    """Save model files to output directory"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_type = trial_info['model_type']
    arch = trial_info['architecture']
    trial_id = trial_info['trial_id']
    
    # Create base filename
    base_name = f"{model_type}_{arch}_{trial_id}"
    
    # Copy checkpoint
    src_checkpoint = trial_info['checkpoint']
    dst_checkpoint = os.path.join(output_dir, f"{base_name}.weights.h5")
    shutil.copy2(src_checkpoint, dst_checkpoint)
    print(f"  📦 Saved checkpoint: {os.path.basename(dst_checkpoint)}")
    
    # Copy trial.json
    src_json = os.path.join(trial_info['trial_dir'], 'trial.json')
    dst_json = os.path.join(output_dir, f"{base_name}_trial.json")
    if os.path.exists(src_json):
        shutil.copy2(src_json, dst_json)
        print(f"  📦 Saved trial.json: {os.path.basename(dst_json)}")
    
    # Copy build_config.json if exists
    src_build = os.path.join(trial_info['trial_dir'], 'build_config.json')
    dst_build = os.path.join(output_dir, f"{base_name}_build_config.json")
    if os.path.exists(src_build):
        shutil.copy2(src_build, dst_build)
        print(f"  📦 Saved build_config: {os.path.basename(dst_build)}")
    
    # Create summary
    summary = {
        'model_type': model_type,
        'timeframe': trial_info.get('timeframe'),
        'architecture': arch,
        'trial_id': trial_id,
        'metrics': {
            'composite_score': trial_info['composite_score'],
            'val_accuracy': trial_info['val_accuracy'],
            'val_auc': trial_info['val_auc'],
            'val_loss': trial_info['val_loss'],
            'accuracy': trial_info['accuracy']
        },
        'hyperparameters': trial_info['hyperparameters'],
        'source': trial_info['trial_dir'],
        'saved_at': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(output_dir, f"{base_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  📊 Saved summary: {os.path.basename(summary_path)}")


def create_comparison_report(all_trials, best_models, output_dir):
    """Create comprehensive comparison report"""
    
    # Create DataFrame
    df_data = []
    for trial in all_trials:
        row = {
            'model_type': trial['model_type'],
            'timeframe': trial.get('timeframe', ''),
            'architecture': trial['architecture'],
            'trial_id': trial['trial_id'],
            'composite_score': trial['composite_score'],
            'val_accuracy': trial['val_accuracy'],
            'val_auc': trial['val_auc'],
            'val_loss': trial['val_loss'],
            'accuracy': trial['accuracy']
        }
        # Add hyperparameters
        for key, value in trial['hyperparameters'].items():
            row[f'hp_{key}'] = value
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'all_trials_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Saved comparison CSV: {csv_path}")
    
    # Create text report
    report = []
    report.append("=" * 80)
    report.append("BEST MODEL SELECTION REPORT")
    report.append("EURUSD Automated Trading System")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total trials evaluated: {len(all_trials)}")
    
    # Count by category
    model_type_counts = df['model_type'].value_counts()
    report.append(f"\nTrials by model type:")
    for mt, count in model_type_counts.items():
        report.append(f"  {mt}: {count}")
    
    report.append(f"\n{'=' * 80}")
    report.append("BEST MODELS BY TYPE")
    report.append("=" * 80)
    
    for model_type, trial in sorted(best_models.items()):
        report.append(f"\n{'─' * 80}")
        report.append(f"{model_type.upper()}")
        report.append(f"{'─' * 80}")
        
        if trial.get('timeframe'):
            report.append(f"  Timeframe: {trial['timeframe']} periods")
        report.append(f"  Architecture: {trial['architecture']}")
        report.append(f"  Trial ID: {trial['trial_id']}")
        report.append(f"\n  Performance Metrics:")
        report.append(f"    Composite Score: {trial['composite_score']:.6f}")
        report.append(f"    Validation Accuracy: {trial['val_accuracy']:.6f}")
        report.append(f"    Validation AUC: {trial['val_auc']:.6f}")
        report.append(f"    Validation Loss: {trial['val_loss']:.6f}")
        report.append(f"    Training Accuracy: {trial['accuracy']:.6f}")
        
        report.append(f"\n  Hyperparameters:")
        for key, value in sorted(trial['hyperparameters'].items()):
            report.append(f"    {key}: {value}")
        
        report.append(f"\n  Source: {trial['trial_dir']}")
    
    # Top 15 overall
    report.append(f"\n{'=' * 80}")
    report.append("TOP 15 MODELS OVERALL (by composite score)")
    report.append("=" * 80)
    
    top15 = df.nlargest(15, 'composite_score')
    report.append("\n" + top15[['model_type', 'timeframe', 'architecture', 'trial_id', 
                                 'composite_score', 'val_accuracy', 'val_auc', 
                                 'val_loss']].to_string(index=False))
    
    # Architecture comparison
    report.append(f"\n{'=' * 80}")
    report.append("ARCHITECTURE COMPARISON")
    report.append("=" * 80)
    
    arch_stats = df.groupby('architecture').agg({
        'composite_score': ['count', 'mean', 'std', 'min', 'max'],
        'val_accuracy': 'mean',
        'val_loss': 'mean'
    }).round(6)
    
    report.append("\n" + arch_stats.to_string())
    
    # Direction timeframe comparison (if applicable)
    direction_trials = df[df['model_type'].str.startswith('direction')]
    if not direction_trials.empty and 'timeframe' in direction_trials.columns:
        report.append(f"\n{'=' * 80}")
        report.append("DIRECTION TIMEFRAME COMPARISON")
        report.append("=" * 80)
        
        timeframe_stats = direction_trials.groupby('timeframe').agg({
            'composite_score': ['count', 'mean', 'std', 'max'],
            'val_accuracy': 'mean',
            'val_loss': 'mean'
        }).round(6)
        
        report.append("\n" + timeframe_stats.to_string())
    
    # Model type comparison
    report.append(f"\n{'=' * 80}")
    report.append("MODEL TYPE COMPARISON")
    report.append("=" * 80)
    
    type_stats = df.groupby('model_type').agg({
        'composite_score': ['count', 'mean', 'std', 'max'],
        'val_accuracy': 'mean',
        'val_auc': 'mean',
        'val_loss': 'mean'
    }).round(6)
    
    report.append("\n" + type_stats.to_string())
    
    # Save report
    report_path = os.path.join(output_dir, 'best_models_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"📄 Saved report: {report_path}")
    
    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    print(f"\n✅ Total trials evaluated: {len(all_trials)}")
    print(f"✅ Best models selected: {len(best_models)}")
    
    print("\n📊 Best models by category:")
    for model_type in sorted(best_models.keys()):
        trial = best_models[model_type]
        print(f"  • {model_type}: {trial['architecture']} "
              f"(score={trial['composite_score']:.4f})")
    
    # Overall best
    overall_best = max(best_models.values(), key=lambda x: x['composite_score'])
    print(f"\n🏆 OVERALL BEST MODEL:")
    print(f"  {overall_best['model_type']} - {overall_best['architecture']}")
    print(f"  Composite Score: {overall_best['composite_score']:.6f}")
    print(f"  Val Accuracy: {overall_best['val_accuracy']:.6f}")
    print(f"  Val AUC: {overall_best['val_auc']:.6f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("EURUSD TRADING SYSTEM - BEST MODEL SELECTION")
    print("Enhanced Version: Handles Multiple Direction Timeframes")
    print("=" * 80)
    
    # Scan all models
    print("\n📁 Scanning model directories...")
    all_trials = scan_all_models()
    
    if not all_trials:
        print("\n❌ No valid trials found!")
        print("\nTroubleshooting:")
        print("1. Check that directories exist")
        print("2. Verify trial.json and checkpoint.weights.h5 exist in trial folders")
        print("3. Check directory naming follows pattern: *_trials_*/trial_*")
        return
    
    print(f"\n✅ Total valid trials collected: {len(all_trials)}")
    
    # Find best models
    best_models = find_best_models(all_trials)
    
    if not best_models:
        print("\n❌ No best models identified!")
        return
    
    # Save best models
    print(f"\n{'='*80}")
    print("SAVING BEST MODELS")
    print(f"{'='*80}")
    
    for model_type, trial in sorted(best_models.items()):
        print(f"\n💾 Saving {model_type}...")
        save_model(trial, OUTPUT_DIR)
    
    # Create report
    print(f"\n{'='*80}")
    print("GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*80}")
    
    create_comparison_report(all_trials, best_models, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("✅ SELECTION COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"📦 Saved {len(best_models)} best models")
    
    print("\n📄 Generated files:")
    print("  • {model_type}_{arch}_{trial}.weights.h5 - Model weights")
    print("  • {model_type}_{arch}_{trial}_summary.json - Performance summary")
    print("  • all_trials_comparison.csv - Full comparison data")
    print("  • best_models_report.txt - Detailed analysis")
    
    print("\n🚀 Next steps:")
    print("  1. Review best_models_report.txt")
    print("  2. Load models in your trading system")
    print("  3. Integrate into Tier 2 (Signal Generation Layer)")
    print("  4. Build ICT detection layer (Tier 1)")
    print("  5. Implement ensemble voting (Tier 4)")
    print("  6. Add risk management (Tier 3)")
    print("  7. Backtest complete system")


if __name__ == "__main__":
    main()