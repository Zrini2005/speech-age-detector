#!/usr/bin/env python3
"""

This script runs multiple age estimation models on a directory of audio files
and generates comprehensive evaluation results.

Usage:
    python run_age_eval.py [--audio-dir /path/to/audio] [--models model1,model2]
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.audio_utils import load_audio, get_audio_duration, find_audio_files
from utils.label_parser import get_ground_truth_labels
from utils.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    format_metrics_table
)
from models import (
    Wav2Vec2AgeModel,
    WavLMAgeModel,
    SpeechBrainAgeModel,
    MFCCBaselineModel
)

# Default audio directory - CHANGE THIS to your audio folder
AUDIO_DIR = "/data/voices_for_age"


class AgeEstimationEvaluator:
    """Main evaluator class for running age estimation models."""
    
    def __init__(self, audio_dir: str, output_dir: str = "results"):
        """
        Initialize evaluator.
        
        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save results
        """
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Find audio files
        self.audio_files = find_audio_files(str(self.audio_dir))
        print(f"Found {len(self.audio_files)} audio files in {self.audio_dir}")
        
        # Load ground truth labels if available
        self.ground_truth = get_ground_truth_labels(str(self.audio_dir))
        
        if self.ground_truth:
            print(f"Loaded ground truth for {len(self.ground_truth)} files")
        else:
            print("No ground truth labels found - will only save predictions")
        
        # Initialize models dictionary
        self.models = {}
        self.results = {}
        
    def register_model(self, model, model_id: str = None):
        """
        Register a model for evaluation.
        
        Args:
            model: Model instance (must have predict_age method)
            model_id: Optional model identifier
        """
        if model_id is None:
            model_id = model.model_name
        
        self.models[model_id] = model
        print(f"Registered model: {model_id}")
    
    def run_inference(self, model_id: str) -> pd.DataFrame:
        """
        Run inference with a specific model on all audio files.
        
        Args:
            model_id: Model identifier
            
        Returns:
            DataFrame with predictions
        """
        model = self.models[model_id]
        
        print(f"\n{'='*60}")
        print(f"Running inference with {model_id}")
        print(f"{'='*60}")
        
        # Try to load model
        try:
            if not model.loaded:
                model.load_model()
        except Exception as e:
            print(f"Failed to load {model_id}: {e}")
            return None
        
        results = []
        total_time = 0
        
        # Process each audio file
        for audio_path in tqdm(self.audio_files, desc=f"{model_id}"):
            filename = Path(audio_path).name
            
            try:
                # Load audio
                waveform, sr = load_audio(audio_path)
                duration = len(waveform) / sr
                
                # Measure inference time
                start_time = time.time()
                predicted_age, confidence = model.predict_age(waveform, sr)
                inference_time = time.time() - start_time
                
                total_time += inference_time
                
                # Store results
                results.append({
                    'filename': filename,
                    'prediction_age': predicted_age,
                    'model_confidence': confidence,
                    'duration_s': duration,
                    'inference_time_s': inference_time
                })
                
            except Exception as e:
                print(f"\nError processing {filename}: {e}")
                results.append({
                    'filename': filename,
                    'prediction_age': np.nan,
                    'model_confidence': 0.0,
                    'duration_s': 0.0,
                    'inference_time_s': 0.0
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Add ground truth if available
        if self.ground_truth:
            df['ground_truth_age'] = df['filename'].map(self.ground_truth)
        
        # Save to CSV
        output_file = self.output_dir / f"{model_id}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved predictions to {output_file}")
        
        # Compute statistics
        avg_time = total_time / len(self.audio_files) if self.audio_files else 0
        print(f"Average inference time: {avg_time:.3f} seconds per file")
        print(f"Total time: {total_time:.2f} seconds")
        
        return df
    
    def evaluate_model(self, model_id: str, df: pd.DataFrame) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            model_id: Model identifier
            df: DataFrame with predictions and ground truth
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'model_name': model_id,
            'num_predictions': len(df),
            'avg_time': df['inference_time_s'].mean(),
            'total_time': df['inference_time_s'].sum(),
        }
        
        # Only compute accuracy metrics if we have ground truth
        if 'ground_truth_age' in df.columns:
            # Remove rows without ground truth
            df_eval = df.dropna(subset=['ground_truth_age', 'prediction_age'])
            
            if len(df_eval) > 0:
                y_true = df_eval['ground_truth_age'].values
                y_pred = df_eval['prediction_age'].values
                
                # Regression metrics
                reg_metrics = compute_regression_metrics(y_true, y_pred)
                metrics.update(reg_metrics)
                
                # Classification metrics (age groups)
                class_metrics = compute_classification_metrics(y_true, y_pred)
                metrics.update(class_metrics)
                
                # Child/Adult differentiation metrics (KEY METRIC)
                from utils.metrics import compute_child_adult_metrics
                child_metrics = compute_child_adult_metrics(y_true, y_pred)
                metrics.update(child_metrics)
                
                print(f"\n{model_id} Performance:")
                print(f"  MAE: {metrics['mae']:.2f} years")
                print(f"  RMSE: {metrics['rmse']:.2f} years")
                print(f"  Pearson r: {metrics['pearson_r']:.3f}")
                print(f"  Accuracy (age groups): {metrics['accuracy']:.2%}")
                print(f"\n  CHILD/ADULT DIFFERENTIATION:")
                print(f"  Child/Adult Accuracy: {metrics['child_adult_accuracy']:.2%}")
                print(f"  Child Precision: {metrics['child_precision']:.2%}")
                print(f"  Child Recall: {metrics['child_recall']:.2%}")
                print(f"  Child F1: {metrics['child_f1']:.2%}")
            else:
                print(f"No valid predictions with ground truth for {model_id}")
        else:
            print(f"No ground truth available - skipping evaluation metrics")
        
        return metrics
    
    def run_all_models(self, model_ids: List[str] = None):
        """
        Run inference and evaluation for all registered models.
        
        Args:
            model_ids: List of model IDs to run (None = all)
        """
        if model_ids is None:
            model_ids = list(self.models.keys())
        
        all_metrics = {}
        
        for model_id in model_ids:
            if model_id not in self.models:
                print(f"Warning: Model {model_id} not registered, skipping")
                continue
            
            try:
                # Run inference
                df = self.run_inference(model_id)
                
                if df is not None:
                    # Evaluate
                    metrics = self.evaluate_model(model_id, df)
                    all_metrics[model_id] = metrics
                    
            except Exception as e:
                print(f"Error running {model_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save all metrics
        self.results = all_metrics
        return all_metrics
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        print(f"\n{'='*60}")
        print("GENERATING FINAL REPORT")
        print(f"{'='*60}\n")
        
        if not self.results:
            print("No results to report")
            return
        
        # Create summary table
        summary_rows = []
        for model_id, metrics in self.results.items():
            row = {
                'Model': model_id,
                'Predictions': metrics.get('num_predictions', 0),
                'Avg Time (s)': f"{metrics.get('avg_time', 0):.3f}",
            }
            
            if 'mae' in metrics:
                row['MAE'] = f"{metrics['mae']:.2f}"
                row['RMSE'] = f"{metrics['rmse']:.2f}"
                row['Pearson r'] = f"{metrics['pearson_r']:.3f}"
                row['Accuracy'] = f"{metrics['accuracy']:.2%}"
            
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Sort by MAE if available
        if 'MAE' in summary_df.columns:
            summary_df['MAE_numeric'] = summary_df['MAE'].astype(float)
            summary_df = summary_df.sort_values('MAE_numeric')
            summary_df = summary_df.drop('MAE_numeric', axis=1)
        
        # Print summary table
        print("Summary Table:")
        print(summary_df.to_string(index=False))
        print()
        
        # Save summary
        summary_df.to_csv(self.output_dir / 'summary.csv', index=False)
        
        # Generate markdown report
        self.generate_markdown_report(summary_df)
        
        # Generate JSON summary
        self.generate_json_summary()
    
    def generate_markdown_report(self, summary_df: pd.DataFrame):
        """Generate markdown report file."""
        report_path = self.output_dir / 'report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Speaker Age Estimation - Evaluation Report\n\n")
            f.write(f"**Audio Directory:** `{self.audio_dir}`\n\n")
            f.write(f"**Total Audio Files:** {len(self.audio_files)}\n\n")
            
            if self.ground_truth:
                f.write(f"**Files with Ground Truth:** {len(self.ground_truth)}\n\n")
            
            f.write("## Summary Table\n\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## Model Details\n\n")
            for model_id, metrics in self.results.items():
                f.write(f"### {model_id}\n\n")
                
                if 'mae' in metrics:
                    f.write(f"- **MAE:** {metrics['mae']:.2f} years\n")
                    f.write(f"- **RMSE:** {metrics['rmse']:.2f} years\n")
                    f.write(f"- **Pearson r:** {metrics['pearson_r']:.3f}\n")
                    f.write(f"- **Accuracy (groups):** {metrics['accuracy']:.2%}\n")
                
                f.write(f"- **Avg Inference Time:** {metrics.get('avg_time', 0):.3f} s\n")
                f.write(f"- **Predictions File:** `results/{model_id}.csv`\n\n")
            
            f.write("## Reproduction Commands\n\n")
            f.write("```bash\n")
            f.write("# Install dependencies\n")
            f.write("pip install -r requirements.txt\n\n")
            f.write("# Run evaluation\n")
            f.write(f"python run_age_eval.py --audio-dir {self.audio_dir}\n")
            f.write("```\n\n")
            
            f.write("## Recommendations\n\n")
            
            if 'mae' in list(self.results.values())[0]:
                # Find best model by MAE
                best_model = min(self.results.items(), key=lambda x: x[1].get('mae', float('inf')))
                f.write(f"**Best Model (by MAE):** {best_model[0]} with MAE of {best_model[1]['mae']:.2f} years\n\n")
                
                # Find fastest model
                fastest_model = min(self.results.items(), key=lambda x: x[1].get('avg_time', float('inf')))
                f.write(f"**Fastest Model:** {fastest_model[0]} with {fastest_model[1]['avg_time']:.3f} s per file\n\n")
            
            f.write("### Production Considerations\n\n")
            f.write("- Consider model quantization for faster inference\n")
            f.write("- Implement calibration for confidence scores\n")
            f.write("- Test on diverse datasets for robustness\n")
            f.write("- Monitor for bias across age groups and demographics\n")
        
        print(f"Markdown report saved to {report_path}")
    
    def generate_json_summary(self):
        """Generate JSON summary file."""
        summary_path = self.output_dir / 'summary.json'
        
        # Find best model
        best_model = None
        best_mae = float('inf')
        
        for model_id, metrics in self.results.items():
            if 'mae' in metrics and metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_model = model_id
        
        summary = {
            'audio_directory': str(self.audio_dir),
            'total_audio_files': len(self.audio_files),
            'files_with_ground_truth': len(self.ground_truth),
            'models_evaluated': list(self.results.keys()),
            'best_model': {
                'name': best_model,
                'mae': best_mae if best_model else None
            },
            'all_results': self.results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"JSON summary saved to {summary_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run age estimation model evaluation'
    )
    parser.add_argument(
        '--audio-dir',
        type=str,
        default=AUDIO_DIR,
        help=f'Directory containing audio files (default: {AUDIO_DIR})'
    )
    parser.add_argument(
        '--models',
        type=str,
        default='all',
        help='Comma-separated list of models to run (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Check if audio directory exists
    if not os.path.exists(args.audio_dir):
        print(f"WARNING: Audio directory {args.audio_dir} does not exist!")
        print("Creating example directory structure...")
        os.makedirs(args.audio_dir, exist_ok=True)
        print(f"Please add audio files to {args.audio_dir} and run again.")
        return
    
    # Initialize evaluator
    evaluator = AgeEstimationEvaluator(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir
    )
    
    # Register models
    print("\nRegistering models...")
    
    # Model 1: Wav2Vec2 age model
    try:
        wav2vec2_model = Wav2Vec2AgeModel(
            model_name="audeering/wav2vec2-large-robust-24-ft-age-gender"
        )
        evaluator.register_model(wav2vec2_model, "wav2vec2-audeering")
    except Exception as e:
        print(f"Failed to register Wav2Vec2 model: {e}")
    
    # Model 2: WavLM model (demonstration - not fully trained)
    try:
        wavlm_model = WavLMAgeModel()
        evaluator.register_model(wavlm_model, "wavlm-base")
    except Exception as e:
        print(f"Failed to register WavLM model: {e}")
    
    # Model 3: SpeechBrain ECAPA model
    try:
        sb_model = SpeechBrainAgeModel()
        evaluator.register_model(sb_model, "speechbrain-ecapa")
    except Exception as e:
        print(f"Failed to register SpeechBrain model: {e}")
    
    # Model 4: MFCC baseline
    try:
        baseline_model = MFCCBaselineModel()
        evaluator.register_model(baseline_model, "mfcc-baseline")
    except Exception as e:
        print(f"Failed to register baseline model: {e}")
    
    # Parse model selection
    if args.models == 'all':
        model_ids = None
    else:
        model_ids = [m.strip() for m in args.models.split(',')]
    
    # Run evaluation
    print(f"\nStarting evaluation...")
    print(f"Models to evaluate: {model_ids if model_ids else 'all'}")
    
    evaluator.run_all_models(model_ids)
    
    # Generate report
    evaluator.generate_report()
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}")
    print(f"- CSV files: {args.output_dir}/*.csv")
    print(f"- Report: {args.output_dir}/report.md")
    print(f"- Summary: {args.output_dir}/summary.json")


if __name__ == '__main__':
    main()
