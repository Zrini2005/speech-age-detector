"""
Predict age and characteristics for audio files in test folder
Scans /mnt/c/Users/srini/Downloads/testing/ and creates predictions for each model
"""

import os
import sys
import pandas as pd
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
import json
from datetime import datetime

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add models to path
sys.path.append(os.path.dirname(__file__))

# Import models
from models.wav2vec2_age_model import Wav2Vec2AgeModel
from models.audeering_6layer_model import Audeering6LayerAgeModel
from models.wavlm_age_model import WavLMAgeModel


class TestFolderPredictor:
    """Predict age for all audio files in test folder"""
    
    def __init__(self, test_folder="/mnt/c/Users/srini/Downloads/testing"):
        self.test_folder = test_folder
        self.models = {}
        self.predictions = {}
        
        # Create results directory
        self.results_dir = "results/test_predictions"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def scan_audio_files(self):
        """Scan test folder for audio files"""
        print(f"\nScanning folder: {self.test_folder}")
        print("=" * 80)
        
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        audio_files = []
        
        if not os.path.exists(self.test_folder):
            print(f"ERROR: Folder not found: {self.test_folder}")
            return []
        
        for root, dirs, files in os.walk(self.test_folder):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in audio_extensions:
                    full_path = os.path.join(root, file)
                    audio_files.append(full_path)
        
        print(f"Found {len(audio_files)} audio files")
        return sorted(audio_files)
    
    def initialize_models(self):
        """Initialize all available models"""
        print("\n" + "=" * 80)
        print("INITIALIZING MODELS")
        print("=" * 80)
        
        # Model 1: Audeering 24-layer Wav2Vec2
        print("\n[1/6] Audeering 24-layer Wav2Vec2...")
        try:
            model1 = Wav2Vec2AgeModel()
            model1.load_model()
            self.models['audeering_24layer'] = model1
            print("✓ Loaded successfully")
        except Exception as e:
            print(f"✗ Failed: {e}")
        
        # Model 2: Audeering 6-layer Wav2Vec2
        print("\n[2/6] Audeering 6-layer Wav2Vec2...")
        try:
            model2 = Audeering6LayerAgeModel()
            model2.load_model()
            self.models['audeering_6layer'] = model2
            print("✓ Loaded successfully")
        except Exception as e:
            print(f"✗ Failed: {e}")
        
        # Model 3: WavLM + Acoustic
        print("\n[3/6] WavLM + Acoustic Analysis...")
        try:
            model3 = WavLMAgeModel()
            model3.load_model()
            self.models['wavlm_acoustic'] = model3
            print("✓ Loaded successfully")
        except Exception as e:
            print(f"✗ Failed: {e}")
        
        # Model 4: Vox-Profile WavLM
        print("\n[4/6] Vox-Profile WavLM Age-Sex...")
        try:
            from models.voxprofile_wavlm_model import VoxProfileWavLMModel
            model4 = VoxProfileWavLMModel()
            model4.load_model()
            self.models['voxprofile_wavlm'] = model4
            print("✓ Loaded successfully")
        except Exception as e:
            print(f"✗ Failed: {e}")
        
        # Model 5: Anvarjon CNN
        print("\n[5/6] Anvarjon CNN + MAM...")
        anvarjon_age_model = "anvarjon_repo/models/age/best_model_age.h5"
        if os.path.exists(anvarjon_age_model):
            try:
                from models.anvarjon_cnn_model import AnvarjonCNNModel
                model5 = AnvarjonCNNModel()
                model5.load_pretrained_weights(age_model_path=anvarjon_age_model)
                self.models['anvarjon_cnn'] = model5
                print("✓ Loaded successfully")
            except Exception as e:
                print(f"✗ Failed: {e}")
        else:
            print("✗ Model weights not found (skipping)")
        
        # Model 6: SpeechBrain ECAPA-TDNN
        print("\n[6/6] SpeechBrain ECAPA-TDNN Age...")
        try:
            from models.speechbrain_age_model import SpeechBrainAgeModel
            model6 = SpeechBrainAgeModel()
            model6.load_model()
            self.models['speechbrain_ecapa'] = model6
            print("✓ Loaded successfully")
        except Exception as e:
            print(f"✗ Failed: {e}")
        
        print(f"\n{'=' * 80}")
        print(f"LOADED {len(self.models)} MODELS")
        print(f"{'=' * 80}\n")
    
    def predict_file(self, model_name, model, audio_path):
        """Make prediction for a single file"""
        try:
            # Load audio file - exactly like run_comprehensive_eval.py
            waveform, sample_rate = sf.read(audio_path)
            
            # Ensure mono
            if waveform.ndim > 1:
                waveform = waveform[:, 0]
            
            # Predict - pass numpy array (models handle it correctly)
            age_pred, confidence = model.predict_age(waveform, sample_rate)
            
            # Extract relevant information
            output = {
                'filename': os.path.basename(audio_path),
                'full_path': audio_path,
                'model': model_name,
                'predicted_age': float(age_pred),
                'confidence': float(confidence)
            }
            
            # Try to get additional details if model has them
            if hasattr(model, 'last_prediction_details') and model.last_prediction_details:
                details = model.last_prediction_details
                
                # Add gender if available
                if 'gender' in details:
                    output['gender'] = details['gender']
                if 'gender_probs' in details:
                    output['gender_probs'] = str(details['gender_probs'])
                
                # Add acoustic features if available
                if 'pitch' in details:
                    output['pitch'] = float(details['pitch'])
                if 'energy' in details:
                    output['energy'] = float(details['energy'])
                if 'is_child' in details:
                    output['is_child'] = details['is_child']
            
            return output
            
        except Exception as e:
            import traceback
            return {
                'filename': os.path.basename(audio_path),
                'full_path': audio_path,
                'model': model_name,
                'predicted_age': 'ERROR',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def run_predictions(self):
        """Run predictions on all files with all models"""
        # Get audio files
        audio_files = self.scan_audio_files()
        
        if not audio_files:
            print("No audio files found!")
            return
        
        # Initialize models
        self.initialize_models()
        
        if not self.models:
            print("No models loaded!")
            return
        
        # Run predictions for each model
        for model_name, model in self.models.items():
            print(f"\n{'=' * 80}")
            print(f"PREDICTIONS: {model_name}")
            print(f"{'=' * 80}")
            
            predictions = []
            
            for i, audio_file in enumerate(audio_files, 1):
                print(f"\n[{i}/{len(audio_files)}] {os.path.basename(audio_file)}")
                
                result = self.predict_file(model_name, model, audio_file)
                predictions.append(result)
                
                # Print result
                age = result.get('predicted_age', 'N/A')
                print(f"  → Age: {age}")
                
                # Print other characteristics
                for key in ['gender', 'pitch', 'energy', 'confidence']:
                    if key in result and result[key]:
                        print(f"  → {key.capitalize()}: {result[key]}")
            
            # Save predictions for this model
            self.save_model_predictions(model_name, predictions)
            self.predictions[model_name] = predictions
        
        # Create summary
        self.create_summary()
    
    def save_model_predictions(self, model_name, predictions):
        """Save predictions to CSV file"""
        output_file = os.path.join(self.results_dir, f"{model_name}_predictions.csv")
        df = pd.DataFrame(predictions)
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved to: {output_file}")
    
    def create_summary(self):
        """Create summary with all models' predictions"""
        print("\n" + "=" * 80)
        print("CREATING SUMMARY")
        print("=" * 80)
        
        # Get all unique files
        all_files = set()
        for model_predictions in self.predictions.values():
            for pred in model_predictions:
                all_files.add(pred['filename'])
        
        all_files = sorted(all_files)
        
        # Create summary data
        summary_data = []
        for filename in all_files:
            row = {'filename': filename}
            
            # Add predictions from each model
            for model_name, model_predictions in self.predictions.items():
                # Find prediction for this file
                pred = next((p for p in model_predictions if p['filename'] == filename), None)
                if pred:
                    age = pred.get('predicted_age', 'N/A')
                    row[f'{model_name}_age'] = age
                    
                    # Add other characteristics
                    for key in ['gender', 'pitch', 'energy', 'confidence']:
                        if key in pred:
                            row[f'{model_name}_{key}'] = pred[key]
                else:
                    row[f'{model_name}_age'] = 'N/A'
            
            summary_data.append(row)
        
        # Save summary
        summary_file = os.path.join(self.results_dir, "all_models_summary.csv")
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_file, index=False)
        print(f"\n✓ Summary saved to: {summary_file}")
        
        # Create readable report
        self.create_report(summary_data)
    
    def create_report(self, summary_data):
        """Create human-readable report"""
        report_file = os.path.join(self.results_dir, "predictions_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AGE PREDICTION REPORT\n")
            f.write(f"Test Folder: {self.test_folder}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models: {len(self.models)}\n")
            f.write(f"Audio Files: {len(summary_data)}\n")
            f.write("=" * 80 + "\n\n")
            
            for item in summary_data:
                f.write(f"\nFile: {item['filename']}\n")
                f.write("-" * 80 + "\n")
                
                # List all model predictions
                for model_name in self.models.keys():
                    age_key = f'{model_name}_age'
                    if age_key in item:
                        age = item[age_key]
                        model_desc = self.models[model_name]['description']
                        f.write(f"  {model_desc:40s}: Age = {age}\n")
                        
                        # Add other characteristics
                        for char in ['gender', 'pitch', 'energy', 'confidence']:
                            char_key = f'{model_name}_{char}'
                            if char_key in item:
                                f.write(f"  {' ':40s}  {char.capitalize()} = {item[char_key]}\n")
                
                f.write("\n")
        
        print(f"✓ Report saved to: {report_file}")
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total files processed: {len(summary_data)}")
        print(f"Models used: {len(self.models)}")
        print(f"\nResults saved in: {self.results_dir}/")
        print("  - Individual model CSVs")
        print("  - all_models_summary.csv (combined)")
        print("  - predictions_report.txt (readable)")


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("TEST FOLDER PREDICTION TOOL")
    print("=" * 80)
    
    # Allow custom test folder path
    test_folder = "/mnt/c/Users/srini/Downloads/testing"
    if len(sys.argv) > 1:
        test_folder = sys.argv[1]
    
    predictor = TestFolderPredictor(test_folder=test_folder)
    predictor.run_predictions()
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
