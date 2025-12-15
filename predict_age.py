"""
Age Prediction from JSON Input
Runs Audeering 24-layer and/or Vox-Profile WavLM models on audio files
Input: JSON array with list of audio file paths
Output: JSON array with predictions (age + adult/child category)

Usage:
    python predict_age_json.py input.json --model audeering --output output.json
    python predict_age_json.py input.json --model voxprofile --output output.json
    python predict_age_json.py input.json --model both --output output.json
"""

import os
import sys
import json
import argparse
import numpy as np
import soundfile as sf
import torch

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add models to path
sys.path.append(os.path.dirname(__file__))

from models.wav2vec2_age_model import Wav2Vec2AgeModel
from models.voxprofile_wavlm_model import VoxProfileWavLMModel


def load_model(model_name):
    """Load specified model"""
    if model_name == 'audeering':
        print("Loading Audeering 24-layer Wav2Vec2...")
        model = Wav2Vec2AgeModel()
        model.load_model()
        return model
    elif model_name == 'voxprofile':
        print("Loading Vox-Profile WavLM...")
        model = VoxProfileWavLMModel()
        model.load_model()
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")


def categorize_age(age):
    """Categorize age as adult or child"""
    return "child" if age < 18 else "adult"


def predict_audio_file(audio_path, model):
    """Make prediction for a single audio file"""
    try:
        # Load audio
        waveform, sample_rate = sf.read(audio_path)
        
        # Ensure mono
        if waveform.ndim > 1:
            waveform = waveform[:, 0]
        
        # Predict age
        age, confidence = model.predict_age(waveform, sample_rate)
        
        return {
            "audio_path": audio_path,
            "age": round(float(age)),
            "category": categorize_age(age)
        }
        
    except Exception as e:
        return {
            "audio_path": audio_path,
            "age": None,
            "category": None,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Predict age from audio files using JSON input/output')
    parser.add_argument('--input', '-i', required=True,
                        help='Input JSON file with array of audio file paths')
    parser.add_argument('--model', '-m', required=True, choices=['audeering', 'voxprofile', 'both'], 
                        help='Model to use')
    parser.add_argument('--output', '-o', required=True, 
                        help='Output JSON file for predictions')
    
    args = parser.parse_args()
    
    # Load input JSON (should be an array of paths)
    print(f"Reading input from {args.input}...")
    with open(args.input, 'r') as f:
        audio_files = json.load(f)
    
    if not isinstance(audio_files, list):
        print("ERROR: Input JSON must be an array of file paths")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio files")
    
    # Load model(s)
    models = {}
    if args.model in ['audeering', 'both']:
        models['audeering_24layer'] = load_model('audeering')
    if args.model in ['voxprofile', 'both']:
        models['voxprofile'] = load_model('voxprofile')
    
    # Process each audio file
    results = []
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_path}")
        
        if args.model == 'both':
            # Run both models and store separate results
            file_result = {"audio_path": audio_path}
            
            for model_name, model in models.items():
                print(f"  Running {model_name}...")
                prediction = predict_audio_file(audio_path, model)
                
                if 'error' in prediction:
                    file_result[model_name] = {"error": prediction['error']}
                else:
                    file_result[model_name] = {
                        "age": prediction['age'],
                        "category": prediction['category']
                    }
            
            results.append(file_result)
        else:
            # Single model - flat output format
            model_name = 'audeering_24layer' if args.model == 'audeering' else 'voxprofile'
            model = models[model_name]
            print(f"  Running {args.model}...")
            prediction = predict_audio_file(audio_path, model)
            results.append(prediction)
    
    # Save output JSON
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Complete! Processed {len(results)} files")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
