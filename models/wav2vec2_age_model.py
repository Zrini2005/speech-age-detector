"""Wav2Vec2-based age estimation models from HuggingFace.

This module implements the CORRECT usage of audeering/wav2vec2-large-robust-24-ft-age-gender
using their custom model architecture (not Wav2Vec2ForSequenceClassification).

The model outputs:
- age: 0-1 range representing 0-100 years (needs *100 scaling)
- gender: 3 classes [female, male, child] as probabilities

Reference: https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Tuple, Dict, Optional
from .base_model import BaseAgeModel


class ModelHead(nn.Module):
    """
    Custom head for the audeering age-gender model.
    """
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderModel(nn.Module):
    """
    Custom Wav2Vec2 model for age and gender prediction.
    Architecture from audeering/wav2vec2-large-robust-24-ft-age-gender.
    
    Outputs:
    - age: single value in [0, 1] range (multiply by 100 for years)
    - gender: 3 values [female_prob, male_prob, child_prob]
    """
    def __init__(self, config):
        super().__init__()
        from transformers import Wav2Vec2Model
        
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)  # Single output for age
        self.gender = ModelHead(config, 3)  # 3 classes: female, male, child

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        # Mean pooling across time dimension
        hidden_states = torch.mean(hidden_states, dim=1)
        
        age_output = self.age(hidden_states)
        gender_output = self.gender(hidden_states)
        
        return hidden_states, age_output, gender_output


class Wav2Vec2AgeModel(BaseAgeModel):
    """
    Wav2Vec2-based age estimation using audeering model.
    
    This implementation uses the correct custom architecture that outputs:
    - Age in 0-1 range (scaled to 0-100 years)
    - Gender probabilities including child detection
    
    Model: audeering/wav2vec2-large-robust-24-ft-age-gender
    """
    
    def __init__(self, model_name: str = "audeering/wav2vec2-large-robust-24-ft-age-gender"):
        """
        Initialize Wav2Vec2 age model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        super().__init__(model_name)
        self.hf_model_id = model_name
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = 16000
        
        # Last prediction details for advanced analysis
        self.last_prediction_details: Dict = {}
        
    def load_model(self):
        """Load the Wav2Vec2 model from HuggingFace using correct architecture."""
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Config
            import os
            
            print(f"Loading {self.model_name} (custom age-gender architecture)...")
            
            # Load processor
            self.processor = Wav2Vec2Processor.from_pretrained(self.hf_model_id)
            
            # Load config and initialize custom model
            config = Wav2Vec2Config.from_pretrained(self.hf_model_id)
            self.model = AgeGenderModel(config)
            
            # Load pretrained weights
            from huggingface_hub import hf_hub_download
            model_file = hf_hub_download(self.hf_model_id, "pytorch_model.bin")
            state_dict = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            print(f"{self.model_name} loaded successfully (custom architecture)")
            
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            print("Falling back to alternative loading method...")
            self._load_alternative()
            
    def _load_alternative(self):
        """Alternative loading using transformers auto classes."""
        try:
            from transformers import AutoProcessor, AutoModel, Wav2Vec2Config
            
            print("Attempting alternative load...")
            self.processor = AutoProcessor.from_pretrained(self.hf_model_id)
            
            # Load base model and manually add heads
            config = Wav2Vec2Config.from_pretrained(self.hf_model_id)
            self.model = AgeGenderModel(config)
            
            # Try loading with trust_remote_code
            from huggingface_hub import hf_hub_download
            try:
                model_file = hf_hub_download(self.hf_model_id, "pytorch_model.bin")
                state_dict = torch.load(model_file, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state_dict)
            except Exception as e2:
                print(f"Could not load weights: {e2}")
                # Try safetensors
                try:
                    from safetensors.torch import load_file
                    model_file = hf_hub_download(self.hf_model_id, "model.safetensors")
                    state_dict = load_file(model_file)
                    self.model.load_state_dict(state_dict)
                except:
                    raise RuntimeError("Could not load model weights")
            
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            print(f"{self.model_name} loaded successfully (alternative method)")
            
        except Exception as e:
            print(f"Alternative loading also failed: {e}")
            self.loaded = False
            raise
    
    def predict_age(self, waveform: np.ndarray, sr: int) -> Tuple[float, float]:
        """
        Predict age from audio waveform.
        
        Args:
            waveform: Audio waveform as numpy array
            sr: Sample rate of the audio
            
        Returns:
            Tuple of (predicted_age_years, confidence_score)
        """
        if not self.loaded:
            self.load_model()
        
        try:
            # Resample to 16kHz if needed
            target_sr = self.sampling_rate
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=target_sr
                )
                waveform_tensor = torch.from_numpy(waveform).float()
                if waveform_tensor.dim() == 1:
                    waveform_tensor = waveform_tensor.unsqueeze(0)
                waveform = resampler(waveform_tensor).squeeze().numpy()
            
            # Process audio
            inputs = self.processor(
                waveform,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            input_values = inputs.input_values.to(self.device)
            
            # Inference
            with torch.no_grad():
                hidden, age_output, gender_output = self.model(input_values)
            
            # Process age output - IMPORTANT: output is in 0-1 range, scale to 0-100 years
            age_raw = age_output.squeeze().item()
            age_years = age_raw * 100.0  # Scale to years
            
            # Clamp to reasonable range
            age_years = max(0.0, min(100.0, age_years))
            
            # Process gender output for confidence and child detection
            gender_probs = torch.softmax(gender_output, dim=-1).squeeze()
            female_prob = gender_probs[0].item()
            male_prob = gender_probs[1].item()
            child_prob = gender_probs[2].item()
            
            # Confidence based on gender prediction certainty
            max_gender_prob = max(female_prob, male_prob, child_prob)
            confidence = max_gender_prob
            
            # Store details for advanced analysis
            self.last_prediction_details = {
                'age_raw': age_raw,
                'age_years': age_years,
                'female_prob': female_prob,
                'male_prob': male_prob,
                'child_prob': child_prob,
                'is_child': child_prob > 0.5,
                'predicted_gender': ['female', 'male', 'child'][gender_probs.argmax().item()]
            }
            
            return float(age_years), float(confidence)
            
        except Exception as e:
            print(f"Error during prediction with {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
            return 35.0, 0.0  # Default fallback
    
    def predict_age_gender(self, waveform: np.ndarray, sr: int) -> Dict:
        """
        Predict both age and gender from audio.
        
        Args:
            waveform: Audio waveform as numpy array
            sr: Sample rate of the audio
            
        Returns:
            Dict with age, gender, and child probability
        """
        age, confidence = self.predict_age(waveform, sr)
        return {
            'age': age,
            'confidence': confidence,
            **self.last_prediction_details
        }
