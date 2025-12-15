"""Lightweight 6-layer audeering age-gender model for faster inference."""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Tuple, Dict
from .base_model import BaseAgeModel


class ModelHead(nn.Module):
    """Custom head for the audeering age-gender model."""
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
    """Custom Wav2Vec2 model for age and gender prediction."""
    def __init__(self, config):
        super().__init__()
        from transformers import Wav2Vec2Model
        
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        hidden_states = torch.mean(hidden_states, dim=1)
        
        age_output = self.age(hidden_states)
        gender_output = self.gender(hidden_states)
        
        return hidden_states, age_output, gender_output


class Audeering6LayerAgeModel(BaseAgeModel):
    """
    Faster 6-layer audeering age-gender model.
    
    This is a lighter version with only 6 transformer layers instead of 24,
    providing ~4x faster inference with similar accuracy.
    
    Model: audeering/wav2vec2-large-robust-6-ft-age-gender
    """
    
    def __init__(self, model_name: str = "audeering/wav2vec2-large-robust-6-ft-age-gender"):
        super().__init__(model_name)
        self.hf_model_id = model_name
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = 16000
        self.last_prediction_details: Dict = {}
        
    def load_model(self):
        """Load the 6-layer Wav2Vec2 model."""
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Config
            from huggingface_hub import hf_hub_download
            
            print(f"Loading {self.model_name} (6-layer, faster)...")
            
            self.processor = Wav2Vec2Processor.from_pretrained(self.hf_model_id)
            config = Wav2Vec2Config.from_pretrained(self.hf_model_id)
            self.model = AgeGenderModel(config)
            
            model_file = hf_hub_download(self.hf_model_id, "pytorch_model.bin")
            state_dict = torch.load(model_file, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            print(f"{self.model_name} loaded successfully (6-layer architecture)")
            
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            self.loaded = False
            raise
    
    def predict_age(self, waveform: np.ndarray, sr: int) -> Tuple[float, float]:
        """Predict age from audio waveform."""
        if not self.loaded:
            self.load_model()
        
        try:
            # Resample to 16kHz if needed
            target_sr = self.sampling_rate
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform_tensor = torch.from_numpy(waveform).float()
                if waveform_tensor.dim() == 1:
                    waveform_tensor = waveform_tensor.unsqueeze(0)
                waveform = resampler(waveform_tensor).squeeze().numpy()
            
            # Process audio
            inputs = self.processor(waveform, sampling_rate=target_sr, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)
            
            # Inference
            with torch.no_grad():
                hidden, age_output, gender_output = self.model(input_values)
            
            # Process outputs
            age_raw = age_output.squeeze().item()
            age_years = age_raw * 100.0
            age_years = max(0.0, min(100.0, age_years))
            
            gender_probs = torch.softmax(gender_output, dim=-1).squeeze()
            female_prob = gender_probs[0].item()
            male_prob = gender_probs[1].item()
            child_prob = gender_probs[2].item()
            
            confidence = max(female_prob, male_prob, child_prob)
            
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
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return 35.0, 0.0
    
    def predict_age_gender(self, waveform: np.ndarray, sr: int) -> Dict:
        """Predict both age and gender."""
        age, confidence = self.predict_age(waveform, sr)
        return {
            'age': age,
            'confidence': confidence,
            **self.last_prediction_details
        }
