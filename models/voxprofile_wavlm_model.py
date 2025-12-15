"""
Vox-Profile WavLM Age-Sex Model
Paper: "Vox-Profile: A Speech Foundation Model Benchmark for Characterizing
Diverse Speaker and Speech Traits" (arXiv:2505.14648)
GitHub: https://github.com/tiantiaf0627/vox-profile-release
HuggingFace: tiantiaf/wavlm-large-age-sex
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
import logging

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Monkey patch for torchaudio compatibility with speechbrain
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: 'soundfile'

# Add vox_profile_repo to path
vox_profile_path = os.path.join(os.path.dirname(__file__), '../vox_profile_repo/src/model/age_sex')
sys.path.insert(0, vox_profile_path)


class VoxProfileWavLMModel:
    """
    WavLM-based age and sex prediction model from Vox-Profile benchmark
    Trained on: CommonVoice + TIMIT + VoxCeleb (age enriched)
    Outputs: Age (0-100 years) + Sex (Female/Male)
    """
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "tiantiaf/wavlm-large-age-sex"
        self.max_audio_length = 15 * 16000  # 15 seconds max
        self.sex_labels = ["Female", "Male"]
        
    def load_model(self):
        """Load pretrained model from HuggingFace"""
        try:
            print(f"Loading Vox-Profile WavLM model from {self.model_name}...")
            
            # Import the custom WavLMWrapper from vox_profile repo
            from wavlm_demographics import WavLMWrapper
            
            # Load the pretrained model
            self.model = WavLMWrapper.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print(f"✓ VoxProfile WavLM model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"✗ Error loading VoxProfile model: {e}")
            print("Make sure all dependencies are installed:")
            print("  pip install loralib speechbrain transformers")
            raise
    
    def predict_age(self, waveform, sample_rate):
        """
        Predict age from audio waveform
        Args:
            waveform: numpy array or torch tensor of audio samples
            sample_rate: sampling rate (will resample to 16kHz if needed)
        Returns:
            (age_years, confidence) tuple
        """
        if self.model is None:
            self.load_model()
        
        # Convert to torch tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        
        # Ensure float32
        waveform = waveform.float()
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Trim or pad to max length
        if waveform.shape[0] > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        
        # Add batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Move to device
        waveform = waveform.to(self.device)
        
        # Predict
        with torch.no_grad():
            age_output, sex_output = self.model(waveform)
            
            # Age is 0-1, multiply by 100 for years
            age_pred = age_output.detach().cpu().numpy()[0][0] * 100
            
            # Sex probabilities
            sex_prob = F.softmax(sex_output, dim=1)
            sex_pred_idx = torch.argmax(sex_prob).detach().cpu().item()
            sex_confidence = sex_prob[0][sex_pred_idx].detach().cpu().item()
        
        # Age confidence: inverse of distance from midpoint (0.5)
        # More confident when closer to 0 (young) or 1 (old)
        age_normalized = age_output.detach().cpu().numpy()[0][0]
        age_confidence = float(abs(age_normalized - 0.5) * 2)  # 0 to 1 scale
        
        return float(age_pred), float(age_confidence)
    
    def predict_sex(self, waveform, sample_rate):
        """
        Predict sex from audio waveform
        Returns: (sex_label, confidence)
        """
        if self.model is None:
            self.load_model()
        
        # Convert to torch tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        
        # Ensure float32
        waveform = waveform.float()
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Trim or pad to max length
        if waveform.shape[0] > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        
        # Add batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Move to device
        waveform = waveform.to(self.device)
        
        # Predict
        with torch.no_grad():
            age_output, sex_output = self.model(waveform)
            
            # Sex probabilities
            sex_prob = F.softmax(sex_output, dim=1)
            sex_pred_idx = torch.argmax(sex_prob).detach().cpu().item()
            sex_confidence = sex_prob[0][sex_pred_idx].detach().cpu().item()
            sex_label = self.sex_labels[sex_pred_idx]
        
        return sex_label, float(sex_confidence)
    
    def predict_combined(self, waveform, sample_rate):
        """
        Predict both age and sex in single forward pass
        Returns: (age, age_conf, sex, sex_conf)
        """
        if self.model is None:
            self.load_model()
        
        # Convert to torch tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        
        # Ensure float32
        waveform = waveform.float()
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Trim or pad to max length
        if waveform.shape[0] > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        
        # Add batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Move to device
        waveform = waveform.to(self.device)
        
        # Predict
        with torch.no_grad():
            age_output, sex_output = self.model(waveform)
            
            # Age is 0-1, multiply by 100 for years
            age_pred = age_output.detach().cpu().numpy()[0][0] * 100
            
            # Age confidence
            age_normalized = age_output.detach().cpu().numpy()[0][0]
            age_confidence = float(abs(age_normalized - 0.5) * 2)
            
            # Sex probabilities
            sex_prob = F.softmax(sex_output, dim=1)
            sex_pred_idx = torch.argmax(sex_prob).detach().cpu().item()
            sex_confidence = sex_prob[0][sex_pred_idx].detach().cpu().item()
            sex_label = self.sex_labels[sex_pred_idx]
        
        return float(age_pred), float(age_confidence), sex_label, float(sex_confidence)


if __name__ == "__main__":
    # Example usage
    model = VoxProfileWavLMModel()
    
    try:
        model.load_model()
        
        # Test with dummy audio (1 second at 16kHz)
        dummy_audio = torch.zeros(16000)
        age, age_conf, sex, sex_conf = model.predict_combined(dummy_audio, 16000)
        
        print(f"\nTest prediction:")
        print(f"Age: {age:.1f} years (confidence: {age_conf:.2f})")
        print(f"Sex: {sex} (confidence: {sex_conf:.2f})")
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("\nModel information:")
        print(f"HuggingFace: {model.model_name}")
        print(f"Max audio length: {model.max_audio_length/16000:.1f} seconds")
        print(f"Output: Age (0-100 years) + Sex (Female/Male)")
