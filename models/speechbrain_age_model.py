"""SpeechBrain-based age estimation using ECAPA-TDNN embeddings."""

import os
import torch
import torchaudio
import numpy as np
from typing import Tuple
from .base_model import BaseAgeModel

# Monkey patch for torchaudio compatibility with speechbrain
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: 'soundfile'


class SpeechBrainAgeModel(BaseAgeModel):
    """
    Age estimation using SpeechBrain ECAPA-TDNN embeddings + regressor.
    
    Uses pretrained speaker embeddings as features for age prediction.
    """
    
    def __init__(self, model_name: str = "speechbrain-ecapa-age"):
        super().__init__(model_name)
        self.embedding_model = None
        self.regressor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load SpeechBrain ECAPA-TDNN model."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            
            print(f"Loading {self.model_name}...")
            # Load pretrained speaker embedding model
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            
            # Simple regression head (not trained - placeholder)
            self.regressor = torch.nn.Sequential(
                torch.nn.Linear(192, 128),  # ECAPA outputs 192-dim embeddings
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(128, 1)
            ).to(self.device)
            
            self.loaded = True
            print(f"{self.model_name} loaded (note: using random regressor - not trained)")
            
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            self.loaded = False
            raise
    
    def predict_age(self, waveform, sr: int) -> Tuple[float, float]:
        """
        Predict age from audio waveform.
        
        Args:
            waveform: Audio waveform as numpy array or torch tensor
            sr: Sample rate of the audio
            
        Returns:
            Tuple of (predicted_age_years, confidence_score)
        """
        if not self.loaded:
            self.load_model()
        
        try:
            # Convert to numpy if tensor
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            
            # Convert to tensor
            waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
            
            # Extract speaker embeddings
            with torch.no_grad():
                embeddings = self.embedding_model.encode_batch(waveform_tensor)
                # Predict age from embeddings
                age_pred = self.regressor(embeddings.to(self.device))
            
            age = torch.clamp(age_pred[0, 0], min=0, max=100).item()
            confidence = 0.3  # Low confidence - not trained
            
            return float(age), float(confidence)
            
        except Exception as e:
            print(f"Error during prediction with {self.model_name}: {e}")
            return 35.0, 0.0
