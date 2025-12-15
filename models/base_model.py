"""Base class for all age estimation models."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseAgeModel(ABC):
    """Abstract base class for age estimation models."""
    
    def __init__(self, model_name: str):
        """
        Initialize the model.
        
        Args:
            model_name: Unique identifier for the model
        """
        self.model_name = model_name
        self.sample_rate = 16000  # Default sample rate
        self.loaded = False
        
    @abstractmethod
    def load_model(self):
        """Load the model and its weights."""
        pass
    
    @abstractmethod
    def predict_age(self, waveform: np.ndarray, sr: int) -> Tuple[float, float]:
        """
        Predict age from audio waveform.
        
        Args:
            waveform: Audio waveform as numpy array
            sr: Sample rate of the audio
            
        Returns:
            Tuple of (predicted_age_years, confidence_score)
        """
        pass
    
    def get_model_info(self) -> dict:
        """Return model metadata."""
        return {
            'name': self.model_name,
            'sample_rate': self.sample_rate,
            'loaded': self.loaded,
        }
