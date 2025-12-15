"""Baseline age estimation models using classical ML."""

import numpy as np
from typing import Tuple
import librosa
from .base_model import BaseAgeModel


class MFCCBaselineModel(BaseAgeModel):
    """
    Baseline age estimation using MFCC features + classical ML.
    
    This serves as a simple baseline for comparison.
    """
    
    def __init__(self, model_name: str = "mfcc-baseline"):
        super().__init__(model_name)
        self.model = None
        self.scaler = None
        
    def load_model(self):
        """Initialize baseline model."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            print(f"Loading {self.model_name}...")
            
            # Create untrained models (in practice, these would be loaded from checkpoint)
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.scaler = StandardScaler()
            
            # Placeholder: fit on dummy data to make it usable
            # In practice, you'd load a trained model
            dummy_features = np.random.randn(100, 40)  # 40 MFCCs
            dummy_ages = np.random.uniform(10, 80, 100)
            
            dummy_features_scaled = self.scaler.fit_transform(dummy_features)
            self.model.fit(dummy_features_scaled, dummy_ages)
            
            self.loaded = True
            print(f"{self.model_name} loaded (note: using random forest - not trained on real data)")
            
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            self.loaded = False
            raise
    
    def extract_mfcc_features(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features from audio."""
        # Extract 40 MFCCs
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
        
        # Compute statistics over time (only mean for consistency)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        return mfcc_mean
    
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
            # Extract features
            features = self.extract_mfcc_features(waveform, sr)
            features = features.reshape(1, -1)
            
            # Scale and predict
            features_scaled = self.scaler.transform(features)
            age = self.model.predict(features_scaled)[0]
            
            # Clip to valid range
            age = np.clip(age, 0, 100)
            
            # Estimate confidence (placeholder)
            confidence = 0.4
            
            return float(age), float(confidence)
            
        except Exception as e:
            print(f"Error during prediction with {self.model_name}: {e}")
            return 35.0, 0.0
