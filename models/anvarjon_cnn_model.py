"""
Anvarjon Age-Gender CNN Model with Multi-Attention Module (MAM)
Paper: "Age and Gender Recognition Using a Convolutional Neural Network with a
Specially Designed Multi-Attention Module through Speech Spectrograms"
GitHub: https://github.com/Anvarjon/Age-Gender-Classification
"""

import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import logging
import tempfile
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class AnvarjonCNNModel:
    """
    CNN-based age/gender classifier using spectrogram input
    Uses Multi-Attention Module (MAM) with time and frequency attention
    """
    
    def __init__(self):
        self.model_age = None
        self.model_gender = None
        
        # Spectrogram parameters
        self.sampling_rate = 16000
        self.n_fft = 256
        self.num_overlap = 128
        self._spec_img_size = (64, 64)
        
        # Age groups mapping (6 classes)
        self._decode_age = {
            0: 'teens', 1: 'twenties', 2: 'thirties',
            3: 'fourties', 4: 'fifties', 5: 'sixties'
        }
        self._age_midpoints = {
            0: 15, 1: 25, 2: 35, 3: 45, 4: 55, 5: 65
        }
        
        # Gender mapping (2 classes)
        self._decode_gender = {0: 'male', 1: 'female'}
        
    def scale_minmax(self, X, min_val=0, max_val=255):
        """Min-max scaling for spectrogram"""
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max_val - min_val) + min_val
        return X_scaled
    
    def save_spectrogram(self, data, fn):
        """Save spectrogram as image"""
        plt.axis('off')
        fig = plt.imshow(data, aspect='auto', origin='lower', interpolation='none')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def extract_spectrogram(self, audio_file):
        """Extract spectrogram from audio file"""
        # Load audio
        y, sr = librosa.load(audio_file, sr=self.sampling_rate)
        
        # Compute STFT
        spec = librosa.stft(y, n_fft=self.n_fft, hop_length=self.num_overlap)
        spec = librosa.amplitude_to_db(np.abs(spec))
        
        # Min-max scale to fit inside 8-bit range
        img = self.scale_minmax(spec).astype(np.uint8)
        
        # Save temporarily and load as image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            out_file = tmp.name
        
        try:
            self.save_spectrogram(img, out_file)
            inp_spec = img_to_array(load_img(out_file, target_size=self._spec_img_size))
            return inp_spec
        finally:
            if os.path.exists(out_file):
                os.remove(out_file)
    
    def build_model(self, num_classes, inp_shape=(64, 64, 3), reg_value=0.01):
        """
        Build CNN model with Multi-Attention Module (MAM)
        Architecture from paper with FLB + MAM + FLB structure
        """
        reg = regularizers.l2(l2=reg_value) if reg_value else None
        
        def FLB(inp):
            """Feature Learning Block"""
            x = layers.Conv2D(filters=120, kernel_size=(9, 9), strides=(2, 2), 
                            activation='relu', kernel_regularizer=reg, padding='same')(inp)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
            x = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), 
                            activation='relu', kernel_regularizer=reg, padding='same')(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Conv2D(filters=384, kernel_size=(3, 3), 
                            activation='relu', kernel_regularizer=reg, padding='same')(x)
            x = layers.BatchNormalization()(x)
            return x
        
        def time_attention(inp):
            """Time attention branch - rectangular kernel (1, width)"""
            x = layers.Conv2D(filters=64, kernel_size=(1, 9), activation='relu', 
                            kernel_regularizer=reg, padding='same')(inp)
            x = layers.Conv2D(filters=64, kernel_size=(1, 3), activation='relu', 
                            kernel_regularizer=reg, padding='same')(x)
            x = layers.Conv2D(filters=64, kernel_size=(1, 3), activation='relu', 
                            kernel_regularizer=reg, padding='same')(x)
            x = layers.BatchNormalization()(x)
            return x
        
        def frequency_attention(inp):
            """Frequency attention branch - rectangular kernel (height, 1)"""
            x = layers.Conv2D(filters=64, kernel_size=(9, 1), activation='relu', 
                            kernel_regularizer=reg, padding='same')(inp)
            x = layers.Conv2D(filters=64, kernel_size=(3, 1), activation='relu', 
                            kernel_regularizer=reg, padding='same')(x)
            x = layers.Conv2D(filters=64, kernel_size=(3, 1), activation='relu', 
                            kernel_regularizer=reg, padding='same')(x)
            x = layers.BatchNormalization()(x)
            return x
        
        def MAM(inp):
            """Multi-Attention Module - combines time and frequency attention"""
            ta = time_attention(inp)
            fa = frequency_attention(inp)
            mam = layers.concatenate([ta, fa])
            mam = layers.BatchNormalization()(mam)
            return mam
        
        # Build full model
        inp = keras.Input(shape=inp_shape)
        
        # First feature learning block (FLB-1)
        x = FLB(inp)
        
        # Multi-attention module (MAM)
        mam = MAM(x)
        
        # Concatenate FLB-1 and MAM outputs
        x = layers.concatenate([x, mam])
        
        # Second feature learning block (FLB-2)
        x = FLB(x)
        
        # Classification head
        x = layers.Flatten()(x)
        x = layers.Dense(80, activation='relu', kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        out = layers.Dense(units=num_classes, activation='softmax')(x)
        
        model = keras.Model(inp, out)
        return model
    
    def load_pretrained_weights(self, age_model_path=None, gender_model_path=None):
        """
        Load pretrained model weights
        Download from: https://1drv.ms/u/s!AtLl-Rpr0uJohKJ6_236uKDuJsLkhA?e=7zmPvM
        """
        if age_model_path and os.path.exists(age_model_path):
            self.model_age = load_model(age_model_path)
            print(f"Loaded age model from {age_model_path}")
        
        if gender_model_path and os.path.exists(gender_model_path):
            self.model_gender = load_model(gender_model_path)
            print(f"Loaded gender model from {gender_model_path}")
    
    def predict_age(self, waveform, sample_rate):
        """
        Predict age group from audio waveform
        Returns: (age_years, confidence)
        """
        if self.model_age is None:
            return 35.0, 0.0  # Default middle age
        
        # Save waveform temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_file = tmp.name
        
        try:
            import soundfile as sf
            sf.write(tmp_file, waveform.numpy() if hasattr(waveform, 'numpy') else waveform, sample_rate)
            
            # Extract spectrogram
            inp_spec = self.extract_spectrogram(tmp_file)
            inp_spec = np.expand_dims(inp_spec/255.0, axis=0)
            inp_spec = inp_spec.reshape(-1, 64, 64, 3)
            
            # Predict
            prediction = self.model_age.predict(inp_spec, verbose=0)
            age_class = int(np.argmax(prediction, axis=1)[0])
            confidence = float(prediction[0][age_class])
            
            # Convert class to approximate age
            age_years = self._age_midpoints.get(age_class, 35)
            
            return float(age_years), float(confidence)
        
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
    
    def predict_gender(self, waveform, sample_rate):
        """Predict gender from audio waveform"""
        if self.model_gender is None:
            return "unknown", 0.0
        
        # Save waveform temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_file = tmp.name
        
        try:
            import soundfile as sf
            sf.write(tmp_file, waveform.numpy() if hasattr(waveform, 'numpy') else waveform, sample_rate)
            
            # Extract spectrogram
            inp_spec = self.extract_spectrogram(tmp_file)
            inp_spec = np.expand_dims(inp_spec/255.0, axis=0)
            inp_spec = inp_spec.reshape(-1, 64, 64, 3)
            
            # Predict
            prediction = self.model_gender.predict(inp_spec, verbose=0)
            gender_class = int(np.argmax(prediction, axis=1)[0])
            confidence = float(prediction[0][gender_class])
            
            gender_label = self._decode_gender.get(gender_class, "unknown")
            
            return gender_label, float(confidence)
        
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
    
    def predict_combined(self, waveform, sample_rate):
        """
        Predict both age and gender
        Returns: (age, age_conf, gender, gender_conf)
        """
        age, age_conf = self.predict_age(waveform, sample_rate)
        gender, gender_conf = self.predict_gender(waveform, sample_rate)
        return age, age_conf, gender, gender_conf


if __name__ == "__main__":
    # Example usage (requires downloaded pretrained models)
    model = AnvarjonCNNModel()
    
    # If you have pretrained models:
    # model.load_pretrained_weights(
    #     age_model_path='models/age/best_model_age.h5',
    #     gender_model_path='models/gender/best_model_gender.h5'
    # )
    
    print("Anvarjon CNN Model initialized")
    print(f"Input: Spectrograms (64x64)")
    print(f"Age output: 6 classes (teens to sixties)")
    print(f"Gender output: 2 classes (male/female)")
