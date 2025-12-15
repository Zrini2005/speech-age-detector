"""Audio loading and preprocessing utilities."""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_audio(
    audio_path: str,
    target_sr: int = 16000,
    max_duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and optionally resample.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        max_duration: Maximum duration in seconds (None for full file)
        
    Returns:
        Tuple of (waveform, sample_rate)
    """
    try:
        # Try soundfile first (faster)
        waveform, sr = sf.read(audio_path, dtype='float32')
    except:
        # Fallback to librosa
        waveform, sr = librosa.load(audio_path, sr=None)
    
    # Convert stereo to mono if needed
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Resample if needed
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Truncate if max_duration specified
    if max_duration is not None:
        max_samples = int(max_duration * sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
    
    return waveform, sr


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    try:
        info = sf.info(audio_path)
        return info.duration
    except:
        waveform, sr = librosa.load(audio_path, sr=None)
        return len(waveform) / sr


def find_audio_files(directory: str, extensions: list = None) -> list:
    """
    Find all audio files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to search for
        
    Returns:
        List of audio file paths
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    audio_files = []
    dir_path = Path(directory)
    
    for ext in extensions:
        audio_files.extend(dir_path.glob(f'*{ext}'))
        audio_files.extend(dir_path.glob(f'*{ext.upper()}'))
    
    return sorted([str(f) for f in audio_files])


def convert_to_wav(input_path: str, output_path: str, target_sr: int = 16000):
    """
    Convert audio file to WAV format.
    
    Args:
        input_path: Input audio file path
        output_path: Output WAV file path
        target_sr: Target sample rate
    """
    waveform, sr = load_audio(input_path, target_sr=target_sr)
    sf.write(output_path, waveform, sr)
