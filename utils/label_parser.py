"""Parse ground truth labels from filenames or label files."""

import re
from pathlib import Path
from typing import Dict, Optional
import pandas as pd


def parse_age_from_filename(filename: str) -> Optional[float]:
    """
    Try to extract age from filename.
    
    Common patterns:
    - speaker_25_utterance.wav (age 25)
    - audio_age35.wav (age 35)
    - 40yo_recording.wav (age 40)
    
    Args:
        filename: Audio filename
        
    Returns:
        Extracted age or None
    """
    # Pattern: _XX_ or _XXX_ where XX is 2-3 digit age
    patterns = [
        r'_(\d{2,3})_',  # _25_
        r'age(\d{2,3})',  # age25
        r'(\d{2,3})yo',   # 25yo
        r'(\d{2,3})y',    # 25y
        r'-(\d{2,3})-',   # -25-
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            # Validate age range
            if 0 <= age <= 100:
                return float(age)
    
    return None


def load_labels_from_csv(csv_path: str) -> Dict[str, float]:
    """
    Load ground truth labels from CSV file.
    
    Expected format:
    filename,age
    audio1.wav,25
    audio2.wav,45
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dictionary mapping filename to age
    """
    df = pd.read_csv(csv_path)
    
    # Try common column names
    filename_col = None
    age_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['filename', 'file', 'audio', 'name']:
            filename_col = col
        if col_lower in ['age', 'label', 'target']:
            age_col = col
    
    if filename_col is None or age_col is None:
        raise ValueError(f"Could not find filename and age columns in {csv_path}")
    
    # Create mapping
    labels = {}
    for _, row in df.iterrows():
        filename = Path(row[filename_col]).name  # Get just the filename, not path
        age = float(row[age_col])
        labels[filename] = age
    
    return labels


def get_ground_truth_labels(audio_dir: str) -> Dict[str, float]:
    """
    Get ground truth labels from either labels.csv or filenames.
    
    Args:
        audio_dir: Directory containing audio files
        
    Returns:
        Dictionary mapping filename to age
    """
    labels = {}
    audio_path = Path(audio_dir)
    
    # First, check for labels.csv
    csv_path = audio_path / 'labels.csv'
    if csv_path.exists():
        print(f"Loading labels from {csv_path}")
        labels = load_labels_from_csv(str(csv_path))
        return labels
    
    # Otherwise, try to parse from filenames
    print("No labels.csv found, attempting to parse ages from filenames...")
    audio_files = list(audio_path.glob('*.wav')) + list(audio_path.glob('*.mp3'))
    
    for audio_file in audio_files:
        age = parse_age_from_filename(audio_file.name)
        if age is not None:
            labels[audio_file.name] = age
    
    if labels:
        print(f"Extracted {len(labels)} ages from filenames")
    else:
        print("Could not extract ages from filenames")
    
    return labels
