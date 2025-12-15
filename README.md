# Voice Age Prediction Tool

A Python-based tool for predicting speaker age from audio files using state-of-the-art deep learning models. This tool supports batch processing through JSON input/output format.

## Features

- 🎯 **Accurate Age Prediction**: Uses pre-trained deep learning models for age estimation
- 🔄 **Batch Processing**: Process multiple audio files at once via JSON input
- 🤖 **Multiple Models**: Choose between Audeering Wav2Vec2 or Vox-Profile WavLM models
- 📊 **Automatic Categorization**: Classifies speakers as "child" (<18) or "adult" (≥18)
- 🎵 **Multiple Audio Formats**: Supports WAV, MP3, FLAC, OGG, and M4A files
- 📁 **Clean JSON Output**: Easy to parse results for downstream processing

## Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows with WSL
- **RAM**: Minimum 8GB (16GB recommended for larger batches)
- **GPU**: Optional but recommended for faster inference

## Installation

### 1. Clone or Download the Repository

```bash
git clone https://github.com/Zrini2005/speech-age-detector.git
cd VoiceLabel
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run will automatically download pre-trained models (~1-2GB) from HuggingFace. This may take several minutes depending on your internet connection.

## Available Models

### 1. Audeering Wav2Vec2 (Recommended) ⭐
- **Model ID**: `audeering`
- **Base**: Wav2Vec2-large-robust-24-ft-age-gender
- **Accuracy**: High accuracy on diverse datasets
- **Speed**: ~1-2 seconds per audio file (CPU)
- **License**: CC-BY-NC-SA 4.0
- **Best for**: Production use, general purpose age prediction

### 2. Vox-Profile WavLM
- **Model ID**: `voxprofile`
- **Base**: WavLM-based speaker profile model
- **Accuracy**: Good for speaker profiling tasks
- **Speed**: ~1-2 seconds per audio file (CPU)
- **Best for**: Research, alternative predictions

### 3. Both Models
- **Model ID**: `both`
- **Output**: Predictions from both models for comparison
- **Use case**: Ensemble predictions or model evaluation

## Usage

### Basic Command Structure

```bash
python predict_age.py -i <input.json> -m <model> -o <output.json>
```

### Command Line Arguments

| Argument | Short | Required | Description | Values |
|----------|-------|----------|-------------|---------|
| `--input` | `-i` | Yes | Input JSON file with audio paths | Path to JSON file |
| `--model` | `-m` | Yes | Model to use for prediction | `audeering`, `voxprofile`, `both` |
| `--output` | `-o` | Yes | Output JSON file for results | Path to JSON file |

### Example 1: Single Model Prediction

```bash
python predict_age.py -i example_input.json -m audeering -o output.json
```

### Example 2: Using Vox-Profile Model

```bash
python predict_age.py -i example_input.json -m voxprofile -o output.json
```

### Example 3: Compare Both Models

```bash
python predict_age.py -i example_input.json -m both -o comparison_output.json
```

## Input Format

The input JSON file should contain an array of absolute or relative paths to audio files:

```json
[
  "/path/to/audio1.wav",
  "/path/to/audio2.mp3",
  "./relative/path/audio3.flac"
]
```

**Example** ([example_input.json](example_input.json)):
```json
[
  "/mnt/c/Users/srini/Downloads/testing/cry1.wav",
  "/mnt/c/Users/srini/Downloads/testing/five.mp3",
  "/mnt/c/Users/srini/Downloads/testing/twenties.mp3",
  "/mnt/c/Users/srini/Downloads/testing/fifties.mp3"
]
```

## Output Format

### Single Model Output

When using a single model (`audeering` or `voxprofile`), the output is an array of prediction objects:

```json
[
  {
    "audio_path": "/path/to/audio1.wav",
    "age": 25,
    "category": "adult"
  },
  {
    "audio_path": "/path/to/audio2.mp3",
    "age": 8,
    "category": "child"
  },
  {
    "audio_path": "/path/to/audio3.wav",
    "age": null,
    "category": null,
    "error": "File not found"
  }
]
```

### Both Models Output

When using `--model both`, each audio file has predictions from both models:

```json
[
  {
    "audio_path": "/path/to/audio1.wav",
    "audeering_24layer": {
      "age": 25,
      "category": "adult"
    },
    "voxprofile": {
      "age": 27,
      "category": "adult"
    }
  },
  {
    "audio_path": "/path/to/audio2.mp3",
    "audeering_24layer": {
      "age": 8,
      "category": "child"
    },
    "voxprofile": {
      "age": 10,
      "category": "child"
    }
  }
]
```

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `audio_path` | string | Path to the processed audio file |
| `age` | integer | Predicted age in years (rounded) |
| `category` | string | "child" (<18) or "adult" (≥18) |
| `error` | string | Error message if prediction failed (only present on error) |


## Audio Requirements

### Supported Formats
- **WAV** (recommended)
- **MP3**
- **FLAC**
- **OGG**
- **M4A**

### Audio Specifications
- **Sample Rate**: Any (automatically resampled to 16kHz)
- **Channels**: Mono or Stereo (stereo automatically converted to mono)
- **Duration**: Recommended 2-10 seconds (shorter clips may be less accurate)
- **Quality**: Higher quality audio generally produces better predictions


## Model Information

### Audeering Wav2Vec2
- **Source**: https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender
- **Parameters**: ~300M
- **Training Data**: Multi-domain age and gender datasets
- **License**: CC-BY-NC-SA 4.0 (Non-commercial use)

### Vox-Profile WavLM
- **Source**: Custom implementation based on Microsoft WavLM
- **Training**: Speaker profiling tasks
- **License**: MIT

## Directory Structure

```
VoiceLabel/
├── predict_age.py           # Main prediction script
├── example_input.json       # Example input file
├── example_output.json      # Example output file
├── requirements.txt         # Python dependencies
├── MODEL_CATALOG.md        # Detailed model documentation
├── models/                  # Model implementations
│   ├── wav2vec2_age_model.py
│   ├── voxprofile_wavlm_model.py
│   └── ...
└── utils/                   # Utility functions
    ├── audio_utils.py
    └── ...
```

