# Speaker Age Estimation Models - Research Catalog

This document catalogs publicly available models and research for speaker/voice age estimation.

## HuggingFace Models

### 1. audeering/wav2vec2-large-robust-24-ft-age-gender ⭐ RECOMMENDED
- **URL**: https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender
- **Type**: Wav2Vec2 + Classification Head
- **Training**: Fine-tuned on age and gender data
- **Output**: Age and gender predictions
- **Sample Rate**: 16 kHz
- **Parameters**: ~300M
- **License**: CC-BY-NC-SA 4.0
- **Status**: INTEGRATED ✓
- **Notes**: Production-ready, high accuracy

### 2. versae/wav2vec2-base-finetuned-coscan-age_group
- **URL**: https://huggingface.co/versae/wav2vec2-base-finetuned-coscan-age_group
- **Type**: Wav2Vec2 + Age Group Classification
- **Training**: Fine-tuned for age group classification
- **Output**: Age groups (child/adult/elderly)
- **Sample Rate**: 16 kHz
- **Parameters**: ~95M
- **Status**: Available (not integrated - outputs groups not continuous age)

### 3. versae/wav2vec2-base-coscan-no-age_group
- **URL**: https://huggingface.co/versae/wav2vec2-base-coscan-no-age_group
- **Type**: Wav2Vec2 base model
- **Status**: Base model without age prediction head

### 4. CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-all-age
- **URL**: https://huggingface.co/CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-all-age
- **Type**: Speech Emotion Recognition with age features
- **Languages**: English, Chinese
- **Status**: Available (multi-task model)

## SpeechBrain Models

### 5. SpeechBrain ECAPA-TDNN Speaker Embeddings
- **Source**: speechbrain/spkrec-ecapa-voxceleb
- **Type**: Speaker embeddings (can be used as features)
- **Training**: VoxCeleb speaker recognition
- **Output**: 192-dim embeddings
- **Use Case**: Extract embeddings → train age regressor
- **Status**: INTEGRATED ✓ (with placeholder regressor)
- **GitHub**: https://github.com/speechbrain/speechbrain

### 6. SpeechBrain X-vectors
- **Source**: speechbrain/spkrec-xvect-voxceleb
- **Type**: X-vector embeddings
- **Training**: VoxCeleb
- **Output**: 512-dim embeddings
- **Status**: Available

## Microsoft Models

### 7. WavLM (Base/Large)
- **HuggingFace**: microsoft/wavlm-base, microsoft/wavlm-large
- **Type**: Self-supervised speech representation
- **Use Case**: Extract features → train age regressor
- **Status**: INTEGRATED ✓ (base model with placeholder regressor)
- **Paper**: WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing
- **GitHub**: https://github.com/microsoft/unilm/tree/master/wavlm

### 8. Wav2Vec2 (Base/Large)
- **HuggingFace**: facebook/wav2vec2-base, facebook/wav2vec2-large
- **Type**: Self-supervised speech representation
- **Use Case**: Extract features → train age regressor
- **Status**: Available (similar to WavLM)
