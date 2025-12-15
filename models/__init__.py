"""Model wrappers for age estimation."""

from .base_model import BaseAgeModel
from .wav2vec2_age_model import Wav2Vec2AgeModel
from .wavlm_age_model import WavLMAgeModel
from .speechbrain_age_model import SpeechBrainAgeModel
from .baseline_models import MFCCBaselineModel
from .audeering_6layer_model import Audeering6LayerAgeModel

__all__ = [
    'BaseAgeModel',
    'Wav2Vec2AgeModel',
    'WavLMAgeModel',
    'SpeechBrainAgeModel',
    'MFCCBaselineModel',
    'Audeering6LayerAgeModel',
]
