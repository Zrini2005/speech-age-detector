# Speaker Age Estimation - Evaluation Report

**Audio Directory:** `data/common_voice/clips`

**Total Audio Files:** 100

**Files with Ground Truth:** 100

## Summary Table

| Model              |   Predictions |   Avg Time (s) |   MAE |   RMSE |   Pearson r | Accuracy   |
|:-------------------|--------------:|---------------:|------:|-------:|------------:|:-----------|
| wav2vec2-audeering |           100 |           0.95 | 13.32 |  16.61 |      -0.07  | 48.00%     |
| wavlm-base         |           100 |           0.35 | 33.71 |  36.26 |       0.031 | 0.00%      |

## Model Details

### wav2vec2-audeering

- **MAE:** 13.32 years
- **RMSE:** 16.61 years
- **Pearson r:** -0.070
- **Accuracy (groups):** 48.00%
- **Avg Inference Time:** 0.950 s
- **Predictions File:** `results/wav2vec2-audeering.csv`

### wavlm-base

- **MAE:** 33.71 years
- **RMSE:** 36.26 years
- **Pearson r:** 0.031
- **Accuracy (groups):** 0.00%
- **Avg Inference Time:** 0.350 s
- **Predictions File:** `results/wavlm-base.csv`

## Reproduction Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation
python run_age_eval.py --audio-dir data/common_voice/clips
```

## Recommendations

**Best Model (by MAE):** wav2vec2-audeering with MAE of 13.32 years

**Fastest Model:** wavlm-base with 0.350 s per file

### Production Considerations

- Consider model quantization for faster inference
- Implement calibration for confidence scores
- Test on diverse datasets for robustness
- Monitor for bias across age groups and demographics
