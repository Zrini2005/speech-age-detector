# Speaker Age Estimation - Evaluation Report

**Audio Directory:** `data/children_real/clips`

**Total Audio Files:** 200

**Files with Ground Truth:** 71

## Summary Table

| Model              |   Predictions |   Avg Time (s) |   MAE |   RMSE |   Pearson r | Accuracy   |
|:-------------------|--------------:|---------------:|------:|-------:|------------:|:-----------|
| wav2vec2-audeering |           200 |          1.345 | 10.76 |  14.75 |       0.228 | 49.30%     |
| wavlm-base         |           200 |          0.873 | 14.53 |  16.15 |       0.127 | 59.15%     |
| mfcc-baseline      |           200 |          0.023 | 26.3  |  27.37 |      -0.208 | 0.00%      |

## Model Details

### wav2vec2-audeering

- **MAE:** 10.76 years
- **RMSE:** 14.75 years
- **Pearson r:** 0.228
- **Accuracy (groups):** 49.30%
- **Avg Inference Time:** 1.345 s
- **Predictions File:** `results/wav2vec2-audeering.csv`

### wavlm-base

- **MAE:** 14.53 years
- **RMSE:** 16.15 years
- **Pearson r:** 0.127
- **Accuracy (groups):** 59.15%
- **Avg Inference Time:** 0.873 s
- **Predictions File:** `results/wavlm-base.csv`

### mfcc-baseline

- **MAE:** 26.30 years
- **RMSE:** 27.37 years
- **Pearson r:** -0.208
- **Accuracy (groups):** 0.00%
- **Avg Inference Time:** 0.023 s
- **Predictions File:** `results/mfcc-baseline.csv`

## Reproduction Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation
python run_age_eval.py --audio-dir data/children_real/clips
```

## Recommendations

**Best Model (by MAE):** wav2vec2-audeering with MAE of 10.76 years

**Fastest Model:** mfcc-baseline with 0.023 s per file

### Production Considerations

- Consider model quantization for faster inference
- Implement calibration for confidence scores
- Test on diverse datasets for robustness
- Monitor for bias across age groups and demographics
