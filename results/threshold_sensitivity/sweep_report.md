# U-3: OOD Threshold Sensitivity Analysis

## 1. Signal Distributions

Percentile values computed from **train+val** data (N=4000).

### d_M

| Split/Dim    |      p50 |      p75 |      p90 |      p95 |      p99 |    p99.5 |
|--------------|----------|----------|----------|----------|----------|----------|
| train+val    |  23.7406 |  26.5144 |  29.2400 |  30.8009 |  34.5998 |  36.3846 |
| test         |  31.4147 |  36.6946 |  42.9312 |  48.4544 |  58.2228 |  61.8526 |
| all          |  24.5588 |  28.1626 |  32.9485 |  36.9926 |  48.4505 |  52.2179 |

### sigma_sq

| Split/Dim    |      p50 |      p75 |      p90 |      p95 |      p99 |    p99.5 |
|--------------|----------|----------|----------|----------|----------|----------|
| train+val_overall |   0.0543 |   0.0648 |   0.0759 |   0.0855 |   0.1074 |   0.1468 |
| train+val_sharpness |   0.0641 |   0.0764 |   0.0894 |   0.1006 |   0.1318 |   0.1747 |
| train+val_color |   0.0607 |   0.0739 |   0.0871 |   0.0968 |   0.1212 |   0.1502 |
| test_overall |   0.0545 |   0.0633 |   0.0748 |   0.0824 |   0.1029 |   0.1074 |
| test_sharpness |   0.0651 |   0.0768 |   0.0880 |   0.0975 |   0.1315 |   0.1319 |
| test_color   |   0.0608 |   0.0720 |   0.0831 |   0.0919 |   0.1172 |   0.1180 |

### entropy

| Split/Dim    |      p50 |      p75 |      p90 |      p95 |      p99 |    p99.5 |
|--------------|----------|----------|----------|----------|----------|----------|
| train+val_overall |   0.4423 |   0.6249 |   0.6828 |   0.6911 |   0.6945 |   0.6972 |
| train+val_sharpness |   0.4578 |   0.6243 |   0.6853 |   0.6928 |   0.6995 |   0.7056 |
| train+val_color |   0.4490 |   0.6253 |   0.6850 |   0.6921 |   0.6989 |   0.7035 |
| test_overall |   0.4130 |   0.6164 |   0.6848 |   0.6922 |   0.6940 |   0.6974 |
| test_sharpness |   0.4257 |   0.5999 |   0.6808 |   0.6911 |   0.6999 |   0.7014 |
| test_color   |   0.4233 |   0.6313 |   0.6850 |   0.6928 |   0.6989 |   0.7055 |

## 2. Named Profiles Comparison (Test Split)

| Profile | Dim | AUTO_ACCEPT% | LOW_WEIGHT% | TIER2% | HARD_REJECT% | Mean Weight | Effective N |
|---------|-----|-------------|------------|--------|-------------|-------------|-------------|
| current         | overall   |        93.7 |        0.0 |    5.4 |         0.9 |       0.937 |       937.0 |
| current         | sharpness |        93.7 |        0.0 |    5.4 |         0.9 |       0.937 |       937.0 |
| current         | color     |        93.7 |        0.0 |    5.4 |         0.9 |       0.937 |       937.0 |
| data_calibrated | overall   |        65.4 |       16.9 |   16.8 |         0.9 |       0.726 |       726.1 |
| data_calibrated | sharpness |        52.2 |       26.0 |   20.9 |         0.9 |       0.641 |       640.6 |
| data_calibrated | color     |        57.8 |       20.2 |   21.1 |         0.9 |       0.667 |       667.4 |
| strict          | overall   |        15.1 |       10.2 |   21.1 |        53.6 |       0.196 |       196.0 |
| strict          | sharpness |         9.3 |       12.0 |   25.1 |        53.6 |       0.148 |       147.8 |
| strict          | color     |         9.7 |       10.9 |   25.8 |        53.6 |       0.145 |       145.4 |
| moderate        | overall   |        30.8 |        9.3 |   26.2 |        33.7 |       0.347 |       346.6 |
| moderate        | sharpness |        26.4 |       10.9 |   29.0 |        33.7 |       0.312 |       312.4 |
| moderate        | color     |        26.2 |        9.8 |   30.3 |        33.7 |       0.306 |       305.8 |
| lenient         | overall   |        61.5 |        4.4 |    8.2 |        25.9 |       0.639 |       638.7 |
| lenient         | sharpness |        60.9 |        3.4 |    9.8 |        25.9 |       0.626 |       626.0 |
| lenient         | color     |        60.0 |        5.3 |    8.8 |        25.9 |       0.627 |       626.6 |
| dm_only         | overall   |        93.7 |        0.0 |    5.4 |         0.9 |       0.937 |       937.0 |
| dm_only         | sharpness |        93.7 |        0.0 |    5.4 |         0.9 |       0.937 |       937.0 |
| dm_only         | color     |        93.7 |        0.0 |    5.4 |         0.9 |       0.937 |       937.0 |
| no_ood          | overall   |        68.2 |       19.1 |   12.7 |         0.0 |       0.764 |       764.0 |
| no_ood          | sharpness |        52.9 |       28.0 |   19.1 |         0.0 |       0.656 |       656.2 |
| no_ood          | color     |        60.1 |       22.7 |   17.2 |         0.0 |       0.703 |       703.1 |

## 3. d_M Percentile Sweep (Test Split, Overall Dimension)

| Config | d_M OOD | d_M Reject | AUTO_ACCEPT% | TIER2% | HARD_REJECT% | Effective N |
|--------|---------|-----------|-------------|--------|-------------|-------------|
| dm_p90     |    29.2 |      36.4 |        25.3 |   40.8 |        25.9 |       285.9 |
| dm_p92     |    29.7 |      36.4 |        26.9 |   39.0 |        25.9 |       302.7 |
| dm_p95     |    30.8 |      36.4 |        30.8 |   34.0 |        25.9 |       346.6 |
| dm_p97     |    32.0 |      36.4 |        36.1 |   27.0 |        25.9 |       407.2 |
| dm_p99     |    34.6 |      36.4 |        45.7 |   16.1 |        25.9 |       509.2 |

## 4. Key Findings

- **Current thresholds**: 93.7% AUTO_ACCEPT on test (σ²/entropy thresholds never trigger)
- **Data-calibrated thresholds**: 65.4% AUTO_ACCEPT on test (σ²/entropy now discriminate)
- **d_M only**: 93.7% AUTO_ACCEPT (σ²/entropy disabled → pure OOD gating)
- **No OOD**: 68.2% AUTO_ACCEPT (d_M disabled → σ²/entropy only)

## 5. Tier-2 VLM Veto Threshold Sweep

### 5.1 VLM Disagreement Distribution (|VLM - SigLIP2|)

| Model | overall | sharpness | color |
|-------|---------|-----------|-------|
| anthropic/claude-haiku-4.5               |   0.691 |     0.731 | 0.735 |
| google/gemini-2.5-pro                    |   0.816 |     0.945 | 0.718 |
| google/gemini-3-flash-preview            |   0.850 |     0.920 | 0.886 |
| google/gemini-3-flash-preview/no_resize  |   0.887 |     1.009 | 0.844 |
| openai/gpt-4.1                           |   1.185 |     1.306 | 1.169 |
| qwen/qwen3-vl-8b-instruct                |   1.340 |     1.296 | 1.351 |
| qwen/qwen3-vl-8b-thinking                |   0.948 |     0.860 | 1.084 |
| qwen/qwen3-vl-8b-thinking/temp0          |   0.946 |     0.855 | 1.082 |
| qwen/qwen3.5-flash-02-23                 |   1.536 |     1.632 | 1.508 |

### 5.2 Veto Rate by Threshold (Overall Dimension)

| Threshold | claude-haiku | gemini-2.5-p | gemini-3-fla |    no_resize |      gpt-4.1 | qwen3-vl-8b- | qwen3-vl-8b- |        temp0 | qwen3.5-flas | ensemble |
|-----------|-------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|-------------|----------|
|       0.3 |        82.9% |        73.0% |        83.9% |        83.7% |        91.5% |        99.1% |        87.6% |        87.6% |        98.3% |    99.3% |
|       0.5 |        68.5% |        60.7% |        73.6% |        71.3% |        86.7% |        97.0% |        78.3% |        76.2% |        96.4% |    94.7% |
|       0.8 |        40.2% |        43.8% |        55.5% |        55.2% |        78.2% |        90.7% |        58.5% |        56.8% |        91.3% |    72.2% |
|       1.0 |        20.3% |        36.9% |        43.4% |        47.5% |        72.0% |        80.6% |        44.6% |        44.3% |        86.9% |    55.5% |
|       1.5 |         0.5% |        18.9% |         6.6% |        11.0% |        27.9% |        38.4% |        16.7% |        18.0% |        60.3% |     5.6% |

### 5.3 Per-Dimension Veto Rates at Current Threshold (1.5)

| Model | overall | sharpness | color |
|-------|---------|-----------|-------|
| claude-haiku-4.5                         |    0.5% |      2.6% |  2.2% |
| gemini-2.5-pro                           |   18.9% |     27.6% | 14.9% |
| gemini-3-flash-preview                   |    6.6% |     14.3% |  9.4% |
| no_resize                                |   11.0% |     29.2% |  6.9% |
| gpt-4.1                                  |   27.9% |     43.7% | 23.1% |
| qwen3-vl-8b-instruct                     |   38.4% |     39.6% | 33.3% |
| qwen3-vl-8b-thinking                     |   16.7% |     20.9% | 25.5% |
| temp0                                    |   18.0% |     21.3% | 26.2% |
| qwen3.5-flash-02-23                      |   60.3% |     68.4% | 55.9% |
| ensemble_majority                        |    5.6% |     17.7% |  6.3% |

## 6. Limitations

- **JSD thresholds not swept**: DeQA predictions unavailable per-image; JSD=0 for all. JSD sensitivity requires separate DeQA inference.
- **GT validation for Tier-2 not available**: DIQA-5000 test set has no ground-truth MOS, so True/False veto accuracy cannot be computed.
- **Calibration split**: All percentile thresholds computed from train+val (N=4000). Test split (N=1000) used for evaluation only.
