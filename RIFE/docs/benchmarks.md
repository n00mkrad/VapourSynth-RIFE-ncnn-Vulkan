# Model Benchmarks (2026-05-06)

Speed of each model to get the delta-1 and delta-2 motion vectors (both forward and backward for each delta) on a 1920x1024 input (500 frames) with block size 16x8 and overlap 8x4.

To ensure VapourSynth actually requests all vector frames (speed will be false if you do not actually utilize the returned clips), they were fed into Merge():

```python
delta1 = core.std.Merge(bwd[0], fwd[0])
delta2 = core.std.Merge(bwd[1], fwd[1])
clip = core.std.Merge(delta1, delta2) # Getting a frame from 'clip' will always evaluate all four vector clips
```

RTX 4090 (2 GPU threads per instance = 4 threads total) - Note that the GPU was never fully utilized due to bandwidth or other bottlenecks, power draw usually topped out at ~260W (out of 450). This means that if you have a slower card, your results may not be that much worse as the utilization is likely higher.

`[rmv] RIFEMV parameters: gpu_thread=2 shared_flow_inflight=8 shared_luma_cache=true flow_scale=1 res_scale=1 cpu_flow_resize=auto perf_stats=true blksize_x=16 blksize_y=8 overlap_x=8 overlap_y=4 pel=1 delta=2 bits=8 sad_multiplier=1 render_sad_mask=false matrix_in_s=709 range_in_s=full hpad=0 vpad=0 block_reduce=1 chroma=false inference_width=1920 inference_height=1024`

### Model Speed Ranking

Includes quality metrics for the fastest 20. Refer to [Model Quality Ranking](#model-quality-ranking) for tables sorted by the respective metric.

| Model Name | FPS | Stability Rank | SAD Rank | Avg SAD | Compensate Rank | Flow Rank |
|---|---|---|---|---|---|---|
| rife-v4.2_ensembleFalse_fastTrue | 43.22 | 19 | 1 | 420.89 | 1 | 19 |
| rife-v4.4_ensembleFalse_fastTrue | 43.14 | 15 | 4 | 427.55 | 3 | 17 |
| rife-v4.3_ensembleFalse_fastTrue | 42.67 | 16 | 3 | 426.43 | 2 | 16 |
| rife-v4.5_ensembleFalse | 40.42 | 9 | 19 | 454.54 | 18 | 2 |
| rife-v4.6_ensembleFalse | 39.73 | 21 | 15 | 437.65 | 11 | 4 |
| rife-v3.1 | 35.65 | 1 | 2 | 421.81 | 12 | 20 |
| rife-v4.16_lite_ensembleFalse | 35.02 | 18 | 11 | 433.96 | 10 | 12 |
| rife-v4.15_lite_ensembleFalse | 33.07 | 17 | 8 | 433.65 | 8 | 13 |
| rife-v4.17_lite_ensembleFalse | 32.83 | 2 | 20 | 462.04 | 4 | 1 |
| rife-v4.13_lite_ensembleFalse | 32.61 | 20 | 5 | 431.16 | 6 | 11 |
| rife-v4.12_lite_ensembleFalse | 31.73 | 22 | 6 | 431.43 | 5 | 6 |
| rife-v4.22_lite_ensembleFalse | 30.45 | 3 | 12 | 434.53 | 19 | 10 |
| rife-v4.18_ensembleFalse | 28.47 | 8 | 13 | 437.57 | 15 | 8 |
| rife-v4.15_ensembleFalse | 28.43 | 10 | 9 | 433.74 | 13 | 18 |
| rife-v4.14_ensembleFalse | 28.42 | 14 | 10 | 433.79 | 9 | 14 |
| rife-v4.20_ensembleFalse | 28.32 | 13 | 16 | 438.01 | 20 | 7 |
| rife-v4.24_ensembleFalse | 28.30 | 4 | 18 | 442.83 | 17 | 3 |
| rife-v4.18 | 28.27 | 7 | 14 | 437.57 | 14 | 9 |
| rife-v4.17_ensembleFalse | 28.25 | 6 | 17 | 439.02 | 16 | 5 |
| rife-v4.13_ensembleFalse | 28.15 | 11 | 7 | 432.24 | 7 | 15 |
| rife-v4.19 | 28.09 | 5 | - | - | - | - |
| rife-v4.20 | 27.89 | 12 | - | - | - | - |
| rife-v4.25-lite_ensembleFalse | 27.88 | - | - | - | - | - |
| rife-v4.26_ensembleFalse | 27.55 | - | - | - | - | - |
| rife-v4.12_ensembleFalse | 27.38 | - | - | - | - | - |
| rife-v4.10_ensembleFalse | 27.27 | - | - | - | - | - |
| rife-v4.11_ensembleFalse | 26.35 | - | - | - | - | - |
| rife-v4.21 | 25.03 | - | - | - | - | - |
| rife-v4.21_ensembleFalse | 24.95 | - | - | - | - | - |
| rife-v4.22 | 24.95 | - | - | - | - | - |
| rife-v4.22_ensembleFalse | 24.82 | - | - | - | - | - |
| rife-v4.4_ensembleTrue_fastFalse | 24.63 | - | - | - | - | - |
| rife-v4.3_ensembleTrue_fastFalse | 24.58 | - | - | - | - | - |
| rife-v4.2_ensembleTrue_fastFalse | 24.45 | - | - | - | - | - |
| rife-v4.14_lite_ensembleFalse | 23.32 | - | - | - | - | - |
| rife-v4.5_ensembleTrue | 22.86 | - | - | - | - | - |
| rife-v4.6_ensembleTrue | 22.75 | - | - | - | - | - |
| rife-v4.9_ensembleFalse **[fp32 fallback]** | 22.01 | - | - | - | - | - |
| rife-v3.9_ensembleFalse_fastTrue | 21.32 | - | - | - | - | - |
| rife-v4.26-large_ensembleFalse | 21.17 | - | - | - | - | - |
| rife-v4.15_lite_ensembleTrue | 20.84 | - | - | - | - | - |
| rife-v4.17_lite_ensembleTrue | 20.65 | - | - | - | - | - |
| rife-v4.16_lite_ensembleTrue | 20.62 | - | - | - | - | - |
| rife-v4.7_ensembleTrue | 19.85 | - | - | - | - | - |
| rife-v4.8_ensembleTrue | 19.81 | - | - | - | - | - |
| rife-v4.9_ensembleTrue | 19.54 | - | - | - | - | - |
| rife-v4.13_ensembleTrue | 17.24 | - | - | - | - | - |
| rife-v4.13_ensembleTrue_fastFalse | 17.20 | - | - | - | - | - |
| rife-v4.24_ensembleTrue | 17.16 | - | - | - | - | - |
| rife-v4.14_ensembleTrue | 17.11 | - | - | - | - | - |
| rife-v4.18_ensembleTrue | 17.03 | - | - | - | - | - |
| rife-v4.12_ensembleTrue | 17.03 | - | - | - | - | - |
| rife-v4.15_ensembleTrue | 16.95 | - | - | - | - | - |
| rife-v4.11_ensembleTrue | 16.82 | - | - | - | - | - |
| rife-v4.10_ensembleTrue | 16.00 | - | - | - | - | - |
| rife-v4.17_ensembleTrue | 15.98 | - | - | - | - | - |
| rife-v4.14_lite_ensembleTrue | 14.12 | - | - | - | - | - |

### Model Quality Ranking

Candidates are the 20 fastest models (from [Model Speed Ranking](#model-speed-ranking)).

#### Average SAD

Best to worst, based on average Sum of Absolute Differences (SAD):
Use case: **Temporal denoising (`mv.Degrain`)**, where ignoring grain and tracking reliable block structures is critical.

| Rank | Model Name | Avg SAD | Avg SAD 10% High |
|---|---|---|---|
| 1 | rife-v4.2_ensembleFalse_fastTrue | 420.89 | 1060.34 |
| 2 | rife-v3.1 | 421.81 | 1084.24 |
| 3 | rife-v4.3_ensembleFalse_fastTrue | 426.43 | 1093.52 |
| 4 | rife-v4.4_ensembleFalse_fastTrue | 427.55 | 1094.64 |
| 5 | rife-v4.13_lite_ensembleFalse | 431.16 | 1121.31 |
| 6 | rife-v4.12_lite_ensembleFalse | 431.43 | 1102.69 |
| 7 | rife-v4.13_ensembleFalse | 432.24 | 1136.22 |
| 8 | rife-v4.15_lite_ensembleFalse | 433.65 | 1150.02 |
| 9 | rife-v4.15_ensembleFalse | 433.74 | 1158.37 |
| 10 | rife-v4.14_ensembleFalse | 433.79 | 1159.71 |
| 11 | rife-v4.16_lite_ensembleFalse | 433.96 | 1149.98 |
| 12 | rife-v4.22_lite_ensembleFalse | 434.53 | 1131.20 |
| 13 | rife-v4.18_ensembleFalse | 437.57 | 1163.27 |
| 14 | rife-v4.18 | 437.57 | 1163.27 |
| 15 | rife-v4.6_ensembleFalse | 437.65 | 1147.22 |
| 16 | rife-v4.20_ensembleFalse | 438.01 | 1157.39 |
| 17 | rife-v4.17_ensembleFalse | 439.02 | 1156.32 |
| 18 | rife-v4.24_ensembleFalse | 442.83 | 1142.34 |
| 19 | rife-v4.5_ensembleFalse | 454.54 | 1200.64 |
| 20 | rife-v4.17_lite_ensembleFalse | 462.04 | 1214.62 |


#### Motion Stability

Best to worst, based on resulting video encoded bitrate when visualizing MVs of a highly grainy clip using mv.Mask. Lower size indicates less noise in motion vector visualization.

| Rank | Name                              | Size (KB) |
|-----:|-----------------------------------|----------:|
|    1 | rife-v3.1                         |     5,911 |
|    2 | rife-v4.17_lite_ensembleFalse     |     7,736 |
|    3 | rife-v4.22_lite_ensembleFalse     |     8,169 |
|    4 | rife-v4.24_ensembleFalse          |     9,735 |
|    5 | rife-v4.19                        |     9,764 |
|    6 | rife-v4.17_ensembleFalse          |    10,012 |
|    7 | rife-v4.18                        |    10,896 |
|    8 | rife-v4.18_ensembleFalse          |    10,896 |
|    9 | rife-v4.5_ensembleFalse           |    11,163 |
|   10 | rife-v4.15_ensembleFalse          |    11,412 |
|   11 | rife-v4.13_ensembleFalse          |    11,579 |
|   12 | rife-v4.20                        |    11,592 |
|   13 | rife-v4.20_ensembleFalse          |    11,592 |
|   14 | rife-v4.14_ensembleFalse          |    12,012 |
|   15 | rife-v4.4_ensembleFalse_fastTrue  |    12,376 |
|   16 | rife-v4.3_ensembleFalse_fastTrue  |    12,916 |
|   17 | rife-v4.15_lite_ensembleFalse     |    13,042 |
|   18 | rife-v4.16_lite_ensembleFalse     |    13,072 |
|   19 | rife-v4.2_ensembleFalse_fastTrue  |    13,124 |
|   20 | rife-v4.13_lite_ensembleFalse     |    13,458 |
|   21 | rife-v4.6_ensembleFalse           |    14,084 |
|   22 | rife-v4.12_lite_ensembleFalse     |    15,233 |


#### Flow MAE

Best to worst, using `mv.Flow` reconstruction of frames using their neighbors, by Mean Absolute Error (MAE).

Use case: **Pixel-level interpolation and framerate conversion (`mv.Flow`)**, where reproducing exact pixel values (even if it means overfitting to noise/grain) is mathematically rewarded.

| Rank | Model Name | Dense MAE |
|---|---|---|
| 1 | rife-v4.17_lite_ensembleFalse | 0.0603 |
| 2 | rife-v4.5_ensembleFalse | 0.0635 |
| 3 | rife-v4.24_ensembleFalse | 0.0649 |
| 4 | rife-v4.6_ensembleFalse | 0.0657 |
| 5 | rife-v4.17_ensembleFalse | 0.0659 |
| 6 | rife-v4.12_lite_ensembleFalse | 0.0661 |
| 7 | rife-v4.20_ensembleFalse | 0.0661 |
| 8 | rife-v4.18_ensembleFalse | 0.0663 |
| 9 | rife-v4.18 | 0.0663 |
| 10 | rife-v4.22_lite_ensembleFalse | 0.0666 |
| 11 | rife-v4.13_lite_ensembleFalse | 0.0667 |
| 12 | rife-v4.16_lite_ensembleFalse | 0.0668 |
| 13 | rife-v4.15_lite_ensembleFalse | 0.0670 |
| 14 | rife-v4.14_ensembleFalse | 0.0670 |
| 15 | rife-v4.13_ensembleFalse | 0.0671 |
| 16 | rife-v4.3_ensembleFalse_fastTrue | 0.0673 |
| 17 | rife-v4.4_ensembleFalse_fastTrue | 0.0673 |
| 18 | rife-v4.15_ensembleFalse | 0.0673 |
| 19 | rife-v4.2_ensembleFalse_fastTrue | 0.0677 |
| 20 | rife-v3.1 | 0.0678 |


#### Compensate MAE

Best to worst, using `mv.Compensate` reconstruction of frames using their neighbors, by Mean Absolute Error (MAE).

Use case: **Block-based compensation (`mv.Compensate`)** and real-world block SAD validation, as this cleanly ships whole, un-smeared grained structures compared to Flow.

| Rank | Model Name | Dense MAE |
|---|---|---|
| 1 | rife-v4.2_ensembleFalse_fastTrue | 0.0180 |
| 2 | rife-v4.3_ensembleFalse_fastTrue | 0.0181 |
| 3 | rife-v4.4_ensembleFalse_fastTrue | 0.0182 |
| 4 | rife-v4.17_lite_ensembleFalse | 0.0182 |
| 5 | rife-v4.12_lite_ensembleFalse | 0.0183 |
| 6 | rife-v4.13_lite_ensembleFalse | 0.0184 |
| 7 | rife-v4.13_ensembleFalse | 0.0186 |
| 8 | rife-v4.15_lite_ensembleFalse | 0.0186 |
| 9 | rife-v4.14_ensembleFalse | 0.0186 |
| 10 | rife-v4.16_lite_ensembleFalse | 0.0186 |
| 11 | rife-v4.6_ensembleFalse | 0.0186 |
| 12 | rife-v3.1 | 0.0187 |
| 13 | rife-v4.15_ensembleFalse | 0.0187 |
| 14 | rife-v4.18 | 0.0187 |
| 15 | rife-v4.18_ensembleFalse | 0.0187 |
| 16 | rife-v4.17_ensembleFalse | 0.0187 |
| 17 | rife-v4.24_ensembleFalse | 0.0187 |
| 18 | rife-v4.5_ensembleFalse | 0.0187 |
| 19 | rife-v4.22_lite_ensembleFalse | 0.0188 |
| 20 | rife-v4.20_ensembleFalse | 0.0188 |