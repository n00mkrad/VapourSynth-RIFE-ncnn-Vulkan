# Model Benchmarks (2026-05-06)

Speed of each model to get the delta-1 and delta-2 motion vectors (both forward and backward for each delta) on a 1920x1024 input (500 frames) with block size 16x8 and overlap 8x4.

To ensure VapourSynth actually requests all vector frames (speed will be false if you do not actually utilize the returned clips), they were fed into Merge():

```python
delta1 = core.std.Merge(bwd[0], fwd[0])
delta2 = core.std.Merge(bwd[1], fwd[1])
clip = core.std.Merge(delta1, delta2) # Getting a frame from clip now requires all four vector clips
```

RTX 4090 (2 GPU threads per instance = 4 threads total) - Note that the GPU was never fully utilized due to bandwidth or other bottlenecks, power draw usually topped out at ~260W (out of 450). This means that if you have a slower card, your results may not be that much worse as the utilization is likely higher.

`[rmv] RIFEMV parameters: gpu_thread=2 shared_flow_inflight=8 shared_luma_cache=true flow_scale=1 res_scale=1 cpu_flow_resize=auto perf_stats=true blksize_x=16 blksize_y=8 overlap_x=8 overlap_y=4 pel=1 delta=2 bits=8 sad_multiplier=1 render_sad_mask=false matrix_in_s=709 range_in_s=full hpad=0 vpad=0 block_reduce=1 chroma=false inference_width=1920 inference_height=1024`

### Model Speed Ranking

| Model Name | FPS |
|---|---|
| rife-v4.2_ensembleFalse_fastTrue | 43.22 |
| rife-v4.4_ensembleFalse_fastTrue | 43.14 |
| rife-v4.3_ensembleFalse_fastTrue | 42.67 |
| rife-v4.5_ensembleFalse | 40.42 |
| rife-v4.6_ensembleFalse | 39.73 |
| rife-v3.1 | 35.65 |
| rife-v4.16_lite_ensembleFalse | 35.02 |
| rife-v4.15_lite_ensembleFalse | 33.07 |
| rife-v4.17_lite_ensembleFalse | 32.83 |
| rife-v4.13_lite_ensembleFalse | 32.61 |
| rife-v4.12_lite_ensembleFalse | 31.73 |
| rife-v4.22_lite_ensembleFalse | 30.45 |
| rife-v4.18_ensembleFalse | 28.47 |
| rife-v4.15_ensembleFalse | 28.43 |
| rife-v4.14_ensembleFalse | 28.42 |
| rife-v4.20_ensembleFalse | 28.32 |
| rife-v4.24_ensembleFalse | 28.30 |
| rife-v4.18 | 28.27 |
| rife-v4.17_ensembleFalse | 28.25 |
| rife-v4.13_ensembleFalse | 28.15 |
| rife-v4.19 | 28.09 |
| rife-v4.20 | 27.89 |
| rife-v4.25-lite_ensembleFalse | 27.88 |
| rife-v4.26_ensembleFalse | 27.55 |
| rife-v4.12_ensembleFalse | 27.38 |
| rife-v4.10_ensembleFalse | 27.27 |
| rife-v4.11_ensembleFalse | 26.35 |
| rife-v4.21 | 25.03 |
| rife-v4.21_ensembleFalse | 24.95 |
| rife-v4.22 | 24.95 |
| rife-v4.22_ensembleFalse | 24.82 |
| rife-v4.4_ensembleTrue_fastFalse | 24.63 |
| rife-v4.3_ensembleTrue_fastFalse | 24.58 |
| rife-v4.2_ensembleTrue_fastFalse | 24.45 |
| rife-v4.14_lite_ensembleFalse | 23.32 |
| rife-v4.5_ensembleTrue | 22.86 |
| rife-v4.6_ensembleTrue | 22.75 |
| rife-v4.9_ensembleFalse **[fp32 fallback]** | 22.01 |
| rife-v3.9_ensembleFalse_fastTrue | 21.32 |
| rife-v4.26-large_ensembleFalse | 21.17 |
| rife-v4.15_lite_ensembleTrue | 20.84 |
| rife-v4.17_lite_ensembleTrue | 20.65 |
| rife-v4.16_lite_ensembleTrue | 20.62 |
| rife-v4.7_ensembleTrue | 19.85 |
| rife-v4.8_ensembleTrue | 19.81 |
| rife-v4.9_ensembleTrue | 19.54 |
| rife-v4.13_ensembleTrue | 17.24 |
| rife-v4.13_ensembleTrue_fastFalse | 17.20 |
| rife-v4.24_ensembleTrue | 17.16 |
| rife-v4.14_ensembleTrue | 17.11 |
| rife-v4.18_ensembleTrue | 17.03 |
| rife-v4.12_ensembleTrue | 17.03 |
| rife-v4.15_ensembleTrue | 16.95 |
| rife-v4.11_ensembleTrue | 16.82 |
| rife-v4.10_ensembleTrue | 16.00 |
| rife-v4.17_ensembleTrue | 15.98 |
| rife-v4.14_lite_ensembleTrue | 14.12 |
| **No result / Currently incompatible** | |
| rife-v3.9_ensembleTrue_fastFalse | — |
| rife-v4.7_ensembleFalse | — |
| rife-v4.8_ensembleFalse | — |
| rife-v4.12_lite_ensembleTrue | — |
| rife-v4.13_lite_ensembleTrue | — |
| rife-v4.20_ensembleTrue | — |
| rife-v4.25_ensembleFalse | — |