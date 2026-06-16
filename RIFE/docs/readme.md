# About this modification

This modified build of the VapourSynth RIFE plugin focuses on MVTools-compatible motion-vector export.

It exports motion vectors in the same binary frame-property format used by MVTools:

- `MVTools_MVAnalysisData`
- `MVTools_vectors`

That makes it possible to feed RIFE-derived motion into MVTools consumers such as `mv.Mask`, `mv.Flow`, or `mv.Degrain*`.

Interpolation is no longer part of this fork. Use the unmodified upstream RIFE plugin when you want frame interpolation output.

## What this build adds

- `rmv.RIFEMV(...)`
  Returns both backward and forward vector clips from one inference pass.

- `rmv.RIFEMVApprox2(...)`
  Returns approximate vectors for deltas 1 and 2 by composing adjacent motions.

- `rmv.RIFEMVApprox3(...)`
  Returns approximate vectors for deltas 1, 2, and 3 by composing adjacent motions.

- `rmv.CropGrid(...)`
  Crops MVTools-compatible vector clips and matching `mv.Super` clips on the vector block-step grid.

## Important limitations

- Motion-vector export supports `rife-v3.1`, `rife-v3.9`, and `rife-v4.2+` model families.
- Legacy `rife-v4` as well as `rife-v4.0` and `rife-v4.1` are not supported for motion-vector export.
- Motion-vector APIs accept either constant-format `RGBS` or constant-format `YUV`. Non-`RGBS` `YUV` input is converted internally to `RGBS` for RIFE inference.
- MVTools metadata is derived from the main `clip`. For non-`RGBS` input, the original input clip is used as the metadata source automatically.
- Vector clips are `Gray8` carrier clips. The motion data still lives in frame properties. By default the pixel plane contains a SAD mask derived from the exported block SADs for that direction, and `render_sad_mask=False` leaves that plane black instead.
- By default the rendered SAD carrier mask is relative: it maps `0` to `0`, maps the largest SAD in that frame to `255`, and bilinearly upsamples the block grid to full-frame `Gray8`.
- If `abs_sad_clip_range > 0`, the rendered carrier mask switches to absolute mode: it quantizes the SAD range starting at `0`, clips values at the requested upper bound, and bilinearly upsamples the block grid to full-frame `Gray8`.
- Frames without a valid reference still export invalid MVTools vectors as before. When the SAD mask is rendered, relative mode makes it solid `255` and absolute mode shows the clipped sentinel SAD, which will usually also be `255`.
- Exported motion-vector frames can optionally include SAD summary frame properties `RMV_AvgSad`, `RMV_MaxSad`, and `RMV_MinSad` (controlled by `sad_stats`), and motion-summary frame properties `RMV_AvgAbsDx`, `RMV_AvgAbsDy`, `RMV_AvgAbsMotion`, and `RMV_PanAmount` (controlled by `motion_stats`). `RMV_PanAmount` is the source-pixel magnitude of the median signed frame motion vector, which is intended to track coherent panning/camera translation better than local object motion.
- Do not resize or colorspace-convert the exported vector clips after creation. Use `rmv.CropGrid(...)` when a vector clip and matching `mv.Super` clip need a metadata-aware crop.

## Motion-vector arguments

- `flow_scale`
  Scales the image before flow estimation and rescales vectors back to the original image coordinates.  
  `flow_scale` replaces the `uhd` bool parameter used in the original plugin. To match the old behavior, use `0.5` for `uhd=True` or `1.0` (default) for `uhd=False`.
  Accepted values are restricted to: `0.25`, `0.5`, `1.0`, `2.0`, `4.0`.

- `res_scale`
  Motion-vector clip-resize factor applied before RIFE flow inference.
  Default: `1.0`.
  The plugin rescales the RGBS inference clip to `round(width * res_scale)` by `round(height * res_scale)` and runs RIFE flow on that resized clip. Motion-vector reduction then happens on an internal block lattice derived from the inference size, and only the final block vectors are scaled back to original-image coordinates for SAD computation and MVTools export.
  This means `blksize_x`, `blksize_y`, `overlap_x`, `overlap_y`, `hpad`, and `vpad` always operate on the original clip geometry, so a given block-size configuration always produces the same block grid regardless of `res_scale`.
  Example: use `res_scale=0.5` to run a 2160p clip internally at about 1080p without changing MVTools block size.

- `cpu_flow_resize`
  Debug control for the internal resize path used by motion-vector export.
  Omit this argument for automatic behavior (GPU resize when available, CPU fallback on failure).
  `False` (`0`) forces the GPU resize path only.
  `True` (`1`) forces the CPU resize fallback path.

- `gpu_mode`
  Manual export backend used by `RIFEMV` after RIFE has produced optical flow.
  Default: `0`.
  This argument is not exposed by `RIFEMVApprox2` or `RIFEMVApprox3`.
  Accepted values:
  - `0` (`cpu`) reads dense flow back to CPU and performs block reduction, vector conversion, SAD, stats, and blob packing on CPU. This is the safest compatibility path.
  - `1` (`gpu_flow_reduce`) performs dense-flow-to-block-flow reduction on GPU, then reads back compact per-block flow. CPU still performs vector conversion, SAD, stats, and blob packing.
  - `2` (`gpu_full`) performs flow reduction, vector conversion, clamping, and raw SAD on GPU (`chroma=0`: luma SAD, `chroma=1`: RGB SAD), then reads back compact vector arrays. CPU computes stats and packs MVTools blobs.
  `2` (`gpu_full`) is currently limited to source-sized inference (`res_scale=1.0`). There is no automatic fallback; unsupported configurations fail instead of silently switching backend.
  String values are not accepted.
  See [GPU export modes](gpu-modes.md) for details and tradeoffs.

- `shared_flow_inflight`
  Global in-flight cap for motion-vector flow inference shared across filter instances on the same GPU.
  This affects `RIFEMV`, `RIFEMVApprox2`, and `RIFEMVApprox3`.
  Default: GPU compute queue count.
  Lower values can reduce CPU contention; higher values can increase throughput on some setups.
  When explicitly set, local admission is relaxed to at least this value (`max(gpu_thread, shared_flow_inflight)`) so the shared cap remains the primary limiter.

- `shared_luma_cache`
  Controls whether source-frame luma planes are shared across compatible filter instances.
  This affects `RIFEMV`, `RIFEMVApprox2`, and `RIFEMVApprox3`.
  Default: `True`.
  `True` / `1` enables automatic sharing for the same logical source clip when synthetic SAD is computed from luma.
  `False` / `0` disables cross-instance sharing and keeps only the local per-instance cache.
  This is only relevant when `chroma=0`.
  Automatic sharing is additionally scoped by the source clip identity, `bits`, and, for YUV input, `matrix_in_s` and `range_in_s`.

- `shared_packed_cache`
  Controls whether prepacked inference RGB frames are shared across compatible filter instances.
  This affects `RIFEMV`, `RIFEMVApprox2`, and `RIFEMVApprox3`.
  Default: `True`.
  `True` / `1` enables cross-instance sharing for the same logical source clip and inference setup.
  `False` / `0` keeps packed-frame reuse local to the current filter instance only.
  Automatic sharing is scoped by source clip identity, inference width and height, and, for YUV input, `matrix_in_s` and `range_in_s`.

- `packed_cache_mib`
  Memory budget (MiB) for packed inference-frame cache entries.
  This affects `RIFEMV`, `RIFEMVApprox2`, and `RIFEMVApprox3`.
  Default: `256`.
  Must be greater than `0`.
  This cache only targets CPU-side inference-input repacking overhead and does not change vector/SAD semantics.

- `blksize_x`, `blksize_y`
  Exported MVTools block size on each axis.
  Default: `blksize_x=16`, `blksize_y=blksize_x`.

- `overlap_x`, `overlap_y`
  Exported MVTools overlap on each axis.
  Default: `overlap_x=blksize_x // 2`, `overlap_y=blksize_y // 2`.

- `pel`
  MVTools pel value written to metadata and used for vector scaling.
  Default: `1`.

- `delta`
  Temporal distance written to `nDeltaFrame` in the MV metadata for `RIFEMV`.
  Default: `1`.

- `bits`
  Synthetic bit depth used when computing exported SAD values.
  Default: `8`.
  Leaving this at `8` keeps exported SAD on an 8-bit scale regardless of source clip bit depth.
  Set a higher value only if you explicitly want larger SAD values that track a higher-bit-depth scale.
  This also sets the exported MVTools `bitsPerSample` metadata so downstream filters scale `thsad` and `thscd1` against the same SAD range.

- `abs_sad_clip_range`
  Controls how SAD values are written into the `Gray8` carrier pixels.
  Default: `0`.
  `0` keeps the relative per-frame normalization mode.
  Values greater than `0` enable absolute mode, where the carrier starts at SAD `0`, clips at the requested upper bound, and uses the available `Gray8` precision to quantize that range.
  Examples: `1024` gives steps of `4`, `2048` gives steps of `8`, `4096` gives steps of `16`.

- `render_sad_mask`
  Controls whether the `Gray8` carrier plane is populated with the direction-specific SAD mask.
  Default: `True`.
  If disabled, the carrier plane is left black while motion vectors and metadata remain unchanged.
  This can reduce CPU overhead when downstream consumers only need the frame properties.

- `sad_stats`
  Controls whether SAD summary frame properties are computed and attached.
  This affects `RIFEMV`, `RIFEMVApprox2`, and `RIFEMVApprox3`.
  Default: `False`.
  If enabled, `RMV_AvgSad`, `RMV_MaxSad`, and `RMV_MinSad` are attached.

- `motion_stats`
  Controls whether motion summary frame properties are computed and attached.
  This affects `RIFEMV`, `RIFEMVApprox2`, and `RIFEMVApprox3`.
  Default: `False`.
  If enabled, `RMV_AvgAbsDx`, `RMV_AvgAbsDy`, `RMV_AvgAbsMotion`, and `RMV_PanAmount` are attached.

- `matrix_in_s`
  Input matrix used when the MV API receives a non-`RGBS` `YUV` clip and performs internal conversion to `RGBS`.
  Default: `"709"`.

- `range_in_s`
  Input range used for that same internal `YUV` -> `RGBS` conversion.
  Default: `"full"`.

- `hpad`, `vpad`
  Horizontal and vertical padding written into MV metadata and used for vector clamping.
  These should match the corresponding `mv.Super` settings when relevant.

- `block_reduce`
  Controls how dense RIFE flow is reduced to one block vector.
  `0` = center sample.
  `1` = average over the whole block.
  Default: `1`.

- `chroma`
  If enabled, synthetic SAD includes all RGB channels. Otherwise it uses luma only.

## `rmv.CropGrid`

`CropGrid` crops MVTools-compatible vector clips and matching `mv.Super` clips without leaving stale dimensions in their frame properties.

### Signature

```python
cropped = core.rmv.CropGrid(clip, left=0, right=0, top=0, bottom=0, vectors=None)
```

`left`, `right`, `top`, and `bottom` are not pixel counts. They are counts of vector block steps:

```text
step_x = nBlkSizeX - nOverlapX
step_y = nBlkSizeY - nOverlapY
```

For example, with `blksize_x=16` and `overlap_x=8`, `left=2` removes two vector columns and crops 16 pixels from the carrier clip.

For vector clips, pass the vector clip as `clip`:

```python
mvbw_crop = core.rmv.CropGrid(mvbw, left=1, right=1)
mvfw_crop = core.rmv.CropGrid(mvfw, left=1, right=1)
```

For `mv.Super` clips, pass the Super clip as `clip` and a matching vector clip as `vectors` so `CropGrid` can derive the block step. The vector clip may be either the original matching vector clip or the vector clip after applying the same `CropGrid` crop:

```python
sup_crop = core.rmv.CropGrid(sup, left=1, right=1, vectors=mvfw)
```

`CropGrid` v1 supports single-level vector clips, which matches the output from this plugin's `RIFEMV`, `RIFEMVApprox2`, and `RIFEMVApprox3` functions. Native multi-level MVTools vector clips are rejected.

## `rmv.RIFEMV`

`rmv.RIFEMV` is the convenience function for the common delta-1 case.

### Signature

```python
mvbw, mvfw = core.rmv.RIFEMV(clip, model_path=..., gpu_id=default_gpu, gpu_thread=2, shared_flow_inflight=None, shared_luma_cache=True, shared_packed_cache=True, packed_cache_mib=256, flow_scale=1.0, res_scale=1.0, cpu_flow_resize=None, gpu_mode=0, perf_stats=False, blksize_x=16, blksize_y=None, overlap_x=None, overlap_y=None, pel=1, delta=1, bits=8, abs_sad_clip_range=0, render_sad_mask=True, sad_stats=False, motion_stats=False, matrix_in_s="709", range_in_s="full", hpad=0, vpad=0, block_reduce=1, chroma=0)
```

### Return value

`RIFEMV` returns `clip:vnode[]` with this ordering:

```python
mvbw, mvfw = core.rmv.RIFEMV(...)
```

- first output: backward vectors
- second output: forward vectors

`cpu_flow_resize`:
- omitted = automatic (GPU resize with CPU fallback)
- `0`/`False` = force GPU resize
- `1`/`True` = force CPU resize

`perf_stats` enables per-filter performance timing. When enabled, a summary is printed to `stderr` when the filter instance is freed (end of clip processing).

`perf_stats` now reports:
- `semaphore_wait_ms` total wait
- `local_wait_ms` wait on per-filter `gpu_thread` limiter
- `shared_wait_ms` wait on the cross-instance `shared_flow_inflight` limiter
- `flow_readback_mib` and `flow_readback_avg_mib` GPU-to-CPU readback size
- `flow_reduce_record_ms` GPU command recording time for `gpu_flow_reduce`
- `flow_vector_record_ms` GPU command recording time for `gpu_full`
- `packed_cache_hits` and `packed_cache_misses` packed inference-frame cache reuse counters
- `packed_build_ms` and `packed_wait_ms` packed-frame build and cache-contention wait time
- `render_sad_mask_ms` time spent rasterizing the `Gray8` SAD carrier plane when `render_sad_mask=True`

### Recommended usage

```python
mvbw, mvfw = core.rmv.RIFEMV(clip, model_path=rife_mdl, matrix_in_s="709", range_in_s="full")

mask = core.mv.Mask(clip, mvfw, kind=5, ml=100.0)
```

### Example with Degrain1

```python
sup = core.mv.Super(clip, pel=1, hpad=0, vpad=0, levels=1)
mvbw, mvfw = core.rmv.RIFEMV(clip, model_path=rife_mdl, matrix_in_s="709", range_in_s="full", pel=1, hpad=0, vpad=0)

den = core.mv.Degrain1(clip, sup, mvbw, mvfw, thsad=500)
```

When using MVTools consumers that depend on `pel`, `hpad`, or `vpad`, keep those values aligned between `mv.Super(...)` and the RIFE exporter.

## `rmv.RIFEMVApprox2` and `rmv.RIFEMVApprox3`

These functions generate approximate larger-delta motion by composing adjacent frame-to-frame displacements.

They are useful when you want delta-2 or delta-3 vectors without running separate direct exporters for each temporal distance.

### `RIFEMVApprox2`

```python
outputs = core.rmv.RIFEMVApprox2(clip, model_path=rife_mdl, matrix_in_s="709", range_in_s="full")
```

Output order:

```python
bw1, fw1, bw2, fw2 = core.rmv.RIFEMVApprox2(...)
```

- `bw1`, `fw1`: approximate delta-1 vectors
- `bw2`, `fw2`: approximate delta-2 vectors

### `RIFEMVApprox3`

```python
outputs = core.rmv.RIFEMVApprox3(clip, model_path=rife_mdl, matrix_in_s="709", range_in_s="full")
```

Output order:

```python
bw1, fw1, bw2, fw2, bw3, fw3 = core.rmv.RIFEMVApprox3(...)
```

- `bw1`, `fw1`: approximate delta-1 vectors
- `bw2`, `fw2`: approximate delta-2 vectors
- `bw3`, `fw3`: approximate delta-3 vectors

### Shared arguments

`RIFEMVApprox2` and `RIFEMVApprox3` accept the same arguments as `RIFEMV`, except they do not expose `delta` because each function has a fixed maximum delta built into it, and they do not expose `gpu_mode` because the approximation path still requires dense displacement data.

### Example with Degrain2

```python
sup = core.mv.Super(clip, pel=1, hpad=0, vpad=0, levels=1)
bw1, fw1, bw2, fw2 = core.rmv.RIFEMVApprox2(clip, model_path=rife_mdl, matrix_in_s="709", range_in_s="full")

den = core.mv.Degrain2(clip, sup, bw1, fw1, bw2, fw2, thsad=500)
```

## Practical notes

- Use `block_reduce=1` as the default starting point for degraining.
- You can pass YUV clips directly; internal conversion to RGBS is done automatically for MV inference.
- MVTools metadata always comes from the main `clip`.
- Keep `pel`, `hpad`, and `vpad` consistent with the `mv.Super` clip you use downstream.
- If a function only needs one direction, call `rmv.RIFEMV(...)` and use either `mvbw` or `mvfw`.
- If you need both directions for delta 1, prefer `rmv.RIFEMV(...)`.
- If you need approximate delta 2 or 3 vectors, use `rmv.RIFEMVApprox2(...)` or `rmv.RIFEMVApprox3(...)`.

## Summary

This fork is now dedicated to exporting MVTools-compatible motion vectors from RIFE optical flow.

The key idea is:

- RIFE still runs on `RGBS` internally (with optional automatic YUV -> RGBS conversion in MV APIs)
- MVTools still consumes its own vector-clip format
- this build bridges the two by exporting MVTools-compatible binary vector properties

Use the upstream RIFE plugin for interpolation and this fork for MVTools workflows.
