# RIFEMV GPU Export Modes

`RIFEMV` can use different export backends through `gpu_mode`.

This setting controls what happens after RIFE has produced dense optical flow. It does not select a different RIFE model and it does not retrain or alter model weights. The goal is to reduce CPU work and GPU-to-CPU readback while still exporting MVTools-compatible vector clips.

The default is:

```python
mvbw, mvfw = core.rmv.RIFEMV(clip, model_path=rife_mdl, gpu_mode=0)
```

Mode mapping:
- `0` = `cpu`
- `1` = `gpu_flow_reduce`
- `2` = `gpu_full`

`RIFEMVApprox2` and `RIFEMVApprox3` do not expose this option. They currently need dense displacement data for temporal composition, so they use the dense CPU export path internally.

## Mode Summary

| Mode | Dense flow readback | GPU work after RIFE | CPU work after readback | Current status |
|---|---:|---|---|---|
| `0` (`cpu`) | Full dense flow | none | flow reduction, vector conversion, SAD, stats, blob packing | safest default |
| `1` (`gpu_flow_reduce`) | compact block flow | flow reduction only | vector conversion, SAD, stats, blob packing | recommended GPU-assisted mode |
| `2` (`gpu_full`) | compact block vectors | flow reduction, vector conversion, clamping, raw SAD | `sad_multiplier`, stats, blob packing | experimental, narrower config support |

For a 1920x1024 clip using the default 16x8 blocks and 8x4 overlap, the approximate readback sizes are:

| Mode | Approximate readback per flow call |
|---|---:|
| `0` (`cpu`) | about 15 MiB |
| `1` (`gpu_flow_reduce`) | about 0.93 MiB |
| `2` (`gpu_full`) | about 1.86 MiB |

`gpu_full` reads back more than `gpu_flow_reduce` because it returns two compact vector arrays, one backward and one forward, including raw SAD. It can still be faster if moving SAD and vector generation to GPU saves enough CPU work.

## `"cpu"`

`"cpu"` is the original dense export path and remains the default.

RIFE produces dense optical flow on GPU. The plugin reads the dense flow tensor back to CPU, then CPU code converts it into MVTools block vectors.

The CPU path performs:

- block flow reduction from dense per-pixel flow
- vector scaling to MVTools units
- `lround()`-style vector rounding
- vector clamping against frame bounds and padding
- synthetic SAD calculation
- `RMV_*` frame-stat calculation
- MVTools blob packing

This mode has the broadest compatibility. It supports `chroma=0`, `chroma=1`, `res_scale`, `flow_scale`, and the existing CPU/GPU flow-resize behavior.

Use this mode when validating correctness, comparing output against other modes, or using settings that the GPU paths do not support yet.

## `"gpu_flow_reduce"`

`"gpu_flow_reduce"` is the lower-risk GPU-assisted path.

RIFE still produces dense optical flow on GPU, but the plugin dispatches a small compute shader before readback. That shader reduces the dense flow into one compact flow tuple per MVTools block:

```text
backward_x, backward_y, forward_x, forward_y
```

The CPU then performs the same post-reduction work as the normal path:

- vector scaling
- vector rounding
- vector clamping
- synthetic SAD
- frame stats
- MVTools blob packing

This mode removes most of the dense-flow readback cost while keeping SAD and MVTools finalization on CPU. It is usually the best first GPU mode to test because it changes less of the MV export semantics than `gpu_full`.

Expected effects:

- `flow_readback_avg_mib` should drop sharply compared to `"cpu"`.
- `flow_export_direct_ms` should usually become much smaller.
- `flow_reduce_record_ms` should become non-zero.
- `vector_pack_ms` can still be significant because CPU still computes SAD and packs vectors.

Parity notes:

- `block_reduce=0` is a direct center sample and should match the CPU path closely.
- `block_reduce=1` averages in the shader using float arithmetic, while the CPU path accumulates using double. Vectors can differ near rounding boundaries.
- SAD is still CPU-computed, so SAD behavior should remain aligned once vector components match.

There is no automatic fallback in manual mode. If the shader cannot handle the current GPU flow layout, the filter fails instead of silently switching to `"cpu"`.

## `"gpu_full"`

`"gpu_full"` moves most of the block-vector export work to GPU.

The shader performs:

- dense-flow block reduction
- vector scaling to MVTools units
- vector rounding
- vector clamping
- pixel displacement conversion
- raw SAD (`chroma=0`: luma SAD, `chroma=1`: RGB SAD)

The CPU still performs:

- `sad_multiplier` application
- `RMV_*` frame-stat calculation
- MVTools blob packing
- frame-property output

The shader outputs compact vectors instead of final MVTools blobs. That keeps the public blob format unchanged and avoids making GPU code responsible for the serialized MVTools layout.

Current limitations:

- `res_scale=1.0` only, because the shader currently computes SAD from the same RGBS frames uploaded for RIFE inference.
- Source dimensions must match inference dimensions.
- Raw SAD must fit in 32-bit storage before `sad_multiplier` is applied.
- No automatic fallback.

Expected effects:

- `flow_vector_record_ms` should become non-zero.
- `flow_readback_avg_mib` should be compact, but usually larger than `gpu_flow_reduce`.
- `luma_build_ms` should be near zero for this path because CPU luma cache construction is skipped.
- `vector_pack_ms` should mostly represent CPU conversion from compact GPU vectors, stats, and blob packing rather than full SAD computation.

Parity notes:

`gpu_full` is more likely to differ from `"cpu"` than `gpu_flow_reduce` because vector rounding, clamping, sample quantization, and raw SAD are all performed in shader code. The intended behavior matches the CPU path, but small differences are possible near rounding boundaries or SAD quantization boundaries.

Use this mode as an explicit experimental backend until it has been validated on the specific settings you care about.

## Choosing A Mode

Use `gpu_mode=0` (`cpu`) when correctness and broad compatibility matter more than speed, or when testing a setting that the GPU modes do not support.

Use `gpu_mode=1` (`gpu_flow_reduce`) when you want the main readback reduction with relatively low semantic risk. This is the best general performance mode to try first.

Use `gpu_mode=2` (`gpu_full`) when you want to test whether moving vector and SAD work to GPU improves your workload. Expect it to be more configuration-sensitive than `gpu_flow_reduce`.

Example:

```python
mvbw, mvfw = core.rmv.RIFEMV(
    clip,
    model_path=rife_mdl,
    gpu_mode=1,
    blksize_x=16,
    blksize_y=8,
    overlap_x=8,
    overlap_y=4,
    render_sad_mask=False,
    perf_stats=True,
)
```

For `gpu_full`:

```python
mvbw, mvfw = core.rmv.RIFEMV(
    clip,
    model_path=rife_mdl,
    gpu_mode=2,
    res_scale=1.0,
    chroma=False,
    perf_stats=True,
)
```

## Performance Counters To Watch

`perf_stats=True` is the easiest way to compare modes.

Useful counters:

- `flow_readback_avg_mib`: average GPU-to-CPU readback size per RIFE flow call.
- `flow_export_direct_ms`: CPU time spent copying mapped readback data into the plugin output buffer.
- `flow_reduce_record_ms`: command-recording time for the `gpu_flow_reduce` shader.
- `flow_vector_record_ms`: command-recording time for the `gpu_full` shader.
- `flow_submit_wait_ms`: time waiting for the GPU work and readback to complete.
- `luma_build_ms`: CPU luma cache construction time. This should matter for `"cpu"` and `"gpu_flow_reduce"`, but not for `"gpu_full"`.
- `vector_pack_ms`: CPU-side vector finalization, stats, and blob packing time. In `"cpu"` this includes dense-flow reduction and SAD. In `"gpu_flow_reduce"` it still includes SAD. In `"gpu_full"` it should be much smaller and mostly represent final conversion and packing.

Interpret these counters together with end-to-end FPS. Some GPU modes move time from CPU counters into GPU wait time, so a single counter can look worse even when total throughput improves.

## Failure Behavior

Backend selection is manual. There is currently no `auto` mode and no fallback mechanism.

If a selected backend is unsupported for the current configuration, the filter reports an error. This is intentional for now because it makes benchmarking and correctness comparison explicit.
