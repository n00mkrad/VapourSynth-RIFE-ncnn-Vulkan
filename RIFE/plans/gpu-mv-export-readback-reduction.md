# GPU Motion Vector Export Readback Reduction Plan

## Problem Sketch

### Symptoms

- Changing MV block size or overlap has little effect on total FPS, even though it changes CPU-side block work.
- Increasing `gpu_thread` or `shared_flow_inflight` eventually plateaus, while GPU utilization still does not look fully saturated.
- CPU-only optimizations reduced CPU load, but wall-clock FPS barely moved.
- Recent stats show `flow_readback_avg_mib=15.000` for 1920x1024 input, per flow call.
- With the default MV layout of 16x8 blocks and 8x4 overlap, the dense flow readback is still full-frame sized, not block-count sized.
- The current full-frame dense flow readback is about 15 MiB per call at 1920x1024 because it transfers 4 fp16 flow channels for every pixel.
- For 1920x1024 with 16x8 blocks and 8x4 overlap, the MV grid is about 239x255 blocks, or 60,945 vectors per direction. Two `MVToolsVector` arrays are roughly 1.9 MiB before blob overhead, much smaller than dense flow.

### Cause

RIFE already produces dense per-pixel optical flow on the GPU. RIFEMV currently reads that whole dense flow back to CPU and only then reduces it to MVTools block vectors in `buildMotionVectorBlobsFromConfig()`.

That means the expensive readback is proportional to `width * height * 4`, not to `blkX * blkY`. Block parameters only affect the later CPU reduction and packing stage. This explains why block-size and overlap changes do not strongly affect total speed.

### Rough Fix Outline

- Keep running the same RIFE model. No model retraining is required.
- Add an optional GPU export path that leaves dense flow on GPU.
- Reduce dense flow into a compact block representation on GPU before readback.
- Read back only compact per-block data, then pack the existing MVTools blobs on CPU.
- Keep the existing dense-flow CPU path as the default-safe fallback and as a comparison path.
- Start with a lower-risk GPU flow pre-reduction path before attempting full GPU SAD/vector export.

## Implementation Strategy

The implementation should be optional from day one. Add a backend switch such as `mv_export_backend` or `gpu_mv_export` with at least these modes:

- `cpu`: current behavior, always read back dense flow and build vectors on CPU.
- `gpu_flow_reduce`: GPU reduces dense flow to per-block flow samples, CPU handles vector clamping, SAD, stats, and blob packing.
- `gpu_full`: GPU builds complete block vectors including SAD, CPU only packs blobs and frame props.
- `auto`: try the best supported GPU path and fall back to `cpu` if unsupported.

The existing path must remain unchanged and callable for correctness checks, perf comparison, and fallback.

## Recommended First Path: GPU Flow Pre-Reduction

This path reduces most of the readback without moving all MVTools semantics to GPU.

### Goal

Move only `reduceBlockFlow()` to GPU:

- Input: dense RIFE flow still resident as `ncnn::VkMat`.
- Output: compact per-block reduced flow values for backward X/Y and forward X/Y.
- CPU still performs:
  - motion scaling
  - `lround()`
  - MVTools component clamping
  - pixel displacement conversion
  - SAD calculation
  - frame stats
  - blob serialization

For default 16x8 blocks with 8x4 overlap at 1920x1024, this changes readback from about 15 MiB per flow call to roughly `blkX * blkY * 4 * sizeof(float)`, or about 0.93 MiB. If fp16 output is acceptable for reduced flow, that can be about 0.46 MiB, but float32 is safer for parity.

### GPU Shader

Add a new compute shader, for example `rife_mv_reduce.comp`.

Each invocation handles one block index:

- Compute `bx`, `by`, and `vectorIndex`.
- Reconstruct the same internal block origin used by CPU:
  - `internalBlockX = bx * internalStepX - internalHPadding`
  - `internalBlockY = by * internalStepY - internalVPadding`
- For `block_reduce=center`, sample one clamped pixel from each flow plane.
- For `block_reduce=average`, iterate over `internalBlockSizeX * internalBlockSizeY`, clamp each sample coordinate, and accumulate each flow plane.
- Write four float32 values per block:
  - backward flow X
  - backward flow Y
  - forward flow X
  - forward flow Y

For the default 16x8/8x4 case, each block averages 128 pixels. With about 60,945 blocks, that is roughly 7.8 million flow samples per direction pair. This is reasonable for GPU compute and avoids transferring the dense flow.

### CPU Integration

Add a new path near `RIFE::process_flow()` or a sibling method that returns compact reduced-flow data instead of dense flow.

Suggested API shape:

```cpp
struct ReducedFlowBlock final {
    float backwardX;
    float backwardY;
    float forwardX;
    float forwardY;
};
```

Then add a CPU builder similar to `buildMotionVectorBlobsFromConfig()`, but taking `ReducedFlowBlock*` instead of dense `float* flow`.

It should reuse the existing CPU logic after `reduceBlockFlow()`:

- Convert reduced flow to MVTools vector components.
- Clamp components with `clampMotionVectorComponent()`.
- Compute pixel dx/dy.
- Compute SAD with the current CPU `computeBlockSAD()`.
- Pack with the existing `packMotionVectorBlob()`.

This is the safest first step because SAD parity stays exactly on the current CPU implementation.

### Parity Notes

`block_reduce=center` should match very closely because it is a direct sample.

`block_reduce=average` may differ slightly because CPU currently accumulates into `double`, while a shader would normally accumulate in float32. That can affect vectors only near an `lround()` boundary. If strict byte-for-byte parity is required, keep this backend optional and add validation tools before promoting it.

Possible mitigations:

- Use float32 output first and measure mismatch frequency.
- Keep `gpu_flow_reduce` disabled by default until validated.
- Allow `gpu_flow_reduce` only for `block_reduce=center` initially if exact parity is required.
- Add a debug comparison mode that runs both CPU dense and GPU-reduced paths for selected frames and reports vector/SAD mismatches.

## Full GPU Path: Block Vectors And SAD On GPU

This path has the highest potential gain but is more complex.

### Goal

Move almost all of `buildMotionVectorBlobsFromConfig()` to GPU:

- Reduce flow to block motion.
- Scale and clamp vectors.
- Compute pixel dx/dy.
- Compute SAD.
- Output `MVToolsVector` arrays for backward and forward directions.

The CPU would then read back only the two vector arrays and use existing blob packing and frame prop code.

### Extra Inputs Needed

For SAD, the GPU path needs image samples matching the current CPU source:

- For `chroma=false`, it needs current and reference luma planes built with the same quantization and RGB-to-luma formula as `buildFrameLumaPlane()`.
- For `chroma=true`, it needs current/reference RGBS planes and the same per-channel quantization behavior as `computeBlockSAD()`.
- It must use the same source frames that CPU SAD uses, not a resized or padded inference-only representation unless those are guaranteed equivalent.

### SAD Representation

The CPU stores `sad` as `int64_t`, but raw SAD for normal block sizes usually fits in 32 bits:

- 16x8 luma, 8-bit: max raw SAD is 32,640.
- 16x8 RGB, 8-bit: max raw SAD is 97,920.
- 16x8 RGB, 16-bit: max raw SAD is 25,165,440.

The GPU shader can output raw `uint32_t` SAD plus vector x/y, and the CPU can apply `sad_multiplier` into `int64_t` to preserve existing overflow validation and stats behavior.

If larger blocks, higher bit depths, or `sad_multiplier` combinations exceed safe 32-bit raw SAD assumptions, fall back to CPU or use a shader path requiring 64-bit integer support only when available.

### Output Format

Do not write MVTools blobs directly on GPU at first. Instead, write a compact GPU vector struct and reuse CPU blob packing:

```cpp
struct GpuMVToolsVector final {
    int32_t x;
    int32_t y;
    uint32_t rawSad;
    uint32_t flagsOrPadding;
};
```

CPU converts this to existing `MVToolsVector`:

- Copy `x` and `y`.
- Apply `sad_multiplier` and rounding to `rawSad`.
- Pack with `packMotionVectorBlob()`.
- Compute stats with existing `computeMotionVectorFrameStats()`.

This keeps the public blob format unchanged.

### Full GPU Path Risks

- Exact luma quantization and SAD rounding must match CPU.
- Boundary clamping must match `computeBlockSAD()`.
- Float-to-int rounding must match `std::lround()` behavior closely enough.
- `sad_multiplier` handling should remain CPU-side unless GPU parity is proven.
- `chroma=true` triples SAD work and input bandwidth.
- Validation needs to cover boundary frames, h/v padding, `pel`, `bits`, `block_reduce`, and both directions.

## Alternative Paths

### GPU Center-Sample Path Only

If average reduction parity is too hard, implement GPU reduction only for `block_reduce=center`.

This is much simpler:

- One clamped flow sample per block per direction.
- No average accumulation mismatch.
- Large readback reduction.
- CPU keeps vector clamping and SAD.

It does not help the current default `block_reduce=average`, but it provides a safe GPU export path and infrastructure.

### GPU Flow Tile Summaries

Instead of outputting one reduced flow per final MV block, output integral-image-like or tile-summary data for flow planes.

Potential benefit:

- CPU can compute block averages from compact summaries.
- Average semantics can be closer to CPU if summaries are high precision.

Drawbacks:

- More complex than direct per-block reduction.
- Still needs careful handling of clamped border samples.
- Less direct benefit than per-block reduced flow.

### Dense Flow Readback In Smaller Tiles

Read back only tiles needed for current block ranges, process them, then discard them.

This is not a strong solution:

- Total transferred data is still close to full dense flow for overlapping blocks.
- It may reduce peak memory or improve cache behavior, but not the main bandwidth cost.
- It adds scheduling complexity without making block count drive readback size.

### Read Back Dense Flow As fp16 Only

The current path already reads about 15 MiB per call, consistent with 4 fp16 channels at 1920x1024. Further CPU-side fp16 handling can reduce CPU expansion costs, but it cannot solve the GPU-to-CPU bandwidth issue.

This path is useful as a micro-optimization, not as the main fix.

## Detailed Implementation Plan

### Phase 1: Backend Plumbing And Stats

- Add an internal enum for MV export backend.
- Add an optional user parameter, or initially an internal/debug parameter, to select `cpu`, `gpu_flow_reduce`, `gpu_full`, or `auto`.
- Default to `cpu` until validation is strong.
- Print selected backend in the existing `[rmv] RIFEMV parameters` log.
- Add perf counters:
  - `gpu_mv_backend`
  - `gpu_mv_fallback_count`
  - `gpu_mv_reduce_record_ms`
  - `gpu_mv_readback_mib`
  - `gpu_mv_readback_wait_ms`
  - `gpu_mv_cpu_finalize_ms`
  - `gpu_mv_validation_mismatch_count`
- Keep the existing dense-flow readback stats for CPU backend.

### Phase 2: Expose A Compact Flow Result From RIFE

- Keep `RIFE::process_flow()` unchanged for the CPU backend.
- Add a new method or internal helper for the GPU-reduction backend.
- The method should run RIFE inference exactly as before.
- Before dense flow readback, dispatch the new reduction shader against the GPU `flow` `VkMat`.
- Read back the compact `ReducedFlowBlock` buffer.
- Return compact reduced-flow blocks plus dimensions and layout metadata.
- If shader creation, dispatch, readback, or validation fails, return a status that lets the caller fall back to `process_flow()`.

### Phase 3: Add The Reduction Shader

- Create a shader source file following the existing ncnn pipeline pattern.
- Compile/create the pipeline in `RIFE::load()`.
- Add specialization or push constants for:
  - flow width and height
  - flow channel layout or packed layout assumptions
  - block grid width/height
  - internal block size
  - internal step
  - internal padding
  - reduce mode
- Prefer float32 output for reduced flow blocks.
- Initially support only the layouts observed in current models.
- If an unsupported `VkMat` layout appears, report unsupported and fall back.

### Phase 4: CPU Finalization From Reduced Flow

- Add `buildMotionVectorBlobsFromReducedFlow()`.
- Keep the existing `buildMotionVectorBlobsFromConfig()` intact.
- Share as much code as practical after the flow-reduction step.
- The new function should:
  - resize reusable vector/blob scratch buffers
  - read `ReducedFlowBlock` for each block
  - apply the same scale, round, clamp, and SAD logic as the CPU dense path
  - pack backward and forward blobs with the existing pack helper
  - produce the same frame props and stats

### Phase 5: Validation Mode

Add a debug-only comparison mode:

- Run the CPU dense path and the selected GPU path on the same frame.
- Compare vector count, x/y, SAD, and serialized blob size.
- Track max x/y mismatch, max SAD mismatch, and mismatch count.
- Optionally fail hard on mismatch for development builds.
- Allow validating only the first N frames to keep benchmarks practical.

Use this to answer whether `block_reduce=average` float32 GPU accumulation is acceptable. If not, either restrict GPU reduction to center mode or keep the average path behind an explicit opt-in.

### Phase 6: Auto Mode Rules

`auto` should be conservative:

- Use `gpu_flow_reduce` when the shader supports the current flow layout and MV config.
- Fall back to `cpu` for unsupported formats, unsupported `block_reduce`, unsafe SAD ranges, failed shader creation, or validation failures.
- Keep `gpu_full` disabled unless it has full parity coverage.
- Never silently change public output semantics.

### Phase 7: Full GPU Path, If Needed

Only attempt this after `gpu_flow_reduce` is measured.

- Add optional GPU luma/RGB preparation matching CPU SAD inputs.
- Add a shader that produces vector x/y and raw SAD per block for both directions.
- Keep `sad_multiplier`, blob packing, and stats on CPU at first.
- Validate against CPU on diverse clips and settings.
- Promote to `auto` only for configurations proven to match.

## Expected Result

For the default 1920x1024, 16x8 block, 8x4 overlap configuration:

- Dense readback path: about 15 MiB per flow call.
- GPU flow pre-reduction path: about 0.93 MiB per flow call with float32 reduced flow blocks.
- Full GPU vector path: about 1.9 MiB per flow call for both vector arrays if reading `MVToolsVector`-sized data, with potential to be smaller if SAD is packed as `uint32_t`.

The first implementation target should be `gpu_flow_reduce`, because it removes most readback bandwidth while preserving current CPU SAD semantics and keeping the old dense-flow path available as a fallback.
