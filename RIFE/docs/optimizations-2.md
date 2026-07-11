# RIFEMV optimization work performed in this development thread

## Scope and purpose

This document records the RIFEMV optimization work performed during the July 2026 development thread that followed the removal of `RIFEMVApprox2` and `RIFEMVApprox3`. It is deliberately limited to work from that thread. Optimizations already present in older commits, even when they are visible in the current source or affect the measurements below, are not claimed as new work here.

The intended audience includes future maintainers and LLM coding agents that can inspect the repository but do not have the conversation that led to the current design. In particular, the sections on reverted experiments and work that should not be repeated are part of the engineering record, not merely historical commentary.

Unless stated otherwise, retained changes described here apply principally to `RIFEMV` with `gpu_mode=2` (`gpu_full`) and `gpu_mode=3` (`gpu_full_packed`). The plugin identifier and plugin version remained unchanged.

## Executive summary

The original supplied benchmark was expected to run at approximately **32.5 FPS** for temporal radius 2. After the retained exporter, cache, graph, packing, and profiling work, `gpu_mode=3` reached **46.20 FPS**, a gain of **42.15%** over that starting reference. Updating the pinned NCNN tree and adapting the local integration raised the same `gpu_mode=3`, temporal-radius-2 workload to **49.77 FPS**, which is:

- **7.73% faster** than the 46.20 FPS pre-update result.
- **53.14% faster** than the original 32.5 FPS reference.

Temporal radius 3 reached **32.75 FPS** with the updated NCNN tree. Before that update, the best measured `gpu_mode=3` result was **30.70 FPS**, so the NCNN update improved that workload by **6.68%**. Tests also showed that temporal-radius-3 scaling was already close to proportional to the amount of RIFE inference requested; there was no hidden radius-3-specific collapse to fix.

The largest lesson is that this workload remains inference-bound. Reducing readback, record conversion, frame-property copying, and redundant uploads helps, but changes that serialize otherwise independent RIFE instances lose more throughput than their reuse saves.

## Benchmark workload and measurement rules

### Canonical command

The principal benchmark was the user-supplied VapourSynth workload below. The build script produces `rifemv.dll`; that is the DLL symlinked into the portable VapourSynth installation.

```bat
D:\Software\.Media\vapoursynth-portable\VSPipe.exe trim_crop.v.py -a i=D:\Temp\MKV\grain\Aliens.1986.Special.Edition.cut2.mkv -a rife_mv=True -a crop=1920x960 -a rmv_tr=2 -a rmv_thsad=500 -a rmv_chroma=True -a fmt_in=420_10 -a fmt=_ -a rmv_pf=scale@0.5 -a rmv_bench=True -a trim=1000+1000 -c y4m --
```

The benchmark was run with VSPipe progress reporting enabled. `rmv_tr` controls the number of separate `RIFEMV` instances used by the script:

- `rmv_tr=1`: one temporal distance and one RIFE inference stream.
- `rmv_tr=2`: two temporal distances and two RIFE inference streams.
- `rmv_tr=3`: three temporal distances and three RIFE inference streams.

`gpu_mode=3` measurements used the script's `rmv_gm=3` argument. Detailed plugin profiling used `rmv_stats=True`, which enables `perf_stats` in the generated `RIFEMV` calls. The exact script arguments and trim length matter: short diagnostic runs and measurements from another GPU mode are not interchangeable with the full 1000-frame result.

### Percentage convention

All percentage changes use:

```text
percentage change = (new FPS / reference FPS - 1) * 100
```

Small changes around one percent should be treated cautiously because process startup, shader warm-up, scheduling, decoding, and ordinary run-to-run variation can be of the same order. Stage timings and repeatability were considered alongside the headline FPS.

## Benchmark result overview

| Configuration | FPS | Comparison | Change |
| --- | ---: | --- | ---: |
| Original supplied temporal-radius-2 reference | 32.50 | Starting reference | — |
| Optimized pre-NCNN-update `gpu_mode=2`, radius 2 | about 45.50 | Versus 32.50 | **+40.00%** |
| Optimized pre-NCNN-update `gpu_mode=3`, radius 2 | 46.20 | Versus 32.50 | **+42.15%** |
| Optimized pre-NCNN-update `gpu_mode=3`, radius 2 | 46.20 | Versus mode 2 at 45.50 | **+1.54%** |
| Updated-NCNN `gpu_mode=3`, radius 2 | 49.77 | Versus 46.20 | **+7.73%** |
| Updated-NCNN `gpu_mode=3`, radius 2 | 49.77 | Versus 32.50 | **+53.14%** |
| Pre-update `gpu_mode=2`, radius 3 | 30.38 | Radius-3 mode comparison | — |
| Pre-update `gpu_mode=3`, radius 3 | 30.70 | Versus mode 2 at 30.38 | **+1.05%** |
| Updated-NCNN `gpu_mode=3`, radius 3 | 32.75 | Versus 30.70 | **+6.68%** |

The 32.5 FPS starting value was an expected baseline supplied by the user, whereas the later values were measured during this thread. It should not be interpreted as an isolated before/after result for any one individual patch.

## Retained implementation changes

### Removal of approximate multi-delta APIs

`RIFEMVApprox2` and `RIFEMVApprox3` were removed completely. Their registrations, creators, callbacks, data structures, cleanup paths, scratch buffers, internal properties, performance counters, and displacement-composition helpers were removed rather than hidden behind compatibility aliases.

The approximate APIs were not materially faster than running independent exact `RIFEMV(delta=N)` instances, and their displacement composition introduced a separate implementation path with additional maintenance and semantic cost. Multiple temporal distances now use explicit calls:

```python
bw1, fw1 = core.rmv.RIFEMV(clip, model_path=rife_mdl, delta=1)
bw2, fw2 = core.rmv.RIFEMV(clip, model_path=rife_mdl, delta=2)
bw3, fw3 = core.rmv.RIFEMV(clip, model_path=rife_mdl, delta=3)
```

No precise isolated FPS result was retained for the removed approximate path, so this document does not invent one. Its removal is a simplification based on the observed absence of a meaningful speed advantage.

### Corrected GPU-full compute dispatch sizing

The GPU-full vector/SAD shader dispatch had been sized through a representation that caused more invocations than the block count required. Bounds checks preserved correctness, but the redundant invocations still consumed scheduling and GPU work.

The retained path dispatches from a host-side shape representing the actual block count. A missing host-`Mat` dispatcher overload was added to the local NCNN command API to support this correctly. This reduced unnecessary shader execution without changing exported vectors, SAD values, or MVTools blob layout.

This correction was bundled with other early optimization work, so no trustworthy standalone FPS percentage is assigned to it.

### Direct GPU-mode-2 record copy into MVTools blobs

For `gpu_mode=2`, the GPU readback record and the MVTools record are both 16 bytes and have compatible field placement:

```text
GPU record:     int32 x, int32 y, uint32 rawSad, uint32 reserved
MVTools record: int32 x, int32 y, int64 sad
```

The shader writes the reserved high 32 bits as zero. Compile-time size and offset assertions verify that the layouts remain compatible. On the supported little-endian Windows target, the records can therefore be copied directly into the MVTools vector blob instead of being converted field by field.

A representative full run improved from **45.12 FPS** to **45.52 FPS**, a measured gain of **0.89%**. Because the difference is small, part of it may be benchmark variance; the retained implementation is still preferable because it demonstrably removes CPU work and is protected by layout assertions.

### Direct blob construction when optional statistics are disabled

The exporter previously materialized intermediate CPU arrays for backward and forward vectors before packing the final MVTools properties. When `sad_stats=False` and `motion_stats=False`, those arrays are not otherwise observable or needed.

The retained fast path writes the final blobs directly from the compact GPU output. The more general intermediate representation remains available when optional statistics require iteration over vector data. This change lowers allocation, memory traffic, and CPU conversion overhead while preserving the statistics-enabled behavior.

### SSE2 expansion for `gpu_mode=3`

`gpu_mode=3` reads each vector back as an 8-byte packed record: signed 16-bit X and Y plus a 32-bit raw SAD. MVTools still requires a 16-byte record containing signed 32-bit X and Y and a 64-bit SAD.

The retained SSE2 fast path expands two packed records per iteration. It performs signed extension of X/Y and zero extension of SAD into the required layout, with a scalar fallback for unsupported targets and a scalar tail for an odd vector count.

The detailed counters showed vector-packing totals falling from roughly **343–371 ms per filter** to approximately **257–291 ms per filter**, a reduction of about **20–25%** in that CPU substage. Overall throughput reached **46.20 FPS**, compared with a nearby pre-fast-path run at **45.39 FPS**, an observed improvement of **1.78%**. That overall percentage should not be attributed entirely to SSE2 because the runs also contain normal variation; the substage reduction is the stronger evidence.

### Reuse inference frames as the SAD source when possible

When `sad_clip` is the same logical clip and geometry as the inference source, the current and reference inference frames already available to the pair callback can also serve as the SAD source. The plugin now reuses those frames instead of requesting the same source frames through a duplicate dependency.

This removes redundant VapourSynth frame requests, dependency tracking, and CPU-side packing/build work. A separate path remains for a genuinely distinct `sad_clip`, which is required to preserve the documented ability to estimate flow from one signal and score SAD against another.

### Bounded GPU-resident input cache

Modes 2 and 3 now retain a small per-RIFE-instance cache of uploaded input frames. Adjacent pair requests repeatedly use the same source frames, so caching avoids uploading and preprocessing both sides from scratch for every pair.

The final design has these important properties:

- Cache identity includes the source frame and compatible input geometry/state.
- Entries become reusable only after the command using or producing them has completed.
- The cache is bounded to eight input frames and uses LRU-style admission/eviction.
- Runtime buffers use a reusable 64 MiB `VkBlobAllocator`, not an append-only weight allocator.
- A per-instance process mutex protects NCNN command state, buffer barriers, cache mutation, and buffer lifetime.

In representative radius-2 profiling, each filter reported approximately **2992–2996 GPU input-cache hits** and **1000 misses**. The misses largely represent genuinely new frames entering the stream; increasing capacity cannot make those compulsory uploads disappear.

The first cache prototype used `VkWeightAllocator`, whose allocation behavior is suitable for persistent model weights but not for a rotating runtime frame cache. A full stress run exhausted Vulkan memory. Replacing it with the reusable `VkBlobAllocator` and a bounded cache fixed the regression.

### Direct backward output from the pair node

The internal pair node already owns the backward result. The output graph was changed so that the pair node itself is the public backward clip, while the forward result continues through the extraction node using the private forward-vector property.

This removed one filter layer and one property-copy/materialization pass for every backward output frame. The earlier intermediate step of reducing the hidden pair carrier to 1x1 was superseded by this direct-output arrangement; the public pair/backward frame must be full-sized because it is now observable.

A nearby full run measured **45.20 FPS** after this graph change versus roughly **45.5 FPS** in adjacent runs. That is effectively neutral at the whole-pipeline level and should not be reported as an FPS win. The change was retained because it removes unneeded CPU and property work, which is visible in profiling and can matter more in less inference-bound workloads.

### Current NCNN upstream update

The pinned NCNN tree was updated from commit `305837fd4a722ebc47c5d72e72d8ec9ae970e932` (2025-05-03) to upstream commit `13b6d5318c73be53bb386fa51e8067615d0eb7c1` (2026-07-08).

The complete old local tree was retained as:

```text
subprojects/ncnn_20250503_305837f_local.7z
```

Its recorded SHA-256 is:

```text
35895646DF0348A9A8B2E6BA42F95DF15BA20B2F403074B7546736F62FB7D77E
```

The archive passed an integrity test. It contains 7,006 files, with 89,838,262 uncompressed bytes and 10,178,588 compressed bytes.

The local Vulkan timestamp/dispatcher patch was rebased rather than discarded. Integration also required:

- Disabling the newly available but unused `rmsnorm` layer in the minimal NCNN build.
- Removing the custom Warp pack8 path because current NCNN removed Vulkan pack8 support; pack1 and pack4 remain.
- Rewriting the custom motion-vector shader record declaration as an explicit structure accepted by the newer shader toolchain.
- Rebuilding the embedded SPIR-V header.

After these adaptations, full property-parity checks passed for `gpu_mode=2` and `gpu_mode=3`. The radius-2 mode-3 benchmark improved from **46.20 FPS** to **49.77 FPS** (**+7.73%**), and radius 3 improved from **30.70 FPS** to **32.75 FPS** (**+6.68%**).

Detailed GPU timing supports the headline result. Representative per-filter inference time fell from about **8.421–8.433 ms** to **7.838–7.849 ms**, approximately **6.9% lower**. GPU total time fell from about **10.815–10.847 ms** to **10.100–10.157 ms**, approximately **6.5% lower**. The GPU vector stage fell from about **0.360–0.376 ms** to **0.230–0.269 ms**, a reduction of roughly **28–36%**, while upload remained near **1.666–1.669 ms**.

## Profiling and benchmarking infrastructure added in this thread

### Opt-in Vulkan timestamps

`perf_stats=True` now records GPU timestamps around the major Vulkan stages without enabling NCNN's global benchmark mode. The local NCNN command interface exposes query-pool creation, timestamp recording, and query retrieval independently of `NCNN_BENCHMARK`.

Seven timestamp boundaries divide the command into:

- Input upload.
- Preprocessing.
- RIFE inference.
- Optional flow resize.
- Flow reduction or full vector/SAD generation.
- Output readback.
- Total GPU interval from upload start through readback completion.

The command uses an all-commands pipeline stage for these measurements. Queries are created and recorded only when plugin performance collection is enabled, so normal operation does not pay the query/readback cost.

The resulting summary fields include:

- `gpu_upload_ms`
- `gpu_preproc_ms`
- `gpu_inference_ms`
- `gpu_flow_resize_ms`
- `gpu_flow_reduce_ms`
- `gpu_flow_vector_ms`
- `gpu_readback_ms`
- `gpu_total_ms`

Both totals and per-call averages are reported where applicable.

### Expanded CPU-side timing and counters

The performance report was expanded to identify time outside the GPU inference interval. The fields newly introduced for this work were the Vulkan stage totals/averages, GPU input-cache hit/miss/wait measurements, and CPU materialization timings for pair carrier creation, pair properties, packed SAD-source construction, public output-frame allocation, and public output properties.

Those additions were interpreted together with the report's pre-existing timing and cache counters. The complete set used during this investigation covers:

- Local and shared semaphore wait time.
- Total flow-processing wall time and unaccounted RIFE process time.
- CPU preparation and command-recording subdivisions for upload, preprocessing, inference, output, reduction/vector generation, and readback.
- Submit/wait time, unpack/export time, direct export, resize, and cleanup.
- Readback byte counts, mapping, and invalidation.
- GPU input-cache hits, misses, wait/admission time, and the cache contribution to total processing time.
- CPU packed-frame cache hits, misses, build time, and contention wait.
- Luma/SAD source build time.
- Vector packing.
- Pair-carrier and pair-property work.
- Packed SAD-source construction.
- Public output-frame allocation and property materialization.

These measurements were essential for distinguishing actual bottlenecks from attractive but low-impact micro-optimizations. For example, mode 3 halves the vector readback representation, but measured readback was only around **0.14 ms**; therefore readback size alone cannot produce a large whole-pipeline speedup.

### Build and benchmark reliability notes

The supplied `build_rmv.bat` remains the reference build path and renames the built plugin to `rifemv.dll`, which is the filename linked into VapourSynth.

During a clean configuration after the NCNN update, `ccache` stalled while compiling CMake probe programs in the managed environment. Building with `CCACHE_DISABLE=1` allowed configuration and compilation to proceed. This was a build-environment/cache issue, not a plugin-source performance regression.

The initial updated-NCNN shader integration rejected vector-component syntax used by the old generated shader. After that shader failure, one subsequent long attempt encountered a transient `VK_ERROR_DEVICE_LOST` around frame 345. Once the shader declaration was corrected and the embedded SPIR-V regenerated, clean 400-frame and 1000-frame runs completed successfully. Do not cite that one contaminated run as evidence of a stable NCNN device-loss regression.

## Temporal-radius scaling analysis

The pre-update full benchmark produced:

| Temporal radius | Separate RIFEMV instances | FPS | Aggregate instance rate (`FPS * radius`) |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 79.50 | 79.50 |
| 2 | 2 | 45.34 | 90.68 |
| 3 | 3 | 30.38 | 91.14 |

If radius 3 scaled perfectly from the radius-2 result according only to the number of RIFE instances, its expected throughput would be:

```text
45.34 * 2 / 3 = 30.23 FPS
```

The measured **30.38 FPS** is **0.50% above** that proportional prediction. In other words, radius 3 did not show a disproportionate penalty. The aggregate processed-instance rate was essentially flat between radius 2 and radius 3.

Switching the pre-update radius-3 workload from mode 2 to mode 3 raised throughput from **30.38 FPS** to **30.70 FPS**, only **1.05%**. Updating NCNN then raised the mode-3 result to **32.75 FPS**, a more meaningful **6.68%** gain.

Short radius-3 tests with `gpu_thread` values 1, 2, and 3 measured approximately **30.53**, **30.7**, and **30.74 FPS** respectively. These differences are not significant enough to justify changing the default or adding radius-specific thread tuning.

The practical conclusion is that radius 3 is limited mainly by running three model inferences. A radius-3-specific exporter optimization is unlikely to transform performance unless it reduces inference cost or permits more inference overlap without serialization.

## Failed experiments and regressions

### Sharing one RIFE object across temporal distances

An experiment allowed compatible delta filters to share the complete RIFE model object and its GPU input cache. Cache reuse improved and short profiling showed GPU totals below 10 ms, but the RIFE object's processing mutex serialized inference that had previously run concurrently across filter instances.

Full throughput fell from approximately **45.50 FPS** to **43.32 FPS**, a **4.79% regression**. The experiment was reverted.

The important result is not that sharing is always impossible; it is that sharing a mutable NCNN/RIFE execution object with one process lock is the wrong throughput design for this workload.

### Cross-instance GPU input cache

A separate attempt to share uploaded GPU input frames across RIFE instances also introduced serialization through shared mutable command/cache state. Avoiding some duplicate uploads did not compensate for reducing inference overlap. The experiment was reverted and the cache remains per RIFE instance.

A viable future design would need immutable cached resources, explicit upload-completion synchronization, reference-counted lifetime/eviction, and independent command queues and RIFE execution state. Merely moving the current per-instance cache into a global object will repeat the regression.

### Append-only allocator for cached runtime frames

The first GPU input-cache implementation allocated rotating runtime frames from `VkWeightAllocator`. A full-length stress test exhausted Vulkan memory because old allocations were not reused as cache entries rotated.

The fix was to use a bounded eight-entry cache backed by a reusable `VkBlobAllocator`. The failed allocator design must not be restored, even if it appears simpler in a short run.

### Enabling Vulkan FP16 arithmetic

Enabling NCNN Vulkan FP16 arithmetic was benchmarked and produced no measurable whole-pipeline improvement, so it was reverted. Existing storage, packing, and device capabilities already captured the useful paths for this model/workload, while changing arithmetic precision creates additional numerical-compatibility risk.

This flag should not be toggled again as a speculative optimization without a new NCNN implementation, new GPU, or profiler evidence that changes the underlying tradeoff.

### Increasing `gpu_thread` for temporal radius 3

Testing values 1, 2, and 3 produced approximately **30.53**, **30.7**, and **30.74 FPS**. The spread is below a meaningful optimization threshold. Radius 3 already reaches essentially the same aggregate RIFE-instance rate as radius 2, so thread-count tuning does not address the dominant inference cost.

### Initial updated-NCNN shader syntax

The first integration retained an old generated-shader vector declaration that the newer shader toolchain rejected. The explicit `MotionVector` structure is the compatible retained representation. Reintroducing the older vector-component syntax will break shader generation rather than improve performance.

## What future optimization agents should not try again

This section is intentionally direct. These ideas have already been tested or invalidated by the architecture and should not be proposed as easy wins without materially new evidence.

### Do not recreate `RIFEMVApprox2` or `RIFEMVApprox3`

The approximation APIs and displacement-composition machinery were removed because they did not offer a meaningful speed advantage over independent exact filters. Do not restore registrations, compatibility aliases, hidden composition helpers, or private frame-property plumbing. Use separate `RIFEMV(delta=N)` instances.

### Do not share the whole mutable RIFE/NCNN object between deltas

The measured result was **43.32 FPS instead of about 45.50 FPS**, a **4.79% loss**. Its process mutex serialized inference. Model-memory sharing and execution-state sharing are different problems; combining them in one locked object sacrifices throughput.

### Do not make the current GPU frame cache global

The naive cross-instance cache serialized command execution and regressed performance. Sharing `VkMat` buffers across queues also requires explicit synchronization and lifetime management; NCNN tracks mutable barrier/layout state. A global map plus a mutex is not a safe or fast solution.

### Do not remove the per-instance cache/process synchronization casually

The mutex protects command state, buffer transitions, cache admission, and eviction. Removing it without resource pinning and Vulkan synchronization can race an upload against reuse or eviction. Any lock-reduction proposal must first replace these guarantees explicitly.

### Do not use `VkWeightAllocator` for the rotating input cache

It exhausted Vulkan memory during the full benchmark. Runtime cache entries require reusable blob allocation and a hard capacity bound.

### Do not increase the eight-frame GPU cache merely to chase misses

The observed pattern already contained roughly 2992–2996 hits against 1000 misses per filter. Most misses are the next unique source frame entering the stream, not capacity evictions that a larger LRU can solve. A larger cache increases VRAM use without removing compulsory uploads.

### Do not expect packed readback by itself to produce a large FPS gain

Mode 3 halves vector readback bytes, but readback was measured at only around **0.14 ms**. Its final advantage over optimized mode 2 was approximately **1.54%** at radius 2 and **1.05%** at radius 3 before the NCNN update. The packed format remains useful, but it is not a hidden multi-fold speedup.

### Do not repeat field-by-field mode-2 packing

The current 16-byte GPU and MVTools layouts are statically verified and copied directly. Replacing the bulk copy with a conversion loop only restores CPU overhead. If either layout changes, update the assertions and deliberately reassess compatibility rather than silently falling back everywhere.

### Do not enable FP16 arithmetic on speculation

It was tested and showed no measurable gain. Revisit only with a new backend/device profile and vector/property parity validation.

### Do not keep retuning `gpu_thread` for radius 3

Values 1 through 3 were effectively tied. The aggregate instance-rate analysis shows that radius 3 is already scaling proportionally from radius 2.

### Do not restore Vulkan pack8 in the current NCNN integration

Upstream NCNN removed the Vulkan pack8 path. The adapted custom Warp implementation supports the current pack1/pack4 model. Restoring stale pack8 code is an API/compatibility regression, not a supported optimization.

### Do not assume that enabling an INT8 build flag quantizes RIFE

The model is not an INT8-calibrated model, and changing an NCNN feature flag does not quantize weights, choose suitable scales, prove that all Vulkan operators use a faster INT8 path, or preserve flow accuracy. Quantization would be a separate model-conversion, calibration, operator-coverage, GPU-support, and quality-validation project. It is not a low-risk compile-time switch.

### Do not treat the one post-shader-failure device loss as a stable regression

The shader incompatibility was fixed, and clean 400-frame and 1000-frame runs then passed. Debug future device loss from a clean process and current shader before drawing conclusions from the contaminated integration run.

### Do not modify plugin code to work around the clean-build `ccache` stall

The workaround was `CCACHE_DISABLE=1` for fresh configuration in the managed environment. The issue occurred in CMake compiler probes and was not caused by the RIFEMV implementation.

## Remaining credible optimization directions

The failed ideas above do not mean the exporter is exhausted, but future work should begin from profiling evidence and preserve concurrency.

### Independent execution with read-only shared model resources

The failed sharing experiment conflated model storage, cache state, and command execution. A more ambitious NCNN-level design might share immutable model weights/pipelines while keeping separate allocators, command buffers, caches, and synchronization per RIFE execution stream. This could reduce VRAM without serializing deltas, but it is a substantial backend change and was not implemented in this thread.

### Eliminate duplicate uploads without sharing mutable execution state

Radius 2 and 3 still upload the same logical source frames independently for different delta instances. A correct solution would need a producer upload timeline/semaphore, immutable image/buffer ownership suitable for all consumers, reference-counted eviction, and independent inference command streams. The potential saving is bounded by the measured upload stage, around **1.67 ms per filter call**, and must be weighed against synchronization overhead.

### Model/backend-level inference work

Inference remains the largest GPU stage. The NCNN update's approximately **6.9%** inference-time reduction produced a correspondingly meaningful overall gain, demonstrating that backend kernels and graph execution have more leverage than another small exporter copy optimization.

Potential areas include newer NCNN kernel improvements, operator fusion supported by upstream, device-specific cooperative-matrix paths, or a carefully validated reduced-precision/quantized model pipeline. Without retraining, quantization is possible only if an appropriate post-training calibration/conversion path and performant Vulkan operator coverage exist; accuracy of motion vectors and downstream SAD behavior would need explicit validation.

### Flow-resize or vector-stage fusion

Fusing flow resize with reduction/vector generation could avoid an intermediate stage in configurations that actually perform flow resize. This was not attempted here. It must preserve coordinate scaling, clamping, block-reduction semantics, and floating-point rounding closely enough to maintain exported property parity.

### Reduce remaining VapourSynth property/materialization overhead

The new counters make this measurable. Any further graph or property work should target a demonstrated total, preserve `MVTools_MVAnalysisData` and `MVTools_vectors` byte compatibility, and be tested with optional statistics enabled and disabled. These CPU changes are most likely to matter on faster GPUs or lower-resolution inference, where inference occupies a smaller share of wall time.

## Validation performed

The retained implementation was validated through combinations of:

- Successful builds using `build_rmv.bat`, with `CCACHE_DISABLE=1` when a clean NCNN configure required it.
- Full 1000-frame VSPipe benchmark runs at temporal radii 2 and 3.
- Shorter diagnostic runs for thread-count and integration checks.
- `gpu_mode=2` versus `gpu_mode=3` vector-property parity checks.
- Detailed CPU timing and Vulkan timestamp inspection.
- Stress testing of the GPU-resident cache, which exposed and then verified the allocator fix.
- Clean updated-NCNN 400-frame and 1000-frame runs after shader adaptation.
- Integrity testing and hashing of the archived previous NCNN tree.

## Final state

The current direction is intentionally conservative:

- Use exact, separate `RIFEMV(delta=N)` instances for multiple temporal distances.
- Preserve independent RIFE execution state so multiple deltas can overlap.
- Reuse uploaded inputs within each instance through a bounded, reusable cache.
- Use `gpu_mode=3` when its signed-16-bit vector constraint is satisfied for the smallest readback and fastest measured exporter path.
- Keep detailed profiling opt-in and use it before proposing another micro-optimization.
- Treat inference/backend improvements as the highest-potential remaining area.

The strongest current measured result for the supplied workload is **49.77 FPS at temporal radius 2** and **32.75 FPS at temporal radius 3** with the updated NCNN tree and `gpu_mode=3`.
