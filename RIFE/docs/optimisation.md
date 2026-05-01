# Optimisation Attempts

## Successes

- None recorded yet.

## Failures

### Native-flow MV export plus SAD luma caching

Attempted to reduce CPU cost in the motion-vector export path by changing two parts of the implementation:

1. SAD generation was changed so luma was precomputed once per frame pair instead of recomputing RGB-to-luma inside every block comparison.
2. Standard MV export paths were changed to avoid building a full-resolution dense flow field before block-vector generation. A new native-flow path returned the half-resolution flow tensor from RIFE, and the vector builder sampled that native flow directly with bilinear interpolation while keeping the approximation/displacement path on the old dense-flow route.

Implementation details:

- Added a native flow export interface in [rife.h](./rife.h) and [rife.cpp](./rife.cpp).
- Refactored the standard MV builders in [plugin.cpp](./plugin.cpp) to consume native flow metadata instead of assuming a full-resolution flow buffer.
- Added cached luma preparation for block SAD computation in [plugin.cpp](./plugin.cpp) so the inner SAD loop could read precomputed luma instead of calling RGB-to-luma repeatedly.
- Left the approximation/displacement path on the previous full-resolution flow path to avoid changing displacement composition behavior in the same pass.

Outcome:

- Performance regressed instead of improving: degrain testing with 2 RIFE MV clips dropped from about 30 FPS to about 26 FPS.
- Visual quality regressed: wobbling and warping artifacts were observed, indicating unstable vectors and/or incorrect SAD behavior.
- CPU and GPU load were roughly unchanged, so the attempted optimizations did not shift the bottleneck in a useful way.

Conclusion:

- This optimization attempt should be treated as a failed experiment.
- This implementation approach is not safe to keep as-is because it both reduced performance and introduced visible motion-vector instability.