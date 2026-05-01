# MVTools VapourSynth Vector Clip Format

This document describes how the attached MVTools VapourSynth port stores motion vectors, based on the source in `src/mvtools`.

It is a source-derived compatibility spec for generating external MV clips that existing MVTools consumers can read.

## Executive Summary

In this VapourSynth port, motion vectors are not stored in frame pixels. They are stored in two per-frame binary frame properties:

- `MVTools_MVAnalysisData`
- `MVTools_vectors`

The vector clip itself is just a normal video node whose image content is ignored by MVTools readers. Readers fetch a frame from the vector clip and only inspect those two frame properties.

The metadata blob is a raw `MVAnalysisData` C struct. The vector blob is a raw byte array containing:

1. A top-level size field
2. A top-level validity field
3. One serialized plane per hierarchy level, ordered from coarsest to finest
4. Each plane containing a size field followed by a row-major array of `VECTOR` records

## Source Basis

The key writer and reader paths are:

- `MVAnalyse.c`: writes the frame properties in `mvanalyseGetFrame()`
- `MVRecalculate.c`: writes the same format in `mvrecalculateGetFrame()`
- `GroupOfPlanes.c`: writes top-level group headers and plane ordering
- `PlaneOfBlocks.cpp`: writes per-plane headers, sizes, defaults, and vector arrays
- `Fakery.c`: deserializes the vector blob into `FakeGroupOfPlanes`
- `MVAnalysisData.c`: reads and validates the metadata property

## Property Names

Defined in `MVAnalysisData.h`:

- `MVTools_MVAnalysisData`
- `MVTools_vectors`

Both are required.

## Metadata Property: `MVTools_MVAnalysisData`

This property must contain exactly `sizeof(MVAnalysisData)` bytes.

The struct definition is:

```c
typedef struct VECTOR {
    int x;
    int y;
    int64_t sad;
} VECTOR;

typedef int MVArraySizeType;

typedef struct MVAnalysisData {
    int nMagicKey;
    int nVersion;
    int nBlkSizeX;
    int nBlkSizeY;
    int nPel;
    int nLvCount;
    int nDeltaFrame;
    int isBackward;
    int nCPUFlags;
    int nMotionFlags;
    int nWidth;
    int nHeight;
    int nOverlapX;
    int nOverlapY;
    int nBlkX;
    int nBlkY;
    int bitsPerSample;
    int yRatioUV;
    int xRatioUV;
    int nHPadding;
    int nVPadding;
} MVAnalysisData;
```

### Required Fields

In current readers, these fields matter:

- `nBlkSizeX`, `nBlkSizeY`
- `nPel`
- `nLvCount`
- `nDeltaFrame`
- `isBackward`
- `nMotionFlags`
- `nWidth`, `nHeight`
- `nOverlapX`, `nOverlapY`
- `nBlkX`, `nBlkY`
- `bitsPerSample`
- `xRatioUV`, `yRatioUV`
- `nHPadding`, `nVPadding`

### Legacy or Effectively Unused Fields

In the current source, `nMagicKey`, `nVersion`, and `nCPUFlags` are not validated by readers. `adataFromVectorClip()` only checks that the property exists and has exactly `sizeof(MVAnalysisData)` bytes.

For compatibility, set:

- `nVersion = 5`
- `nMagicKey = 0` or any fixed value
- `nCPUFlags = 0`

### Important Flags in `nMotionFlags`

Useful flags from `MVAnalysisData.h`:

- `MOTION_USE_SIMD = 0x00000001`
- `MOTION_IS_BACKWARD = 0x00000002`
- `MOTION_SMALLEST_PLANE = 0x00000004`
- `MOTION_USE_CHROMA_MOTION = 0x00000008`
- `MOTION_USE_SSD = 0x00000010`
- `MOTION_USE_SATD = 0x00000020`

For external generation, the practically relevant ones are:

- `MOTION_IS_BACKWARD` if `isBackward = 1`
- `MOTION_USE_CHROMA_MOTION` if your `sad` values include chroma contribution and you want scene-change threshold scaling to match native output

The readers do not require the other flags.

## Vector Property: `MVTools_vectors`

This property is a raw binary byte buffer.

It is serialized with native C integer layout by direct `memcpy()` and pointer casts. There is no portable byte-order or packing layer.

For Windows builds of this plugin, the practical layout is:

- `MVArraySizeType`: 32-bit little-endian signed integer
- `VECTOR.x`: 32-bit little-endian signed integer
- `VECTOR.y`: 32-bit little-endian signed integer
- `VECTOR.sad`: 64-bit little-endian signed integer

That makes `sizeof(VECTOR) == 16` bytes on normal Windows builds.

## Top-Level Vector Blob Layout

The full blob is:

```text
group_size      : int32
validity        : int32
plane[N-1]      : serialized coarsest plane
plane[N-2]      : next finer plane
...
plane[0]        : serialized finest plane
```

where `N = nLvCount` from `MVAnalysisData`.

### `group_size`

`group_size` is the total byte size of the entire `MVTools_vectors` blob, including:

- the `group_size` field itself
- the `validity` field
- every serialized plane

It is produced by `gopGetArraySize()`.

### `validity`

- `1` means the vector frame is valid
- `0` means invalid, usually used near clip edges when a reference frame does not exist

Consumers typically treat a frame as usable only if:

- `validity == 1`
- the finest-plane scene-change test does not trigger

## Per-Plane Layout

Each plane is serialized as:

```text
plane_size      : int32
vectors[]       : VECTOR[block_count]
```

where:

- `plane_size = sizeof(int32) + block_count * sizeof(VECTOR)`
- there is no additional per-plane metadata

`plane_size` includes its own 4-byte header.

## Block Ordering Inside a Plane

Vectors are stored in row-major block order:

```text
index = by * nBlkX + bx
```

with:

- `bx`: horizontal block index
- `by`: vertical block index

Native analysis may search blocks in meander order internally, but the serialized output is still written into row-major index order.

## Block Coordinates Are Not Stored

The blob stores only vectors and SAD values.

Block origins are reconstructed from metadata as:

```text
stepX = nBlkSizeX - nOverlapX
stepY = nBlkSizeY - nOverlapY

block_x = bx * stepX
block_y = by * stepY
```

The fake reader computes these coordinates when building `FakePlaneOfBlocks`.

## Finest and Coarser Plane Geometry

For the normal non-divided hierarchy, the metadata describes the finest plane.

Let:

```text
stepX    = nBlkSizeX - nOverlapX
stepY    = nBlkSizeY - nOverlapY
width_B  = stepX * nBlkX + nOverlapX
height_B = stepY * nBlkY + nOverlapY
```

Then the fake reader derives block counts for level `i` as:

```text
level 0: nBlkX_level = nBlkX
         nBlkY_level = nBlkY

level i > 0:
    nBlkX_level = ((width_B  >> i) - nOverlapX) / stepX
    nBlkY_level = ((height_B >> i) - nOverlapY) / stepY
```

In the fake reader:

- level `0` is always the finest plane
- higher indices are coarser planes
- only level `0` uses `nPel` from metadata
- coarser levels are read with effective `pel = 1`

## Optional Divide Mode

`Analyse` and `Recalculate` can optionally emit an extra divided finest level.

### What Native Writer Does

When `divide` is enabled:

1. Normal hierarchy planes are written first, coarsest to finest
2. An extra plane is appended after the normal finest plane
3. The metadata blob is modified before being attached to the frame:

```text
nBlkX     *= 2
nBlkY     *= 2
nBlkSizeX /= 2
nBlkSizeY /= 2
nOverlapX /= 2
nOverlapY /= 2
nLvCount  += 1
```

This causes the fake reader to interpret the appended plane as the new level 0 finest plane, and the original finest plane becomes level 1.

### Layout of the Extra Divided Plane

The appended extra plane is a normal plane blob:

```text
plane_size = sizeof(int32) + (original_nBlkX * original_nBlkY * 4) * sizeof(VECTOR)
```

Its grid dimensions are:

- `div_nBlkX = original_nBlkX * 2`
- `div_nBlkY = original_nBlkY * 2`

### Extra Plane Ordering

Within each original block, the native writer emits four subblocks arranged as:

```text
top-left, top-right,
bottom-left, bottom-right
```

and the resulting divided plane is still row-major over the doubled grid.

### Vector and SAD Values in Native Divide Mode

For `divide=1`:

- all four subblocks inherit the parent vector
- `sad` is set to `parent.sad >> 2`

For `divide=2`:

- `sad` is still `parent.sad >> 2`
- the subblock vectors may be replaced by medians of the parent block and its neighbors

If you generate divided external vectors directly, you do not need to follow the parent-copy rule. You only need the metadata and serialized plane payload to agree.

## Invalid Frames and Default Payload

When native code has no valid reference frame, it writes:

- `validity = 0`
- every vector in every plane set to:

```text
x   = 0
y   = 0
sad = verybigSAD
```

where:

```text
verybigSAD = nBlkSizeX * nBlkSizeY * (1 << bitsPerSample)
```

For native divide mode, the extra appended plane also gets filled with the same default vector repeated.

Consumers usually short-circuit on `validity == 0`, so the exact default `sad` value mostly matters for native parity, not acceptance.

## Scene Change Semantics

There is no explicit scene-change bit in the vector blob.

Scene changes are inferred from the finest plane only.

The fake reader counts how many finest-plane blocks satisfy:

```text
vector.sad > thscd1_scaled
```

and compares that count to the scaled `thscd2` threshold.

So if you want native-like scene-cut behavior from external vectors, you must encode suitable `sad` values in the finest plane.

## Forward and Backward Semantics

The metadata fields `nDeltaFrame` and `isBackward` define how a frame index maps to its reference frame.

Native consumers interpret them as:

- backward vectors: `nref = n + nDeltaFrame`
- forward vectors: `nref = n - nDeltaFrame`

If `nDeltaFrame <= 0`, some filters treat that as absolute reference mode:

- `nref = -nDeltaFrame`

This mode is not accepted everywhere. For broad compatibility, use:

- `nDeltaFrame > 0`
- `isBackward = 0` for forward clips
- `isBackward = 1` for backward clips

Also mirror the direction in `nMotionFlags` by setting or clearing `MOTION_IS_BACKWARD`.

## What Consumers Actually Use

In the current codebase, runtime consumers primarily use only the finest plane, which is plane 0 after deserialization.

However, `fgopUpdate()` always walks exactly `nLvCount` serialized planes. That means:

- you may set `nLvCount = 1` and provide only a finest plane
- but if you advertise `nLvCount > 1`, the blob must contain all planes implied by that count

This is the simplest external producer strategy.

## Minimum Viable External Producer

For current MVTools consumers in this source tree, the simplest robust external format is:

1. Create a vector clip with at least as many frames as the consumer will request, usually the same frame count as the source clip.
2. The actual pixel content and pixel format of that vector clip do not matter to MVTools readers.
3. Attach `MVTools_MVAnalysisData` to at least the first frame. In practice, attach it to every frame.
4. Attach `MVTools_vectors` to every frame.
5. Use `nLvCount = 1` unless you specifically want to mimic native multi-level output.
6. Encode one plane blob for the finest level.
7. Store vectors row-major.
8. Set `nBlkX`, `nBlkY`, block sizes, overlap, pel, padding, subsampling, and delta consistently with the source clip the vectors are meant for.
9. Set `validity = 0` on frames where no valid vectors exist.

## Native-Compatible Single-Level Example

For a non-divided finest-only external clip, the blob layout is:

```text
int32 group_size
int32 validity
int32 plane0_size
VECTOR plane0_vectors[nBlkX * nBlkY]
```

with:

```text
group_size  = 8 + plane0_size
plane0_size = 4 + (nBlkX * nBlkY * sizeof(VECTOR))
```

## Recommended Metadata Values for External Clips

If you are not trying to emulate native `Analyse` byte-for-byte, this is a good baseline:

- `nMagicKey = 0`
- `nVersion = 5`
- `nCPUFlags = 0`
- `nMotionFlags` containing:
  - `MOTION_IS_BACKWARD` if needed
  - `MOTION_USE_CHROMA_MOTION` if your `sad` uses chroma too
- `nLvCount = 1`
- `nBlkX`, `nBlkY` computed from your chosen block geometry
- `nHPadding`, `nVPadding` matching the super/finest clip geometry expected by the consumer

## Compatibility Notes

- The format is ABI-native, not an explicitly versioned portable container.
- `group_size` is written by native code but not strongly validated by readers. You should still set it correctly.
- Per-plane `plane_size` is used by readers to step through the buffer and must be correct.
- `adataCheckSimilarity()` compares width, height, block size, pel, overlap, subsampling, and bit depth across MV clips, but it does not compare `nLvCount`.
- Current consumers do not inspect vector-clip pixels.

## Practical Recommendation

If your goal is to feed external motion vectors into existing MVTools consumers, target the following first:

- a one-level vector clip
- correct finest-plane geometry metadata
- correct `nDeltaFrame` and `isBackward`
- a row-major finest-plane `VECTOR` array
- sensible `sad` values for thresholding and scene-change behavior

That is the smallest contract that matches the current source.