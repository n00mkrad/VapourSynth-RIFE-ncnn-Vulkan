# Synthetic SAD computation

`SAD` means sum of absolute differences. This plugin does not get a matching cost from RIFE itself, so it synthesizes the MVTools `sad` field after a motion vector has already been chosen from RIFE flow or from a composed displacement field.

## Scope

The same `sad` formula is used in all vector-export paths:

- `RIFEMV()`: two-output API that returns both backward and forward vector clips
- `buildMVToolsVectorBlob()`: helper that converts one frame pair plus motion data into the binary MVTools vector blob, including `x`, `y`, and `sad` for each block
- `buildMotionVectorBlobFromConfig()`: thin wrapper that builds the same blob using a `MotionVectorConfig` settings object
- `RIFEMVApprox2()` / `RIFEMVApprox3()`: approximate exporters for larger temporal distances, built by composing 2 or 3 adjacent motions
- `buildMotionVectorBlobFromDisplacement()`: helper that builds the blob from already-composed pixel displacements instead of directly from raw flow

Only the way `pixelDx` and `pixelDy` are obtained differs.

## Name glossary

- `round(x)`: C++ `std::lround`, meaning nearest integer with halfway cases rounded away from zero
- `clamp(x, lo, hi)`: limit `x` to the closed interval `[lo, hi]`
- `clampPixel(...)`: frame-edge clamp used for pixel coordinates, so any out-of-range access reads the nearest border pixel
- `bilinearSample(...)`: bilinear interpolation on one float plane after clamping the sample position to the frame bounds
- `luma(r, g, b)`: Rec.709 luma computed as `0.2126 * r + 0.7152 * g + 0.0722 * b`
- `flowX`, `flowY`: horizontal and vertical motion read from the RIFE flow tensor
- `dispX`, `dispY`: horizontal and vertical motion expressed in pixel units rather than raw flow units
- `vx`, `vy`: final stored MVTools vector components, in units of `1 / pel` pixels
- `pixelDx`, `pixelDy`: whole-pixel offsets derived from `vx` and `vy`, used only when computing synthetic SAD

## Inputs

- `current`, `reference`: planar RGB `float32` SAD reference frames being compared; `current` is the frame whose block is being scored, `reference` is the frame sampled at the motion-shifted position. By default these come from `clip`, but when `sad_clip` is provided they come from `sad_clip`; RIFE motion estimation still uses `clip`.
- `width`, `height`: dimensions of a single plane in pixels
- `stride`: distance between the start of one row and the next, measured in float samples rather than bytes
- `blockSize = mv_block_size`: square block size used for each exported motion vector
- `overlap = mv_overlap`: shared region between neighboring blocks
- `step = blockSize - overlap`: distance between neighboring block origins
- `pel = mv_pel`: subpixel scale used by the stored vector representation; for example, `pel=2` means vector units are half-pixels
- `bits = mv_bits`: synthetic bit depth used to quantize comparison samples, scale exported SAD, and set the MVTools `bitsPerSample` metadata used by downstream threshold scaling; default `8` keeps thresholds and SAD on an 8-bit-equivalent scale unless you override it explicitly
- `sadMultiplier = sad_multiplier`: positive post-scale applied to the final exported SAD values
- `useChroma = mv_chroma`: if `1`, SAD uses all RGB channels; if `0`, SAD uses luma only
- `sad_y`, `sad_uv`: optional `RIFEMV` GPU-full synthetic Rec.709 Y/Cb/Cr SAD multipliers. When either is provided, `mv_chroma=1` is required and missing values default to `1.0`. Omitting both preserves the legacy RGB-channel SAD path.
- `hPadding = mv_hpad`, `vPadding = mv_vpad`: virtual horizontal and vertical analysis padding used for block placement and vector clamping
- `blockReduce`: how per-pixel motion inside a block is reduced to a single block vector, where `0 = center sample` and `1 = average of the whole block`

Block coordinates are derived from block-grid coordinates `(bx, by)`, where `bx` and `by` are integer block indices in the horizontal and vertical block grid:

```text
blockX = bx * step - hPadding
blockY = by * step - vPadding
```

Any pixel read outside the frame is edge-clamped:

```text
clampPixel(v, limit) = min(max(v, 0), limit - 1)
```

## Motion vector to pixel displacement

### Direct flow path

For normal export, a block vector is obtained by reducing the flow plane on the inference lattice. If `res_scale != 1.0`, RIFE runs on the resized inference clip and the block reduction uses an internal block geometry derived from that inference size. After reduction, horizontal motion is scaled by `sourceWidth / inferenceWidth` and vertical motion by `sourceHeight / inferenceHeight` so the exported vectors are back in source-pixel units. The block reduction is:

- `center`: sample the flow at the internal block center `(internalBlockX + internalBlockSizeX / 2, internalBlockY + internalBlockSizeY / 2)`, edge-clamped
- `average`: average all `internalBlockSizeX * internalBlockSizeY` flow samples covered by the internal block, with each sample position edge-clamped

Channel selection in the 4-channel exported flow tensor is:

- backward vectors use flow channels `0, 1`
- forward vectors use flow channels `2, 3`

The exported vector components are:

```text
vx = round(-2 * flowX * (sourceWidth  / inferenceWidth)  * pel)
vy = round(-2 * flowY * (sourceHeight / inferenceHeight) * pel)
```

They are then clamped so the referenced block stays within the padded analysis area. In other words, the chosen vector is limited so that the motion-shifted block cannot move beyond the configured padded bounds:

```text
minDx = (-hPadding - blockX) * pel
maxDx = (width  - blockSize + hPadding - blockX) * pel
minDy = (-vPadding - blockY) * pel
maxDy = (height - blockSize + vPadding - blockY) * pel

vx = clamp(vx, minDx, maxDx)
vy = clamp(vy, minDy, maxDy)
```

The synthetic SAD uses whole-pixel offsets derived from the clamped vector. This means the stored vector may keep subpixel precision through `pel`, but the actual synthetic SAD lookup always compares integer pixel positions:

```text
pixelDx = round(vx / pel)
pixelDy = round(vy / pel)
```

### Approximate displacement path

For `RIFEMVApprox2/3`, each adjacent-pair flow field is first converted to displacement on the inference lattice, meaning motion measured directly in inference-frame pixels:

```text
dispX = -2 * flowX
dispY = -2 * flowY
```

Multiple displacement fields are composed in sequence. Starting from the first field:

```text
composedX = dispX[0]
composedY = dispY[0]

for each later field i:
    sampleX = x + composedX[x, y]
    sampleY = y + composedY[x, y]
    composedX[x, y] += bilinearSample(dispX[i], sampleX, sampleY)
    composedY[x, y] += bilinearSample(dispY[i], sampleX, sampleY)
```

The bilinear sampler clamps sample coordinates to the frame before interpolation. Here `sampleX` and `sampleY` are floating-point lookup positions reached by following the already-composed motion.

Block reduction is then applied directly to `composedX` and `composedY` on the inference lattice. The reduced block displacement is then scaled back to source-pixel units before export. `pixelBlockDx` and `pixelBlockDy` mean the reduced horizontal and vertical block displacements measured in source pixels after scaling. Exported vectors are:

```text
vx = round(pixelBlockDx * pel)
vy = round(pixelBlockDy * pel)
```

The same vector clamp is applied as above, and the whole-pixel offsets used by the SAD are again:

```text
pixelDx = round(vx / pel)
pixelDy = round(vy / pel)
```

## Synthetic SAD formula

For each pixel in the block:

```text
currentY   = clampPixel(blockY + y, height)
referenceY = clampPixel(currentY + pixelDy, height)
currentX   = clampPixel(blockX + x, width)
referenceX = clampPixel(currentX + pixelDx, width)

currentIndex   = currentY   * stride + currentX
referenceIndex = referenceY * stride + referenceX
scale = (1 << bits) - 1
q(v) = round(v * scale) / scale
```

Here `currentX/currentY` are the edge-clamped coordinates inside the source block, `referenceX/referenceY` are the corresponding motion-shifted coordinates in the reference frame, and `currentIndex/referenceIndex` are row-major array indices into the planar float buffers. `scale` is the maximum integer sample value for the chosen synthetic SAD bit depth. `q(v)` is the synthetic-bit-depth quantizer applied to comparison samples before the absolute difference is measured.

If `mv_chroma=1` and `sad_y` / `sad_uv` are omitted, the legacy per-pixel contribution is:

```text
round((abs(q(Rc) - q(Rr)) + abs(q(Gc) - q(Gr)) + abs(q(Bc) - q(Br))) * scale)
```

If `mv_chroma=1` and either `sad_y` or `sad_uv` is provided for `RIFEMV` `gpu_mode=2` or `gpu_mode=3`, RGB is converted to synthetic Rec.709 Y/Cb/Cr after sample quantization:

```text
Y  = 0.2126 * R + 0.7152 * G + 0.0722 * B
Cb = 0.5 * (B - Y) / (1 - 0.0722)
Cr = 0.5 * (R - Y) / (1 - 0.2126)
```

and the per-pixel contribution is:

```text
round((abs(Yc - Yr) * sad_y + (abs(Cbc - Cbr) + abs(Crc - Crr)) * sad_uv) * scale)
```

If `mv_chroma=0`, RGB is converted to luma first:

```text
luma(r, g, b) = 0.2126 * r + 0.7152 * g + 0.0722 * b
```

and the per-pixel contribution is:

```text
round(abs(luma(q(Rc), q(Gc), q(Bc)) - luma(q(Rr), q(Gr), q(Br))) * scale)
```

The block `sad` is the sum of those rounded per-pixel contributions over the full `blockSize x blockSize` block. This is a synthetic matching cost intended to populate MVTools metadata, not a native RIFE confidence or loss value.

After that block sum is computed, the plugin applies the exported SAD calibration multiplier:

```text
sad = round(sad * sadMultiplier)
```

With the default `sad_multiplier=1.0`, the multiplier itself does not add any extra calibration beyond the chosen SAD bit scale.

With the default `bits=8`, exported SAD stays on an 8-bit-equivalent scale even when the metadata source clip is 10-bit or higher. Because the comparison samples are also quantized to that same synthetic bit depth before differencing and the exported MVTools bit-depth metadata matches that same scale, downstream `thsad` and `thscd1` thresholds stay aligned with the exported SAD values.

Important implementation detail: rounding happens per pixel before accumulation, not once at the end.

## Block-size normalization and frame props

The exported `VECTOR.sad` values stored in `MVTools_vectors` remain raw block SAD values. They therefore grow with larger block sizes, which is expected and required for MVTools compatibility because MVTools already rescales user thresholds such as `thsad` and `thscd1` using block size, chroma mode, and `bitsPerSample`.

That means:

- `RMV_AvgSadNorm` preserves the previous raw mean exported block SAD and will generally increase with larger blocks.
- `RMV_AvgSad` is the mean of the per-block SADs after each block has been converted into the implicit 8x8-equivalent threshold space used by MVTools user parameters.
- `RMV_AvgSadHigh2Pct`, `RMV_AvgSadHigh10Pct`, `RMV_AvgSadHigh25Pct`, `RMV_AvgSadHigh50Pct`, and `RMV_AvgSadHigh75Pct` are the rounded means of the highest `ceil(N * p)` per-block 8x8-equivalent SADs for that frame.
- `RMV_AvgSadLow2Pct`, `RMV_AvgSadLow10Pct`, and `RMV_AvgSadLow25Pct` are the rounded means of the lowest `ceil(N * p)` per-block 8x8-equivalent SADs for that frame.
- `RMV_MaxSad` and `RMV_MinSad` are the maximum and minimum per-block 8x8-equivalent SADs for that frame.
- `RMV_SadAvgDeviation` is `abs(maxSad8x8 - RMV_AvgSad)`, again using per-block 8x8-equivalent SADs.

Because MVTools performs its own block-size normalization internally, changing the actual exported `VECTOR.sad` values to an 8x8-equivalent scale would break downstream threshold behavior.

## Gray8 carrier SAD mask

The public vector clips returned by `RIFEMV`, `RIFEMVApprox2`, and `RIFEMVApprox3` are `Gray8` carrier clips. Their MVTools compatibility still comes entirely from frame properties, but the exposed pixel plane is now populated with a direction-specific SAD mask instead of dummy zeroes.

The mask is generated from the selected direction's exported finest-plane block `sad` values already stored in `MVTools_vectors`. The pixel plane does not use `RMV_AvgSad`, `RMV_AvgSadNorm`, or any other summary property.

### Relative mode

If `abs_sad_clip_range = 0`, the mask uses the relative per-frame mode.

Let `sad[i]` be the raw exported `VECTOR.sad` for block `i` in that frame, and let:

```text
frameMaxSad = max(sad[i])
```

If `frameMaxSad <= 0`, the whole `Gray8` plane is filled with `0`.

Otherwise each block-grid sample is normalized as:

```text
maskSmall[i] = round(sad[i] * 255 / frameMaxSad)
```

with the result clamped to `[0, 255]`.

This means:

- absolute zero SAD stays black
- the largest SAD present in that frame becomes white
- the contrast is frame-local rather than globally calibrated

### Absolute mode

If `abs_sad_clip_range > 0`, the mask switches to an absolute clipped range that still starts at SAD `0`.

Let:

```text
clipRange = abs_sad_clip_range
clippedSad[i] = min(max(sad[i], 0), clipRange)
maskSmall[i] = min(floor(clippedSad[i] * 256 / clipRange), 255)
```

This means:

- SAD values are quantized against a fixed range instead of the current frame maximum
- values at or above `clipRange` map to white
- when `clipRange` is a multiple of `256`, each code step corresponds to `clipRange / 256` SAD units

Examples:

- `abs_sad_clip_range = 1024` gives steps of `4`
- `abs_sad_clip_range = 2048` gives steps of `8`
- `abs_sad_clip_range = 4096` gives steps of `16`

### Full-frame rasterization

One normalized value is produced for each exported block, so the intermediate mask is a `nBlkX x nBlkY` grid. That small grid is then bilinearly upscaled directly to the full output frame size.

This rasterization is intentionally smooth and visualization-oriented. It is meant to make the `Gray8` carrier clip directly useful as a SAD mask, not to reproduce MVTools `mv.Mask(kind=1)` byte-for-byte.

## Invalid vectors

If no valid reference frame exists, the vector is marked invalid and uses:

```text
vx = 0
vy = 0
sad = round(blockSize * blockSize * (1 << bits) * sadMultiplier)
```

This sentinel is stored directly without running the per-pixel SAD loop.

Because every block in an invalid frame receives the same nonzero sentinel SAD, relative mode produces a solid `255` `Gray8` carrier mask for that direction. In absolute mode the mask shows that same sentinel after clipping to `abs_sad_clip_range`, which will usually also produce `255`.
