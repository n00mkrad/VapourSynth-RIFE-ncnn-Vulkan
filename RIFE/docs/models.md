# IFNet model architecture summary

This summary covers the legacy `rife_3.1/IFNet_HDv3.py` model plus every `IFNet_HDv3_v4_*.py` file currently in `vsrife/`.

When a file is listed as identical to another one, that means the layer topology, block widths, encoder shape, IFBlock outputs, and scale schedule match. I only call out code-organization differences separately when they do not change the model architecture.

## Architecture groups

### `rife_3.1/IFNet_HDv3.py`

Legacy pre-v4 branch.

- 3-stage cascade with fixed block width `80 / 80 / 80`
- `IFBlock` uses plain conv stacks plus `PReLU`; there is no `ResConv`
- No external feature encoder and no timestep input; the network takes a concatenated 6-channel `(img0, img1)` tensor and predicts the centered midpoint flow directly
- `IFBlock` outputs only 4 flow channels; there is no mask path in `IFNet` itself
- Uses the scale schedule `4 / 2 / 1`
- Returns the final 4-channel flow plus intermediate flow states, not an interpolated RGB frame
- `rife_3.1/RIFE_HDv3.py` wraps this flownet with a separate `ContextNet` and `FusionNet` to produce the final interpolated image

### `IFNet_HDv3_v4_0.py`, `IFNet_HDv3_v4_1.py`

The original baseline.

- 4-stage cascade with block widths `192 / 128 / 96 / 64`
- `IFBlock` uses plain conv stacks plus `PReLU`; there is no `ResConv`
- No external feature encoder and no `f0` / `f1` inputs
- `lastconv` is a plain `ConvTranspose2d` that outputs 5 channels (`flow[4] + mask[1]`)
- Includes the early large-motion retry path inside `forward()`
- Ensemble is supported

### `IFNet_HDv3_v4_2.py`, `IFNet_HDv3_v4_3.py`, `IFNet_HDv3_v4_4.py`

Same high-level 4-block baseline as `v4_0` / `v4_1`, but simplified.

- Switches from `PReLU` to `LeakyReLU(0.2)`
- Keeps plain conv blocks instead of `ResConv`
- Still no external encoder and no `f0` / `f1`
- Removes the earlier `convblock(feat) + feat` pattern and the large-motion retry logic
- Ensemble is still supported

### `IFNet_HDv3_v4_5.py`, `IFNet_HDv3_v4_6.py`

First `ResConv` generation.

- Still 4 blocks at `192 / 128 / 96 / 64`
- Replaces the plain conv body with 8 `ResConv` layers per block
- `IFBlock.lastconv` becomes `ConvTranspose2d(..., 4 * 6)` followed by `PixelShuffle(2)`
- No external encoder yet; `forward()` still only takes the two images plus timestep
- Ensemble is supported

### `IFNet_HDv3_v4_7.py`, `IFNet_HDv3_v4_8.py`

Same `ResConv` 4-block architecture as `v4_5` / `v4_6`, but now with a tiny inline encoder.

- Adds an inline encoder `Conv2d(3, 16, stride=2) -> ConvTranspose2d(16, 4)`
- `forward()` now accepts `f0` and `f1`
- Block inputs become `7 + 8` for `block0` and `8 + 4 + 8` for later blocks
- Everything else stays at the `v4_5` / `v4_6` shape

### `IFNet_HDv3_v4_9.py`, `IFNet_HDv3_v4_10.py`, `IFNet_HDv3_v4_11.py`, `IFNet_HDv3_v4_12.py`

Architecturally identical to each other.

- Keeps the 4-block `ResConv` / `PixelShuffle(4 * 6)` design
- Replaces the tiny encoder with a deeper inline encoder: `32 -> 32 -> 32 -> 8`
- Because the encoder now emits 8 channels per image, block inputs grow to `7 + 16` and `8 + 4 + 16`
- Block widths remain `192 / 128 / 96 / 64`
- Ensemble is supported

### `IFNet_HDv3_v4_12_lite.py`, `IFNet_HDv3_v4_13_lite.py`

Lite inline-encoder branch.

- Same overall pattern as `v4_9` through `v4_12`, but still uses an inline encoder instead of a reusable `Head` module
- Inline encoder shape is `32 -> 32 -> 32 -> 4`
- Block widths shrink to `128 / 96 / 64 / 48`
- Block inputs are `7 + 8` and `8 + 4 + 8`
- Ensemble is supported

### `IFNet_HDv3_v4_13.py`, `IFNet_HDv3_v4_14.py`, `IFNet_HDv3_v4_15.py`, `IFNet_HDv3_v4_17.py`, `IFNet_HDv3_v4_18.py`, `IFNet_HDv3_v4_19.py`, `IFNet_HDv3_v4_24.py`

Standard Head-based 4-block branch.

- Replaces the inline encoder with a reusable `Head` module
- `Head` shape is `32 -> 32 -> 32 -> 8`
- Block widths are `192 / 128 / 96 / 64`
- `IFBlock` still returns only `(flow, mask)` through `4 * 6` channels after `PixelShuffle`
- Later blocks still use `8 + 4 + 16` inputs
- Ensemble is supported

### `IFNet_HDv3_v4_14_lite.py`

Unique one-off variant.

- Uses the same full-size `Head(32 -> 32 -> 32 -> 8)` and full block widths `192 / 128 / 96 / 64`
- The distinguishing change is `ResConv(..., groups=2)` instead of `groups=1`
- Despite the `_lite` name, this is not the reduced-width lite branch
- Ensemble is supported

### `IFNet_HDv3_v4_15_lite.py`, `IFNet_HDv3_v4_16_lite.py`, `IFNet_HDv3_v4_17_lite.py`

Standard Head-based lite branch.

- Uses a smaller `Head(16 -> 16 -> 16 -> 4)`
- Block widths are `128 / 96 / 64 / 48`
- `IFBlock` still returns only `(flow, mask)` through `4 * 6` channels after `PixelShuffle`
- Ensemble is supported

### `IFNet_HDv3_v4_20.py`

Wider 4-block Head-based model.

- Same Head-based `(flow, mask)` design as the standard full branch
- `Head` is still `32 -> 32 -> 32 -> 8`
- Block widths change to `384 / 192 / 96 / 48`
- Ensemble is supported

### `IFNet_HDv3_v4_21.py`, `IFNet_HDv3_v4_22.py`, `IFNet_HDv3_v4_23.py`

Feature-propagating 4-block branch.

- `Head` stays `32 -> 32 -> 32 -> 8`
- `IFBlock` now outputs `(flow, mask, feat)` via `4 * 13` channels before `PixelShuffle`
- The feature tensor from one block is fed into the next block
- Block widths are `256 / 192 / 96 / 48`
- Later block inputs become `8 + 4 + 16 + 8`
- Ensemble is disabled and raises `ValueError`

### `IFNet_HDv3_v4_22_lite.py`

Lite feature-propagating 4-block model.

- Same feature-returning idea as `v4_21` through `v4_23`
- Uses `Head(16 -> 16 -> 16 -> 4)`
- Block widths are `192 / 128 / 64 / 32`
- Later block inputs become `8 + 4 + 8 + 8`
- Ensemble is disabled

### `IFNet_HDv3_v4_25.py`, `IFNet_HDv3_v4_26.py`

Architecturally identical 5-block models.

- Extend the feature-propagating branch from 4 blocks to 5 blocks
- Use `Head(16 -> 16 -> 16 -> 4)`
- Block widths are `192 / 128 / 96 / 64 / 32`
- `IFBlock` still returns `(flow, mask, feat)` through `4 * 13` channels
- Scale schedule changes to `16 / 8 / 4 / 2 / 1`
- Ensemble is disabled
- `IFNet_HDv3_v4_26.py` only refactors the flow path into `forward_flow()`; the layer architecture itself matches `IFNet_HDv3_v4_25.py`

### `IFNet_HDv3_v4_25_lite.py`

Lite 5-block feature-propagating branch.

- Same family as `v4_25` / `v4_26`
- Block widths are `192 / 128 / 96 / 64 / 24`
- Uses the more aggressive scale schedule `32 / 16 / 8 / 4 / 1`
- Ensemble is disabled

### `IFNet_HDv3_v4_25_heavy.py`

Heavy 5-block feature-propagating branch.

- Same family as `v4_25` / `v4_26`
- Keeps `Head(16 -> 16 -> 16 -> 4)`
- Doubles the block widths to `384 / 256 / 192 / 128 / 64`
- Ensemble is disabled

### `IFNet_HDv3_v4_26_heavy.py`

Unique heavy 5-block variant.

- Still uses 5 feature-propagating blocks and the `16 / 8 / 4 / 2 / 1` scale schedule
- Changes the Head output from 4 channels to 16 channels: `Head(16 -> 16 -> 16 -> 16)`
- That expands the IFNet inputs to `7 + 32` for `block0` and `8 + 4 + 8 + 32` for later blocks
- Block widths themselves stay at `192 / 128 / 96 / 64 / 32`
- Includes the `forward_flow()` refactor from `v4_26`
- Ensemble is disabled

## Short version

The main architectural evolution is:

1. Legacy pre-v4 3-block flow-only IFNet with separate context/fusion refinement (`rife_3.1`)
2. Plain conv blocks with no encoder (`v4_0` to `v4_4`)
3. `ResConv` blocks with no encoder (`v4_5`, `v4_6`)
4. `ResConv` blocks with inline encoders (`v4_7` to `v4_12`, plus the lite inline branch)
5. Reusable `Head` encoder with 4-block `(flow, mask)` models (`v4_13` family)
6. Feature-propagating 4-block models that return `(flow, mask, feat)` (`v4_21` family)
7. Feature-propagating 5-block models (`v4_25` family)

The naming is not perfectly consistent with the internals: `IFNet_HDv3_v4_14_lite.py` is not the reduced-width lite branch, and `IFNet_HDv3_v4_26.py` is architecturally the same model as `IFNet_HDv3_v4_25.py` with a refactored forward path.