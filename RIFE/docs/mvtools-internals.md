The VapourSynth mvtools plugin stores all of its working data for motion vectors as binary data in vapoursynth frame props on the clip which results from calling `mv.Analyse()`. Specifically there are two props of interest `MVTools_MVAnalysisData` and `MVTools_vectors`.

All of this data is serialized rather implictly from C++ structs. Most of these structs contain only signed integers (even for fields which do not have logical negative interpretations) and bytes are written with native endianness. For deserialization I have chosen to interpret fields that should not be negative (e.g. width, height, size) as unsigned and always use little endian byte order. These nuances are hopefully not relevant in practice as the positive integer range of a signed 32 bit integer is still much larger than practical video sizes and almost every host running MVTools is likely to be little endian natively. Still it would be nice conceptually if future motion vector work could make these conventions explicit; for this reason the types below will be listed with the signedness I think they should have.

### MVTools_MVAnalysisData

This just contains some metadata about the context in which the vectors were generated. The length of this data is expected to always be 84 bytes (21 32-bit integers) in the following order:

Note: The `magic_key` and `version` appear to be uninitialized and so have no usable data in them.

```
u32 magic_key
u32 version
u32 block_size_x
u32 block_size_y
u32 pel
u32 level_count
i32 delta_frame
u32 backwards
u32 cpu_flags
u32 motion_flags
u32 width
u32 height
u32 overlap_x
u32 overlap_y
u32 block_count_x
u32 block_count_y
u32 bits_per_sample
u32 chroma_ratio_y
u32 chroma_ratio_x
u32 padding_x
u32 padding_y
```

### MVTools_vectors

This contains the actual motion vector data for all levels of motion vector calculation. The structure of a single motion vector is simply:

```
i32 x
i32 y
u64 sad
```

These are serialized without any padding (16 bytes per vector). Each level is structured as:

```
u32 size
[] vectors
```

Again without any padding. So for example a level with 10 vectors would have a size value of 164 (16 bytes per vector * 10 vectors + 4 bytes for the size).

This is ultimately structured as:

```
u32 size
u32 valid
[] levels
```

Again without any padding. The value stored in size is therefore expected to be equivalent to the size which vapoursynth reports for the frame property. The valid value is expected to be 0 for situations where motion vectors are not present (e.g. the first frame of forwards vectors) and 1 for situations where motion vectors are present.
