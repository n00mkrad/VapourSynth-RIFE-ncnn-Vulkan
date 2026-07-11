static const char rife_degrain_sad_comp_data[] = R"glsl(#version 450

#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif

layout (binding = 0) readonly buffer flow_blob { sfpvec4 flow_data[]; };
layout (binding = 1) readonly buffer current_blob { float current_data[]; };
layout (binding = 2) readonly buffer reference_blob { float reference_data[]; };
layout (binding = 3) writeonly buffer sad_blob { float sad_data[]; };

layout (push_constant) uniform parameter
{
int flow_physical_w;
int flow_w;
int flow_h;
int image_w;
int image_h;
float motion_scale_x;
float motion_scale_y;
int sad_center;
float sad_center_floor;
} p;

int clampi(int value, int limit)
{
return min(max(value, 0), limit - 1);
}

vec2 load_flow(float x, float y)
{
float flow_x = (x + 0.5f) * float(p.flow_w) / float(p.image_w) - 0.5f;
float flow_y = (y + 0.5f) * float(p.flow_h) / float(p.image_h) - 0.5f;
int x0 = clampi(int(floor(flow_x)), p.flow_w);
int y0 = clampi(int(floor(flow_y)), p.flow_h);
int x1 = clampi(x0 + 1, p.flow_w);
int y1 = clampi(y0 + 1, p.flow_h);
float ax = flow_x - floor(flow_x);
float ay = flow_y - floor(flow_y);
vec4 f00 = vec4(buffer_ld4(flow_data, y0 * p.flow_physical_w + x0));
vec4 f10 = vec4(buffer_ld4(flow_data, y0 * p.flow_physical_w + x1));
vec4 f01 = vec4(buffer_ld4(flow_data, y1 * p.flow_physical_w + x0));
vec4 f11 = vec4(buffer_ld4(flow_data, y1 * p.flow_physical_w + x1));
return mix(mix(f00.xy, f10.xy, ax), mix(f01.xy, f11.xy, ax), ay) * vec2(p.motion_scale_x, p.motion_scale_y);
}

float sample_reference(float x, float y)
{
float sx = clamp(x, 0.f, float(p.image_w - 1));
float sy = clamp(y, 0.f, float(p.image_h - 1));
int x0 = int(floor(sx));
int y0 = int(floor(sy));
int x1 = min(x0 + 1, p.image_w - 1);
int y1 = min(y0 + 1, p.image_h - 1);
float ax = sx - float(x0);
float ay = sy - float(y0);
float v00 = reference_data[y0 * p.image_w + x0];
float v10 = reference_data[y0 * p.image_w + x1];
float v01 = reference_data[y1 * p.image_w + x0];
float v11 = reference_data[y1 * p.image_w + x1];
return mix(mix(v00, v10, ax), mix(v01, v11, ax), ay);
}

float delta_at(int x, int y)
{
int cx = clampi(x, p.image_w);
int cy = clampi(y, p.image_h);
vec2 flow = load_flow(float(cx), float(cy));
return current_data[cy * p.image_w + cx] - sample_reference(float(cx) + flow.x, float(cy) + flow.y);
}

void main()
{
int x = int(gl_GlobalInvocationID.x);
int y = int(gl_GlobalInvocationID.y);
if (x >= p.image_w || y >= p.image_h)
return;

// Use a fixed clamped 8x8 neighborhood so thresholds retain MVTools-like units at frame borders.
float delta_sum = 0.f;
float raw_sad = 0.f;
for (int wy = -3; wy <= 4; wy++)
{
for (int wx = -3; wx <= 4; wx++)
{
float delta = delta_at(x + wx, y + wy);
delta_sum += delta;
raw_sad += abs(delta) * 255.f;
}
}

float effective_sad = raw_sad;
if (p.sad_center != 0)
{
float mean_delta = delta_sum / 64.f;
float centered_sad = 0.f;
for (int wy = -3; wy <= 4; wy++)
{
for (int wx = -3; wx <= 4; wx++)
centered_sad += abs(delta_at(x + wx, y + wy) - mean_delta) * 255.f;
}
effective_sad = max(min(centered_sad, raw_sad), raw_sad * p.sad_center_floor);
}

sad_data[y * p.image_w + x] = effective_sad;
}
)glsl";

static const char rife_degrain_accumulate_comp_data[] = R"glsl(#version 450

#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif

layout (binding = 0) readonly buffer flow_blob { sfpvec4 flow_data[]; };
layout (binding = 1) readonly buffer reference_blob { float reference_data[]; };
layout (binding = 2) readonly buffer sad_blob { float sad_data[]; };
layout (binding = 3) buffer numerator_blob { float numerator_data[]; };
layout (binding = 4) buffer denominator_blob { float denominator_data[]; };

layout (push_constant) uniform parameter
{
int flow_physical_w;
int flow_w;
int flow_h;
int luma_w;
int luma_h;
int plane_w;
int plane_h;
int sub_x;
int sub_y;
float motion_scale_x;
float motion_scale_y;
float threshold;
float flow_consistency;
} p;

int clampi(int value, int limit)
{
return min(max(value, 0), limit - 1);
}

vec4 load_flow(float lx, float ly)
{
float flow_x = (lx + 0.5f) * float(p.flow_w) / float(p.luma_w) - 0.5f;
float flow_y = (ly + 0.5f) * float(p.flow_h) / float(p.luma_h) - 0.5f;
int x0 = clampi(int(floor(flow_x)), p.flow_w);
int y0 = clampi(int(floor(flow_y)), p.flow_h);
int x1 = clampi(x0 + 1, p.flow_w);
int y1 = clampi(y0 + 1, p.flow_h);
float ax = flow_x - floor(flow_x);
float ay = flow_y - floor(flow_y);
vec4 f00 = vec4(buffer_ld4(flow_data, y0 * p.flow_physical_w + x0));
vec4 f10 = vec4(buffer_ld4(flow_data, y0 * p.flow_physical_w + x1));
vec4 f01 = vec4(buffer_ld4(flow_data, y1 * p.flow_physical_w + x0));
vec4 f11 = vec4(buffer_ld4(flow_data, y1 * p.flow_physical_w + x1));
return mix(mix(f00, f10, ax), mix(f01, f11, ax), ay) * vec4(p.motion_scale_x, p.motion_scale_y, p.motion_scale_x, p.motion_scale_y);
}

float sample_reference(float x, float y)
{
float sx = clamp(x, 0.f, float(p.plane_w - 1));
float sy = clamp(y, 0.f, float(p.plane_h - 1));
int x0 = int(floor(sx));
int y0 = int(floor(sy));
int x1 = min(x0 + 1, p.plane_w - 1);
int y1 = min(y0 + 1, p.plane_h - 1);
float ax = sx - float(x0);
float ay = sy - float(y0);
float v00 = reference_data[y0 * p.plane_w + x0];
float v10 = reference_data[y0 * p.plane_w + x1];
float v01 = reference_data[y1 * p.plane_w + x0];
float v11 = reference_data[y1 * p.plane_w + x1];
return mix(mix(v00, v10, ax), mix(v01, v11, ax), ay);
}

void main()
{
int x = int(gl_GlobalInvocationID.x);
int y = int(gl_GlobalInvocationID.y);
if (x >= p.plane_w || y >= p.plane_h)
return;

int scale_x = 1 << p.sub_x;
int scale_y = 1 << p.sub_y;
int luma_x0 = x * scale_x;
int luma_y0 = y * scale_y;
float sad = 0.f;
for (int oy = 0; oy < scale_y; oy++)
{
for (int ox = 0; ox < scale_x; ox++)
sad += sad_data[clampi(luma_y0 + oy, p.luma_h) * p.luma_w + clampi(luma_x0 + ox, p.luma_w)];
}
sad /= float(scale_x * scale_y);

// Preserve the MVTools Degrain reference-weight curve in floating-point form.
float weight = 0.f;
if (sad < p.threshold && p.threshold > 0.f)
{
float threshold2 = p.threshold * p.threshold;
float sad2 = sad * sad;
weight = (threshold2 - sad2) / (threshold2 + sad2);
}

float luma_x = float(luma_x0) + float(scale_x - 1) * 0.5f;
float luma_y = float(luma_y0) + float(scale_y - 1) * 0.5f;
vec4 flow = load_flow(luma_x, luma_y);
float reference_luma_x = luma_x + flow.x;
float reference_luma_y = luma_y + flow.y;
float consistency_weight = 1.f;
if (p.flow_consistency > 0.f)
{
vec4 reverse_flow = load_flow(reference_luma_x, reference_luma_y);
float round_trip_error = length(flow.xy + reverse_flow.zw);
consistency_weight = clamp(1.f - round_trip_error / p.flow_consistency, 0.f, 1.f);
consistency_weight *= consistency_weight;
}
weight *= consistency_weight;
float reference = sample_reference(float(x) + flow.x / float(scale_x), float(y) + flow.y / float(scale_y));
int index = y * p.plane_w + x;
numerator_data[index] += reference * weight;
denominator_data[index] += weight;
}
)glsl";

static const char rife_degrain_finish_comp_data[] = R"glsl(#version 450

layout (binding = 0) readonly buffer current_blob { float current_data[]; };
layout (binding = 1) readonly buffer numerator_blob { float numerator_data[]; };
layout (binding = 2) readonly buffer denominator_blob { float denominator_data[]; };
layout (binding = 3) writeonly buffer output_blob { float output_data[]; };

layout (push_constant) uniform parameter
{
int width;
int height;
float limit;
} p;

void main()
{
int x = int(gl_GlobalInvocationID.x);
int y = int(gl_GlobalInvocationID.y);
if (x >= p.width || y >= p.height)
return;
int index = y * p.width + x;
float current = current_data[index];
float result = numerator_data[index] / denominator_data[index];
output_data[index] = clamp(result, max(0.f, current - p.limit), min(1.f, current + p.limit));
}
)glsl";

static const char rife_degrain_stats_comp_data[] = R"glsl(#version 450

layout (binding = 0) readonly buffer input_blob { float input_data[]; };
layout (binding = 1) writeonly buffer stats_blob { vec2 stats_data[]; };

layout (push_constant) uniform parameter
{
int count;
int pairs;
} p;

void main()
{
int output_index = int(gl_GlobalInvocationID.x);
if (p.pairs != 0 && output_index != 0)
return;
int chunk_start = p.pairs != 0 ? 0 : output_index * 256;
if (chunk_start >= p.count)
return;
float sum = 0.f;
float maximum = 0.f;
int chunk_end = p.pairs != 0 ? p.count : min(chunk_start + 256, p.count);
for (int i = chunk_start; i < chunk_end; i++)
{
if (p.pairs != 0)
{
sum += input_data[i * 2];
maximum = max(maximum, input_data[i * 2 + 1]);
}
else
{
float value = input_data[i];
sum += value;
maximum = max(maximum, value);
}
}
stats_data[output_index] = vec2(sum, maximum);
}
)glsl";
