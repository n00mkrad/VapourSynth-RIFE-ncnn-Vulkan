static const char rife_mv_full_comp_data[] = R"glsl(#version 450

#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif

layout (binding = 0) readonly buffer flow_blob { sfpvec4 flow_blob_data[]; };
layout (binding = 1) readonly buffer current_blob { float current_blob_data[]; };
layout (binding = 2) readonly buffer reference_blob { float reference_blob_data[]; };
layout (binding = 3) writeonly buffer vector_blob { uint vector_blob_data[]; };

layout (push_constant) uniform parameter
{
int physical_w;
int flow_w;
int flow_h;
int image_w;

int image_h;
int image_cstep;
int blk_x;
int blk_y;

int block_size_x;
int block_size_y;
int step_x;
int step_y;

int pad_x;
int pad_y;
int internal_block_size_x;
int internal_block_size_y;

int internal_step_x;
int internal_step_y;
int internal_pad_x;
int internal_pad_y;

int pel;
int reduce_mode;
int use_chroma;
int output_words_per_vector;

float motion_scale_x;
float motion_scale_y;
float max_sample;
float reserved2;
} p;

int clampi(int value, int limit)
{
return min(max(value, 0), limit - 1);
}

int round_away(float value)
{
return value >= 0.f ? int(floor(value + 0.5f)) : int(ceil(value - 0.5f));
}

uint round_positive(float value)
{
return uint(floor(value + 0.5f));
}

float round_sample(float value)
{
float scaled = value * p.max_sample;
float rounded = scaled >= 0.f ? floor(scaled + 0.5f) : ceil(scaled - 0.5f);
return rounded / p.max_sample;
}

vec3 load_current_rgb(int x, int y)
{
int idx = y * p.image_w + x;
return vec3(round_sample(current_blob_data[idx]),
round_sample(current_blob_data[p.image_cstep + idx]),
round_sample(current_blob_data[p.image_cstep * 2 + idx]));
}

vec3 load_reference_rgb(int x, int y)
{
int idx = y * p.image_w + x;
return vec3(round_sample(reference_blob_data[idx]),
round_sample(reference_blob_data[p.image_cstep + idx]),
round_sample(reference_blob_data[p.image_cstep * 2 + idx]));
}

vec4 load_flow(int x, int y)
{
int sx = clampi(x, p.flow_w);
int sy = clampi(y, p.flow_h);
return vec4(buffer_ld4(flow_blob_data, sy * p.physical_w + sx));
}

vec4 reduce_flow_block(int block_x, int block_y)
{
if (p.reduce_mode == 0)
{
int sample_x = block_x + p.internal_block_size_x / 2;
int sample_y = block_y + p.internal_block_size_y / 2;
return load_flow(sample_x, sample_y);
}

vec4 sum = vec4(0.f);
for (int y = 0; y < p.internal_block_size_y; y++)
{
for (int x = 0; x < p.internal_block_size_x; x++)
{
sum += load_flow(block_x + x, block_y + y);
}
}

float sample_count = float(p.internal_block_size_x * p.internal_block_size_y);
return sum / sample_count;
}

int clamp_mv_component(int value, int block_coord, int block_size, int size, int padding)
{
int min_pixel_delta = -padding - block_coord;
int max_pixel_delta = size - block_size + padding - block_coord;
return min(max(value, min_pixel_delta * p.pel), max_pixel_delta * p.pel);
}

float load_current_luma(int x, int y)
{
vec3 rgb = load_current_rgb(x, y);
return rgb.r * 0.2126f + rgb.g * 0.7152f + rgb.b * 0.0722f;
}

float load_reference_luma(int x, int y)
{
vec3 rgb = load_reference_rgb(x, y);
return rgb.r * 0.2126f + rgb.g * 0.7152f + rgb.b * 0.0722f;
}

vec3 load_sad_current_rgb(int x, int y, bool swap_images)
{
return swap_images ? load_reference_rgb(x, y) : load_current_rgb(x, y);
}

vec3 load_sad_reference_rgb(int x, int y, bool swap_images)
{
return swap_images ? load_current_rgb(x, y) : load_reference_rgb(x, y);
}

float load_sad_current_luma(int x, int y, bool swap_images)
{
return swap_images ? load_reference_luma(x, y) : load_current_luma(x, y);
}

float load_sad_reference_luma(int x, int y, bool swap_images)
{
return swap_images ? load_current_luma(x, y) : load_reference_luma(x, y);
}

uint compute_sad(int pixel_dx, int pixel_dy, int block_x, int block_y, bool swap_images)
{
uint sad = 0;
int current_x0 = block_x;
int current_y0 = block_y;
int reference_x0 = block_x + pixel_dx;
int reference_y0 = block_y + pixel_dy;
bool interior = current_x0 >= 0 && current_y0 >= 0 &&
reference_x0 >= 0 && reference_y0 >= 0 &&
current_x0 + p.block_size_x <= p.image_w &&
current_y0 + p.block_size_y <= p.image_h &&
reference_x0 + p.block_size_x <= p.image_w &&
reference_y0 + p.block_size_y <= p.image_h;

if (interior)
{
if (p.use_chroma != 0)
{
for (int y = 0; y < p.block_size_y; y++)
{
for (int x = 0; x < p.block_size_x; x++)
{
vec3 current_rgb = load_sad_current_rgb(current_x0 + x, current_y0 + y, swap_images);
vec3 reference_rgb = load_sad_reference_rgb(reference_x0 + x, reference_y0 + y, swap_images);
vec3 diff = abs(current_rgb - reference_rgb);
sad += round_positive((diff.r + diff.g + diff.b) * p.max_sample);
}
}
}
else
{
for (int y = 0; y < p.block_size_y; y++)
{
for (int x = 0; x < p.block_size_x; x++)
{
float current_luma = load_sad_current_luma(current_x0 + x, current_y0 + y, swap_images);
float reference_luma = load_sad_reference_luma(reference_x0 + x, reference_y0 + y, swap_images);
sad += round_positive(abs(current_luma - reference_luma) * p.max_sample);
}
}
}
return sad;
}

for (int y = 0; y < p.block_size_y; y++)
{
int current_y = clampi(block_y + y, p.image_h);
int reference_y = clampi(current_y + pixel_dy, p.image_h);
for (int x = 0; x < p.block_size_x; x++)
{
int current_x = clampi(block_x + x, p.image_w);
int reference_x = clampi(current_x + pixel_dx, p.image_w);
if (p.use_chroma != 0)
{
vec3 current_rgb = load_sad_current_rgb(current_x, current_y, swap_images);
vec3 reference_rgb = load_sad_reference_rgb(reference_x, reference_y, swap_images);
vec3 diff = abs(current_rgb - reference_rgb);
sad += round_positive((diff.r + diff.g + diff.b) * p.max_sample);
}
else
{
float current_luma = load_sad_current_luma(current_x, current_y, swap_images);
float reference_luma = load_sad_reference_luma(reference_x, reference_y, swap_images);
sad += round_positive(abs(current_luma - reference_luma) * p.max_sample);
}
}
}

return sad;
}

uvec3 make_vector(float flow_x, float flow_y, int block_x, int block_y, bool swap_sad_images)
{
int x = round_away(-2.f * flow_x * p.motion_scale_x * float(p.pel));
int y = round_away(-2.f * flow_y * p.motion_scale_y * float(p.pel));
x = clamp_mv_component(x, block_x, p.block_size_x, p.image_w, p.pad_x);
y = clamp_mv_component(y, block_y, p.block_size_y, p.image_h, p.pad_y);
int pixel_dx = round_away(float(x) / float(p.pel));
int pixel_dy = round_away(float(y) / float(p.pel));
uint sad = compute_sad(pixel_dx, pixel_dy, block_x, block_y, swap_sad_images);
return uvec3(uint(x), uint(y), sad);
}

void store_vector(int vector_index, uvec3 vector)
{
int output_index = vector_index * p.output_words_per_vector;
if (p.output_words_per_vector == 2)
{
// Pack only the final output; vector and SAD calculations remain 32-bit.
vector_blob_data[output_index] = (vector.x & 0xffffu) | ((vector.y & 0xffffu) << 16);
vector_blob_data[output_index + 1] = vector.z;
return;
}

vector_blob_data[output_index] = vector.x;
vector_blob_data[output_index + 1] = vector.y;
vector_blob_data[output_index + 2] = vector.z;
vector_blob_data[output_index + 3] = 0u;
}

void main()
{
int block_index = int(gl_GlobalInvocationID.x);
int block_count = p.blk_x * p.blk_y;
if (block_index >= block_count)
return;

int by = block_index / p.blk_x;
int bx = block_index - by * p.blk_x;
int block_x = bx * p.step_x - p.pad_x;
int block_y = by * p.step_y - p.pad_y;
int internal_block_x = bx * p.internal_step_x - p.internal_pad_x;
int internal_block_y = by * p.internal_step_y - p.internal_pad_y;
vec4 reduced = reduce_flow_block(internal_block_x, internal_block_y);

store_vector(block_index, make_vector(reduced.x, reduced.y, block_x, block_y, false));
store_vector(block_count + block_index, make_vector(reduced.z, reduced.w, block_x, block_y, true));
}
)glsl";
