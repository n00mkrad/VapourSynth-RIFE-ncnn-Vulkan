static const char rife_mv_reduce_comp_data[] = R"glsl(#version 450

#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif

layout (binding = 0) readonly buffer flow_blob { sfpvec4 flow_blob_data[]; };
layout (binding = 1) writeonly buffer reduced_blob { vec4 reduced_blob_data[]; };

layout (push_constant) uniform parameter
{
int physical_w;
int flow_w;
int flow_h;
int blk_x;

int blk_y;
int block_size_x;
int block_size_y;
int step_x;

int step_y;
int pad_x;
int pad_y;
int reduce_mode;
} p;

int clampi(int value, int limit)
{
return min(max(value, 0), limit - 1);
}

vec4 load_flow(int x, int y)
{
int sx = clampi(x, p.flow_w);
int sy = clampi(y, p.flow_h);
return vec4(buffer_ld4(flow_blob_data, sy * p.physical_w + sx));
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

if (p.reduce_mode == 0)
{
int sample_x = block_x + p.block_size_x / 2;
int sample_y = block_y + p.block_size_y / 2;
reduced_blob_data[block_index] = load_flow(sample_x, sample_y);
return;
}

vec4 sum = vec4(0.f);
for (int y = 0; y < p.block_size_y; y++)
{
for (int x = 0; x < p.block_size_x; x++)
{
sum += load_flow(block_x + x, block_y + y);
}
}

float sample_count = float(p.block_size_x * p.block_size_y);
reduced_blob_data[block_index] = sum / sample_count;
}
)glsl";
