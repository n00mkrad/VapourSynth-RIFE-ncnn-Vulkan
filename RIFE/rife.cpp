// rife implemented with ncnn library

#include "rife.h"
//#include <iostream>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "allocator.h"
#include "benchmark.h"

#include "rife_preproc.comp.hex.h"
#include "rife_v4_timestep.comp.hex.h"
#include "rife_mv_reduce.comp.hex.h"

#include "rife_ops.h"

DEFINE_LAYER_CREATOR(Warp)

RIFE::RIFE(int gpuid, float _flow_scale, int _num_threads, bool _rife_v2, bool _rife_v4, int _padding,
           FlowResizeMode _flow_resize_mode, bool _disable_vulkan_fp16)
{
    vkdev = gpuid == -1 ? 0 : ncnn::get_gpu_device(gpuid);

    rife_preproc = 0;
    rife_v4_timestep = 0;
    rife_mv_reduce = 0;
    rife_flow_scale_image = 0;
    rife_flow_resize_flow = 0;
    rife_flow_scale_vectors = 0;
    rife_flow_resize_output = 0;
    rife_flow_double_vectors = 0;
    rife_v2_slice_flow = 0;
    use_flow_scale = std::abs(_flow_scale - 1.f) > 1e-6f;
    flow_scale = _flow_scale;
    flow_scale_inv = 1.f / _flow_scale;
    flow_resize_mode = _flow_resize_mode;
    num_threads = _num_threads;
    rife_v2 = _rife_v2;
    rife_v4 = _rife_v4;
    disable_vulkan_fp16 = _disable_vulkan_fp16;
    padding = _padding;
    rife_v4_flow_blob_name.clear();
}

RIFE::~RIFE()
{
    delete rife_preproc;
    delete rife_v4_timestep;
    delete rife_mv_reduce;

    if (use_flow_scale)
    {
        rife_flow_scale_image->destroy_pipeline(flownet.opt);
        delete rife_flow_scale_image;

        rife_flow_resize_flow->destroy_pipeline(flownet.opt);
        delete rife_flow_resize_flow;

        rife_flow_scale_vectors->destroy_pipeline(flownet.opt);
        delete rife_flow_scale_vectors;
    }

    if (rife_flow_resize_output)
    {
        rife_flow_resize_output->destroy_pipeline(flownet.opt);
        delete rife_flow_resize_output;
    }

    if (rife_flow_double_vectors)
    {
        rife_flow_double_vectors->destroy_pipeline(flownet.opt);
        delete rife_flow_double_vectors;
    }

    if (rife_v2)
    {
        rife_v2_slice_flow->destroy_pipeline(flownet.opt);
        delete rife_v2_slice_flow;
    }
}

#if _WIN32
static void load_param_model(ncnn::Net& net, const std::wstring& modeldir, const wchar_t* name)
{
    wchar_t parampath[256];
    wchar_t modelpath[256];
    swprintf(parampath, 256, L"%s/%s.param", modeldir.c_str(), name);
    swprintf(modelpath, 256, L"%s/%s.bin", modeldir.c_str(), name);

    {
        FILE* fp = _wfopen(parampath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", parampath);
        }

        net.load_param(fp);

        fclose(fp);
    }
    {
        FILE* fp = _wfopen(modelpath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", modelpath);
        }

        net.load_model(fp);

        fclose(fp);
    }
}
#else
static void load_param_model(ncnn::Net& net, const std::string& modeldir, const char* name)
{
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s/%s.param", modeldir.c_str(), name);
    sprintf(modelpath, "%s/%s.bin", modeldir.c_str(), name);

    net.load_param(parampath);
    net.load_model(modelpath);
}
#endif

static bool starts_with(const std::string& value, const char* prefix)
{
    const auto prefix_len = std::strlen(prefix);
    return value.size() >= prefix_len && value.compare(0, prefix_len, prefix) == 0;
}

static std::string detect_v4_flow_blob_name(const ncnn::Net& flownet)
{
    const auto& blobs = flownet.blobs();
    const auto& layers = flownet.layers();
    if (blobs.empty() || layers.empty())
        return {};

    auto is_valid_blob = [&](const int blob_idx) {
        return blob_idx >= 0 && blob_idx < static_cast<int>(blobs.size());
    };
    auto is_valid_layer = [&](const int layer_idx) {
        return layer_idx >= 0 && layer_idx < static_cast<int>(layers.size()) && layers[layer_idx];
    };

    auto resolve_flow_blob_from_warp_layer = [&](const int warp_layer_idx) {
        if (!is_valid_layer(warp_layer_idx))
            return -1;

        const auto* const warp_layer = layers[warp_layer_idx];
        if (warp_layer->type != "rife.Warp" || warp_layer->bottoms.size() < 2)
            return -1;

        const auto image_blob_idx = warp_layer->bottoms[0];
        if (!is_valid_blob(image_blob_idx))
            return -1;

        const auto& image_blob_name = blobs[image_blob_idx].name;
        if (!starts_with(image_blob_name, "in0") && !starts_with(image_blob_name, "in1"))
            return -1;

        const auto flow_slice_blob_idx = warp_layer->bottoms[1];
        if (!is_valid_blob(flow_slice_blob_idx))
            return -1;

        const auto crop_layer_idx = blobs[flow_slice_blob_idx].producer;
        if (!is_valid_layer(crop_layer_idx))
            return -1;

        const auto* const crop_layer = layers[crop_layer_idx];
        if (crop_layer->type != "Crop" || crop_layer->bottoms.empty())
            return -1;

        const auto split_output_blob_idx = crop_layer->bottoms[0];
        if (!is_valid_blob(split_output_blob_idx))
            return -1;

        const auto split_layer_idx = blobs[split_output_blob_idx].producer;
        if (!is_valid_layer(split_layer_idx))
            return -1;

        const auto* const split_layer = layers[split_layer_idx];
        if (split_layer->type != "Split" || split_layer->bottoms.empty())
            return -1;

        return split_layer->bottoms[0];
    };

    // Primary path: infer the flow blob from the two branches that feed out0.
    int out0_blob_idx = -1;
    for (int i = 0; i < static_cast<int>(blobs.size()); i++)
    {
        if (blobs[i].name == "out0")
        {
            out0_blob_idx = i;
            break;
        }
    }

    auto resolve_branch_flow_blob = [&](const int blend_branch_blob_idx) {
        if (!is_valid_blob(blend_branch_blob_idx))
            return -1;

        const auto mul_layer_idx = blobs[blend_branch_blob_idx].producer;
        if (!is_valid_layer(mul_layer_idx))
            return -1;

        const auto* const mul_layer = layers[mul_layer_idx];
        for (const auto mul_input_blob_idx : mul_layer->bottoms)
        {
            if (!is_valid_blob(mul_input_blob_idx))
                continue;

            const auto warp_layer_idx = blobs[mul_input_blob_idx].producer;
            const auto flow_blob_idx = resolve_flow_blob_from_warp_layer(warp_layer_idx);
            if (is_valid_blob(flow_blob_idx))
                return flow_blob_idx;
        }

        return -1;
    };

    if (out0_blob_idx >= 0)
    {
        const auto final_layer_idx = blobs[out0_blob_idx].producer;
        if (is_valid_layer(final_layer_idx))
        {
            const auto* const final_layer = layers[final_layer_idx];
            if (final_layer->bottoms.size() >= 2)
            {
                const auto flow_blob_from_first_branch = resolve_branch_flow_blob(final_layer->bottoms[0]);
                const auto flow_blob_from_second_branch = resolve_branch_flow_blob(final_layer->bottoms[1]);

                if (is_valid_blob(flow_blob_from_first_branch) &&
                    is_valid_blob(flow_blob_from_second_branch) &&
                    flow_blob_from_first_branch == flow_blob_from_second_branch)
                {
                    return blobs[flow_blob_from_first_branch].name;
                }

                if (is_valid_blob(flow_blob_from_first_branch))
                    return blobs[flow_blob_from_first_branch].name;

                if (is_valid_blob(flow_blob_from_second_branch))
                    return blobs[flow_blob_from_second_branch].name;
            }
        }
    }

    // Fallback path for graphs where out0 is produced by a post-processed RGB head.
    for (int layer_idx = static_cast<int>(layers.size()) - 1; layer_idx >= 0; layer_idx--)
    {
        const auto flow_blob_idx = resolve_flow_blob_from_warp_layer(layer_idx);
        if (is_valid_blob(flow_blob_idx))
            return blobs[flow_blob_idx].name;
    }

    return {};
}

#if _WIN32
int RIFE::load(const std::wstring& modeldir)
#else
int RIFE::load(const std::string& modeldir)
#endif
{
    ncnn::Option opt;
    opt.num_threads = num_threads;
    opt.use_vulkan_compute = vkdev ? true : false;
    opt.use_fp16_packed = vkdev && !disable_vulkan_fp16;
    opt.use_fp16_storage = vkdev && !disable_vulkan_fp16;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;

    flownet.opt = opt;

    flownet.set_vulkan_device(vkdev);

    flownet.register_custom_layer("rife.Warp", Warp_layer_creator);

#if _WIN32
    load_param_model(flownet, modeldir, L"flownet");
#else
    load_param_model(flownet, modeldir, "flownet");
#endif

    if (rife_v4)
        rife_v4_flow_blob_name = detect_v4_flow_blob_name(flownet);

    // initialize preprocess pipeline
    if (vkdev)
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        {
            std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                    compile_spirv_module(rife_preproc_comp_data, static_cast<int>(sizeof(rife_preproc_comp_data) - 1), opt, spirv);
            }

            rife_preproc = new ncnn::Pipeline(vkdev);
            rife_preproc->set_optimal_local_size_xyz(8, 8, 3);
            rife_preproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                    compile_spirv_module(rife_mv_reduce_comp_data, static_cast<int>(sizeof(rife_mv_reduce_comp_data) - 1), opt, spirv);
            }

            rife_mv_reduce = new ncnn::Pipeline(vkdev);
            rife_mv_reduce->set_optimal_local_size_xyz(64, 1, 1);
            std::vector<ncnn::vk_specialization_type> reduceSpecializations;
            rife_mv_reduce->create(spirv.data(), spirv.size() * 4, reduceSpecializations);
        }

    }

    if (use_flow_scale)
    {
        {
            rife_flow_scale_image = ncnn::create_layer("Interp");
            rife_flow_scale_image->vkdev = vkdev;

            ncnn::ParamDict pd;
            pd.set(0, 2);// bilinear
            pd.set(1, flow_scale);
            pd.set(2, flow_scale);
            rife_flow_scale_image->load_param(pd);

            rife_flow_scale_image->create_pipeline(opt);
        }
        {
            rife_flow_resize_flow = ncnn::create_layer("Interp");
            rife_flow_resize_flow->vkdev = vkdev;

            ncnn::ParamDict pd;
            pd.set(0, 2);// bilinear
            pd.set(1, flow_scale_inv);
            pd.set(2, flow_scale_inv);
            rife_flow_resize_flow->load_param(pd);

            rife_flow_resize_flow->create_pipeline(opt);
        }
        {
            rife_flow_scale_vectors = ncnn::create_layer("BinaryOp");
            rife_flow_scale_vectors->vkdev = vkdev;

            ncnn::ParamDict pd;
            pd.set(0, 2);// mul
            pd.set(1, 1);// with_scalar
            pd.set(2, flow_scale_inv);// b
            rife_flow_scale_vectors->load_param(pd);

            rife_flow_scale_vectors->create_pipeline(opt);
        }
    }

    if (vkdev)
    {
        rife_flow_resize_output = ncnn::create_layer("Interp");
        rife_flow_resize_output->vkdev = vkdev;
        {
            ncnn::ParamDict pd;
            pd.set(0, 2);// bilinear
            pd.set(1, 2.f);
            pd.set(2, 2.f);
            rife_flow_resize_output->load_param(pd);
        }
        rife_flow_resize_output->create_pipeline(opt);

        rife_flow_double_vectors = ncnn::create_layer("BinaryOp");
        rife_flow_double_vectors->vkdev = vkdev;
        {
            ncnn::ParamDict pd;
            pd.set(0, 2);// mul
            pd.set(1, 1);// with_scalar
            pd.set(2, 2.f);
            rife_flow_double_vectors->load_param(pd);
        }
        rife_flow_double_vectors->create_pipeline(opt);
    }

    if (rife_v2)
    {
        {
            rife_v2_slice_flow = ncnn::create_layer("Slice");
            rife_v2_slice_flow->vkdev = vkdev;

            ncnn::Mat slice_points(2);
            slice_points.fill<int>(-233);

            ncnn::ParamDict pd;
            pd.set(0, slice_points);
            pd.set(1, 0);// axis

            rife_v2_slice_flow->load_param(pd);

            rife_v2_slice_flow->create_pipeline(opt);
        }
    }

    if (rife_v4)
    {
        if (vkdev)
        {
            std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    compile_spirv_module(rife_v4_timestep_comp_data, sizeof(rife_v4_timestep_comp_data), opt, spirv);
                }
            }

            std::vector<ncnn::vk_specialization_type> specializations;

            rife_v4_timestep = new ncnn::Pipeline(vkdev);
            rife_v4_timestep->set_optimal_local_size_xyz(8, 8, 1);
            rife_v4_timestep->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    return 0;
}

static float convert_fp16_to_float32(const uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t mantissa = value & 0x03ffu;
    uint32_t bits{};

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            bits = sign;
        }
        else
        {
            auto normalized_mantissa = mantissa;
            int adjusted_exponent = -14;
            while ((normalized_mantissa & 0x0400u) == 0)
            {
                normalized_mantissa <<= 1;
                adjusted_exponent -= 1;
            }

            normalized_mantissa &= 0x03ffu;
            bits = sign | (static_cast<uint32_t>(adjusted_exponent + 127) << 23) | (normalized_mantissa << 13);
        }
    }
    else if (exponent == 0x1fu)
    {
        bits = sign | 0x7f800000u | (mantissa << 13);
    }
    else
    {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static float read_ncnn_scalar(const unsigned char* const data, const size_t index, const int scalar_size)
{
    if (scalar_size == static_cast<int>(sizeof(float)))
    {
        float value;
        std::memcpy(&value, data + index * sizeof(float), sizeof(value));
        return value;
    }

    uint16_t value;
    std::memcpy(&value, data + index * sizeof(uint16_t), sizeof(value));
    return convert_fp16_to_float32(value);
}

struct BilinearAxisEntry {
    int index0;
    int index1;
    float alpha;
};

static void build_bilinear_axis_table(const int src_size, const int virtual_out_size, const int dst_size,
                                      std::vector<BilinearAxisEntry>& table) {
    table.resize(dst_size);
    const auto scale = src_size / static_cast<float>(virtual_out_size);
    for (int i = 0; i < dst_size; i++)
    {
        auto sample = (i + 0.5f) * scale - 0.5f;
        sample = std::max(0.0f, std::min(sample, static_cast<float>(src_size - 1)));
        const auto i0 = static_cast<int>(std::floor(sample));
        const auto i1 = std::min(i0 + 1, src_size - 1);
        table[i] = { i0, i1, sample - i0 };
    }
}

static int unpack_flow_channels(const ncnn::Mat& flow_cpu, ncnn::Mat& flow_cpu_unpacked)
{
    const auto scalar_size = flow_cpu.elempack > 0 ? static_cast<int>(flow_cpu.elemsize / flow_cpu.elempack) : 0;
    if (flow_cpu.elempack < 1 || (scalar_size != static_cast<int>(sizeof(float)) && scalar_size != static_cast<int>(sizeof(uint16_t))))
        return -1;

    if (flow_cpu.elempack == 1 && scalar_size == static_cast<int>(sizeof(float)))
    {
        flow_cpu_unpacked = flow_cpu;
        return 0;
    }

    const int ep = flow_cpu.elempack;
    const int actual_c = flow_cpu.c * ep;
    flow_cpu_unpacked.create(flow_cpu.w, flow_cpu.h, actual_c, sizeof(float));
    const auto pixel_count = static_cast<size_t>(flow_cpu.w) * flow_cpu.h;
    for (int cg = 0; cg < flow_cpu.c; cg++)
    {
        const auto* packed = static_cast<const unsigned char*>(flow_cpu.channel(cg));
        for (int ep_idx = 0; ep_idx < ep; ep_idx++)
        {
            auto* dst = static_cast<float*>(flow_cpu_unpacked.channel(cg * ep + ep_idx));
            for (size_t i = 0; i < pixel_count; i++)
                dst[i] = read_ncnn_scalar(packed, i * ep + ep_idx, scalar_size);
        }
    }

    return 0;
}

static void copy_scalar_channel_to_flow_output(const unsigned char* const packed, const int scalar_size, const int elempack,
                                               const int pack_index, const int src_w, float* dst, const int dst_w, const int dst_h) noexcept
{
    if (scalar_size == static_cast<int>(sizeof(float)))
    {
        const auto* src = reinterpret_cast<const float*>(packed) + pack_index;
        for (int y = 0; y < dst_h; y++)
        {
            const auto* src_row = src + static_cast<size_t>(y) * src_w * elempack;
            auto* dst_row = dst + static_cast<size_t>(y) * dst_w;
            for (int x = 0; x < dst_w; x++)
                dst_row[x] = src_row[static_cast<size_t>(x) * elempack];
        }
        return;
    }

    const auto* src = reinterpret_cast<const uint16_t*>(packed) + pack_index;
    for (int y = 0; y < dst_h; y++)
    {
        const auto* src_row = src + static_cast<size_t>(y) * src_w * elempack;
        auto* dst_row = dst + static_cast<size_t>(y) * dst_w;
        for (int x = 0; x < dst_w; x++)
            dst_row[x] = convert_fp16_to_float32(src_row[static_cast<size_t>(x) * elempack]);
    }
}

static void copy_first_four_packed_scalars_to_flow_output(const unsigned char* const packed, const int scalar_size, const int elempack,
                                                          const int src_w, float* flow_out, const int dst_w, const int dst_h) noexcept
{
    const auto plane_size = static_cast<size_t>(dst_w) * dst_h;
    auto* dst0 = flow_out;
    auto* dst1 = flow_out + plane_size;
    auto* dst2 = flow_out + plane_size * 2;
    auto* dst3 = flow_out + plane_size * 3;

    if (scalar_size == static_cast<int>(sizeof(float)))
    {
        const auto* src = reinterpret_cast<const float*>(packed);
        for (int y = 0; y < dst_h; y++)
        {
            const auto* src_row = src + static_cast<size_t>(y) * src_w * elempack;
            const auto dst_row = static_cast<size_t>(y) * dst_w;
            for (int x = 0; x < dst_w; x++)
            {
                const auto* src_pixel = src_row + static_cast<size_t>(x) * elempack;
                const auto dst_index = dst_row + x;
                dst0[dst_index] = src_pixel[0];
                dst1[dst_index] = src_pixel[1];
                dst2[dst_index] = src_pixel[2];
                dst3[dst_index] = src_pixel[3];
            }
        }
        return;
    }

    const auto* src = reinterpret_cast<const uint16_t*>(packed);
    for (int y = 0; y < dst_h; y++)
    {
        const auto* src_row = src + static_cast<size_t>(y) * src_w * elempack;
        const auto dst_row = static_cast<size_t>(y) * dst_w;
        for (int x = 0; x < dst_w; x++)
        {
            const auto* src_pixel = src_row + static_cast<size_t>(x) * elempack;
            const auto dst_index = dst_row + x;
            dst0[dst_index] = convert_fp16_to_float32(src_pixel[0]);
            dst1[dst_index] = convert_fp16_to_float32(src_pixel[1]);
            dst2[dst_index] = convert_fp16_to_float32(src_pixel[2]);
            dst3[dst_index] = convert_fp16_to_float32(src_pixel[3]);
        }
    }
}

static int copy_flow_output_direct_from_ncnn(const ncnn::Mat& flow_cpu, float* flow_out, const int w, const int h)
{
    const auto scalar_size = flow_cpu.elempack > 0 ? static_cast<int>(flow_cpu.elemsize / flow_cpu.elempack) : 0;
    if (flow_cpu.elempack < 1 || (scalar_size != static_cast<int>(sizeof(float)) && scalar_size != static_cast<int>(sizeof(uint16_t))))
        return -1;
    if (flow_cpu.c * flow_cpu.elempack < 4 || flow_cpu.w < w || flow_cpu.h < h)
        return -1;

    if (flow_cpu.elempack == 1 && scalar_size == static_cast<int>(sizeof(float)))
    {
        for (int c = 0; c < 4; c++)
        {
            const auto* src = static_cast<const float*>(flow_cpu.channel(c));
            auto* dst = flow_out + static_cast<size_t>(c) * w * h;
            for (int y = 0; y < h; y++)
                std::memcpy(dst + static_cast<size_t>(y) * w, src + static_cast<size_t>(y) * flow_cpu.w, static_cast<size_t>(w) * sizeof(float));
        }
        return 0;
    }

    if (flow_cpu.elempack >= 4)
    {
        const auto* packed = static_cast<const unsigned char*>(flow_cpu.channel(0));
        copy_first_four_packed_scalars_to_flow_output(packed, scalar_size, flow_cpu.elempack, flow_cpu.w, flow_out, w, h);
        return 0;
    }

    for (int c = 0; c < 4; c++)
    {
        const auto channel_index = c / flow_cpu.elempack;
        const auto pack_index = c - channel_index * flow_cpu.elempack;
        const auto* packed = static_cast<const unsigned char*>(flow_cpu.channel(channel_index));
        auto* dst = flow_out + static_cast<size_t>(c) * w * h;
        copy_scalar_channel_to_flow_output(packed, scalar_size, flow_cpu.elempack, pack_index, flow_cpu.w, dst, w, h);
    }

    return 0;
}

static int copy_flow_output_resized_cpu(const ncnn::Mat& flow_cpu_unpacked, float* flow_out, const int w, const int h)
{
    if (flow_cpu_unpacked.c < 4)
        return -1;

    const auto flow_w = flow_cpu_unpacked.w;
    const auto flow_h = flow_cpu_unpacked.h;
    const auto out_w = flow_w * 2;
    const auto out_h = flow_h * 2;

    struct AxisCache {
        int srcW{};
        int srcH{};
        int virtualOutW{};
        int virtualOutH{};
        int dstW{};
        int dstH{};
        std::vector<BilinearAxisEntry> xTable;
        std::vector<BilinearAxisEntry> yTable;
    };
    static thread_local AxisCache axisCache;
    if (axisCache.srcW != flow_w || axisCache.srcH != flow_h ||
        axisCache.virtualOutW != out_w || axisCache.virtualOutH != out_h ||
        axisCache.dstW != w || axisCache.dstH != h)
    {
        axisCache.srcW = flow_w;
        axisCache.srcH = flow_h;
        axisCache.virtualOutW = out_w;
        axisCache.virtualOutH = out_h;
        axisCache.dstW = w;
        axisCache.dstH = h;
        build_bilinear_axis_table(flow_w, out_w, w, axisCache.xTable);
        build_bilinear_axis_table(flow_h, out_h, h, axisCache.yTable);
    }

    for (int c = 0; c < 4; c++)
    {
        const auto* src = static_cast<const float*>(flow_cpu_unpacked.channel(c));
        auto* dst = flow_out + static_cast<size_t>(c) * w * h;
        for (int y = 0; y < h; y++)
        {
            const auto& yEntry = axisCache.yTable[y];
            const auto row0 = static_cast<size_t>(yEntry.index0) * flow_w;
            const auto row1 = static_cast<size_t>(yEntry.index1) * flow_w;
            for (int x = 0; x < w; x++)
            {
                const auto& xEntry = axisCache.xTable[x];
                const auto top = src[row0 + xEntry.index0] * (1.0f - xEntry.alpha) + src[row0 + xEntry.index1] * xEntry.alpha;
                const auto bottom = src[row1 + xEntry.index0] * (1.0f - xEntry.alpha) + src[row1 + xEntry.index1] * xEntry.alpha;
                dst[static_cast<size_t>(y) * w + x] = 2.0f * (top * (1.0f - yEntry.alpha) + bottom * yEntry.alpha);
            }
        }
    }

    return 0;
}

static int copy_reduced_flow_output_direct_from_ncnn(const ncnn::Mat& reduced_flow_cpu, RIFEReducedFlowBlock* reduced_flow_out,
                                                     const int block_count)
{
    if (reduced_flow_cpu.dims != 1 || reduced_flow_cpu.w < block_count ||
        reduced_flow_cpu.elempack != 4 || reduced_flow_cpu.elemsize != sizeof(RIFEReducedFlowBlock))
        return -1;

    std::memcpy(reduced_flow_out, reduced_flow_cpu.data, static_cast<size_t>(block_count) * sizeof(RIFEReducedFlowBlock));
    return 0;
}

static bool extract_v4_flow_blob(ncnn::Extractor& ex, ncnn::VkCompute& cmd, ncnn::VkMat& flow,
                                 const std::string& preferred_flow_blob_name)
{
    if (!preferred_flow_blob_name.empty())
    {
        if (ex.extract(preferred_flow_blob_name.c_str(), flow, cmd) == 0 && !flow.empty())
            return true;
    }

    static constexpr std::array<const char*, 8> flow_blob_names{
        "/Add_4_output_0",
        "/Add_3_output_0",
        "Add_4_output_0",
        "Add_3_output_0",
        "/Add_2_output_0",
        "Add_2_output_0",
        "flow",
        "/flow"
    };

    for (const auto* const blob_name : flow_blob_names)
    {
        if (ex.extract(blob_name, flow, cmd) == 0 && !flow.empty())
            return true;
    }

    return false;
}

static int64_t monotonic_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static void copy_plane_to_ncnn_input(const float* src, const ptrdiff_t src_stride, float* dst, const int w, const int h) noexcept
{
    if (src_stride == w)
    {
        std::memcpy(dst, src, static_cast<size_t>(w) * h * sizeof(float));
        return;
    }

    for (auto y = 0; y < h; y++)
        std::memcpy(dst + static_cast<size_t>(y) * w, src + src_stride * y, static_cast<size_t>(w) * sizeof(float));
}

static void scale_plane_to_ncnn_input(const float* src, const ptrdiff_t src_stride, float* dst, const int w, const int h) noexcept
{
    constexpr float inputScale = 255.0f;
    if (src_stride == w)
    {
        const auto pixelCount = static_cast<size_t>(w) * h;
        for (size_t i = 0; i < pixelCount; i++)
            dst[i] = src[i] * inputScale;
        return;
    }

    for (auto y = 0; y < h; y++)
    {
        const auto* srcRow = src + src_stride * y;
        auto* dstRow = dst + static_cast<size_t>(y) * w;
        for (auto x = 0; x < w; x++)
            dstRow[x] = srcRow[x] * inputScale;
    }
}

int RIFE::process_flow_internal(const float* src0R, const float* src0G, const float* src0B,
                                const float* src1R, const float* src1G, const float* src1B,
                                float* flow_out, RIFEReducedFlowBlock* reduced_flow_out, const RIFEFlowReduceConfig* reduce_config,
                                const int w, const int h, const ptrdiff_t stride,
                                FlowPerfBreakdown* perf) const
{
    const bool collect_perf = perf != nullptr;
    if (collect_perf)
        *perf = {};

    const auto setup_start_ns = collect_perf ? monotonic_now_ns() : 0;
    const int channels = 3;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();
    const auto finish = [&](const int status) {
        const auto cleanup_start_ns = collect_perf ? monotonic_now_ns() : 0;
        vkdev->reclaim_blob_allocator(blob_vkallocator);
        vkdev->reclaim_staging_allocator(staging_vkallocator);
        if (collect_perf)
            perf->cleanupNs += monotonic_now_ns() - cleanup_start_ns;

        return status;
    };

    ncnn::Option opt = flownet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    const auto w_padded = (w + padding - 1) / padding * padding;
    const auto h_padded = (h + padding - 1) / padding * padding;
    const auto in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::Mat in0;
    ncnn::Mat in1;
    in0.create(w, h, channels, sizeof(float), 1);
    in1.create(w, h, channels, sizeof(float), 1);
    auto* in0_r = static_cast<float*>(in0.channel(0));
    auto* in0_g = static_cast<float*>(in0.channel(1));
    auto* in0_b = static_cast<float*>(in0.channel(2));
    auto* in1_r = static_cast<float*>(in1.channel(0));
    auto* in1_g = static_cast<float*>(in1.channel(1));
    auto* in1_b = static_cast<float*>(in1.channel(2));
    if (collect_perf)
        perf->setupNs += monotonic_now_ns() - setup_start_ns;

    const auto cpu_prep_start_ns = collect_perf ? monotonic_now_ns() : 0;
    if (opt.use_int8_storage)
    {
        scale_plane_to_ncnn_input(src0R, stride, in0_r, w, h);
        scale_plane_to_ncnn_input(src0G, stride, in0_g, w, h);
        scale_plane_to_ncnn_input(src0B, stride, in0_b, w, h);
        scale_plane_to_ncnn_input(src1R, stride, in1_r, w, h);
        scale_plane_to_ncnn_input(src1G, stride, in1_g, w, h);
        scale_plane_to_ncnn_input(src1B, stride, in1_b, w, h);
    }
    else
    {
        copy_plane_to_ncnn_input(src0R, stride, in0_r, w, h);
        copy_plane_to_ncnn_input(src0G, stride, in0_g, w, h);
        copy_plane_to_ncnn_input(src0B, stride, in0_b, w, h);
        copy_plane_to_ncnn_input(src1R, stride, in1_r, w, h);
        copy_plane_to_ncnn_input(src1G, stride, in1_g, w, h);
        copy_plane_to_ncnn_input(src1B, stride, in1_b, w, h);
    }
    if (collect_perf)
        perf->cpuPrepNs += monotonic_now_ns() - cpu_prep_start_ns;

    ncnn::VkCompute cmd(vkdev);
    const auto command_record_start_ns = collect_perf ? monotonic_now_ns() : 0;

    ncnn::VkMat in0_gpu;
    ncnn::VkMat in1_gpu;
    const auto upload_record_start_ns = collect_perf ? monotonic_now_ns() : 0;
    cmd.record_clone(in0, in0_gpu, opt);
    cmd.record_clone(in1, in1_gpu, opt);
    if (collect_perf)
        perf->uploadRecordNs += monotonic_now_ns() - upload_record_start_ns;

    ncnn::VkMat in0_gpu_padded;
    ncnn::VkMat in1_gpu_padded;
    const auto preproc_record_start_ns = collect_perf ? monotonic_now_ns() : 0;
    {
        in0_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = in0_gpu;
        bindings[1] = in0_gpu_padded;

        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].i = in0_gpu.w;
        constants[1].i = in0_gpu.h;
        constants[2].i = in0_gpu.cstep;
        constants[3].i = in0_gpu_padded.w;
        constants[4].i = in0_gpu_padded.h;
        constants[5].i = in0_gpu_padded.cstep;

        cmd.record_pipeline(rife_preproc, bindings, constants, in0_gpu_padded);
    }
    {
        in1_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = in1_gpu;
        bindings[1] = in1_gpu_padded;

        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].i = in1_gpu.w;
        constants[1].i = in1_gpu.h;
        constants[2].i = in1_gpu.cstep;
        constants[3].i = in1_gpu_padded.w;
        constants[4].i = in1_gpu_padded.h;
        constants[5].i = in1_gpu_padded.cstep;

        cmd.record_pipeline(rife_preproc, bindings, constants, in1_gpu_padded);
    }
    if (collect_perf)
        perf->preprocRecordNs += monotonic_now_ns() - preproc_record_start_ns;

    ncnn::VkMat flow;
    {
        const auto inference_record_start_ns = collect_perf ? monotonic_now_ns() : 0;
        ncnn::Extractor ex = flownet.create_extractor();
        ex.set_blob_vkallocator(blob_vkallocator);
        ex.set_workspace_vkallocator(blob_vkallocator);
        ex.set_staging_vkallocator(staging_vkallocator);

        if (rife_v4)
        {
            ncnn::VkMat timestep_gpu_padded;
            timestep_gpu_padded.create(w_padded, h_padded, 1, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(1);
            bindings[0] = timestep_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].i = timestep_gpu_padded.w;
            constants[1].i = timestep_gpu_padded.h;
            constants[2].i = timestep_gpu_padded.cstep;
            constants[3].f = 0.5f;

            cmd.record_pipeline(rife_v4_timestep, bindings, constants, timestep_gpu_padded);

            ex.input("in0", in0_gpu_padded);
            ex.input("in1", in1_gpu_padded);
            ex.input("in2", timestep_gpu_padded);

            if (!extract_v4_flow_blob(ex, cmd, flow, rife_v4_flow_blob_name))
            {
                if (collect_perf)
                    perf->inferenceRecordNs += monotonic_now_ns() - inference_record_start_ns;
                return finish(-1);
            }
        }
        else if (use_flow_scale)
        {
            ncnn::VkMat in0_gpu_padded_downscaled;
            ncnn::VkMat in1_gpu_padded_downscaled;
            rife_flow_scale_image->forward(in0_gpu_padded, in0_gpu_padded_downscaled, cmd, opt);
            rife_flow_scale_image->forward(in1_gpu_padded, in1_gpu_padded_downscaled, cmd, opt);

            ex.input("input0", in0_gpu_padded_downscaled);
            ex.input("input1", in1_gpu_padded_downscaled);

            ncnn::VkMat flow_downscaled;
            ex.extract("flow", flow_downscaled, cmd);

            ncnn::VkMat flow_half;
            rife_flow_resize_flow->forward(flow_downscaled, flow_half, cmd, opt);
            rife_flow_scale_vectors->forward(flow_half, flow, cmd, opt);
        }
        else
        {
            ex.input("input0", in0_gpu_padded);
            ex.input("input1", in1_gpu_padded);
            ex.extract("flow", flow, cmd);
        }
        if (collect_perf)
            perf->inferenceRecordNs += monotonic_now_ns() - inference_record_start_ns;
    }

    ncnn::VkMat flow_readback_source = flow;
    ncnn::VkMat flow_staging;
    bool used_gpu_resize{};
    const bool flow_needs_resize = flow.w < w || flow.h < h;
    const bool can_try_gpu_resize = flow_needs_resize &&
                                    flow_resize_mode != FlowResizeMode::ForceCPU &&
                                    vkdev && rife_flow_resize_output && rife_flow_double_vectors;
    const bool require_gpu_resize = flow_needs_resize && flow_resize_mode == FlowResizeMode::ForceGPU;

    const auto output_record_start_ns = collect_perf ? monotonic_now_ns() : 0;
    if (can_try_gpu_resize)
    {
        ncnn::VkMat flow_resized_gpu;
        if (rife_flow_resize_output->forward(flow, flow_resized_gpu, cmd, opt) == 0)
        {
            ncnn::VkMat flow_scaled_gpu;
            if (rife_flow_double_vectors->forward(flow_resized_gpu, flow_scaled_gpu, cmd, opt) == 0)
            {
                flow_readback_source = flow_scaled_gpu;
                used_gpu_resize = true;
            }
        }
    }

    if (!used_gpu_resize && require_gpu_resize)
        return finish(-1);

    const auto reduced_output = reduce_config != nullptr;
    if (reduced_output)
    {
        if (!rife_mv_reduce || !reduced_flow_out || reduce_config->blockCountX <= 0 || reduce_config->blockCountY <= 0)
            return finish(-1);
        if (flow_readback_source.w < w || flow_readback_source.h < h || flow_readback_source.elempack != 4)
            return finish(-1);

        const auto reduce_record_start_ns = collect_perf ? monotonic_now_ns() : 0;
        const auto block_count = reduce_config->blockCountX * reduce_config->blockCountY;
        ncnn::VkMat reduced_flow_gpu;
        reduced_flow_gpu.create(block_count, sizeof(RIFEReducedFlowBlock), 4, blob_vkallocator);
        if (reduced_flow_gpu.empty())
            return finish(-1);

        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = flow_readback_source;
        bindings[1] = reduced_flow_gpu;

        std::vector<ncnn::vk_constant_type> constants(12);
        constants[0].i = flow_readback_source.w;
        constants[1].i = w;
        constants[2].i = h;
        constants[3].i = reduce_config->blockCountX;
        constants[4].i = reduce_config->blockCountY;
        constants[5].i = reduce_config->internalBlockSizeX;
        constants[6].i = reduce_config->internalBlockSizeY;
        constants[7].i = reduce_config->internalStepX;
        constants[8].i = reduce_config->internalStepY;
        constants[9].i = reduce_config->internalHPadding;
        constants[10].i = reduce_config->internalVPadding;
        constants[11].i = reduce_config->blockReduce;

        cmd.record_pipeline(rife_mv_reduce, bindings, constants, reduced_flow_gpu);
        flow_readback_source = reduced_flow_gpu;
        if (collect_perf)
            perf->flowReduceRecordNs += monotonic_now_ns() - reduce_record_start_ns;
    }

    ncnn::Option opt_staging = opt;
    opt_staging.blob_vkallocator = staging_vkallocator;
    const auto readback_record_start_ns = collect_perf ? monotonic_now_ns() : 0;
    cmd.record_clone(flow_readback_source, flow_staging, opt_staging);
    if (collect_perf)
        perf->readbackRecordNs += monotonic_now_ns() - readback_record_start_ns;
    if (flow_staging.empty())
        return finish(-1);
    if (collect_perf)
        perf->readbackBytes += static_cast<int64_t>(flow_staging.total() * flow_staging.elemsize);
    if (collect_perf)
        perf->outputRecordNs += monotonic_now_ns() - output_record_start_ns;
    if (collect_perf)
        perf->commandRecordNs += monotonic_now_ns() - command_record_start_ns;

    const auto submit_wait_start_ns = collect_perf ? monotonic_now_ns() : 0;
    if (cmd.submit_and_wait() != 0)
        return finish(-1);
    if (collect_perf)
        perf->submitWaitNs += monotonic_now_ns() - submit_wait_start_ns;

    if (!flow_staging.allocator || !flow_staging.data || !flow_staging.mapped_ptr())
        return finish(-1);
    flow_staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    flow_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    const auto readback_invalidate_start_ns = collect_perf ? monotonic_now_ns() : 0;
    if (flow_staging.allocator->invalidate(flow_staging.data) != 0)
        return finish(-1);
    if (collect_perf)
        perf->readbackInvalidateNs += monotonic_now_ns() - readback_invalidate_start_ns;

    const auto readback_map_start_ns = collect_perf ? monotonic_now_ns() : 0;
    const auto flow_cpu = flow_staging.mapped();
    if (flow_cpu.empty())
        return finish(-1);
    if (collect_perf)
        perf->readbackMapNs += monotonic_now_ns() - readback_map_start_ns;

    int export_status{};
    if (reduced_output)
    {
        const auto export_start_ns = collect_perf ? monotonic_now_ns() : 0;
        export_status = copy_reduced_flow_output_direct_from_ncnn(flow_cpu, reduced_flow_out, reduce_config->blockCountX * reduce_config->blockCountY);
        if (collect_perf)
            perf->exportDirectNs += monotonic_now_ns() - export_start_ns;
    }
    else if (flow_cpu.w >= w && flow_cpu.h >= h)
    {
        const auto export_start_ns = collect_perf ? monotonic_now_ns() : 0;
        export_status = copy_flow_output_direct_from_ncnn(flow_cpu, flow_out, w, h);
        if (collect_perf)
            perf->exportDirectNs += monotonic_now_ns() - export_start_ns;
    }
    else
    {
        ncnn::Mat flow_cpu_unpacked;
        const auto unpack_start_ns = collect_perf ? monotonic_now_ns() : 0;
        if (unpack_flow_channels(flow_cpu, flow_cpu_unpacked) != 0)
            return finish(-1);
        if (collect_perf)
            perf->unpackNs += monotonic_now_ns() - unpack_start_ns;

        if (flow_cpu_unpacked.w * 2 >= w && flow_cpu_unpacked.h * 2 >= h)
        {
            const auto export_start_ns = collect_perf ? monotonic_now_ns() : 0;
            export_status = copy_flow_output_resized_cpu(flow_cpu_unpacked, flow_out, w, h);
            if (collect_perf)
                perf->exportResizeNs += monotonic_now_ns() - export_start_ns;
        }
        else
            export_status = -1;
    }
    if (export_status != 0)
        return finish(-1);

    return finish(0);
}

int RIFE::process_flow(const float* src0R, const float* src0G, const float* src0B,
                       const float* src1R, const float* src1G, const float* src1B,
                       float* flow_out, const int w, const int h, const ptrdiff_t stride,
                       FlowPerfBreakdown* perf) const
{
    return process_flow_internal(src0R, src0G, src0B, src1R, src1G, src1B, flow_out, nullptr, nullptr, w, h, stride, perf);
}

int RIFE::process_flow_reduced(const float* src0R, const float* src0G, const float* src0B,
                               const float* src1R, const float* src1G, const float* src1B,
                               RIFEReducedFlowBlock* reduced_flow_out, const RIFEFlowReduceConfig& reduce_config,
                               const int w, const int h, const ptrdiff_t stride,
                               FlowPerfBreakdown* perf) const
{
    return process_flow_internal(src0R, src0G, src0B, src1R, src1G, src1B, nullptr, reduced_flow_out, &reduce_config, w, h, stride, perf);
}
