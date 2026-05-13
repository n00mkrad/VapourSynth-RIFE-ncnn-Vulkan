// rife implemented with ncnn library

#ifndef RIFE_H
#define RIFE_H

#include <string>
#include <vector>
#include <cstdint>

// ncnn
#include "net.h"

enum class FlowResizeMode {
    Auto,
    ForceGPU,
    ForceCPU
};

struct FlowPerfBreakdown final {
    int64_t setupNs{};
    int64_t cpuPrepNs{};
    int64_t commandRecordNs{};
    int64_t uploadRecordNs{};
    int64_t preprocRecordNs{};
    int64_t inferenceRecordNs{};
    int64_t outputRecordNs{};
    int64_t flowReduceRecordNs{};
    int64_t flowVectorRecordNs{};
    int64_t readbackRecordNs{};
    int64_t submitWaitNs{};
    int64_t readbackInvalidateNs{};
    int64_t readbackMapNs{};
    int64_t unpackNs{};
    int64_t exportDirectNs{};
    int64_t exportResizeNs{};
    int64_t cleanupNs{};
    int64_t readbackBytes{};
};

struct RIFEFlowReduceConfig final {
    int blockCountX;
    int blockCountY;
    int internalBlockSizeX;
    int internalBlockSizeY;
    int internalStepX;
    int internalStepY;
    int internalHPadding;
    int internalVPadding;
    int blockReduce;
};

struct RIFEReducedFlowBlock final {
    float backwardX;
    float backwardY;
    float forwardX;
    float forwardY;
};

struct RIFEGpuMotionVectorConfig final {
    int blockCountX;
    int blockCountY;
    int blockSizeX;
    int blockSizeY;
    int stepX;
    int stepY;
    int hPadding;
    int vPadding;
    int internalBlockSizeX;
    int internalBlockSizeY;
    int internalStepX;
    int internalStepY;
    int internalHPadding;
    int internalVPadding;
    int pel;
    int blockReduce;
    int bits;
    int useChroma;
    float motionScaleX;
    float motionScaleY;
};

struct RIFEGpuMotionVector final {
    int32_t x;
    int32_t y;
    uint32_t rawSad;
    uint32_t reserved;
};

class RIFE
{
public:
    RIFE(int gpuid, float flow_scale = 1.f, int num_threads = 1, bool rife_v2 = false, bool rife_v4 = false,
         int padding = 32, FlowResizeMode flow_resize_mode = FlowResizeMode::Auto, bool disable_vulkan_fp16 = false,
         bool enable_gpu_mv_full = false);
    ~RIFE();

#if _WIN32
    int load(const std::wstring& modeldir);
#else
    int load(const std::string& modeldir);
#endif

    int process_flow(const float* src0R, const float* src0G, const float* src0B,
                     const float* src1R, const float* src1G, const float* src1B,
                     float* flow, const int w, const int h, const ptrdiff_t stride,
                     FlowPerfBreakdown* perf = nullptr) const;
    int process_flow_reduced(const float* src0R, const float* src0G, const float* src0B,
                             const float* src1R, const float* src1G, const float* src1B,
                             RIFEReducedFlowBlock* reducedFlow, const RIFEFlowReduceConfig& reduceConfig,
                             const int w, const int h, const ptrdiff_t stride,
                             FlowPerfBreakdown* perf = nullptr) const;
    int process_motion_vectors_gpu(const float* src0R, const float* src0G, const float* src0B,
                                   const float* src1R, const float* src1G, const float* src1B,
                                   RIFEGpuMotionVector* vectors, const RIFEGpuMotionVectorConfig& vectorConfig,
                                   const int w, const int h, const ptrdiff_t stride,
                                   FlowPerfBreakdown* perf = nullptr) const;

private:
    int process_flow_internal(const float* src0R, const float* src0G, const float* src0B,
                              const float* src1R, const float* src1G, const float* src1B,
                              float* flow, RIFEReducedFlowBlock* reducedFlow, const RIFEFlowReduceConfig* reduceConfig,
                              RIFEGpuMotionVector* vectors, const RIFEGpuMotionVectorConfig* vectorConfig,
                              const int w, const int h, const ptrdiff_t stride,
                              FlowPerfBreakdown* perf = nullptr) const;

    ncnn::VulkanDevice* vkdev;
    ncnn::Net flownet;
    ncnn::Pipeline* rife_preproc;
    ncnn::Pipeline* rife_v4_timestep;
    ncnn::Pipeline* rife_mv_reduce;
    ncnn::Pipeline* rife_mv_full;
    ncnn::Layer* rife_flow_scale_image;
    ncnn::Layer* rife_flow_resize_flow;
    ncnn::Layer* rife_flow_scale_vectors;
    ncnn::Layer* rife_flow_resize_output;
    ncnn::Layer* rife_flow_double_vectors;
    ncnn::Layer* rife_v2_slice_flow;
    bool use_flow_scale;
    float flow_scale;
    float flow_scale_inv;
    FlowResizeMode flow_resize_mode;
    int num_threads;
    bool rife_v2;
    bool rife_v4;
    bool disable_vulkan_fp16;
    bool enable_gpu_mv_full;
    int padding;
    std::string rife_v4_flow_blob_name;
};

#endif // RIFE_H
