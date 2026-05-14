// Copyright (c) 2021-2022 HolyWu
// SPDX-License-Identifier: MIT

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include "VapourSynth4.h"
#include "VSHelper4.h"

#include "rife.h"

using namespace std::literals;

static std::atomic<int> numGPUInstances{ 0 };

struct MotionVectorPerfStats final {
    std::atomic<int64_t> pairFrames{ 0 };
    std::atomic<int64_t> outputFrames{ 0 };
    std::atomic<int64_t> flowCalls{ 0 };
    std::atomic<int64_t> pairTotalNs{ 0 };
    std::atomic<int64_t> outputTotalNs{ 0 };
    std::atomic<int64_t> semaphoreWaitNs{ 0 };
    std::atomic<int64_t> localSemaphoreWaitNs{ 0 };
    std::atomic<int64_t> sharedSemaphoreWaitNs{ 0 };
    std::atomic<int64_t> processFlowNs{ 0 };
    std::atomic<int64_t> rifeProcessWallNs{ 0 };
    std::atomic<int64_t> flowSetupNs{ 0 };
    std::atomic<int64_t> flowCpuPrepNs{ 0 };
    std::atomic<int64_t> flowCommandRecordNs{ 0 };
    std::atomic<int64_t> flowUploadRecordNs{ 0 };
    std::atomic<int64_t> flowPreprocRecordNs{ 0 };
    std::atomic<int64_t> flowInferenceRecordNs{ 0 };
    std::atomic<int64_t> flowOutputRecordNs{ 0 };
    std::atomic<int64_t> flowReduceRecordNs{ 0 };
    std::atomic<int64_t> flowVectorRecordNs{ 0 };
    std::atomic<int64_t> flowReadbackRecordNs{ 0 };
    std::atomic<int64_t> flowSubmitWaitNs{ 0 };
    std::atomic<int64_t> flowReadbackInvalidateNs{ 0 };
    std::atomic<int64_t> flowReadbackMapNs{ 0 };
    std::atomic<int64_t> flowUnpackNs{ 0 };
    std::atomic<int64_t> flowExportDirectNs{ 0 };
    std::atomic<int64_t> flowExportResizeNs{ 0 };
    std::atomic<int64_t> flowCleanupNs{ 0 };
    std::atomic<int64_t> flowReadbackBytes{ 0 };
    std::atomic<int64_t> packedCacheHitCount{ 0 };
    std::atomic<int64_t> packedCacheMissCount{ 0 };
    std::atomic<int64_t> packedBuildNs{ 0 };
    std::atomic<int64_t> packedWaitNs{ 0 };
    std::atomic<int64_t> lumaBuildNs{ 0 };
    std::atomic<int64_t> vectorPackNs{ 0 };
    std::atomic<int64_t> renderSadMaskNs{ 0 };
    std::atomic<int64_t> displacementBuildNs{ 0 };
    std::atomic<int64_t> composeNs{ 0 };
};

struct MotionVectorLumaCacheEntry final {
    int frameNumber;
    int stride;
    int height;
    bool building;
    std::shared_ptr<const std::vector<float>> luma;
};

struct MotionVectorLumaCache final {
    std::mutex mutex;
    std::condition_variable condition;
    std::unordered_map<int, MotionVectorLumaCacheEntry> entries;
    std::deque<int> lru;
    size_t maxEntries;
};

struct MotionVectorPackedCacheEntry final {
    int frameNumber;
    int width;
    int height;
    bool building;
    std::shared_ptr<const ncnn::Mat> packed;
};

struct MotionVectorPackedCache final {
    std::mutex mutex;
    std::condition_variable condition;
    std::unordered_map<int, MotionVectorPackedCacheEntry> entries;
    std::deque<int> lru;
    size_t maxEntries;
};

struct SharedMotionVectorLumaCacheKey final {
    uintptr_t sourceIdentity;
    int width;
    int height;
    int bits;
    bool convertedFromYUV;
    std::string matrixIn;
    std::string rangeIn;

    bool operator==(const SharedMotionVectorLumaCacheKey& other) const noexcept {
        return sourceIdentity == other.sourceIdentity &&
               width == other.width &&
               height == other.height &&
               bits == other.bits &&
               convertedFromYUV == other.convertedFromYUV &&
               matrixIn == other.matrixIn &&
               rangeIn == other.rangeIn;
    }
};

struct SharedMotionVectorLumaCacheKeyHash final {
    size_t operator()(const SharedMotionVectorLumaCacheKey& key) const noexcept {
        size_t hash = std::hash<uintptr_t>{}(key.sourceIdentity);
        hash ^= std::hash<int>{}(key.width) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(key.height) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(key.bits) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<bool>{}(key.convertedFromYUV) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<std::string>{}(key.matrixIn) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<std::string>{}(key.rangeIn) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct SharedMotionVectorPackedCacheKey final {
    uintptr_t sourceIdentity;
    int inferenceWidth;
    int inferenceHeight;
    bool convertedFromYUV;
    std::string matrixIn;
    std::string rangeIn;

    bool operator==(const SharedMotionVectorPackedCacheKey& other) const noexcept {
        return sourceIdentity == other.sourceIdentity &&
               inferenceWidth == other.inferenceWidth &&
               inferenceHeight == other.inferenceHeight &&
               convertedFromYUV == other.convertedFromYUV &&
               matrixIn == other.matrixIn &&
               rangeIn == other.rangeIn;
    }
};

struct SharedMotionVectorPackedCacheKeyHash final {
    size_t operator()(const SharedMotionVectorPackedCacheKey& key) const noexcept {
        size_t hash = std::hash<uintptr_t>{}(key.sourceIdentity);
        hash ^= std::hash<int>{}(key.inferenceWidth) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(key.inferenceHeight) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<bool>{}(key.convertedFromYUV) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<std::string>{}(key.matrixIn) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<std::string>{}(key.rangeIn) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

namespace {

constexpr auto MVToolsAnalysisDataKey = "MVTools_MVAnalysisData";
constexpr auto MVToolsVectorsKey = "MVTools_vectors";
constexpr auto RIFEMVBackwardVectorsInternalKey = "_RMVBackwardVectors";
constexpr auto RIFEMVForwardVectorsInternalKey = "_RMVForwardVectors";
constexpr auto RIFEMVBackwardDisplacementInternalKey = "_RMVBackwardDisplacement";
constexpr auto RIFEMVForwardDisplacementInternalKey = "_RMVForwardDisplacement";
constexpr auto RIFEMVBackwardAvgSad8x8InternalKey = "_RMVBackwardAvgSad8x8";
constexpr auto RIFEMVForwardAvgSad8x8InternalKey = "_RMVForwardAvgSad8x8";
constexpr auto RIFEMVBackwardMaxSadInternalKey = "_RMVBackwardMaxSad";
constexpr auto RIFEMVForwardMaxSadInternalKey = "_RMVForwardMaxSad";
constexpr auto RIFEMVBackwardMinSadInternalKey = "_RMVBackwardMinSad";
constexpr auto RIFEMVForwardMinSadInternalKey = "_RMVForwardMinSad";
constexpr auto RIFEMVBackwardAvgAbsDxInternalKey = "_RMVBackwardAvgAbsDx";
constexpr auto RIFEMVForwardAvgAbsDxInternalKey = "_RMVForwardAvgAbsDx";
constexpr auto RIFEMVBackwardAvgAbsDyInternalKey = "_RMVBackwardAvgAbsDy";
constexpr auto RIFEMVForwardAvgAbsDyInternalKey = "_RMVForwardAvgAbsDy";
constexpr auto RIFEMVBackwardAvgAbsMotionInternalKey = "_RMVBackwardAvgAbsMotion";
constexpr auto RIFEMVForwardAvgAbsMotionInternalKey = "_RMVForwardAvgAbsMotion";
constexpr auto RIFEMVBackwardPanAmountInternalKey = "_RMVBackwardPanAmount";
constexpr auto RIFEMVForwardPanAmountInternalKey = "_RMVForwardPanAmount";
constexpr auto RIFEMVAvgSadKey = "RMV_AvgSad";
constexpr auto RIFEMVMaxSadKey = "RMV_MaxSad";
constexpr auto RIFEMVMinSadKey = "RMV_MinSad";
constexpr auto RIFEMVAvgAbsDxKey = "RMV_AvgAbsDx";
constexpr auto RIFEMVAvgAbsDyKey = "RMV_AvgAbsDy";
constexpr auto RIFEMVAvgAbsMotionKey = "RMV_AvgAbsMotion";
constexpr auto RIFEMVPanAmountKey = "RMV_PanAmount";
constexpr int MotionIsBackward = 0x00000002;
constexpr int MotionUseChromaMotion = 0x00000008;
constexpr int MVBlockReduceCenter = 0;
constexpr int MVBlockReduceAverage = 1;

using MVArraySizeType = int;

struct MVToolsVector final {
    int x;
    int y;
    int64_t sad;
};

struct MotionVectorScratchBuffers final {
    std::vector<float> flow;
    std::vector<float> backwardDisplacement;
    std::vector<float> forwardDisplacement;
    std::vector<float> composedX;
    std::vector<float> composedY;
    std::vector<RIFEReducedFlowBlock> reducedFlow;
    std::vector<RIFEGpuMotionVector> gpuVectors;
    std::vector<MVToolsVector> backwardVectors;
    std::vector<MVToolsVector> forwardVectors;
    std::vector<char> backwardBlob;
    std::vector<char> forwardBlob;
};

struct MVAnalysisData final {
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
};

struct MotionVectorFrameStats final {
    int64_t averageSad8x8;
    int64_t maxSad8x8;
    int64_t minSad8x8;
    double averageAbsDx;
    double averageAbsDy;
    double averageAbsMotion;
    double panAmount;
};

struct BilinearAxisEntry final {
    int index0;
    int index1;
    float alpha;
};

enum class MotionVectorSADMaskMode : uint8_t {
    Relative,
    Absolute,
};

enum class MotionVectorExportBackend : uint8_t {
    Cpu,
    GpuFlowReduce,
    GpuFull,
};

struct MotionVectorFrameStatsKeys final {
    const char* averageSad8x8;
    const char* maxSad;
    const char* minSad;
    const char* averageAbsDx;
    const char* averageAbsDy;
    const char* averageAbsMotion;
    const char* panAmount;
};

struct MotionVectorConfig final {
    bool useChroma;
    int blockSizeX;
    int blockSizeY;
    int overlapX;
    int overlapY;
    int stepX;
    int stepY;
    int internalBlockSizeX;
    int internalBlockSizeY;
    int internalOverlapX;
    int internalOverlapY;
    int internalStepX;
    int internalStepY;
    int pel;
    int delta;
    int bits;
    int hPadding;
    int vPadding;
    int internalHPadding;
    int internalVPadding;
    int blkX;
    int blkY;
    int inferenceWidth;
    int inferenceHeight;
    int blockReduce;
    float motionScaleX;
    float motionScaleY;
    double sadMultiplier;
    int64_t invalidSad;
    MVAnalysisData backwardAnalysisData;
    MVAnalysisData forwardAnalysisData;
};

struct MotionVectorInternalGeometry final {
    float motionScaleX;
    float motionScaleY;
    int inferenceWidth;
    int inferenceHeight;
    int internalBlockSizeX;
    int internalBlockSizeY;
    int internalOverlapX;
    int internalOverlapY;
    int internalStepX;
    int internalStepY;
    int internalHPadding;
    int internalVPadding;
};

struct ResolvedRIFEModel final {
    std::string modelPath;
    int padding;
    bool rifeV2;
    bool rifeV4;
    bool disableVulkanFp16;
};

constexpr auto RIFEMVModelRequirementError =
    "motion-vector export requires the rife-v3.1/rife-v3.9 model or a rife-v4.2+ model";
constexpr auto RIFEMVUnsupportedEarlyV4Error =
    "legacy rife-v4, rife-v4.0, and rife-v4.1 are not supported for motion-vector export; use rife-v4.2 or newer";

static_assert(sizeof(MVArraySizeType) == 4);
static_assert(sizeof(MVToolsVector) == 16);
static_assert(sizeof(RIFEGpuMotionVector) == 16);
static_assert(sizeof(MVAnalysisData) == 84);

static int64_t monotonicNowNs() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static void accumulatePerfStat(std::atomic<int64_t>& stat, const int64_t value) noexcept {
    stat.fetch_add(value, std::memory_order_relaxed);
}

static double nsToMs(const int64_t ns) noexcept {
    return static_cast<double>(ns) / 1'000'000.0;
}

static double bytesToMiB(const int64_t bytes) noexcept {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static int64_t roundPositiveAverageToInt64(const int64_t sum, const int64_t count) noexcept {
    if (count <= 0)
        return 0;

    return (sum + count / 2) / count;
}

static void printMotionVectorPerfSummary(const MotionVectorPerfStats& stats, const std::string& label) {
    const auto pairFrames = stats.pairFrames.load(std::memory_order_relaxed);
    const auto outputFrames = stats.outputFrames.load(std::memory_order_relaxed);
    const auto flowCalls = stats.flowCalls.load(std::memory_order_relaxed);
    const auto pairTotalNs = stats.pairTotalNs.load(std::memory_order_relaxed);
    const auto outputTotalNs = stats.outputTotalNs.load(std::memory_order_relaxed);
    const auto semaphoreWaitNs = stats.semaphoreWaitNs.load(std::memory_order_relaxed);
    const auto localSemaphoreWaitNs = stats.localSemaphoreWaitNs.load(std::memory_order_relaxed);
    const auto sharedSemaphoreWaitNs = stats.sharedSemaphoreWaitNs.load(std::memory_order_relaxed);
    const auto processFlowNs = stats.processFlowNs.load(std::memory_order_relaxed);
    const auto rifeProcessWallNs = stats.rifeProcessWallNs.load(std::memory_order_relaxed);
    const auto flowSetupNs = stats.flowSetupNs.load(std::memory_order_relaxed);
    const auto flowCpuPrepNs = stats.flowCpuPrepNs.load(std::memory_order_relaxed);
    const auto flowCommandRecordNs = stats.flowCommandRecordNs.load(std::memory_order_relaxed);
    const auto flowUploadRecordNs = stats.flowUploadRecordNs.load(std::memory_order_relaxed);
    const auto flowPreprocRecordNs = stats.flowPreprocRecordNs.load(std::memory_order_relaxed);
    const auto flowInferenceRecordNs = stats.flowInferenceRecordNs.load(std::memory_order_relaxed);
    const auto flowOutputRecordNs = stats.flowOutputRecordNs.load(std::memory_order_relaxed);
    const auto flowReduceRecordNs = stats.flowReduceRecordNs.load(std::memory_order_relaxed);
    const auto flowVectorRecordNs = stats.flowVectorRecordNs.load(std::memory_order_relaxed);
    const auto flowReadbackRecordNs = stats.flowReadbackRecordNs.load(std::memory_order_relaxed);
    const auto flowSubmitWaitNs = stats.flowSubmitWaitNs.load(std::memory_order_relaxed);
    const auto flowReadbackInvalidateNs = stats.flowReadbackInvalidateNs.load(std::memory_order_relaxed);
    const auto flowReadbackMapNs = stats.flowReadbackMapNs.load(std::memory_order_relaxed);
    const auto flowUnpackNs = stats.flowUnpackNs.load(std::memory_order_relaxed);
    const auto flowExportDirectNs = stats.flowExportDirectNs.load(std::memory_order_relaxed);
    const auto flowExportResizeNs = stats.flowExportResizeNs.load(std::memory_order_relaxed);
    const auto flowCleanupNs = stats.flowCleanupNs.load(std::memory_order_relaxed);
    const auto flowReadbackBytes = stats.flowReadbackBytes.load(std::memory_order_relaxed);
    const auto packedCacheHitCount = stats.packedCacheHitCount.load(std::memory_order_relaxed);
    const auto packedCacheMissCount = stats.packedCacheMissCount.load(std::memory_order_relaxed);
    const auto packedBuildNs = stats.packedBuildNs.load(std::memory_order_relaxed);
    const auto packedWaitNs = stats.packedWaitNs.load(std::memory_order_relaxed);
    const auto lumaBuildNs = stats.lumaBuildNs.load(std::memory_order_relaxed);
    const auto vectorPackNs = stats.vectorPackNs.load(std::memory_order_relaxed);
    const auto renderSadMaskNs = stats.renderSadMaskNs.load(std::memory_order_relaxed);
    const auto displacementBuildNs = stats.displacementBuildNs.load(std::memory_order_relaxed);
    const auto composeNs = stats.composeNs.load(std::memory_order_relaxed);
    const auto rifeProcessAccountedNs = flowSetupNs + flowCpuPrepNs + flowCommandRecordNs + flowSubmitWaitNs +
                                        flowReadbackInvalidateNs + flowReadbackMapNs + flowUnpackNs +
                                        flowExportDirectNs + flowExportResizeNs + flowCleanupNs;
    const auto rifeProcessUnaccountedNs = rifeProcessWallNs - rifeProcessAccountedNs;

    std::cerr << std::fixed << std::setprecision(3);
    std::cerr << "[rmv] perf_stats summary (" << label << ")\n";
    std::cerr << "  pair_frames=" << pairFrames
              << " pair_total_ms=" << nsToMs(pairTotalNs)
              << " pair_avg_ms=" << (pairFrames > 0 ? nsToMs(pairTotalNs) / pairFrames : 0.0) << '\n';
    std::cerr << "  output_frames=" << outputFrames
              << " output_total_ms=" << nsToMs(outputTotalNs)
              << " output_avg_ms=" << (outputFrames > 0 ? nsToMs(outputTotalNs) / outputFrames : 0.0) << '\n';
    std::cerr << "  flow_calls=" << flowCalls
              << " process_flow_ms=" << nsToMs(processFlowNs)
              << " process_flow_avg_ms=" << (flowCalls > 0 ? nsToMs(processFlowNs) / flowCalls : 0.0)
              << " rife_process_wall_ms=" << nsToMs(rifeProcessWallNs)
              << " rife_process_wall_avg_ms=" << (flowCalls > 0 ? nsToMs(rifeProcessWallNs) / flowCalls : 0.0)
              << " rife_process_unaccounted_ms=" << nsToMs(rifeProcessUnaccountedNs) << '\n';
    std::cerr << "  flow_setup_ms=" << nsToMs(flowSetupNs)
              << " flow_cpu_prep_ms=" << nsToMs(flowCpuPrepNs)
              << " flow_record_ms=" << nsToMs(flowCommandRecordNs)
              << " flow_submit_wait_ms=" << nsToMs(flowSubmitWaitNs)
              << " flow_unpack_ms=" << nsToMs(flowUnpackNs)
              << " flow_export_direct_ms=" << nsToMs(flowExportDirectNs)
              << " flow_export_resize_ms=" << nsToMs(flowExportResizeNs)
              << " flow_cleanup_ms=" << nsToMs(flowCleanupNs) << '\n';
    std::cerr << "  flow_upload_record_ms=" << nsToMs(flowUploadRecordNs)
              << " flow_preproc_record_ms=" << nsToMs(flowPreprocRecordNs)
              << " flow_inference_record_ms=" << nsToMs(flowInferenceRecordNs)
              << " flow_output_record_ms=" << nsToMs(flowOutputRecordNs)
              << " flow_reduce_record_ms=" << nsToMs(flowReduceRecordNs)
              << " flow_vector_record_ms=" << nsToMs(flowVectorRecordNs)
              << " flow_readback_record_ms=" << nsToMs(flowReadbackRecordNs) << '\n';
    std::cerr << "  flow_readback_mib=" << bytesToMiB(flowReadbackBytes)
              << " flow_readback_avg_mib=" << (flowCalls > 0 ? bytesToMiB(flowReadbackBytes) / flowCalls : 0.0)
              << " flow_readback_invalidate_ms=" << nsToMs(flowReadbackInvalidateNs)
              << " flow_readback_map_ms=" << nsToMs(flowReadbackMapNs) << '\n';
    std::cerr << "  flow_setup_avg_ms=" << (flowCalls > 0 ? nsToMs(flowSetupNs) / flowCalls : 0.0)
              << " flow_cpu_prep_avg_ms=" << (flowCalls > 0 ? nsToMs(flowCpuPrepNs) / flowCalls : 0.0)
              << " flow_record_avg_ms=" << (flowCalls > 0 ? nsToMs(flowCommandRecordNs) / flowCalls : 0.0)
              << " flow_reduce_record_avg_ms=" << (flowCalls > 0 ? nsToMs(flowReduceRecordNs) / flowCalls : 0.0)
              << " flow_vector_record_avg_ms=" << (flowCalls > 0 ? nsToMs(flowVectorRecordNs) / flowCalls : 0.0)
              << " flow_readback_record_avg_ms=" << (flowCalls > 0 ? nsToMs(flowReadbackRecordNs) / flowCalls : 0.0)
              << " flow_submit_wait_avg_ms=" << (flowCalls > 0 ? nsToMs(flowSubmitWaitNs) / flowCalls : 0.0)
              << " flow_readback_invalidate_avg_ms=" << (flowCalls > 0 ? nsToMs(flowReadbackInvalidateNs) / flowCalls : 0.0)
              << " flow_readback_map_avg_ms=" << (flowCalls > 0 ? nsToMs(flowReadbackMapNs) / flowCalls : 0.0)
              << " flow_unpack_avg_ms=" << (flowCalls > 0 ? nsToMs(flowUnpackNs) / flowCalls : 0.0)
              << " flow_export_direct_avg_ms=" << (flowCalls > 0 ? nsToMs(flowExportDirectNs) / flowCalls : 0.0)
              << " flow_export_resize_avg_ms=" << (flowCalls > 0 ? nsToMs(flowExportResizeNs) / flowCalls : 0.0)
              << " flow_cleanup_avg_ms=" << (flowCalls > 0 ? nsToMs(flowCleanupNs) / flowCalls : 0.0) << '\n';
    std::cerr << "  semaphore_wait_ms=" << nsToMs(semaphoreWaitNs)
              << " local_wait_ms=" << nsToMs(localSemaphoreWaitNs)
              << " shared_wait_ms=" << nsToMs(sharedSemaphoreWaitNs)
              << " packed_cache_hits=" << packedCacheHitCount
              << " packed_cache_misses=" << packedCacheMissCount
              << " packed_build_ms=" << nsToMs(packedBuildNs)
              << " packed_wait_ms=" << nsToMs(packedWaitNs)
              << " luma_build_ms=" << nsToMs(lumaBuildNs)
              << " vector_pack_ms=" << nsToMs(vectorPackNs)
              << " render_sad_mask_ms=" << nsToMs(renderSadMaskNs)
              << " displacement_build_ms=" << nsToMs(displacementBuildNs)
              << " compose_ms=" << nsToMs(composeNs) << std::endl;
    std::cerr << "  local_wait_avg_ms=" << (flowCalls > 0 ? nsToMs(localSemaphoreWaitNs) / flowCalls : 0.0)
              << " shared_wait_avg_ms=" << (flowCalls > 0 ? nsToMs(sharedSemaphoreWaitNs) / flowCalls : 0.0)
              << " render_sad_mask_avg_ms=" << (outputFrames > 0 ? nsToMs(renderSadMaskNs) / outputFrames : 0.0) << std::endl;
}

static const char* flowResizeModeName(const FlowResizeMode mode) noexcept {
    switch (mode) {
    case FlowResizeMode::Auto:
        return "auto";
    case FlowResizeMode::ForceCPU:
        return "force_cpu";
    case FlowResizeMode::ForceGPU:
        return "force_gpu";
    default:
        return "unknown";
    }
}

static const char* motionVectorExportBackendName(const MotionVectorExportBackend backend) noexcept {
    switch (backend) {
    case MotionVectorExportBackend::Cpu:
        return "cpu";
    case MotionVectorExportBackend::GpuFlowReduce:
        return "gpu_flow_reduce";
    case MotionVectorExportBackend::GpuFull:
        return "gpu_full";
    default:
        return "unknown";
    }
}

static MotionVectorExportBackend parseMotionVectorExportBackend(const int value) {
    switch (value) {
    case 0:
        return MotionVectorExportBackend::Cpu;
    case 1:
        return MotionVectorExportBackend::GpuFlowReduce;
    case 2:
        return MotionVectorExportBackend::GpuFull;
    default:
        throw "gpu_mode must be 0 (cpu), 1 (gpu_flow_reduce), or 2 (gpu_full)";
    }
}

static void printMotionVectorInvocation(const char* const functionName, const int gpuId, const int gpuThread,
                                        const int sharedFlowInFlight, const bool sharedLumaCache, const float flowScale,
                                        const FlowResizeMode flowResizeMode, const MotionVectorExportBackend mvExportBackend,
                                        const bool sharedPackedCache, const int packedCacheMiB,
                                        const bool sadStats, const bool motionStats, const bool perfStats,
                                        const MotionVectorConfig& config, const float resScale,
                                        const int inferenceWidth, const int inferenceHeight,
                                        const int absSadClipRange, const bool renderSadMask,
                                        const char* const matrixIn, const char* const rangeIn,
                                        const bool includeDelta) {
    std::ostringstream message;
    message << std::boolalpha
            << "[rmv] " << functionName << " parameters: gpu_id=" << gpuId
            << " gpu_thread=" << gpuThread
            << " shared_flow_inflight=" << sharedFlowInFlight
            << " shared_luma_cache=" << sharedLumaCache
            << " shared_packed_cache=" << sharedPackedCache
            << " packed_cache_mib=" << packedCacheMiB
            << " flow_scale=" << flowScale
            << " res_scale=" << resScale
            << " cpu_flow_resize=" << flowResizeModeName(flowResizeMode)
            << " gpu_mode=" << static_cast<int>(mvExportBackend)
            << " gpu_mode_name=" << motionVectorExportBackendName(mvExportBackend)
            << " sad_stats=" << sadStats
            << " motion_stats=" << motionStats
            << " perf_stats=" << perfStats
            << " blksize_x=" << config.blockSizeX
            << " blksize_y=" << config.blockSizeY
            << " overlap_x=" << config.overlapX
            << " overlap_y=" << config.overlapY
            << " pel=" << config.pel;
    if (includeDelta)
        message << " delta=" << config.delta;
    message << " bits=" << config.bits
            << " sad_multiplier=" << config.sadMultiplier
            << " abs_sad_clip_range=" << absSadClipRange
            << " render_sad_mask=" << renderSadMask
            << " matrix_in_s=" << matrixIn
            << " range_in_s=" << rangeIn
            << " hpad=" << config.hPadding
            << " vpad=" << config.vPadding
            << " block_reduce=" << config.blockReduce
            << " chroma=" << config.useChroma
            << " inference_width=" << inferenceWidth
            << " inference_height=" << inferenceHeight;
    std::cerr << message.str() << std::endl;
}

static double computeMotionVectorSadThresholdScale(const MVAnalysisData& analysisData) noexcept {
    auto scale = static_cast<double>(analysisData.nBlkSizeX) * static_cast<double>(analysisData.nBlkSizeY) / 64.0;
    if (analysisData.nMotionFlags & MotionUseChromaMotion)
        scale *= 1.0 + 2.0 / static_cast<double>(analysisData.xRatioUV * analysisData.yRatioUV);

    scale *= static_cast<double>((1ULL << analysisData.bitsPerSample) - 1ULL) / 255.0;
    return scale;
}

static int64_t normalizeMotionVectorSadTo8x8(const int64_t sad, const MVAnalysisData& analysisData) noexcept {
    const auto scale = computeMotionVectorSadThresholdScale(analysisData);
    if (scale <= 0.0)
        return sad;

    return static_cast<int64_t>(static_cast<long double>(sad) / scale + 0.5L);
}

static MotionVectorFrameStats computeMotionVectorFrameStats(const std::vector<MVToolsVector>& vectors,
                                                            const MVAnalysisData& analysisData,
                                                            const bool includeSadStats,
                                                            const bool includeMotionStats) noexcept {
    MotionVectorFrameStats stats{};
    if (vectors.empty() || (!includeSadStats && !includeMotionStats))
        return stats;

    int64_t sad8x8Sum{};
    int64_t absDxSum{};
    int64_t absDySum{};
    int64_t maxSad8x8{};
    int64_t minSad8x8{ std::numeric_limits<int64_t>::max() };
    double absMotionSum{};
    std::vector<int> signedDxs;
    std::vector<int> signedDys;
    if (includeMotionStats) {
        signedDxs.reserve(vectors.size());
        signedDys.reserve(vectors.size());
    }
    for (const auto& vector : vectors) {
        if (includeSadStats) {
            const auto sad8x8 = normalizeMotionVectorSadTo8x8(vector.sad, analysisData);
            sad8x8Sum += sad8x8;
            maxSad8x8 = std::max(maxSad8x8, sad8x8);
            minSad8x8 = std::min(minSad8x8, sad8x8);
        }
        if (includeMotionStats) {
            absDxSum += std::llabs(static_cast<int64_t>(vector.x));
            absDySum += std::llabs(static_cast<int64_t>(vector.y));
            absMotionSum += std::hypot(static_cast<double>(vector.x), static_cast<double>(vector.y));
            signedDxs.push_back(vector.x);
            signedDys.push_back(vector.y);
        }
    }

    const auto vectorCount = static_cast<int64_t>(vectors.size());
    if (includeSadStats) {
        stats.averageSad8x8 = roundPositiveAverageToInt64(sad8x8Sum, vectorCount);
        stats.maxSad8x8 = maxSad8x8;
        stats.minSad8x8 = minSad8x8 == std::numeric_limits<int64_t>::max() ? 0 : minSad8x8;
    }
    if (!includeMotionStats)
        return stats;

    stats.averageAbsDx = static_cast<double>(absDxSum) / static_cast<double>(vectorCount);
    stats.averageAbsDy = static_cast<double>(absDySum) / static_cast<double>(vectorCount);
    stats.averageAbsMotion = absMotionSum / static_cast<double>(vectorCount);

    const auto computeMedianComponent = [&](std::vector<int>& values) {
        const auto mid = values.begin() + static_cast<ptrdiff_t>(values.size() / 2);
        std::nth_element(values.begin(), mid, values.end());
        if ((values.size() & 1U) != 0)
            return static_cast<double>(*mid);

        const auto upper = *mid;
        const auto lowerMid = std::max_element(values.begin(), mid);
        return (static_cast<double>(*lowerMid) + static_cast<double>(upper)) * 0.5;
    };

    const auto pelScale = analysisData.nPel > 0 ? static_cast<double>(analysisData.nPel) : 1.0;
    const auto medianDx = computeMedianComponent(signedDxs) / pelScale;
    const auto medianDy = computeMedianComponent(signedDys) / pelScale;
    stats.panAmount = std::hypot(medianDx, medianDy);
    return stats;
}

static void setMotionVectorProperties(VSMap* props, const MVAnalysisData& analysisData,
                                      const char* vectorBlob, const int vectorBlobSize,
                                      const MotionVectorFrameStats& stats,
                                      const bool includeSadStats, const bool includeMotionStats,
                                      const VSAPI* vsapi) {
    vsapi->mapSetData(props, MVToolsAnalysisDataKey, reinterpret_cast<const char*>(&analysisData), sizeof(analysisData), dtBinary, maReplace);
    vsapi->mapSetData(props, MVToolsVectorsKey, vectorBlob, vectorBlobSize, dtBinary, maReplace);
    if (includeSadStats) {
        vsapi->mapSetInt(props, RIFEMVAvgSadKey, stats.averageSad8x8, maReplace);
        vsapi->mapSetInt(props, RIFEMVMaxSadKey, stats.maxSad8x8, maReplace);
        vsapi->mapSetInt(props, RIFEMVMinSadKey, stats.minSad8x8, maReplace);
    }
    if (includeMotionStats) {
        vsapi->mapSetFloat(props, RIFEMVAvgAbsDxKey, stats.averageAbsDx, maReplace);
        vsapi->mapSetFloat(props, RIFEMVAvgAbsDyKey, stats.averageAbsDy, maReplace);
        vsapi->mapSetFloat(props, RIFEMVAvgAbsMotionKey, stats.averageAbsMotion, maReplace);
        vsapi->mapSetFloat(props, RIFEMVPanAmountKey, stats.panAmount, maReplace);
    }
}

static MotionVectorFrameStatsKeys getMotionVectorInternalFrameStatsKeys(const bool backward) noexcept {
    if (backward) {
        return MotionVectorFrameStatsKeys{
            RIFEMVBackwardAvgSad8x8InternalKey,
            RIFEMVBackwardMaxSadInternalKey,
            RIFEMVBackwardMinSadInternalKey,
            RIFEMVBackwardAvgAbsDxInternalKey,
            RIFEMVBackwardAvgAbsDyInternalKey,
            RIFEMVBackwardAvgAbsMotionInternalKey,
            RIFEMVBackwardPanAmountInternalKey
        };
    }

    return MotionVectorFrameStatsKeys{
        RIFEMVForwardAvgSad8x8InternalKey,
        RIFEMVForwardMaxSadInternalKey,
        RIFEMVForwardMinSadInternalKey,
        RIFEMVForwardAvgAbsDxInternalKey,
        RIFEMVForwardAvgAbsDyInternalKey,
        RIFEMVForwardAvgAbsMotionInternalKey,
        RIFEMVForwardPanAmountInternalKey
    };
}

static void setMotionVectorInternalFrameStats(VSMap* props, const MotionVectorFrameStats& stats,
                                              const bool includeSadStats, const bool includeMotionStats,
                                              const bool backward, const VSAPI* vsapi) {
    const auto keys = getMotionVectorInternalFrameStatsKeys(backward);
    if (includeSadStats) {
        vsapi->mapSetInt(props, keys.averageSad8x8, stats.averageSad8x8, maReplace);
        vsapi->mapSetInt(props, keys.maxSad, stats.maxSad8x8, maReplace);
        vsapi->mapSetInt(props, keys.minSad, stats.minSad8x8, maReplace);
    }
    if (includeMotionStats) {
        vsapi->mapSetFloat(props, keys.averageAbsDx, stats.averageAbsDx, maReplace);
        vsapi->mapSetFloat(props, keys.averageAbsDy, stats.averageAbsDy, maReplace);
        vsapi->mapSetFloat(props, keys.averageAbsMotion, stats.averageAbsMotion, maReplace);
        vsapi->mapSetFloat(props, keys.panAmount, stats.panAmount, maReplace);
    }
}

static MotionVectorFrameStats getMotionVectorInternalFrameStats(const VSMap* props, const bool backward,
                                                                const bool includeSadStats, const bool includeMotionStats,
                                                                const VSAPI* vsapi) noexcept {
    const auto keys = getMotionVectorInternalFrameStatsKeys(backward);
    MotionVectorFrameStats stats{};
    if (includeSadStats) {
        stats.averageSad8x8 = vsapi->mapGetInt(props, keys.averageSad8x8, 0, nullptr);
        stats.maxSad8x8 = vsapi->mapGetInt(props, keys.maxSad, 0, nullptr);
        stats.minSad8x8 = vsapi->mapGetInt(props, keys.minSad, 0, nullptr);
    }
    if (includeMotionStats) {
        stats.averageAbsDx = vsapi->mapGetFloat(props, keys.averageAbsDx, 0, nullptr);
        stats.averageAbsDy = vsapi->mapGetFloat(props, keys.averageAbsDy, 0, nullptr);
        stats.averageAbsMotion = vsapi->mapGetFloat(props, keys.averageAbsMotion, 0, nullptr);
        stats.panAmount = vsapi->mapGetFloat(props, keys.panAmount, 0, nullptr);
    }
    return stats;
}

static void deleteMotionVectorInternalFrameStats(VSMap* props, const VSAPI* vsapi) {
    for (const auto backward : { true, false }) {
        const auto keys = getMotionVectorInternalFrameStatsKeys(backward);
        vsapi->mapDeleteKey(props, keys.averageSad8x8);
        vsapi->mapDeleteKey(props, keys.maxSad);
        vsapi->mapDeleteKey(props, keys.minSad);
        vsapi->mapDeleteKey(props, keys.averageAbsDx);
        vsapi->mapDeleteKey(props, keys.averageAbsDy);
        vsapi->mapDeleteKey(props, keys.averageAbsMotion);
        vsapi->mapDeleteKey(props, keys.panAmount);
    }
}

static MotionVectorScratchBuffers& getMotionVectorScratchBuffers() noexcept {
    static thread_local MotionVectorScratchBuffers scratch;
    return scratch;
}

static std::shared_ptr<std::counting_semaphore<>> acquireSharedFlowSemaphore(const int gpuId, const int maxInFlight) {
    static std::mutex mutex;
    static std::unordered_map<uint64_t, std::weak_ptr<std::counting_semaphore<>>> semaphores;

    std::lock_guard<std::mutex> lock(mutex);
    const auto capacity = std::max(1, maxInFlight);
    const auto key = (static_cast<uint64_t>(static_cast<uint32_t>(gpuId)) << 32) |
                     static_cast<uint32_t>(capacity);
    auto it = semaphores.find(key);
    if (it != semaphores.end()) {
        if (auto existing = it->second.lock())
            return existing;
    }

    auto created = std::make_shared<std::counting_semaphore<>>(capacity);
    semaphores[key] = created;
    return created;
}

static std::shared_ptr<MotionVectorLumaCache> createMotionVectorLumaCache(const size_t maxEntries) {
    auto cache = std::make_shared<MotionVectorLumaCache>();
    cache->maxEntries = maxEntries;
    return cache;
}

static std::shared_ptr<MotionVectorLumaCache> acquireSharedLumaCache(const SharedMotionVectorLumaCacheKey& key,
                                                                     const size_t maxEntries) {
    static std::mutex mutex;
    static std::unordered_map<SharedMotionVectorLumaCacheKey,
                              std::weak_ptr<MotionVectorLumaCache>,
                              SharedMotionVectorLumaCacheKeyHash> caches;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = caches.find(key);
    if (it != caches.end()) {
        if (auto existing = it->second.lock())
            return existing;
    }

    auto created = createMotionVectorLumaCache(maxEntries);
    caches[key] = created;
    return created;
}

static size_t computePackedFrameBytes(const int width, const int height) {
    const auto planeBytes = static_cast<size_t>(width) * height * sizeof(float);
    const auto alignedPlaneBytes = (planeBytes + 15u) & ~static_cast<size_t>(15u);
    return alignedPlaneBytes * 3u;
}

static size_t computePackedCacheMaxEntries(const int width, const int height, const int packedCacheMiB) {
    constexpr size_t minEntries = 2;
    constexpr size_t maxEntries = 4096;
    const auto frameBytes = computePackedFrameBytes(width, height);
    const auto budgetBytes = static_cast<size_t>(packedCacheMiB) * 1024u * 1024u;
    if (frameBytes == 0)
        return minEntries;

    const auto entries = std::max(minEntries, budgetBytes / frameBytes);
    return std::min(entries, maxEntries);
}

static std::shared_ptr<MotionVectorPackedCache> createMotionVectorPackedCache(const size_t maxEntries) {
    auto cache = std::make_shared<MotionVectorPackedCache>();
    cache->maxEntries = maxEntries;
    return cache;
}

static std::shared_ptr<MotionVectorPackedCache> acquireSharedPackedCache(const SharedMotionVectorPackedCacheKey& key,
                                                                         const size_t maxEntries) {
    static std::mutex mutex;
    static std::unordered_map<SharedMotionVectorPackedCacheKey,
                              std::weak_ptr<MotionVectorPackedCache>,
                              SharedMotionVectorPackedCacheKeyHash> caches;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = caches.find(key);
    if (it != caches.end()) {
        if (auto existing = it->second.lock())
            return existing;
    }

    auto created = createMotionVectorPackedCache(maxEntries);
    caches[key] = created;
    return created;
}

static void touchMotionVectorPackedCacheEntry(const std::shared_ptr<MotionVectorPackedCache>& cache,
                                              const int frameNumber) {
    for (auto it = cache->lru.begin(); it != cache->lru.end(); ++it) {
        if (*it != frameNumber)
            continue;

        cache->lru.erase(it);
        break;
    }

    cache->lru.push_back(frameNumber);
}

static void pruneMotionVectorPackedCache(const std::shared_ptr<MotionVectorPackedCache>& cache) {
    while (cache->entries.size() > cache->maxEntries && !cache->lru.empty()) {
        const auto frameNumber = cache->lru.front();
        cache->lru.pop_front();
        const auto it = cache->entries.find(frameNumber);
        if (it == cache->entries.end())
            continue;

        if (it->second.building) {
            cache->lru.push_back(frameNumber);
            break;
        }

        cache->entries.erase(it);
    }
}

static void buildPackedInferenceFrame(const VSFrame* frame, const int width, const int height,
                                      const VSAPI* vsapi, ncnn::Mat& packed) noexcept {
    packed.create(width, height, 3, sizeof(float), 1);
    for (auto plane = 0; plane < 3; plane++) {
        const auto stride = static_cast<int>(vsapi->getStride(frame, plane) / vsapi->getVideoFrameFormat(frame)->bytesPerSample);
        const auto* src = reinterpret_cast<const float*>(vsapi->getReadPtr(frame, plane));
        auto planeMat = packed.channel(plane);
        auto* dst = static_cast<float*>(planeMat);
        if (stride == width) {
            std::memcpy(dst, src, static_cast<size_t>(width) * height * sizeof(float));
            continue;
        }

        for (auto y = 0; y < height; y++) {
            const auto offset = static_cast<size_t>(y) * width;
            std::memcpy(dst + offset, src + static_cast<size_t>(y) * stride, static_cast<size_t>(width) * sizeof(float));
        }
    }
}

static std::shared_ptr<const ncnn::Mat> getOrCreatePackedInferenceFrame(const std::shared_ptr<MotionVectorPackedCache>& cache,
                                                                         const VSFrame* frame, const int frameNumber,
                                                                         const int width, const int height,
                                                                         MotionVectorPerfStats* const perf,
                                                                         const VSAPI* vsapi) {
    if (!cache) {
        auto packed = std::make_shared<ncnn::Mat>();
        buildPackedInferenceFrame(frame, width, height, vsapi, *packed);
        return packed;
    }

    while (true) {
        std::unique_lock<std::mutex> lock(cache->mutex);
        auto& entry = cache->entries[frameNumber];

        if (entry.packed && !entry.building && entry.width == width && entry.height == height) {
            auto packed = entry.packed;
            touchMotionVectorPackedCacheEntry(cache, frameNumber);
            if (perf)
                accumulatePerfStat(perf->packedCacheHitCount, 1);
            return packed;
        }

        if (entry.building) {
            const auto waitStartNs = perf ? monotonicNowNs() : 0;
            cache->condition.wait(lock, [&]() {
                const auto it = cache->entries.find(frameNumber);
                return it == cache->entries.end() || !it->second.building;
            });
            if (perf)
                accumulatePerfStat(perf->packedWaitNs, monotonicNowNs() - waitStartNs);
            continue;
        }

        entry.frameNumber = frameNumber;
        entry.width = width;
        entry.height = height;
        entry.building = true;
        entry.packed.reset();
        if (perf)
            accumulatePerfStat(perf->packedCacheMissCount, 1);
        lock.unlock();

        auto packed = std::make_shared<ncnn::Mat>();
        const auto buildStartNs = perf ? monotonicNowNs() : 0;
        buildPackedInferenceFrame(frame, width, height, vsapi, *packed);
        if (perf)
            accumulatePerfStat(perf->packedBuildNs, monotonicNowNs() - buildStartNs);

        lock.lock();
        auto& stored = cache->entries[frameNumber];
        stored.frameNumber = frameNumber;
        stored.width = width;
        stored.height = height;
        stored.building = false;
        stored.packed = packed;
        touchMotionVectorPackedCacheEntry(cache, frameNumber);
        pruneMotionVectorPackedCache(cache);
        lock.unlock();
        cache->condition.notify_all();
        return packed;
    }
}

static int processFlowWithSemaphores(const RIFE* const rife,
                                     std::counting_semaphore<>* const localSemaphore,
                                     std::counting_semaphore<>* const sharedSemaphore,
                                     const float* src0R, const float* src0G, const float* src0B,
                                     const float* src1R, const float* src1G, const float* src1B,
                                     float* flowOut, const int width, const int height, const ptrdiff_t stride,
                                     int64_t* waitNs = nullptr,
                                     int64_t* localWaitNs = nullptr,
                                     int64_t* sharedWaitNs = nullptr,
                                     int64_t* rifeProcessWallNs = nullptr,
                                     FlowPerfBreakdown* flowPerf = nullptr) noexcept {
    int64_t localWait{};
    int64_t sharedWait{};
    if (localWaitNs || waitNs) {
        const auto localWaitStartNs = monotonicNowNs();
        localSemaphore->acquire();
        localWait = monotonicNowNs() - localWaitStartNs;
    } else {
        localSemaphore->acquire();
    }

    if (sharedSemaphore) {
        if (sharedWaitNs || waitNs) {
            const auto sharedWaitStartNs = monotonicNowNs();
            sharedSemaphore->acquire();
            sharedWait = monotonicNowNs() - sharedWaitStartNs;
        } else {
            sharedSemaphore->acquire();
        }
    }

    if (localWaitNs)
        *localWaitNs = localWait;
    if (sharedWaitNs)
        *sharedWaitNs = sharedWait;
    if (waitNs)
        *waitNs = localWait + sharedWait;

    const auto rifeProcessStartNs = rifeProcessWallNs ? monotonicNowNs() : 0;
    const auto status = rife->process_flow(src0R, src0G, src0B, src1R, src1G, src1B, flowOut, width, height, stride, flowPerf);
    if (rifeProcessWallNs)
        *rifeProcessWallNs = monotonicNowNs() - rifeProcessStartNs;

    if (sharedSemaphore)
        sharedSemaphore->release();
    localSemaphore->release();
    return status;
}

static int processFlowWithSemaphores(const RIFE* const rife,
                                     std::counting_semaphore<>* const localSemaphore,
                                     std::counting_semaphore<>* const sharedSemaphore,
                                     const ncnn::Mat& src0Packed, const ncnn::Mat& src1Packed,
                                     float* flowOut,
                                     int64_t* waitNs = nullptr,
                                     int64_t* localWaitNs = nullptr,
                                     int64_t* sharedWaitNs = nullptr,
                                     int64_t* rifeProcessWallNs = nullptr,
                                     FlowPerfBreakdown* flowPerf = nullptr) noexcept {
    int64_t localWait{};
    int64_t sharedWait{};
    if (localWaitNs || waitNs) {
        const auto localWaitStartNs = monotonicNowNs();
        localSemaphore->acquire();
        localWait = monotonicNowNs() - localWaitStartNs;
    } else {
        localSemaphore->acquire();
    }

    if (sharedSemaphore) {
        if (sharedWaitNs || waitNs) {
            const auto sharedWaitStartNs = monotonicNowNs();
            sharedSemaphore->acquire();
            sharedWait = monotonicNowNs() - sharedWaitStartNs;
        } else {
            sharedSemaphore->acquire();
        }
    }

    if (localWaitNs)
        *localWaitNs = localWait;
    if (sharedWaitNs)
        *sharedWaitNs = sharedWait;
    if (waitNs)
        *waitNs = localWait + sharedWait;

    const auto rifeProcessStartNs = rifeProcessWallNs ? monotonicNowNs() : 0;
    const auto status = rife->process_flow(src0Packed, src1Packed, flowOut, flowPerf);
    if (rifeProcessWallNs)
        *rifeProcessWallNs = monotonicNowNs() - rifeProcessStartNs;

    if (sharedSemaphore)
        sharedSemaphore->release();
    localSemaphore->release();
    return status;
}

static int processReducedFlowWithSemaphores(const RIFE* const rife,
                                            std::counting_semaphore<>* const localSemaphore,
                                            std::counting_semaphore<>* const sharedSemaphore,
                                            const float* src0R, const float* src0G, const float* src0B,
                                            const float* src1R, const float* src1G, const float* src1B,
                                            RIFEReducedFlowBlock* reducedFlowOut, const RIFEFlowReduceConfig& reduceConfig,
                                            const int width, const int height, const ptrdiff_t stride,
                                            int64_t* waitNs = nullptr,
                                            int64_t* localWaitNs = nullptr,
                                            int64_t* sharedWaitNs = nullptr,
                                            int64_t* rifeProcessWallNs = nullptr,
                                            FlowPerfBreakdown* flowPerf = nullptr) noexcept {
    int64_t localWait{};
    int64_t sharedWait{};
    if (localWaitNs || waitNs) {
        const auto localWaitStartNs = monotonicNowNs();
        localSemaphore->acquire();
        localWait = monotonicNowNs() - localWaitStartNs;
    } else {
        localSemaphore->acquire();
    }

    if (sharedSemaphore) {
        if (sharedWaitNs || waitNs) {
            const auto sharedWaitStartNs = monotonicNowNs();
            sharedSemaphore->acquire();
            sharedWait = monotonicNowNs() - sharedWaitStartNs;
        } else {
            sharedSemaphore->acquire();
        }
    }

    if (localWaitNs)
        *localWaitNs = localWait;
    if (sharedWaitNs)
        *sharedWaitNs = sharedWait;
    if (waitNs)
        *waitNs = localWait + sharedWait;

    const auto rifeProcessStartNs = rifeProcessWallNs ? monotonicNowNs() : 0;
    const auto status = rife->process_flow_reduced(src0R, src0G, src0B, src1R, src1G, src1B,
                                                   reducedFlowOut, reduceConfig, width, height, stride, flowPerf);
    if (rifeProcessWallNs)
        *rifeProcessWallNs = monotonicNowNs() - rifeProcessStartNs;

    if (sharedSemaphore)
        sharedSemaphore->release();
    localSemaphore->release();
    return status;
}

static int processReducedFlowWithSemaphores(const RIFE* const rife,
                                            std::counting_semaphore<>* const localSemaphore,
                                            std::counting_semaphore<>* const sharedSemaphore,
                                            const ncnn::Mat& src0Packed, const ncnn::Mat& src1Packed,
                                            RIFEReducedFlowBlock* reducedFlowOut, const RIFEFlowReduceConfig& reduceConfig,
                                            int64_t* waitNs = nullptr,
                                            int64_t* localWaitNs = nullptr,
                                            int64_t* sharedWaitNs = nullptr,
                                            int64_t* rifeProcessWallNs = nullptr,
                                            FlowPerfBreakdown* flowPerf = nullptr) noexcept {
    int64_t localWait{};
    int64_t sharedWait{};
    if (localWaitNs || waitNs) {
        const auto localWaitStartNs = monotonicNowNs();
        localSemaphore->acquire();
        localWait = monotonicNowNs() - localWaitStartNs;
    } else {
        localSemaphore->acquire();
    }

    if (sharedSemaphore) {
        if (sharedWaitNs || waitNs) {
            const auto sharedWaitStartNs = monotonicNowNs();
            sharedSemaphore->acquire();
            sharedWait = monotonicNowNs() - sharedWaitStartNs;
        } else {
            sharedSemaphore->acquire();
        }
    }

    if (localWaitNs)
        *localWaitNs = localWait;
    if (sharedWaitNs)
        *sharedWaitNs = sharedWait;
    if (waitNs)
        *waitNs = localWait + sharedWait;

    const auto rifeProcessStartNs = rifeProcessWallNs ? monotonicNowNs() : 0;
    const auto status = rife->process_flow_reduced(src0Packed, src1Packed, reducedFlowOut, reduceConfig, flowPerf);
    if (rifeProcessWallNs)
        *rifeProcessWallNs = monotonicNowNs() - rifeProcessStartNs;

    if (sharedSemaphore)
        sharedSemaphore->release();
    localSemaphore->release();
    return status;
}

static int processGpuMotionVectorsWithSemaphores(const RIFE* const rife,
                                                 std::counting_semaphore<>* const localSemaphore,
                                                 std::counting_semaphore<>* const sharedSemaphore,
                                                 const float* src0R, const float* src0G, const float* src0B,
                                                 const float* src1R, const float* src1G, const float* src1B,
                                                 RIFEGpuMotionVector* vectorsOut, const RIFEGpuMotionVectorConfig& vectorConfig,
                                                 const int width, const int height, const ptrdiff_t stride,
                                                 int64_t* waitNs = nullptr,
                                                 int64_t* localWaitNs = nullptr,
                                                 int64_t* sharedWaitNs = nullptr,
                                                 int64_t* rifeProcessWallNs = nullptr,
                                                 FlowPerfBreakdown* flowPerf = nullptr) noexcept {
    int64_t localWait{};
    int64_t sharedWait{};
    if (localWaitNs || waitNs) {
        const auto localWaitStartNs = monotonicNowNs();
        localSemaphore->acquire();
        localWait = monotonicNowNs() - localWaitStartNs;
    } else {
        localSemaphore->acquire();
    }

    if (sharedSemaphore) {
        if (sharedWaitNs || waitNs) {
            const auto sharedWaitStartNs = monotonicNowNs();
            sharedSemaphore->acquire();
            sharedWait = monotonicNowNs() - sharedWaitStartNs;
        } else {
            sharedSemaphore->acquire();
        }
    }

    if (localWaitNs)
        *localWaitNs = localWait;
    if (sharedWaitNs)
        *sharedWaitNs = sharedWait;
    if (waitNs)
        *waitNs = localWait + sharedWait;

    const auto rifeProcessStartNs = rifeProcessWallNs ? monotonicNowNs() : 0;
    const auto status = rife->process_motion_vectors_gpu(src0R, src0G, src0B, src1R, src1G, src1B,
                                                         vectorsOut, vectorConfig, width, height, stride, flowPerf);
    if (rifeProcessWallNs)
        *rifeProcessWallNs = monotonicNowNs() - rifeProcessStartNs;

    if (sharedSemaphore)
        sharedSemaphore->release();
    localSemaphore->release();
    return status;
}

static int processGpuMotionVectorsWithSemaphores(const RIFE* const rife,
                                                 std::counting_semaphore<>* const localSemaphore,
                                                 std::counting_semaphore<>* const sharedSemaphore,
                                                 const ncnn::Mat& src0Packed, const ncnn::Mat& src1Packed,
                                                 RIFEGpuMotionVector* vectorsOut, const RIFEGpuMotionVectorConfig& vectorConfig,
                                                 int64_t* waitNs = nullptr,
                                                 int64_t* localWaitNs = nullptr,
                                                 int64_t* sharedWaitNs = nullptr,
                                                 int64_t* rifeProcessWallNs = nullptr,
                                                 FlowPerfBreakdown* flowPerf = nullptr) noexcept {
    int64_t localWait{};
    int64_t sharedWait{};
    if (localWaitNs || waitNs) {
        const auto localWaitStartNs = monotonicNowNs();
        localSemaphore->acquire();
        localWait = monotonicNowNs() - localWaitStartNs;
    } else {
        localSemaphore->acquire();
    }

    if (sharedSemaphore) {
        if (sharedWaitNs || waitNs) {
            const auto sharedWaitStartNs = monotonicNowNs();
            sharedSemaphore->acquire();
            sharedWait = monotonicNowNs() - sharedWaitStartNs;
        } else {
            sharedSemaphore->acquire();
        }
    }

    if (localWaitNs)
        *localWaitNs = localWait;
    if (sharedWaitNs)
        *sharedWaitNs = sharedWait;
    if (waitNs)
        *waitNs = localWait + sharedWait;

    const auto rifeProcessStartNs = rifeProcessWallNs ? monotonicNowNs() : 0;
    const auto status = rife->process_motion_vectors_gpu(src0Packed, src1Packed, vectorsOut, vectorConfig, flowPerf);
    if (rifeProcessWallNs)
        *rifeProcessWallNs = monotonicNowNs() - rifeProcessStartNs;

    if (sharedSemaphore)
        sharedSemaphore->release();
    localSemaphore->release();
    return status;
}

static int computeBlockCount(const int size, const int blockSize, const int overlap, const int padding) noexcept {
    const auto step = blockSize - overlap;
    const auto paddedSize = size + padding * 2;

    return std::max(1, (paddedSize - overlap + step - 1) / step);
}

static double rgbToLuma(const float r, const float g, const float b) noexcept {
    return r * 0.2126 + g * 0.7152 + b * 0.0722;
}

static int clampPixel(const int value, const int limit) noexcept {
    return std::clamp(value, 0, limit - 1);
}

static int clampMotionVectorComponent(const int value, const int pel, const int blockCoord,
                                      const int blockSize, const int size, const int padding) noexcept {
    const auto minPixelDelta = -padding - blockCoord;
    const auto maxPixelDelta = size - blockSize + padding - blockCoord;

    return std::clamp(value, minPixelDelta * pel, maxPixelDelta * pel);
}

static MotionVectorConfig createMotionVectorConfig(const VSVideoInfo& inputVi, const VSVideoInfo* const metadataVi,
                                                   const MotionVectorInternalGeometry& internalGeometry,
                                                   const bool useChroma, const int blockSizeX, const int blockSizeY,
                                                   const int overlapX, const int overlapY,
                                                   const int pel, const int delta, const int bits, const int hPadding,
                                                   const int vPadding, const int blockReduce,
                                                   const double sadMultiplier) {
    MotionVectorConfig config{};
    config.useChroma = useChroma;
    config.blockSizeX = blockSizeX;
    config.blockSizeY = blockSizeY;
    config.overlapX = overlapX;
    config.overlapY = overlapY;
    config.stepX = blockSizeX - overlapX;
    config.stepY = blockSizeY - overlapY;
    config.internalBlockSizeX = internalGeometry.internalBlockSizeX;
    config.internalBlockSizeY = internalGeometry.internalBlockSizeY;
    config.internalOverlapX = internalGeometry.internalOverlapX;
    config.internalOverlapY = internalGeometry.internalOverlapY;
    config.internalStepX = internalGeometry.internalStepX;
    config.internalStepY = internalGeometry.internalStepY;
    config.pel = pel;
    config.delta = delta;
    config.bits = bits;
    config.hPadding = hPadding;
    config.vPadding = vPadding;
    config.internalHPadding = internalGeometry.internalHPadding;
    config.internalVPadding = internalGeometry.internalVPadding;
    config.blkX = computeBlockCount(inputVi.width, blockSizeX, overlapX, hPadding);
    config.blkY = computeBlockCount(inputVi.height, blockSizeY, overlapY, vPadding);
    config.inferenceWidth = internalGeometry.inferenceWidth;
    config.inferenceHeight = internalGeometry.inferenceHeight;
    config.blockReduce = blockReduce;
    config.motionScaleX = internalGeometry.motionScaleX;
    config.motionScaleY = internalGeometry.motionScaleY;
    config.sadMultiplier = sadMultiplier;

    const auto scaleLimit = static_cast<long double>((1LL << bits) - 1LL);
    const auto blockArea = static_cast<long double>(blockSizeX) * blockSizeY;
    const auto maxValidSad = blockArea * (useChroma ? 3.0L * scaleLimit : scaleLimit);
    const auto maxInvalidSad = blockArea * static_cast<long double>(1LL << bits);
    const auto maxScaledSad = std::max(maxValidSad, maxInvalidSad) * sadMultiplier;
    if (maxScaledSad > static_cast<long double>(std::numeric_limits<int64_t>::max()) - 0.5L)
        throw "sad_multiplier results in an overflowed SAD value";

    const auto invalidSad = static_cast<int64_t>(blockSizeX) * blockSizeY * (1LL << bits);
    config.invalidSad = static_cast<int64_t>(static_cast<long double>(invalidSad) * sadMultiplier + 0.5L);

    const auto& analysisVi = metadataVi ? *metadataVi : inputVi;
    const auto xRatioUV = 1 << analysisVi.format.subSamplingW;
    const auto yRatioUV = 1 << analysisVi.format.subSamplingH;
    const auto makeAnalysisData = [&](const bool backward) {
        MVAnalysisData analysisData{};
        analysisData.nVersion = 5;
        analysisData.nBlkSizeX = config.blockSizeX;
        analysisData.nBlkSizeY = config.blockSizeY;
        analysisData.nPel = config.pel;
        analysisData.nLvCount = 1;
        analysisData.nDeltaFrame = config.delta;
        analysisData.isBackward = backward ? 1 : 0;
        analysisData.nMotionFlags = backward ? MotionIsBackward : 0;
        if (config.useChroma)
            analysisData.nMotionFlags |= MotionUseChromaMotion;
        analysisData.nWidth = inputVi.width;
        analysisData.nHeight = inputVi.height;
        analysisData.nOverlapX = config.overlapX;
        analysisData.nOverlapY = config.overlapY;
        analysisData.nBlkX = config.blkX;
        analysisData.nBlkY = config.blkY;
        analysisData.bitsPerSample = config.bits;
        analysisData.yRatioUV = yRatioUV;
        analysisData.xRatioUV = xRatioUV;
        analysisData.nHPadding = config.hPadding;
        analysisData.nVPadding = config.vPadding;
        return analysisData;
    };

    config.backwardAnalysisData = makeAnalysisData(true);
    config.forwardAnalysisData = makeAnalysisData(false);
    return config;
}

static ResolvedRIFEModel resolveRIFEModel(std::string modelPath) {
    ResolvedRIFEModel resolved{};
    resolved.modelPath = std::move(modelPath);
    resolved.padding = 32;

    if (resolved.modelPath.empty())
        throw "model_path must be specified";

    std::ifstream ifs{ resolved.modelPath + "/flownet.param" };
    if (!ifs.is_open())
        throw "failed to load model";

    if (resolved.modelPath.find("rife-v2") != std::string::npos)
        resolved.rifeV2 = true;
    else if (resolved.modelPath.find("rife-v3.9") != std::string::npos)
        resolved.rifeV4 = true;
    else if (resolved.modelPath.find("rife-v3") != std::string::npos)
        resolved.rifeV2 = true;
    else if (resolved.modelPath.find("rife-v4") != std::string::npos)
        resolved.rifeV4 = true;
    else if (resolved.modelPath.find("rife4") != std::string::npos)
        resolved.rifeV4 = true;

    if (resolved.modelPath.find("rife-v4.25") != std::string::npos)
        resolved.padding = 64;
    if (resolved.modelPath.find("rife-v4.25-lite") != std::string::npos)
        resolved.padding = 128;
    if (resolved.modelPath.find("rife-v4.26") != std::string::npos)
        resolved.padding = 64;
    // This specific export trips ncnn Vulkan fp16 paths and loses the device during submit.
    if (resolved.modelPath.find("rife-v4.9_ensembleFalse") != std::string::npos)
        resolved.disableVulkanFp16 = true;
    else if (resolved.modelPath.find("rife") == std::string::npos)
        throw "unknown model dir type";

    return resolved;
}

static bool isEarlyUnsupportedRIFEV4Model(const std::string& modelPath) {
    const auto containsVersionToken = [&](const char* token) {
        const auto tokenLength = std::strlen(token);
        auto tokenPos = modelPath.find(token);

        while (tokenPos != std::string::npos) {
            const auto tokenEndPos = tokenPos + tokenLength;
            const auto hasNumericSuffix = tokenEndPos < modelPath.size() &&
                                          modelPath[tokenEndPos] >= '0' &&
                                          modelPath[tokenEndPos] <= '9';
            if (!hasNumericSuffix)
                return true;

            tokenPos = modelPath.find(token, tokenPos + 1);
        }

        return false;
    };

    const auto plainRifeV4Path = modelPath.find("rife-v4") != std::string::npos &&
                                 modelPath.find("rife-v4.") == std::string::npos;
    if (plainRifeV4Path)
        return true;

    return containsVersionToken("rife-v4.0") ||
           containsVersionToken("rife-v4.1") ||
           containsVersionToken("rife4.0") ||
           containsVersionToken("rife4.1");
}

static bool supportsMotionVectorExport(const ResolvedRIFEModel& resolvedModel) {
    if (resolvedModel.modelPath.find("rife-v3.1") != std::string::npos)
        return true;
    if (resolvedModel.modelPath.find("rife-v3.9") != std::string::npos)
        return true;

    const auto isV4FamilyPath = resolvedModel.modelPath.find("rife-v4") != std::string::npos ||
                                resolvedModel.modelPath.find("rife4") != std::string::npos;
    if (!isV4FamilyPath)
        return false;

    return !isEarlyUnsupportedRIFEV4Model(resolvedModel.modelPath);
}

static void validateAndNormalizeFlowScale(float& flowScale) {
    if (!std::isfinite(flowScale) || flowScale <= 0.f)
        throw "flow_scale must be finite and greater than 0";

    static constexpr float allowedFlowScales[]{ 0.25f, 0.5f, 1.f, 2.f, 4.f };
    static constexpr float flowScaleEpsilon = 1e-5f;

    for (const auto allowedFlowScale : allowedFlowScales) {
        if (std::abs(flowScale - allowedFlowScale) <= flowScaleEpsilon) {
            flowScale = allowedFlowScale;
            return;
        }
    }

    throw "flow_scale must be one of: 0.25, 0.5, 1.0, 2.0, 4.0";
}

static void validateResScale(const float resScale) {
    if (!std::isfinite(resScale) || resScale <= 0.f)
        throw "res_scale must be finite and greater than 0";
}

static int computeInferenceDimension(const int sourceDimension, const float resScale,
                                     const char* const name) {
    const auto scaled = static_cast<long double>(sourceDimension) * static_cast<long double>(resScale);
    if (!std::isfinite(static_cast<double>(scaled)))
        throw std::runtime_error(std::string("res_scale results in an invalid ") + name);

    auto rounded = static_cast<long long>(std::llround(scaled));
    if (rounded < 1)
        rounded = 1;
    if (rounded > std::numeric_limits<int>::max())
        throw std::runtime_error(std::string("res_scale results in an overflowed ") + name);

    return static_cast<int>(rounded);
}

static int scaleMotionVectorGeometryValue(const int value, const float scale) noexcept {
    return static_cast<int>(std::lround(static_cast<double>(value) * static_cast<double>(scale)));
}

static MotionVectorInternalGeometry createMotionVectorInternalGeometry(const int sourceWidth, const int sourceHeight,
                                                                      const int inferenceWidth, const int inferenceHeight,
                                                                      const int blockSizeX, const int blockSizeY,
                                                                      const int overlapX, const int overlapY,
                                                                      const int hPadding, const int vPadding) noexcept {
    MotionVectorInternalGeometry geometry{};
    const auto scaleX = static_cast<float>(inferenceWidth) / static_cast<float>(sourceWidth);
    const auto scaleY = static_cast<float>(inferenceHeight) / static_cast<float>(sourceHeight);

    geometry.motionScaleX = static_cast<float>(sourceWidth) / static_cast<float>(inferenceWidth);
    geometry.motionScaleY = static_cast<float>(sourceHeight) / static_cast<float>(inferenceHeight);
    geometry.inferenceWidth = inferenceWidth;
    geometry.inferenceHeight = inferenceHeight;

    geometry.internalBlockSizeX = std::max(1, scaleMotionVectorGeometryValue(blockSizeX, scaleX));
    geometry.internalBlockSizeY = std::max(1, scaleMotionVectorGeometryValue(blockSizeY, scaleY));
    geometry.internalOverlapX = std::clamp(scaleMotionVectorGeometryValue(overlapX, scaleX), 0, geometry.internalBlockSizeX - 1);
    geometry.internalOverlapY = std::clamp(scaleMotionVectorGeometryValue(overlapY, scaleY), 0, geometry.internalBlockSizeY - 1);
    geometry.internalStepX = std::max(1, geometry.internalBlockSizeX - geometry.internalOverlapX);
    geometry.internalStepY = std::max(1, geometry.internalBlockSizeY - geometry.internalOverlapY);
    geometry.internalHPadding = std::max(0, scaleMotionVectorGeometryValue(hPadding, scaleX));
    geometry.internalVPadding = std::max(0, scaleMotionVectorGeometryValue(vPadding, scaleY));
    return geometry;
}

static void validateSadMultiplier(const double sadMultiplier) {
    if (!std::isfinite(sadMultiplier) || sadMultiplier <= 0.0)
        throw "sad_multiplier must be finite and greater than 0";
}

static void validateAbsSADClipRange(const int absSadClipRange) {
    if (absSadClipRange < 0)
        throw "abs_sad_clip_range must be greater than or equal to 0";
}

static void validateGpuFullMotionVectorBackend(const MotionVectorConfig& config) {
    if (config.inferenceWidth != config.backwardAnalysisData.nWidth ||
        config.inferenceHeight != config.backwardAnalysisData.nHeight)
        throw "gpu_mode=2 currently requires inference dimensions to match the source dimensions";

    const auto maxSample = (1ULL << config.bits) - 1ULL;
    const auto chromaScale = config.useChroma ? 3ULL : 1ULL;
    const auto maxRawSad = static_cast<unsigned long long>(config.blockSizeX) * config.blockSizeY * maxSample * chromaScale;
    if (maxRawSad > std::numeric_limits<uint32_t>::max())
        throw "gpu_mode=2 raw SAD exceeds 32-bit storage for this block size and bit depth";
}

static void loadRIFEModel(RIFE& rife, const std::string& modelPath) {
#ifdef _WIN32
    const auto bufferSize = MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wbuffer(bufferSize);
    MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, wbuffer.data(), bufferSize);
    rife.load(wbuffer.data());
#else
    rife.load(modelPath);
#endif
}

struct MotionVectorInferenceClip final {
    VSNode* node;
    VSVideoInfo vi;
    bool convertedFromYUV;
};

struct MotionVectorClipSet final {
    VSNode* sourceNode;
    VSVideoInfo sourceVi;
    VSNode* inferenceNode;
    VSVideoInfo inferenceVi;
    bool convertedFromYUV;
};

static bool isRGBSVideoFormat(const VSVideoInfo& vi) noexcept {
    return vsh::isConstantVideoFormat(&vi) &&
           vi.format.colorFamily == cfRGB &&
           vi.format.sampleType == stFloat &&
           vi.format.bitsPerSample == 32;
}

static VSNode* convertMotionVectorClipToRGBS(const VSMap* in, VSNode* sourceNode,
                                             VSCore* core, const VSAPI* vsapi) {
    auto resizePlugin = vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core);
    if (!resizePlugin)
        throw "resize plugin is required for internal YUV->RGBS conversion";

    int err{};
    const auto matrixInValue = vsapi->mapGetData(in, "matrix_in_s", 0, &err);
    const auto* matrixIn = err ? "709" : matrixInValue;
    const auto rangeInValue = vsapi->mapGetData(in, "range_in_s", 0, &err);
    const auto* rangeIn = err ? "full" : rangeInValue;

    auto args = vsapi->createMap();
    vsapi->mapSetNode(args, "clip", sourceNode, maReplace);
    vsapi->mapSetInt(args, "format", pfRGBS, maReplace);
    vsapi->mapSetData(args, "matrix_in_s", matrixIn, -1, dtUtf8, maReplace);
    vsapi->mapSetData(args, "range_in_s", rangeIn, -1, dtUtf8, maReplace);

    auto ret = vsapi->invoke(resizePlugin, "Bicubic", args);
    if (const auto* invokeError = vsapi->mapGetError(ret)) {
        const auto errorMessage = std::string("failed to convert clip to RGBS: ") + invokeError;
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        throw std::runtime_error(errorMessage);
    }

    auto rgbNode = vsapi->mapGetNode(ret, "clip", 0, &err);
    if (err || !rgbNode) {
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        throw "resize.Bicubic did not return a clip";
    }

    const auto rgbVi = *vsapi->getVideoInfo(rgbNode);
    if (!isRGBSVideoFormat(rgbVi)) {
        vsapi->freeNode(rgbNode);
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        throw "internal YUV->RGBS conversion did not produce a constant RGBS clip";
    }

    vsapi->freeMap(args);
    vsapi->freeMap(ret);
    return rgbNode;
}

static VSNode* resizeMotionVectorClip(VSNode* sourceNode, const int width, const int height,
                                      VSCore* core, const VSAPI* vsapi) {
    auto resizePlugin = vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core);
    if (!resizePlugin)
        throw "resize plugin is required for motion-vector subsampling";

    auto args = vsapi->createMap();
    vsapi->mapSetNode(args, "clip", sourceNode, maReplace);
    vsapi->mapSetInt(args, "width", width, maReplace);
    vsapi->mapSetInt(args, "height", height, maReplace);

    auto ret = vsapi->invoke(resizePlugin, "Bicubic", args);
    if (const auto* invokeError = vsapi->mapGetError(ret)) {
        const auto errorMessage = std::string("failed to resize motion-vector inference clip: ") + invokeError;
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        throw std::runtime_error(errorMessage);
    }

    int err{};
    auto resizedNode = vsapi->mapGetNode(ret, "clip", 0, &err);
    if (err || !resizedNode) {
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        throw "resize.Bicubic did not return a clip";
    }

    const auto resizedVi = *vsapi->getVideoInfo(resizedNode);
    if (!isRGBSVideoFormat(resizedVi) || resizedVi.width != width || resizedVi.height != height) {
        vsapi->freeNode(resizedNode);
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        throw "motion-vector subsample resize did not produce the expected RGBS clip";
    }

    vsapi->freeMap(args);
    vsapi->freeMap(ret);
    return resizedNode;
}

static MotionVectorInferenceClip buildMotionVectorInferenceClip(const VSMap* in, VSNode* sourceNode,
                                                                const VSVideoInfo& sourceVi,
                                                                VSCore* core, const VSAPI* vsapi) {
    if (!vsh::isConstantVideoFormat(&sourceVi))
        throw "clip must have a constant format";

    if (isRGBSVideoFormat(sourceVi))
        return { vsapi->addNodeRef(sourceNode), sourceVi, false };

    if (sourceVi.format.colorFamily != cfYUV)
        throw "motion-vector APIs require a constant RGBS clip or a constant YUV clip";

    auto resizePlugin = vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core);
    if (!resizePlugin)
        throw "resize plugin is required for internal YUV->RGBS conversion";

    int err{};
    const auto matrixInValue = vsapi->mapGetData(in, "matrix_in_s", 0, &err);
    const auto* matrixIn = err ? "709" : matrixInValue;
    const auto rangeInValue = vsapi->mapGetData(in, "range_in_s", 0, &err);
    const auto* rangeIn = err ? "full" : rangeInValue;

    auto args = vsapi->createMap();
    vsapi->mapSetNode(args, "clip", sourceNode, maReplace);
    vsapi->mapSetInt(args, "format", pfRGBS, maReplace);
    vsapi->mapSetData(args, "matrix_in_s", matrixIn, -1, dtUtf8, maReplace);
    vsapi->mapSetData(args, "range_in_s", rangeIn, -1, dtUtf8, maReplace);

    auto ret = vsapi->invoke(resizePlugin, "Bicubic", args);
    if (const auto* invokeError = vsapi->mapGetError(ret)) {
        const auto errorMessage = std::string("failed to convert clip to RGBS: ") + invokeError;
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        throw std::runtime_error(errorMessage);
    }

    auto rgbNode = vsapi->mapGetNode(ret, "clip", 0, &err);
    if (err || !rgbNode) {
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        throw "resize.Bicubic did not return a clip";
    }

    const auto rgbVi = *vsapi->getVideoInfo(rgbNode);
    if (!isRGBSVideoFormat(rgbVi)) {
        vsapi->freeNode(rgbNode);
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        throw "internal YUV->RGBS conversion did not produce a constant RGBS clip";
    }

    vsapi->freeMap(args);
    vsapi->freeMap(ret);
    return { rgbNode, rgbVi, true };
}

static MotionVectorClipSet buildMotionVectorClipSet(const VSMap* in, VSNode* sourceNode,
                                                    const VSVideoInfo& sourceVi,
                                                    const int inferenceWidth, const int inferenceHeight,
                                                    VSCore* core, const VSAPI* vsapi) {
    MotionVectorClipSet clips{};

    try {
        if (!vsh::isConstantVideoFormat(&sourceVi))
            throw "clip must have a constant format";

        if (isRGBSVideoFormat(sourceVi)) {
            clips.sourceNode = vsapi->addNodeRef(sourceNode);
            clips.sourceVi = sourceVi;
            clips.convertedFromYUV = false;
        } else {
            if (sourceVi.format.colorFamily != cfYUV)
                throw "motion-vector APIs require a constant RGBS clip or a constant YUV clip";

            clips.sourceNode = convertMotionVectorClipToRGBS(in, sourceNode, core, vsapi);
            clips.sourceVi = *vsapi->getVideoInfo(clips.sourceNode);
            clips.convertedFromYUV = true;
        }

        if (inferenceWidth == clips.sourceVi.width &&
            inferenceHeight == clips.sourceVi.height) {
            clips.inferenceNode = vsapi->addNodeRef(clips.sourceNode);
            clips.inferenceVi = clips.sourceVi;
        } else {
            clips.inferenceNode = resizeMotionVectorClip(clips.sourceNode, inferenceWidth, inferenceHeight, core, vsapi);
            clips.inferenceVi = *vsapi->getVideoInfo(clips.inferenceNode);
        }
    } catch (...) {
        vsapi->freeNode(clips.sourceNode);
        vsapi->freeNode(clips.inferenceNode);
        throw;
    }

    return clips;
}

} // namespace

struct MotionVectorExportContext final {
    bool useChroma;
    bool backward;
    int blockSizeX;
    int blockSizeY;
    int stepX;
    int stepY;
    int internalBlockSizeX;
    int internalBlockSizeY;
    int internalStepX;
    int internalStepY;
    int pel;
    int bits;
    int hPadding;
    int vPadding;
    int internalHPadding;
    int internalVPadding;
    int blkX;
    int blkY;
    int blockReduce;
    float motionScaleX;
    float motionScaleY;
    double sadMultiplier;
    int64_t invalidSad;
};

struct RIFEMVPairData final {
    VSNode* node;
    VSNode* sourceNode;
    VSVideoInfo vi;
    MotionVectorConfig mvConfig;
    std::unique_ptr<RIFE> rife;
    std::unique_ptr<std::counting_semaphore<>> semaphore;
    std::shared_ptr<std::counting_semaphore<>> sharedFlowSemaphore;
    std::shared_ptr<MotionVectorLumaCache> lumaCache;
    std::shared_ptr<MotionVectorPackedCache> packedCache;
    MotionVectorExportBackend mvExportBackend{ MotionVectorExportBackend::Cpu };
    bool sadStats;
    bool motionStats;
    bool perfStats;
    std::shared_ptr<MotionVectorPerfStats> perf;
    std::string perfLabel;
};

struct RIFEMVOutputData final {
    VSNode* node;
    VSVideoInfo vi;
    MVAnalysisData analysisData;
    std::vector<char> invalidBlob;
    MotionVectorFrameStats invalidStats;
    int absSadClipRange;
    bool backward;
    bool renderSadMask;
    bool sadStats;
    bool motionStats;
    bool perfStats;
    std::shared_ptr<MotionVectorPerfStats> perf;
};

struct RIFEMVApproxPairData final {
    VSNode* node;
    VSNode* sourceNode;
    VSVideoInfo vi;
    MotionVectorConfig mvConfig;
    std::unique_ptr<RIFE> rife;
    std::unique_ptr<std::counting_semaphore<>> semaphore;
    std::shared_ptr<std::counting_semaphore<>> sharedFlowSemaphore;
    std::shared_ptr<MotionVectorLumaCache> lumaCache;
    std::shared_ptr<MotionVectorPackedCache> packedCache;
    bool sadStats;
    bool motionStats;
    bool perfStats;
    std::shared_ptr<MotionVectorPerfStats> perf;
    std::string perfLabel;
};

struct RIFEMVApproxOutputData final {
    VSNode* node;
    VSNode* sourceNode;
    VSVideoInfo vi;
    MotionVectorConfig mvConfig;
    MVAnalysisData analysisData;
    std::vector<char> invalidBlob;
    MotionVectorFrameStats invalidStats;
    int absSadClipRange;
    bool backward;
    bool renderSadMask;
    bool sadStats;
    bool motionStats;
    bool perfStats;
    std::shared_ptr<MotionVectorPerfStats> perf;
};

static float reduceBlockFlow(const float* flowPlane, const int width, const int height,
                             const int blockX, const int blockY, const MotionVectorExportContext& ctx) noexcept {
    if (ctx.blockReduce == MVBlockReduceCenter) {
        const auto sampleY = clampPixel(blockY + ctx.internalBlockSizeY / 2, height);
        const auto sampleX = clampPixel(blockX + ctx.internalBlockSizeX / 2, width);

        return flowPlane[sampleY * width + sampleX];
    }

    double sum{};
    for (auto y = 0; y < ctx.internalBlockSizeY; y++) {
        const auto sampleY = clampPixel(blockY + y, height);
        for (auto x = 0; x < ctx.internalBlockSizeX; x++) {
            const auto sampleX = clampPixel(blockX + x, width);
            sum += flowPlane[sampleY * width + sampleX];
        }
    }

    return static_cast<float>(sum / static_cast<double>(ctx.internalBlockSizeX * ctx.internalBlockSizeY));
}

struct SADContext final {
    int width;
    int height;
    int stride;
    int blockSizeX;
    int blockSizeY;
    bool useChroma;
    double maxSample;
    double sadMultiplier;
    const float* currentR;
    const float* currentG;
    const float* currentB;
    const float* referenceR;
    const float* referenceG;
    const float* referenceB;
    const float* currentLuma;
    const float* referenceLuma;
};

static inline int64_t roundPositiveToInt64(const double value) noexcept {
    return static_cast<int64_t>(value + 0.5);
}

static inline float quantizeSyntheticSample(const float value, const double maxSample) noexcept {
    if (maxSample <= 0.0)
        return value;

    return static_cast<float>(std::round(static_cast<double>(value) * maxSample) / maxSample);
}

static void buildFrameLumaPlane(const VSFrame* frame, const int width, const int height, const int stride,
                                std::vector<float>& luma, const double maxSample, const VSAPI* vsapi) noexcept {
    luma.resize(static_cast<size_t>(stride) * height);
    const auto* planeR = reinterpret_cast<const float*>(vsapi->getReadPtr(frame, 0));
    const auto* planeG = reinterpret_cast<const float*>(vsapi->getReadPtr(frame, 1));
    const auto* planeB = reinterpret_cast<const float*>(vsapi->getReadPtr(frame, 2));

    for (auto y = 0; y < height; y++) {
        const auto row = static_cast<size_t>(y) * stride;
        for (auto x = 0; x < width; x++) {
            const auto idx = row + x;
            luma[idx] = static_cast<float>(rgbToLuma(quantizeSyntheticSample(planeR[idx], maxSample),
                                                     quantizeSyntheticSample(planeG[idx], maxSample),
                                                     quantizeSyntheticSample(planeB[idx], maxSample)));
        }
    }
}

static void touchMotionVectorLumaCacheEntry(const std::shared_ptr<MotionVectorLumaCache>& cache,
                                            const int frameNumber) {
    for (auto it = cache->lru.begin(); it != cache->lru.end(); ++it) {
        if (*it != frameNumber)
            continue;

        cache->lru.erase(it);
        break;
    }

    cache->lru.push_back(frameNumber);
}

static void pruneMotionVectorLumaCache(const std::shared_ptr<MotionVectorLumaCache>& cache) {
    while (cache->entries.size() > cache->maxEntries && !cache->lru.empty()) {
        const auto frameNumber = cache->lru.front();
        cache->lru.pop_front();
        const auto it = cache->entries.find(frameNumber);
        if (it == cache->entries.end())
            continue;

        if (it->second.building) {
            cache->lru.push_back(frameNumber);
            break;
        }

        cache->entries.erase(it);
    }
}

static std::shared_ptr<const std::vector<float>> getOrCreateFrameLumaPlane(const std::shared_ptr<MotionVectorLumaCache>& cache,
                                                                            const VSFrame* frame, const int frameNumber,
                                                                            const int width, const int height, const int stride,
                                                                            const double maxSample, const VSAPI* vsapi) {
    if (!cache) {
        auto luma = std::make_shared<std::vector<float>>();
        buildFrameLumaPlane(frame, width, height, stride, *luma, maxSample, vsapi);
        return luma;
    }

    const auto expectedSize = static_cast<size_t>(stride) * height;
    while (true) {
        std::unique_lock<std::mutex> lock(cache->mutex);
        auto& entry = cache->entries[frameNumber];

        if (entry.luma && !entry.building &&
            entry.stride == stride &&
            entry.height == height &&
            entry.luma->size() == expectedSize) {
            auto luma = entry.luma;
            touchMotionVectorLumaCacheEntry(cache, frameNumber);
            return luma;
        }

        if (entry.building) {
            cache->condition.wait(lock, [&]() {
                const auto it = cache->entries.find(frameNumber);
                return it == cache->entries.end() || !it->second.building;
            });
            continue;
        }

        entry.frameNumber = frameNumber;
        entry.stride = stride;
        entry.height = height;
        entry.building = true;
        entry.luma.reset();
        lock.unlock();

        auto luma = std::make_shared<std::vector<float>>();
        buildFrameLumaPlane(frame, width, height, stride, *luma, maxSample, vsapi);

        lock.lock();
        auto& stored = cache->entries[frameNumber];
        stored.frameNumber = frameNumber;
        stored.stride = stride;
        stored.height = height;
        stored.building = false;
        stored.luma = luma;
        touchMotionVectorLumaCacheEntry(cache, frameNumber);
        pruneMotionVectorLumaCache(cache);
        lock.unlock();
        cache->condition.notify_all();
        return luma;
    }
}

static SADContext makeSADContext(const VSFrame* current, const VSFrame* reference, const MotionVectorExportContext& ctx,
                                 const VSAPI* vsapi, const float* currentLuma, const float* referenceLuma) noexcept {
    const auto stride = static_cast<int>(vsapi->getStride(current, 0) / vsapi->getVideoFrameFormat(current)->bytesPerSample);
    SADContext context{};
    context.width = vsapi->getFrameWidth(current, 0);
    context.height = vsapi->getFrameHeight(current, 0);
    context.stride = stride;
    context.blockSizeX = ctx.blockSizeX;
    context.blockSizeY = ctx.blockSizeY;
    context.useChroma = ctx.useChroma;
    context.maxSample = static_cast<double>((1ULL << ctx.bits) - 1ULL);
    context.sadMultiplier = ctx.sadMultiplier;
    context.currentR = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 0));
    context.currentG = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 1));
    context.currentB = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 2));
    context.referenceR = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 0));
    context.referenceG = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 1));
    context.referenceB = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 2));
    context.currentLuma = currentLuma;
    context.referenceLuma = referenceLuma;
    return context;
}

static int64_t computeBlockSAD(const SADContext& context, const int pixelDx, const int pixelDy,
                               const int blockX, const int blockY) noexcept {
    int64_t sad{};
    const auto currentX0 = blockX;
    const auto currentY0 = blockY;
    const auto referenceX0 = blockX + pixelDx;
    const auto referenceY0 = blockY + pixelDy;
    const auto interior = currentX0 >= 0 && currentY0 >= 0 &&
                          referenceX0 >= 0 && referenceY0 >= 0 &&
                          currentX0 + context.blockSizeX <= context.width &&
                          currentY0 + context.blockSizeY <= context.height &&
                          referenceX0 + context.blockSizeX <= context.width &&
                          referenceY0 + context.blockSizeY <= context.height;

    if (interior) {
        if (context.useChroma) {
            for (auto y = 0; y < context.blockSizeY; y++) {
                const auto currentRow = (currentY0 + y) * context.stride + currentX0;
                const auto referenceRow = (referenceY0 + y) * context.stride + referenceX0;
                const auto* currentRRow = context.currentR + currentRow;
                const auto* currentGRow = context.currentG + currentRow;
                const auto* currentBRow = context.currentB + currentRow;
                const auto* referenceRRow = context.referenceR + referenceRow;
                const auto* referenceGRow = context.referenceG + referenceRow;
                const auto* referenceBRow = context.referenceB + referenceRow;
                for (auto x = 0; x < context.blockSizeX; x++) {
                    const auto currentR = quantizeSyntheticSample(currentRRow[x], context.maxSample);
                    const auto currentG = quantizeSyntheticSample(currentGRow[x], context.maxSample);
                    const auto currentB = quantizeSyntheticSample(currentBRow[x], context.maxSample);
                    const auto referenceR = quantizeSyntheticSample(referenceRRow[x], context.maxSample);
                    const auto referenceG = quantizeSyntheticSample(referenceGRow[x], context.maxSample);
                    const auto referenceB = quantizeSyntheticSample(referenceBRow[x], context.maxSample);
                    const auto diff =
                        static_cast<double>(std::abs(currentR - referenceR) +
                                            std::abs(currentG - referenceG) +
                                            std::abs(currentB - referenceB));
                    sad += roundPositiveToInt64(diff * context.maxSample);
                }
            }
        } else {
            for (auto y = 0; y < context.blockSizeY; y++) {
                const auto currentRow = (currentY0 + y) * context.stride + currentX0;
                const auto referenceRow = (referenceY0 + y) * context.stride + referenceX0;
                const auto* currentLumaRow = context.currentLuma + currentRow;
                const auto* referenceLumaRow = context.referenceLuma + referenceRow;
                for (auto x = 0; x < context.blockSizeX; x++) {
                    const auto diff = static_cast<double>(std::abs(currentLumaRow[x] - referenceLumaRow[x]));
                    sad += roundPositiveToInt64(diff * context.maxSample);
                }
            }
        }

        return static_cast<int64_t>(static_cast<long double>(sad) * context.sadMultiplier + 0.5L);
    }

    for (auto y = 0; y < context.blockSizeY; y++) {
        const auto currentY = clampPixel(blockY + y, context.height);
        const auto referenceY = clampPixel(currentY + pixelDy, context.height);
        for (auto x = 0; x < context.blockSizeX; x++) {
            const auto currentX = clampPixel(blockX + x, context.width);
            const auto referenceX = clampPixel(currentX + pixelDx, context.width);
            const auto currentIndex = currentY * context.stride + currentX;
            const auto referenceIndex = referenceY * context.stride + referenceX;

            if (context.useChroma) {
                const auto currentR = quantizeSyntheticSample(context.currentR[currentIndex], context.maxSample);
                const auto currentG = quantizeSyntheticSample(context.currentG[currentIndex], context.maxSample);
                const auto currentB = quantizeSyntheticSample(context.currentB[currentIndex], context.maxSample);
                const auto referenceR = quantizeSyntheticSample(context.referenceR[referenceIndex], context.maxSample);
                const auto referenceG = quantizeSyntheticSample(context.referenceG[referenceIndex], context.maxSample);
                const auto referenceB = quantizeSyntheticSample(context.referenceB[referenceIndex], context.maxSample);
                const auto diff =
                    static_cast<double>(std::abs(currentR - referenceR) +
                                        std::abs(currentG - referenceG) +
                                        std::abs(currentB - referenceB));
                sad += roundPositiveToInt64(diff * context.maxSample);
            } else {
                const auto diff = static_cast<double>(std::abs(context.currentLuma[currentIndex] - context.referenceLuma[referenceIndex]));
                sad += roundPositiveToInt64(diff * context.maxSample);
            }
        }
    }

    return static_cast<int64_t>(static_cast<long double>(sad) * context.sadMultiplier + 0.5L);
}

static const MVAnalysisData& getMotionVectorAnalysisData(const MotionVectorConfig& config, const bool backward) noexcept {
    return backward ? config.backwardAnalysisData : config.forwardAnalysisData;
}

static void packMotionVectorBlob(const std::vector<MVToolsVector>& vectors, const bool valid,
                                 const MVAnalysisData& analysisData, std::vector<char>& blob,
                                 MotionVectorFrameStats* const stats = nullptr,
                                 const bool includeSadStats = true,
                                 const bool includeMotionStats = true) {
    const auto planeSize = static_cast<MVArraySizeType>(sizeof(MVArraySizeType) + vectors.size() * sizeof(MVToolsVector));
    const auto groupSize = static_cast<MVArraySizeType>(sizeof(MVArraySizeType) * 2 + planeSize);
    blob.resize(groupSize);
    size_t offset{};
    const auto writeScalar = [&](const auto value) {
        std::memcpy(blob.data() + offset, &value, sizeof(value));
        offset += sizeof(value);
    };

    writeScalar(groupSize);
    writeScalar(valid ? MVArraySizeType{ 1 } : MVArraySizeType{ 0 });
    writeScalar(planeSize);
    std::memcpy(blob.data() + offset, vectors.data(), vectors.size() * sizeof(MVToolsVector));
    if (stats && (includeSadStats || includeMotionStats))
        *stats = computeMotionVectorFrameStats(vectors, analysisData, includeSadStats, includeMotionStats);
}

static void buildMaskResizeAxisTable(const int srcSize, const int dstSize,
                                     std::vector<BilinearAxisEntry>& table) {
    table.resize(dstSize);
    if (srcSize <= 1) {
        for (int i = 0; i < dstSize; i++)
            table[i] = { 0, 0, 0.0f };

        return;
    }

    const auto scale = static_cast<float>(srcSize) / static_cast<float>(dstSize);
    for (int i = 0; i < dstSize; i++) {
        auto sample = (i + 0.5f) * scale - 0.5f;
        sample = std::max(0.0f, std::min(sample, static_cast<float>(srcSize - 1)));
        const auto index0 = static_cast<int>(std::floor(sample));
        const auto index1 = std::min(index0 + 1, srcSize - 1);
        table[i] = { index0, index1, sample - index0 };
    }
}

static bool unpackMotionVectorBlob(const char* vectorBlob, const int vectorBlobSize,
                                   const MVAnalysisData& analysisData, std::vector<MVToolsVector>& vectors) {
    if (!vectorBlob || vectorBlobSize < static_cast<int>(sizeof(MVArraySizeType) * 3) ||
        analysisData.nBlkX <= 0 || analysisData.nBlkY <= 0)
        return false;

    const auto vectorCount = static_cast<size_t>(analysisData.nBlkX) * analysisData.nBlkY;
    const auto expectedPlaneSize = static_cast<MVArraySizeType>(sizeof(MVArraySizeType) + vectorCount * sizeof(MVToolsVector));
    const auto expectedGroupSize = static_cast<MVArraySizeType>(sizeof(MVArraySizeType) * 2 + expectedPlaneSize);
    MVArraySizeType groupSize{};
    MVArraySizeType planeSize{};
    std::memcpy(&groupSize, vectorBlob, sizeof(groupSize));
    std::memcpy(&planeSize, vectorBlob + sizeof(MVArraySizeType) * 2, sizeof(planeSize));

    if (groupSize < expectedGroupSize || groupSize > vectorBlobSize ||
        planeSize < expectedPlaneSize || planeSize > groupSize - static_cast<MVArraySizeType>(sizeof(MVArraySizeType) * 2) ||
        vectorBlobSize < expectedGroupSize)
        return false;

    vectors.resize(vectorCount);
    std::memcpy(vectors.data(), vectorBlob + sizeof(MVArraySizeType) * 3, vectorCount * sizeof(MVToolsVector));
    return true;
}

static void renderMotionVectorSADMask(VSFrame* frame, const VSVideoInfo& vi,
                                      const char* vectorBlob, const int vectorBlobSize,
                                      const MVAnalysisData& analysisData, const int absSadClipRange,
                                      const VSAPI* vsapi) {
    auto* dstp = vsapi->getWritePtr(frame, 0);
    const auto dstStride = vsapi->getStride(frame, 0);
    std::vector<MVToolsVector> vectors;
    if (!unpackMotionVectorBlob(vectorBlob, vectorBlobSize, analysisData, vectors)) {
        for (int y = 0; y < vi.height; y++)
            std::memset(dstp + static_cast<size_t>(y) * dstStride, 0, dstStride);

        return;
    }

    std::vector<uint8_t> smallMask(vectors.size());
    const auto maskMode = absSadClipRange > 0 ? MotionVectorSADMaskMode::Absolute : MotionVectorSADMaskMode::Relative;
    if (maskMode == MotionVectorSADMaskMode::Relative) {
        int64_t frameMaxSad{};
        for (const auto& vector : vectors)
            frameMaxSad = std::max(frameMaxSad, vector.sad);

        if (frameMaxSad <= 0) {
            for (int y = 0; y < vi.height; y++)
                std::memset(dstp + static_cast<size_t>(y) * dstStride, 0, dstStride);

            return;
        }

        for (size_t i = 0; i < vectors.size(); i++) {
            const auto sad = std::max<int64_t>(vectors[i].sad, 0);
            const auto scaled = static_cast<int>(static_cast<long double>(sad) * 255.0L / frameMaxSad + 0.5L);
            smallMask[i] = static_cast<uint8_t>(std::clamp(scaled, 0, 255));
        }
    } else {
        const auto clipRange = static_cast<int64_t>(absSadClipRange);
        for (size_t i = 0; i < vectors.size(); i++) {
            const auto sad = std::clamp<int64_t>(vectors[i].sad, 0, clipRange);
            const auto scaled = static_cast<int>((sad * 256) / clipRange);
            smallMask[i] = static_cast<uint8_t>(std::clamp(scaled, 0, 255));
        }
    }

    std::vector<BilinearAxisEntry> xTable;
    std::vector<BilinearAxisEntry> yTable;
    buildMaskResizeAxisTable(analysisData.nBlkX, vi.width, xTable);
    buildMaskResizeAxisTable(analysisData.nBlkY, vi.height, yTable);

    for (int y = 0; y < vi.height; y++) {
        auto* dstRow = dstp + static_cast<size_t>(y) * dstStride;
        std::memset(dstRow, 0, dstStride);
        const auto& yEntry = yTable[static_cast<size_t>(y)];
        const auto* row0 = smallMask.data() + static_cast<size_t>(yEntry.index0) * analysisData.nBlkX;
        const auto* row1 = smallMask.data() + static_cast<size_t>(yEntry.index1) * analysisData.nBlkX;

        for (int x = 0; x < vi.width; x++) {
            const auto& xEntry = xTable[static_cast<size_t>(x)];
            const auto top = row0[xEntry.index0] * (1.0f - xEntry.alpha) + row0[xEntry.index1] * xEntry.alpha;
            const auto bottom = row1[xEntry.index0] * (1.0f - xEntry.alpha) + row1[xEntry.index1] * xEntry.alpha;
            dstRow[x] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(top * (1.0f - yEntry.alpha) + bottom * yEntry.alpha)), 0, 255));
        }
    }
}

static void buildMVToolsVectorBlob(const VSFrame* current, const VSFrame* reference, const float* flow,
                                   const int flowWidth, const int flowHeight,
                                   const bool valid, const MotionVectorExportContext& ctx,
                                   const MVAnalysisData& analysisData, std::vector<MVToolsVector>& vectors,
                                   std::vector<char>& blob, const VSAPI* vsapi,
                                   const std::vector<float>* currentLumaCache = nullptr,
                                   const std::vector<float>* referenceLumaCache = nullptr,
                                   MotionVectorFrameStats* const stats = nullptr,
                                   const bool includeSadStats = true,
                                   const bool includeMotionStats = true) {
    const auto vectorCount = static_cast<size_t>(ctx.blkX) * ctx.blkY;
    vectors.resize(vectorCount);

    if (!valid) {
        for (auto& vector : vectors) {
            vector.x = 0;
            vector.y = 0;
            vector.sad = ctx.invalidSad;
        }

        packMotionVectorBlob(vectors, false, analysisData, blob, stats, includeSadStats, includeMotionStats);
        return;
    }

    const auto width = vsapi->getFrameWidth(current, 0);
    const auto height = vsapi->getFrameHeight(current, 0);
    const auto stride = static_cast<int>(vsapi->getStride(current, 0) / vsapi->getVideoFrameFormat(current)->bytesPerSample);
    const auto sadMaxSample = static_cast<double>((1ULL << ctx.bits) - 1ULL);
    std::vector<float> currentLuma;
    std::vector<float> referenceLuma;
    const float* currentLumaPtr = nullptr;
    const float* referenceLumaPtr = nullptr;
    if (!ctx.useChroma) {
        if (currentLumaCache && currentLumaCache->size() >= static_cast<size_t>(stride) * height) {
            currentLumaPtr = currentLumaCache->data();
        } else {
            buildFrameLumaPlane(current, width, height, stride, currentLuma, sadMaxSample, vsapi);
            currentLumaPtr = currentLuma.data();
        }

        if (referenceLumaCache && referenceLumaCache->size() >= static_cast<size_t>(stride) * height) {
            referenceLumaPtr = referenceLumaCache->data();
        } else {
            buildFrameLumaPlane(reference, width, height, stride, referenceLuma, sadMaxSample, vsapi);
            referenceLumaPtr = referenceLuma.data();
        }
    }

    const auto sadContext = makeSADContext(current, reference, ctx, vsapi, currentLumaPtr, referenceLumaPtr);
    const auto channelOffset = ctx.backward ? 0 : 2;
    const auto flowPlaneSize = flowWidth * flowHeight;
    const auto* flowXPlane = flow + (channelOffset + 0) * flowPlaneSize;
    const auto* flowYPlane = flow + (channelOffset + 1) * flowPlaneSize;

    for (auto by = 0; by < ctx.blkY; by++) {
        const auto blockY = by * ctx.stepY - ctx.vPadding;
        const auto internalBlockY = by * ctx.internalStepY - ctx.internalVPadding;
        for (auto bx = 0; bx < ctx.blkX; bx++) {
            const auto blockX = bx * ctx.stepX - ctx.hPadding;
            const auto internalBlockX = bx * ctx.internalStepX - ctx.internalHPadding;
            auto& vector = vectors[static_cast<size_t>(by) * ctx.blkX + bx];
            const auto flowX = reduceBlockFlow(flowXPlane, flowWidth, flowHeight, internalBlockX, internalBlockY, ctx);
            const auto flowY = reduceBlockFlow(flowYPlane, flowWidth, flowHeight, internalBlockX, internalBlockY, ctx);

            vector.x = static_cast<int>(std::lround(-2.0f * flowX * ctx.motionScaleX * ctx.pel));
            vector.y = static_cast<int>(std::lround(-2.0f * flowY * ctx.motionScaleY * ctx.pel));
            vector.x = clampMotionVectorComponent(vector.x, ctx.pel, blockX, ctx.blockSizeX, width, ctx.hPadding);
            vector.y = clampMotionVectorComponent(vector.y, ctx.pel, blockY, ctx.blockSizeY, height, ctx.vPadding);
            const auto pixelDx = static_cast<int>(std::lround(static_cast<double>(vector.x) / ctx.pel));
            const auto pixelDy = static_cast<int>(std::lround(static_cast<double>(vector.y) / ctx.pel));
            vector.sad = computeBlockSAD(sadContext, pixelDx, pixelDy, blockX, blockY);
        }
    }

    packMotionVectorBlob(vectors, true, analysisData, blob, stats, includeSadStats, includeMotionStats);
}

static MotionVectorExportContext createMotionVectorExportContext(const MotionVectorConfig& config, const bool backward) {
    MotionVectorExportContext ctx{};
    ctx.useChroma = config.useChroma;
    ctx.backward = backward;
    ctx.blockSizeX = config.blockSizeX;
    ctx.blockSizeY = config.blockSizeY;
    ctx.stepX = config.stepX;
    ctx.stepY = config.stepY;
    ctx.internalBlockSizeX = config.internalBlockSizeX;
    ctx.internalBlockSizeY = config.internalBlockSizeY;
    ctx.internalStepX = config.internalStepX;
    ctx.internalStepY = config.internalStepY;
    ctx.pel = config.pel;
    ctx.bits = config.bits;
    ctx.hPadding = config.hPadding;
    ctx.vPadding = config.vPadding;
    ctx.internalHPadding = config.internalHPadding;
    ctx.internalVPadding = config.internalVPadding;
    ctx.blkX = config.blkX;
    ctx.blkY = config.blkY;
    ctx.blockReduce = config.blockReduce;
    ctx.motionScaleX = config.motionScaleX;
    ctx.motionScaleY = config.motionScaleY;
    ctx.sadMultiplier = config.sadMultiplier;
    ctx.invalidSad = config.invalidSad;
    return ctx;
}

static RIFEFlowReduceConfig createRIFEFlowReduceConfig(const MotionVectorConfig& config) noexcept {
    RIFEFlowReduceConfig reduceConfig{};
    reduceConfig.blockCountX = config.blkX;
    reduceConfig.blockCountY = config.blkY;
    reduceConfig.internalBlockSizeX = config.internalBlockSizeX;
    reduceConfig.internalBlockSizeY = config.internalBlockSizeY;
    reduceConfig.internalStepX = config.internalStepX;
    reduceConfig.internalStepY = config.internalStepY;
    reduceConfig.internalHPadding = config.internalHPadding;
    reduceConfig.internalVPadding = config.internalVPadding;
    reduceConfig.blockReduce = config.blockReduce;
    return reduceConfig;
}

static RIFEGpuMotionVectorConfig createRIFEGpuMotionVectorConfig(const MotionVectorConfig& config) noexcept {
    RIFEGpuMotionVectorConfig vectorConfig{};
    vectorConfig.blockCountX = config.blkX;
    vectorConfig.blockCountY = config.blkY;
    vectorConfig.blockSizeX = config.blockSizeX;
    vectorConfig.blockSizeY = config.blockSizeY;
    vectorConfig.stepX = config.stepX;
    vectorConfig.stepY = config.stepY;
    vectorConfig.hPadding = config.hPadding;
    vectorConfig.vPadding = config.vPadding;
    vectorConfig.internalBlockSizeX = config.internalBlockSizeX;
    vectorConfig.internalBlockSizeY = config.internalBlockSizeY;
    vectorConfig.internalStepX = config.internalStepX;
    vectorConfig.internalStepY = config.internalStepY;
    vectorConfig.internalHPadding = config.internalHPadding;
    vectorConfig.internalVPadding = config.internalVPadding;
    vectorConfig.pel = config.pel;
    vectorConfig.blockReduce = config.blockReduce;
    vectorConfig.bits = config.bits;
    vectorConfig.useChroma = config.useChroma ? 1 : 0;
    vectorConfig.motionScaleX = config.motionScaleX;
    vectorConfig.motionScaleY = config.motionScaleY;
    return vectorConfig;
}

static void buildMotionVectorBlobFromConfig(const VSFrame* current, const VSFrame* reference, const float* flow,
                                            const int flowWidth, const int flowHeight,
                                            const bool valid, const MotionVectorConfig& config,
                                            const bool backward, std::vector<MVToolsVector>& vectors,
                                            std::vector<char>& blob, const VSAPI* vsapi,
                                            const std::vector<float>* currentLumaCache = nullptr,
                                            const std::vector<float>* referenceLumaCache = nullptr,
                                            MotionVectorFrameStats* const stats = nullptr,
                                            const bool includeSadStats = true,
                                            const bool includeMotionStats = true) {
    const auto ctx = createMotionVectorExportContext(config, backward);
    buildMVToolsVectorBlob(current, reference, flow, flowWidth, flowHeight, valid, ctx,
                           getMotionVectorAnalysisData(config, backward), vectors, blob, vsapi,
                           currentLumaCache, referenceLumaCache, stats, includeSadStats, includeMotionStats);
}

static void buildMotionVectorBlobsFromConfig(const VSFrame* current, const VSFrame* reference, const float* flow,
                                             const int flowWidth, const int flowHeight,
                                             const bool valid, const MotionVectorConfig& config, const VSAPI* vsapi,
                                             const std::vector<float>* currentLumaCache,
                                             const std::vector<float>* referenceLumaCache,
                                             std::vector<MVToolsVector>& backwardVectors,
                                             std::vector<MVToolsVector>& forwardVectors,
                                             std::vector<char>& backwardBlob, std::vector<char>& forwardBlob,
                                             MotionVectorFrameStats* const backwardStats = nullptr,
                                             MotionVectorFrameStats* const forwardStats = nullptr,
                                             const bool includeSadStats = true,
                                             const bool includeMotionStats = true) {
    const auto backwardCtx = createMotionVectorExportContext(config, true);
    const auto forwardCtx = createMotionVectorExportContext(config, false);
    const auto vectorCount = static_cast<size_t>(backwardCtx.blkX) * backwardCtx.blkY;
    backwardVectors.resize(vectorCount);
    forwardVectors.resize(vectorCount);

    if (!valid) {
        for (size_t i = 0; i < vectorCount; i++) {
            backwardVectors[i] = { 0, 0, backwardCtx.invalidSad };
            forwardVectors[i] = { 0, 0, forwardCtx.invalidSad };
        }

        packMotionVectorBlob(backwardVectors, false, getMotionVectorAnalysisData(config, true), backwardBlob, backwardStats, includeSadStats, includeMotionStats);
        packMotionVectorBlob(forwardVectors, false, getMotionVectorAnalysisData(config, false), forwardBlob, forwardStats, includeSadStats, includeMotionStats);
        return;
    }

    const auto width = vsapi->getFrameWidth(current, 0);
    const auto height = vsapi->getFrameHeight(current, 0);
    const auto currentLumaPtr = currentLumaCache ? currentLumaCache->data() : nullptr;
    const auto referenceLumaPtr = referenceLumaCache ? referenceLumaCache->data() : nullptr;
    const auto backwardSadContext = makeSADContext(current, reference, backwardCtx, vsapi, currentLumaPtr, referenceLumaPtr);
    const auto forwardSadContext = makeSADContext(reference, current, forwardCtx, vsapi, referenceLumaPtr, currentLumaPtr);
    const auto flowPlaneSize = flowWidth * flowHeight;
    const auto* backwardFlowXPlane = flow + flowPlaneSize * 0;
    const auto* backwardFlowYPlane = flow + flowPlaneSize * 1;
    const auto* forwardFlowXPlane = flow + flowPlaneSize * 2;
    const auto* forwardFlowYPlane = flow + flowPlaneSize * 3;

    for (auto by = 0; by < backwardCtx.blkY; by++) {
        const auto blockY = by * backwardCtx.stepY - backwardCtx.vPadding;
        const auto internalBlockY = by * backwardCtx.internalStepY - backwardCtx.internalVPadding;
        for (auto bx = 0; bx < backwardCtx.blkX; bx++) {
            const auto blockX = bx * backwardCtx.stepX - backwardCtx.hPadding;
            const auto internalBlockX = bx * backwardCtx.internalStepX - backwardCtx.internalHPadding;
            const auto vectorIndex = static_cast<size_t>(by) * backwardCtx.blkX + bx;
            auto& backwardVector = backwardVectors[vectorIndex];
            auto& forwardVector = forwardVectors[vectorIndex];

            const auto backwardFlowX = reduceBlockFlow(backwardFlowXPlane, flowWidth, flowHeight, internalBlockX, internalBlockY, backwardCtx);
            const auto backwardFlowY = reduceBlockFlow(backwardFlowYPlane, flowWidth, flowHeight, internalBlockX, internalBlockY, backwardCtx);
            backwardVector.x = static_cast<int>(std::lround(-2.0f * backwardFlowX * backwardCtx.motionScaleX * backwardCtx.pel));
            backwardVector.y = static_cast<int>(std::lround(-2.0f * backwardFlowY * backwardCtx.motionScaleY * backwardCtx.pel));
            backwardVector.x = clampMotionVectorComponent(backwardVector.x, backwardCtx.pel, blockX, backwardCtx.blockSizeX, width, backwardCtx.hPadding);
            backwardVector.y = clampMotionVectorComponent(backwardVector.y, backwardCtx.pel, blockY, backwardCtx.blockSizeY, height, backwardCtx.vPadding);
            const auto backwardPixelDx = static_cast<int>(std::lround(static_cast<double>(backwardVector.x) / backwardCtx.pel));
            const auto backwardPixelDy = static_cast<int>(std::lround(static_cast<double>(backwardVector.y) / backwardCtx.pel));
            backwardVector.sad = computeBlockSAD(backwardSadContext, backwardPixelDx, backwardPixelDy, blockX, blockY);

            const auto forwardFlowX = reduceBlockFlow(forwardFlowXPlane, flowWidth, flowHeight, internalBlockX, internalBlockY, forwardCtx);
            const auto forwardFlowY = reduceBlockFlow(forwardFlowYPlane, flowWidth, flowHeight, internalBlockX, internalBlockY, forwardCtx);
            forwardVector.x = static_cast<int>(std::lround(-2.0f * forwardFlowX * forwardCtx.motionScaleX * forwardCtx.pel));
            forwardVector.y = static_cast<int>(std::lround(-2.0f * forwardFlowY * forwardCtx.motionScaleY * forwardCtx.pel));
            forwardVector.x = clampMotionVectorComponent(forwardVector.x, forwardCtx.pel, blockX, forwardCtx.blockSizeX, width, forwardCtx.hPadding);
            forwardVector.y = clampMotionVectorComponent(forwardVector.y, forwardCtx.pel, blockY, forwardCtx.blockSizeY, height, forwardCtx.vPadding);
            const auto forwardPixelDx = static_cast<int>(std::lround(static_cast<double>(forwardVector.x) / forwardCtx.pel));
            const auto forwardPixelDy = static_cast<int>(std::lround(static_cast<double>(forwardVector.y) / forwardCtx.pel));
            forwardVector.sad = computeBlockSAD(forwardSadContext, forwardPixelDx, forwardPixelDy, blockX, blockY);
        }
    }

    packMotionVectorBlob(backwardVectors, true, getMotionVectorAnalysisData(config, true), backwardBlob, backwardStats, includeSadStats, includeMotionStats);
    packMotionVectorBlob(forwardVectors, true, getMotionVectorAnalysisData(config, false), forwardBlob, forwardStats, includeSadStats, includeMotionStats);
}

static void buildMotionVectorBlobsFromReducedFlow(const VSFrame* current, const VSFrame* reference, const RIFEReducedFlowBlock* reducedFlow,
                                                  const bool valid, const MotionVectorConfig& config, const VSAPI* vsapi,
                                                  const std::vector<float>* currentLumaCache,
                                                  const std::vector<float>* referenceLumaCache,
                                                  std::vector<MVToolsVector>& backwardVectors,
                                                  std::vector<MVToolsVector>& forwardVectors,
                                                  std::vector<char>& backwardBlob, std::vector<char>& forwardBlob,
                                                  MotionVectorFrameStats* const backwardStats = nullptr,
                                                  MotionVectorFrameStats* const forwardStats = nullptr,
                                                  const bool includeSadStats = true,
                                                  const bool includeMotionStats = true) {
    const auto backwardCtx = createMotionVectorExportContext(config, true);
    const auto forwardCtx = createMotionVectorExportContext(config, false);
    const auto vectorCount = static_cast<size_t>(backwardCtx.blkX) * backwardCtx.blkY;
    backwardVectors.resize(vectorCount);
    forwardVectors.resize(vectorCount);

    if (!valid) {
        for (size_t i = 0; i < vectorCount; i++) {
            backwardVectors[i] = { 0, 0, backwardCtx.invalidSad };
            forwardVectors[i] = { 0, 0, forwardCtx.invalidSad };
        }

        packMotionVectorBlob(backwardVectors, false, getMotionVectorAnalysisData(config, true), backwardBlob, backwardStats, includeSadStats, includeMotionStats);
        packMotionVectorBlob(forwardVectors, false, getMotionVectorAnalysisData(config, false), forwardBlob, forwardStats, includeSadStats, includeMotionStats);
        return;
    }

    const auto width = vsapi->getFrameWidth(current, 0);
    const auto height = vsapi->getFrameHeight(current, 0);
    const auto currentLumaPtr = currentLumaCache ? currentLumaCache->data() : nullptr;
    const auto referenceLumaPtr = referenceLumaCache ? referenceLumaCache->data() : nullptr;
    const auto backwardSadContext = makeSADContext(current, reference, backwardCtx, vsapi, currentLumaPtr, referenceLumaPtr);
    const auto forwardSadContext = makeSADContext(reference, current, forwardCtx, vsapi, referenceLumaPtr, currentLumaPtr);

    for (auto by = 0; by < backwardCtx.blkY; by++) {
        const auto blockY = by * backwardCtx.stepY - backwardCtx.vPadding;
        for (auto bx = 0; bx < backwardCtx.blkX; bx++) {
            const auto blockX = bx * backwardCtx.stepX - backwardCtx.hPadding;
            const auto vectorIndex = static_cast<size_t>(by) * backwardCtx.blkX + bx;
            const auto& reduced = reducedFlow[vectorIndex];
            auto& backwardVector = backwardVectors[vectorIndex];
            auto& forwardVector = forwardVectors[vectorIndex];

            backwardVector.x = static_cast<int>(std::lround(-2.0f * reduced.backwardX * backwardCtx.motionScaleX * backwardCtx.pel));
            backwardVector.y = static_cast<int>(std::lround(-2.0f * reduced.backwardY * backwardCtx.motionScaleY * backwardCtx.pel));
            backwardVector.x = clampMotionVectorComponent(backwardVector.x, backwardCtx.pel, blockX, backwardCtx.blockSizeX, width, backwardCtx.hPadding);
            backwardVector.y = clampMotionVectorComponent(backwardVector.y, backwardCtx.pel, blockY, backwardCtx.blockSizeY, height, backwardCtx.vPadding);
            const auto backwardPixelDx = static_cast<int>(std::lround(static_cast<double>(backwardVector.x) / backwardCtx.pel));
            const auto backwardPixelDy = static_cast<int>(std::lround(static_cast<double>(backwardVector.y) / backwardCtx.pel));
            backwardVector.sad = computeBlockSAD(backwardSadContext, backwardPixelDx, backwardPixelDy, blockX, blockY);

            forwardVector.x = static_cast<int>(std::lround(-2.0f * reduced.forwardX * forwardCtx.motionScaleX * forwardCtx.pel));
            forwardVector.y = static_cast<int>(std::lround(-2.0f * reduced.forwardY * forwardCtx.motionScaleY * forwardCtx.pel));
            forwardVector.x = clampMotionVectorComponent(forwardVector.x, forwardCtx.pel, blockX, forwardCtx.blockSizeX, width, forwardCtx.hPadding);
            forwardVector.y = clampMotionVectorComponent(forwardVector.y, forwardCtx.pel, blockY, forwardCtx.blockSizeY, height, forwardCtx.vPadding);
            const auto forwardPixelDx = static_cast<int>(std::lround(static_cast<double>(forwardVector.x) / forwardCtx.pel));
            const auto forwardPixelDy = static_cast<int>(std::lround(static_cast<double>(forwardVector.y) / forwardCtx.pel));
            forwardVector.sad = computeBlockSAD(forwardSadContext, forwardPixelDx, forwardPixelDy, blockX, blockY);
        }
    }

    packMotionVectorBlob(backwardVectors, true, getMotionVectorAnalysisData(config, true), backwardBlob, backwardStats, includeSadStats, includeMotionStats);
    packMotionVectorBlob(forwardVectors, true, getMotionVectorAnalysisData(config, false), forwardBlob, forwardStats, includeSadStats, includeMotionStats);
}

static void buildMotionVectorBlobsFromGpuVectors(const RIFEGpuMotionVector* gpuVectors,
                                                 const bool valid, const MotionVectorConfig& config,
                                                 std::vector<MVToolsVector>& backwardVectors,
                                                 std::vector<MVToolsVector>& forwardVectors,
                                                 std::vector<char>& backwardBlob, std::vector<char>& forwardBlob,
                                                 MotionVectorFrameStats* const backwardStats = nullptr,
                                                 MotionVectorFrameStats* const forwardStats = nullptr,
                                                 const bool includeSadStats = true,
                                                 const bool includeMotionStats = true) {
    const auto vectorCount = static_cast<size_t>(config.blkX) * config.blkY;
    backwardVectors.resize(vectorCount);
    forwardVectors.resize(vectorCount);

    if (!valid) {
        for (size_t i = 0; i < vectorCount; i++) {
            backwardVectors[i] = { 0, 0, config.invalidSad };
            forwardVectors[i] = { 0, 0, config.invalidSad };
        }

        packMotionVectorBlob(backwardVectors, false, getMotionVectorAnalysisData(config, true), backwardBlob, backwardStats, includeSadStats, includeMotionStats);
        packMotionVectorBlob(forwardVectors, false, getMotionVectorAnalysisData(config, false), forwardBlob, forwardStats, includeSadStats, includeMotionStats);
        return;
    }

    const auto applySadMultiplier = [&](const uint32_t rawSad) {
        return static_cast<int64_t>(static_cast<long double>(rawSad) * config.sadMultiplier + 0.5L);
    };

    const auto* backwardGpuVectors = gpuVectors;
    const auto* forwardGpuVectors = gpuVectors + vectorCount;
    for (size_t i = 0; i < vectorCount; i++) {
        backwardVectors[i] = { backwardGpuVectors[i].x, backwardGpuVectors[i].y, applySadMultiplier(backwardGpuVectors[i].rawSad) };
        forwardVectors[i] = { forwardGpuVectors[i].x, forwardGpuVectors[i].y, applySadMultiplier(forwardGpuVectors[i].rawSad) };
    }

    packMotionVectorBlob(backwardVectors, true, getMotionVectorAnalysisData(config, true), backwardBlob, backwardStats, includeSadStats, includeMotionStats);
    packMotionVectorBlob(forwardVectors, true, getMotionVectorAnalysisData(config, false), forwardBlob, forwardStats, includeSadStats, includeMotionStats);
}

static std::vector<char> buildInvalidMotionVectorBlob(const MotionVectorConfig& config, const bool backward,
                                                      MotionVectorFrameStats* const stats = nullptr,
                                                      const bool includeSadStats = true,
                                                      const bool includeMotionStats = true) {
    std::vector<MVToolsVector> vectors;
    std::vector<char> blob;
    buildMotionVectorBlobFromConfig(nullptr, nullptr, nullptr, 0, 0, false, config, backward,
                                    vectors, blob, nullptr, nullptr, nullptr, stats,
                                    includeSadStats, includeMotionStats);
    return blob;
}

static float sampleBilinearPlane(const float* data, const int width, const int height, float x, float y) noexcept {
    x = std::clamp(x, 0.0f, static_cast<float>(width - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(height - 1));

    const auto x0 = static_cast<int>(std::floor(x));
    const auto y0 = static_cast<int>(std::floor(y));
    const auto x1 = std::min(x0 + 1, width - 1);
    const auto y1 = std::min(y0 + 1, height - 1);
    const auto alpha = x - x0;
    const auto beta = y - y0;
    const auto row0 = static_cast<size_t>(y0) * width;
    const auto row1 = static_cast<size_t>(y1) * width;
    const auto top = data[row0 + x0] * (1.0f - alpha) + data[row0 + x1] * alpha;
    const auto bottom = data[row1 + x0] * (1.0f - alpha) + data[row1 + x1] * alpha;

    return top * (1.0f - beta) + bottom * beta;
}

static void buildDisplacementFromFlow(const float* flow, const int width, const int height,
                                      const int channelOffset, std::vector<float>& displacement) {
    const auto planeSize = static_cast<size_t>(width) * height;
    displacement.resize(planeSize * 2);

    for (size_t i = 0; i < planeSize; i++) {
        displacement[i] = -2.0f * flow[(static_cast<size_t>(channelOffset) + 0) * planeSize + i];
        displacement[planeSize + i] = -2.0f * flow[(static_cast<size_t>(channelOffset) + 1) * planeSize + i];
    }
}

static bool getDisplacementPlanes(const VSFrame* frame, const char* key, const int width, const int height,
                                  const float*& displacementX, const float*& displacementY,
                                  const VSAPI* vsapi) noexcept {
    const auto props = vsapi->getFramePropertiesRO(frame);
    int err{};
    const auto* data = vsapi->mapGetData(props, key, 0, &err);
    if (err)
        return false;

    const auto expectedSize = static_cast<int>(sizeof(float) * static_cast<size_t>(width) * height * 2);
    if (vsapi->mapGetDataSize(props, key, 0, nullptr) != expectedSize)
        return false;

    displacementX = reinterpret_cast<const float*>(data);
    displacementY = displacementX + static_cast<size_t>(width) * height;
    return true;
}

static void composeDisplacementSequence(const std::vector<const float*>& displacementXs,
                                        const std::vector<const float*>& displacementYs,
                                        const int width, const int height,
                                        std::vector<float>& composedX,
                                        std::vector<float>& composedY) {
    const auto planeSize = static_cast<size_t>(width) * height;
    composedX.assign(displacementXs.front(), displacementXs.front() + planeSize);
    composedY.assign(displacementYs.front(), displacementYs.front() + planeSize);

    for (size_t sequenceIndex = 1; sequenceIndex < displacementXs.size(); sequenceIndex++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                const auto index = static_cast<size_t>(y) * width + x;
                const auto sampleX = static_cast<float>(x) + composedX[index];
                const auto sampleY = static_cast<float>(y) + composedY[index];
                composedX[index] += sampleBilinearPlane(displacementXs[sequenceIndex], width, height, sampleX, sampleY);
                composedY[index] += sampleBilinearPlane(displacementYs[sequenceIndex], width, height, sampleX, sampleY);
            }
        }
    }
}

static void buildMotionVectorBlobFromDisplacement(const VSFrame* current, const VSFrame* reference,
                                                  const float* displacementX, const float* displacementY,
                                                  const int displacementWidth, const int displacementHeight,
                                                  const bool valid, const MotionVectorConfig& config,
                                                  const bool backward, std::vector<MVToolsVector>& vectors,
                                                  std::vector<char>& blob, const VSAPI* vsapi,
                                                  const std::vector<float>* currentLumaCache = nullptr,
                                                  const std::vector<float>* referenceLumaCache = nullptr,
                                                  MotionVectorFrameStats* const stats = nullptr,
                                                  const bool includeSadStats = true,
                                                  const bool includeMotionStats = true) {
    const auto ctx = createMotionVectorExportContext(config, backward);
    const auto vectorCount = static_cast<size_t>(ctx.blkX) * ctx.blkY;
    vectors.resize(vectorCount);

    if (!valid) {
        for (auto& vector : vectors) {
            vector.x = 0;
            vector.y = 0;
            vector.sad = ctx.invalidSad;
        }
        packMotionVectorBlob(vectors, false, getMotionVectorAnalysisData(config, backward), blob, stats, includeSadStats, includeMotionStats);
        return;
    }

    const auto width = vsapi->getFrameWidth(current, 0);
    const auto height = vsapi->getFrameHeight(current, 0);
    const auto stride = static_cast<int>(vsapi->getStride(current, 0) / vsapi->getVideoFrameFormat(current)->bytesPerSample);
    const auto sadMaxSample = static_cast<double>((1ULL << ctx.bits) - 1ULL);
    std::vector<float> currentLuma;
    std::vector<float> referenceLuma;
    const float* currentLumaPtr = nullptr;
    const float* referenceLumaPtr = nullptr;
    if (!ctx.useChroma) {
        if (currentLumaCache && currentLumaCache->size() >= static_cast<size_t>(stride) * height) {
            currentLumaPtr = currentLumaCache->data();
        } else {
            buildFrameLumaPlane(current, width, height, stride, currentLuma, sadMaxSample, vsapi);
            currentLumaPtr = currentLuma.data();
        }

        if (referenceLumaCache && referenceLumaCache->size() >= static_cast<size_t>(stride) * height) {
            referenceLumaPtr = referenceLumaCache->data();
        } else {
            buildFrameLumaPlane(reference, width, height, stride, referenceLuma, sadMaxSample, vsapi);
            referenceLumaPtr = referenceLuma.data();
        }
    }

    const auto sadContext = makeSADContext(current, reference, ctx, vsapi, currentLumaPtr, referenceLumaPtr);
    for (auto by = 0; by < ctx.blkY; by++) {
        const auto blockY = by * ctx.stepY - ctx.vPadding;
        const auto internalBlockY = by * ctx.internalStepY - ctx.internalVPadding;
        for (auto bx = 0; bx < ctx.blkX; bx++) {
            const auto blockX = bx * ctx.stepX - ctx.hPadding;
            const auto internalBlockX = bx * ctx.internalStepX - ctx.internalHPadding;
            auto& vector = vectors[static_cast<size_t>(by) * ctx.blkX + bx];
            const auto pixelDx = reduceBlockFlow(displacementX, displacementWidth, displacementHeight, internalBlockX, internalBlockY, ctx) * ctx.motionScaleX;
            const auto pixelDy = reduceBlockFlow(displacementY, displacementWidth, displacementHeight, internalBlockX, internalBlockY, ctx) * ctx.motionScaleY;

            vector.x = static_cast<int>(std::lround(pixelDx * ctx.pel));
            vector.y = static_cast<int>(std::lround(pixelDy * ctx.pel));
            vector.x = clampMotionVectorComponent(vector.x, ctx.pel, blockX, ctx.blockSizeX, width, ctx.hPadding);
            vector.y = clampMotionVectorComponent(vector.y, ctx.pel, blockY, ctx.blockSizeY, height, ctx.vPadding);
            const auto vectorPixelDx = static_cast<int>(std::lround(static_cast<double>(vector.x) / ctx.pel));
            const auto vectorPixelDy = static_cast<int>(std::lround(static_cast<double>(vector.y) / ctx.pel));
            vector.sad = computeBlockSAD(sadContext, vectorPixelDx, vectorPixelDy, blockX, blockY);
        }
    }

    packMotionVectorBlob(vectors, true, getMotionVectorAnalysisData(config, backward), blob, stats, includeSadStats, includeMotionStats);
}

static void zeroMotionVectorFrame(VSFrame* frame, const VSVideoInfo& vi, const VSAPI* vsapi);

static VSFrame* createMotionVectorFrame(const VSVideoInfo& vi, const MVAnalysisData& analysisData,
                                        const char* vectorBlob, const int vectorBlobSize,
                                        const MotionVectorFrameStats& stats, const int absSadClipRange,
                                        const bool renderSadMask,
                                        const bool includeSadStats, const bool includeMotionStats,
                                        MotionVectorPerfStats* const perf,
                                        VSCore* core, const VSAPI* vsapi) {
    auto dst = vsapi->newVideoFrame(&vi.format, vi.width, vi.height, nullptr, core);
    if (renderSadMask) {
        const auto renderSadMaskStartNs = perf ? monotonicNowNs() : 0;
        renderMotionVectorSADMask(dst, vi, vectorBlob, vectorBlobSize, analysisData, absSadClipRange, vsapi);
        if (perf)
            accumulatePerfStat(perf->renderSadMaskNs, monotonicNowNs() - renderSadMaskStartNs);
    } else {
        zeroMotionVectorFrame(dst, vi, vsapi);
    }
    auto props = vsapi->getFramePropertiesRW(dst);
    setMotionVectorProperties(props, analysisData, vectorBlob, vectorBlobSize, stats, includeSadStats, includeMotionStats, vsapi);
    return dst;
}

static void zeroMotionVectorFrame(VSFrame* frame, const VSVideoInfo& vi, const VSAPI* vsapi) {
    auto* dstp = vsapi->getWritePtr(frame, 0);
    const auto dstStride = vsapi->getStride(frame, 0);
    for (auto y = 0; y < vi.height; y++)
        std::memset(dstp + static_cast<size_t>(y) * dstStride, 0, vi.width * vi.format.bytesPerSample);
}

static const VSFrame* VS_CC rifeMVPairGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData,
                                               VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEMVPairData*>(instanceData) };
    const auto delta = d->mvConfig.delta;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        vsapi->requestFrameFilter(n, d->sourceNode, frameCtx);
        if (n + delta < d->vi.numFrames) {
            vsapi->requestFrameFilter(n + delta, d->node, frameCtx);
            vsapi->requestFrameFilter(n + delta, d->sourceNode, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const auto pairStartNs = d->perfStats ? monotonicNowNs() : 0;
        auto current = vsapi->getFrameFilter(n, d->node, frameCtx);
        auto currentSource = vsapi->getFrameFilter(n, d->sourceNode, frameCtx);
        const VSFrame* reference = n + delta < d->vi.numFrames ? vsapi->getFrameFilter(n + delta, d->node, frameCtx) : nullptr;
        const VSFrame* referenceSource = n + delta < d->vi.numFrames ? vsapi->getFrameFilter(n + delta, d->sourceNode, frameCtx) : nullptr;

        auto dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, nullptr, core);
        zeroMotionVectorFrame(dst, d->vi, vsapi);
        auto props = vsapi->getFramePropertiesRW(dst);
        auto& scratch = getMotionVectorScratchBuffers();

        auto& backwardVectors = scratch.backwardVectors;
        auto& forwardVectors = scratch.forwardVectors;
        auto& backwardBlob = scratch.backwardBlob;
        auto& forwardBlob = scratch.forwardBlob;
        MotionVectorFrameStats backwardStats{};
        MotionVectorFrameStats forwardStats{};
        const auto needStats = d->sadStats || d->motionStats;
        if (reference) {
            const auto width = vsapi->getFrameWidth(current, 0);
            const auto height = vsapi->getFrameHeight(current, 0);
            const auto useGpuFlowReduce = d->mvExportBackend == MotionVectorExportBackend::GpuFlowReduce;
            const auto useGpuFull = d->mvExportBackend == MotionVectorExportBackend::GpuFull;
            RIFEFlowReduceConfig reduceConfig{};
            RIFEGpuMotionVectorConfig gpuVectorConfig{};
            if (useGpuFlowReduce) {
                reduceConfig = createRIFEFlowReduceConfig(d->mvConfig);
                scratch.reducedFlow.resize(static_cast<size_t>(reduceConfig.blockCountX) * reduceConfig.blockCountY);
            } else if (useGpuFull) {
                gpuVectorConfig = createRIFEGpuMotionVectorConfig(d->mvConfig);
                scratch.gpuVectors.resize(static_cast<size_t>(gpuVectorConfig.blockCountX) * gpuVectorConfig.blockCountY * 2);
            } else {
                const auto flowSize = static_cast<size_t>(width) * height * 4;
                scratch.flow.resize(flowSize);
            }
            std::shared_ptr<const std::vector<float>> currentLumaPlane;
            std::shared_ptr<const std::vector<float>> referenceLumaPlane;
            const std::vector<float>* currentLumaCache = nullptr;
            const std::vector<float>* referenceLumaCache = nullptr;
            const auto currentPacked = getOrCreatePackedInferenceFrame(d->packedCache, current, n, width, height,
                                                                       d->perfStats ? d->perf.get() : nullptr, vsapi);
            const auto referencePacked = getOrCreatePackedInferenceFrame(d->packedCache, reference, n + delta, width, height,
                                                                         d->perfStats ? d->perf.get() : nullptr, vsapi);

            int64_t semaphoreWaitNs{};
            int64_t localSemaphoreWaitNs{};
            int64_t sharedSemaphoreWaitNs{};
            int64_t rifeProcessWallNs{};
            FlowPerfBreakdown flowPerf{};
            const auto processFlowStartNs = d->perfStats ? monotonicNowNs() : 0;
            int status{};
            if (useGpuFlowReduce) {
                status = processReducedFlowWithSemaphores(d->rife.get(), d->semaphore.get(), d->sharedFlowSemaphore.get(),
                                                          *currentPacked, *referencePacked,
                                                          scratch.reducedFlow.data(), reduceConfig,
                                                          d->perfStats ? &semaphoreWaitNs : nullptr,
                                                          d->perfStats ? &localSemaphoreWaitNs : nullptr,
                                                          d->perfStats ? &sharedSemaphoreWaitNs : nullptr,
                                                          d->perfStats ? &rifeProcessWallNs : nullptr,
                                                          d->perfStats ? &flowPerf : nullptr);
            } else if (useGpuFull) {
                status = processGpuMotionVectorsWithSemaphores(d->rife.get(), d->semaphore.get(), d->sharedFlowSemaphore.get(),
                                                               *currentPacked, *referencePacked,
                                                               scratch.gpuVectors.data(), gpuVectorConfig,
                                                               d->perfStats ? &semaphoreWaitNs : nullptr,
                                                               d->perfStats ? &localSemaphoreWaitNs : nullptr,
                                                               d->perfStats ? &sharedSemaphoreWaitNs : nullptr,
                                                               d->perfStats ? &rifeProcessWallNs : nullptr,
                                                               d->perfStats ? &flowPerf : nullptr);
            } else {
                status = processFlowWithSemaphores(d->rife.get(), d->semaphore.get(), d->sharedFlowSemaphore.get(),
                                                   *currentPacked, *referencePacked,
                                                   scratch.flow.data(),
                                                   d->perfStats ? &semaphoreWaitNs : nullptr,
                                                   d->perfStats ? &localSemaphoreWaitNs : nullptr,
                                                   d->perfStats ? &sharedSemaphoreWaitNs : nullptr,
                                                   d->perfStats ? &rifeProcessWallNs : nullptr,
                                                   d->perfStats ? &flowPerf : nullptr);
            }
            if (d->perfStats) {
                accumulatePerfStat(d->perf->semaphoreWaitNs, semaphoreWaitNs);
                accumulatePerfStat(d->perf->localSemaphoreWaitNs, localSemaphoreWaitNs);
                accumulatePerfStat(d->perf->sharedSemaphoreWaitNs, sharedSemaphoreWaitNs);
                accumulatePerfStat(d->perf->flowCalls, 1);
                accumulatePerfStat(d->perf->processFlowNs, monotonicNowNs() - processFlowStartNs);
                accumulatePerfStat(d->perf->rifeProcessWallNs, rifeProcessWallNs);
                accumulatePerfStat(d->perf->flowSetupNs, flowPerf.setupNs);
                accumulatePerfStat(d->perf->flowCpuPrepNs, flowPerf.cpuPrepNs);
                accumulatePerfStat(d->perf->flowCommandRecordNs, flowPerf.commandRecordNs);
                accumulatePerfStat(d->perf->flowUploadRecordNs, flowPerf.uploadRecordNs);
                accumulatePerfStat(d->perf->flowPreprocRecordNs, flowPerf.preprocRecordNs);
                accumulatePerfStat(d->perf->flowInferenceRecordNs, flowPerf.inferenceRecordNs);
                accumulatePerfStat(d->perf->flowOutputRecordNs, flowPerf.outputRecordNs);
                accumulatePerfStat(d->perf->flowReduceRecordNs, flowPerf.flowReduceRecordNs);
                accumulatePerfStat(d->perf->flowVectorRecordNs, flowPerf.flowVectorRecordNs);
                accumulatePerfStat(d->perf->flowReadbackRecordNs, flowPerf.readbackRecordNs);
                accumulatePerfStat(d->perf->flowSubmitWaitNs, flowPerf.submitWaitNs);
                accumulatePerfStat(d->perf->flowReadbackInvalidateNs, flowPerf.readbackInvalidateNs);
                accumulatePerfStat(d->perf->flowReadbackMapNs, flowPerf.readbackMapNs);
                accumulatePerfStat(d->perf->flowUnpackNs, flowPerf.unpackNs);
                accumulatePerfStat(d->perf->flowExportDirectNs, flowPerf.exportDirectNs);
                accumulatePerfStat(d->perf->flowExportResizeNs, flowPerf.exportResizeNs);
                accumulatePerfStat(d->perf->flowCleanupNs, flowPerf.cleanupNs);
                accumulatePerfStat(d->perf->flowReadbackBytes, flowPerf.readbackBytes);
            }
            if (status != 0) {
                vsapi->freeFrame(current);
                vsapi->freeFrame(reference);
                vsapi->freeFrame(currentSource);
                vsapi->freeFrame(referenceSource);
                vsapi->freeFrame(dst);
                vsapi->setFilterError("RIFEMV: failed to export motion vectors", frameCtx);
                return nullptr;
            }

            if (!useGpuFull && !d->mvConfig.useChroma) {
                const auto lumaStartNs = d->perfStats ? monotonicNowNs() : 0;
                const auto sourceWidth = vsapi->getFrameWidth(currentSource, 0);
                const auto sourceHeight = vsapi->getFrameHeight(currentSource, 0);
                const auto sourceStride = static_cast<int>(vsapi->getStride(currentSource, 0) / vsapi->getVideoFrameFormat(currentSource)->bytesPerSample);
                const auto maxSample = static_cast<double>((1ULL << d->mvConfig.bits) - 1ULL);
                currentLumaPlane = getOrCreateFrameLumaPlane(d->lumaCache, currentSource, n, sourceWidth, sourceHeight, sourceStride, maxSample, vsapi);
                referenceLumaPlane = getOrCreateFrameLumaPlane(d->lumaCache, referenceSource, n + delta, sourceWidth, sourceHeight, sourceStride, maxSample, vsapi);
                currentLumaCache = currentLumaPlane.get();
                referenceLumaCache = referenceLumaPlane.get();
                if (d->perfStats)
                    accumulatePerfStat(d->perf->lumaBuildNs, monotonicNowNs() - lumaStartNs);
            }

            const auto vectorPackStartNs = d->perfStats ? monotonicNowNs() : 0;
            if (useGpuFlowReduce) {
                buildMotionVectorBlobsFromReducedFlow(currentSource, referenceSource, scratch.reducedFlow.data(), true, d->mvConfig, vsapi,
                                                      currentLumaCache, referenceLumaCache, backwardVectors, forwardVectors,
                                                      backwardBlob, forwardBlob,
                                                      needStats ? &backwardStats : nullptr,
                                                      needStats ? &forwardStats : nullptr,
                                                      d->sadStats, d->motionStats);
            } else if (useGpuFull) {
                buildMotionVectorBlobsFromGpuVectors(scratch.gpuVectors.data(), true, d->mvConfig,
                                                     backwardVectors, forwardVectors, backwardBlob, forwardBlob,
                                                     needStats ? &backwardStats : nullptr,
                                                     needStats ? &forwardStats : nullptr,
                                                     d->sadStats, d->motionStats);
            } else {
                buildMotionVectorBlobsFromConfig(currentSource, referenceSource, scratch.flow.data(), width, height, true, d->mvConfig, vsapi,
                                                 currentLumaCache, referenceLumaCache, backwardVectors, forwardVectors,
                                                 backwardBlob, forwardBlob,
                                                 needStats ? &backwardStats : nullptr,
                                                 needStats ? &forwardStats : nullptr,
                                                 d->sadStats, d->motionStats);
            }
            if (d->perfStats)
                accumulatePerfStat(d->perf->vectorPackNs, monotonicNowNs() - vectorPackStartNs);
        } else {
            buildMotionVectorBlobFromConfig(nullptr, nullptr, nullptr, 0, 0, false, d->mvConfig, true,
                                            backwardVectors, backwardBlob, nullptr, nullptr, nullptr,
                                            needStats ? &backwardStats : nullptr, d->sadStats, d->motionStats);
            buildMotionVectorBlobFromConfig(nullptr, nullptr, nullptr, 0, 0, false, d->mvConfig, false,
                                            forwardVectors, forwardBlob, nullptr, nullptr, nullptr,
                                            needStats ? &forwardStats : nullptr, d->sadStats, d->motionStats);
        }

        vsapi->mapSetData(props, RIFEMVBackwardVectorsInternalKey, backwardBlob.data(), static_cast<int>(backwardBlob.size()), dtBinary, maReplace);
        vsapi->mapSetData(props, RIFEMVForwardVectorsInternalKey, forwardBlob.data(), static_cast<int>(forwardBlob.size()), dtBinary, maReplace);
        if (needStats) {
            setMotionVectorInternalFrameStats(props, backwardStats, d->sadStats, d->motionStats, true, vsapi);
            setMotionVectorInternalFrameStats(props, forwardStats, d->sadStats, d->motionStats, false, vsapi);
        }

        vsapi->freeFrame(current);
        vsapi->freeFrame(reference);
        vsapi->freeFrame(currentSource);
        vsapi->freeFrame(referenceSource);
        if (d->perfStats) {
            accumulatePerfStat(d->perf->pairFrames, 1);
            accumulatePerfStat(d->perf->pairTotalNs, monotonicNowNs() - pairStartNs);
        }
        return dst;
    }

    return nullptr;
}

static const VSFrame* VS_CC rifeMVOutputGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData,
                                                 VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEMVOutputData*>(instanceData) };
    const auto delta = d->analysisData.nDeltaFrame;
    const auto pairIndex = d->backward ? n : n - delta;

    if (activationReason == arInitial) {
        if (pairIndex >= 0 && pairIndex < d->vi.numFrames) {
            vsapi->requestFrameFilter(pairIndex, d->node, frameCtx);
        } else {
            return createMotionVectorFrame(d->vi, d->analysisData, d->invalidBlob.data(),
                                           static_cast<int>(d->invalidBlob.size()), d->invalidStats,
                                           d->absSadClipRange, d->renderSadMask, d->sadStats, d->motionStats, d->perf.get(), core, vsapi);
        }
    } else if (activationReason == arAllFramesReady) {
        const auto outputStartNs = d->perfStats ? monotonicNowNs() : 0;
        const VSFrame* pairFrame{};
        if (pairIndex >= 0 && pairIndex < d->vi.numFrames)
            pairFrame = vsapi->getFrameFilter(pairIndex, d->node, frameCtx);

        const char* vectorBlob = nullptr;
        int vectorBlobSize{};
        auto stats = d->invalidStats;
        if (pairFrame) {
            const auto pairProps = vsapi->getFramePropertiesRO(pairFrame);
            const auto blobKey = d->backward ? RIFEMVBackwardVectorsInternalKey : RIFEMVForwardVectorsInternalKey;
            vectorBlob = vsapi->mapGetData(pairProps, blobKey, 0, nullptr);
            vectorBlobSize = vsapi->mapGetDataSize(pairProps, blobKey, 0, nullptr);
            if (d->sadStats || d->motionStats)
                stats = getMotionVectorInternalFrameStats(pairProps, d->backward, d->sadStats, d->motionStats, vsapi);
        } else {
            vectorBlob = d->invalidBlob.data();
            vectorBlobSize = static_cast<int>(d->invalidBlob.size());
        }

        auto dst = createMotionVectorFrame(d->vi, d->analysisData, vectorBlob, vectorBlobSize, stats,
                                           d->absSadClipRange, d->renderSadMask, d->sadStats, d->motionStats, d->perf.get(), core, vsapi);

        vsapi->freeFrame(pairFrame);
        if (d->perfStats) {
            accumulatePerfStat(d->perf->outputFrames, 1);
            accumulatePerfStat(d->perf->outputTotalNs, monotonicNowNs() - outputStartNs);
        }
        return dst;
    }

    return nullptr;
}

static const VSFrame* VS_CC rifeMVApproxPairGetFrame(int n, int activationReason, void* instanceData,
                                                     [[maybe_unused]] void** frameData,
                                                     VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEMVApproxPairData*>(instanceData) };

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        vsapi->requestFrameFilter(n, d->sourceNode, frameCtx);
        if (n + 1 < d->vi.numFrames) {
            vsapi->requestFrameFilter(n + 1, d->node, frameCtx);
            vsapi->requestFrameFilter(n + 1, d->sourceNode, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const auto pairStartNs = d->perfStats ? monotonicNowNs() : 0;
        auto current = vsapi->getFrameFilter(n, d->node, frameCtx);
        auto currentSource = vsapi->getFrameFilter(n, d->sourceNode, frameCtx);
        const VSFrame* reference = n + 1 < d->vi.numFrames ? vsapi->getFrameFilter(n + 1, d->node, frameCtx) : nullptr;
        const VSFrame* referenceSource = n + 1 < d->vi.numFrames ? vsapi->getFrameFilter(n + 1, d->sourceNode, frameCtx) : nullptr;

        auto dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, nullptr, core);
        zeroMotionVectorFrame(dst, d->vi, vsapi);
        auto props = vsapi->getFramePropertiesRW(dst);
        auto& scratch = getMotionVectorScratchBuffers();

        auto& backwardVectors = scratch.backwardVectors;
        auto& forwardVectors = scratch.forwardVectors;
        auto& backwardBlob = scratch.backwardBlob;
        auto& forwardBlob = scratch.forwardBlob;
        MotionVectorFrameStats backwardStats{};
        MotionVectorFrameStats forwardStats{};
        auto& backwardDisplacement = scratch.backwardDisplacement;
        auto& forwardDisplacement = scratch.forwardDisplacement;
        const auto planeSize = static_cast<size_t>(d->mvConfig.inferenceWidth) * d->mvConfig.inferenceHeight;
        const auto needStats = d->sadStats || d->motionStats;

        if (reference) {
            const auto width = vsapi->getFrameWidth(current, 0);
            const auto height = vsapi->getFrameHeight(current, 0);
            const auto flowSize = static_cast<size_t>(width) * height * 4;
            scratch.flow.resize(flowSize);
            std::shared_ptr<const std::vector<float>> currentLumaPlane;
            std::shared_ptr<const std::vector<float>> referenceLumaPlane;
            const std::vector<float>* currentLumaCache = nullptr;
            const std::vector<float>* referenceLumaCache = nullptr;
            const auto currentPacked = getOrCreatePackedInferenceFrame(d->packedCache, current, n, width, height,
                                                                       d->perfStats ? d->perf.get() : nullptr, vsapi);
            const auto referencePacked = getOrCreatePackedInferenceFrame(d->packedCache, reference, n + 1, width, height,
                                                                         d->perfStats ? d->perf.get() : nullptr, vsapi);

            int64_t semaphoreWaitNs{};
            int64_t localSemaphoreWaitNs{};
            int64_t sharedSemaphoreWaitNs{};
            int64_t rifeProcessWallNs{};
            FlowPerfBreakdown flowPerf{};
            const auto processFlowStartNs = d->perfStats ? monotonicNowNs() : 0;
            const auto status = processFlowWithSemaphores(d->rife.get(), d->semaphore.get(), d->sharedFlowSemaphore.get(),
                                                          *currentPacked, *referencePacked,
                                                          scratch.flow.data(),
                                                          d->perfStats ? &semaphoreWaitNs : nullptr,
                                                          d->perfStats ? &localSemaphoreWaitNs : nullptr,
                                                          d->perfStats ? &sharedSemaphoreWaitNs : nullptr,
                                                          d->perfStats ? &rifeProcessWallNs : nullptr,
                                                          d->perfStats ? &flowPerf : nullptr);
            if (d->perfStats) {
                accumulatePerfStat(d->perf->semaphoreWaitNs, semaphoreWaitNs);
                accumulatePerfStat(d->perf->localSemaphoreWaitNs, localSemaphoreWaitNs);
                accumulatePerfStat(d->perf->sharedSemaphoreWaitNs, sharedSemaphoreWaitNs);
                accumulatePerfStat(d->perf->flowCalls, 1);
                accumulatePerfStat(d->perf->processFlowNs, monotonicNowNs() - processFlowStartNs);
                accumulatePerfStat(d->perf->rifeProcessWallNs, rifeProcessWallNs);
                accumulatePerfStat(d->perf->flowSetupNs, flowPerf.setupNs);
                accumulatePerfStat(d->perf->flowCpuPrepNs, flowPerf.cpuPrepNs);
                accumulatePerfStat(d->perf->flowCommandRecordNs, flowPerf.commandRecordNs);
                accumulatePerfStat(d->perf->flowUploadRecordNs, flowPerf.uploadRecordNs);
                accumulatePerfStat(d->perf->flowPreprocRecordNs, flowPerf.preprocRecordNs);
                accumulatePerfStat(d->perf->flowInferenceRecordNs, flowPerf.inferenceRecordNs);
                accumulatePerfStat(d->perf->flowOutputRecordNs, flowPerf.outputRecordNs);
                accumulatePerfStat(d->perf->flowReduceRecordNs, flowPerf.flowReduceRecordNs);
                accumulatePerfStat(d->perf->flowVectorRecordNs, flowPerf.flowVectorRecordNs);
                accumulatePerfStat(d->perf->flowReadbackRecordNs, flowPerf.readbackRecordNs);
                accumulatePerfStat(d->perf->flowSubmitWaitNs, flowPerf.submitWaitNs);
                accumulatePerfStat(d->perf->flowReadbackInvalidateNs, flowPerf.readbackInvalidateNs);
                accumulatePerfStat(d->perf->flowReadbackMapNs, flowPerf.readbackMapNs);
                accumulatePerfStat(d->perf->flowUnpackNs, flowPerf.unpackNs);
                accumulatePerfStat(d->perf->flowExportDirectNs, flowPerf.exportDirectNs);
                accumulatePerfStat(d->perf->flowExportResizeNs, flowPerf.exportResizeNs);
                accumulatePerfStat(d->perf->flowCleanupNs, flowPerf.cleanupNs);
                accumulatePerfStat(d->perf->flowReadbackBytes, flowPerf.readbackBytes);
            }
            if (status != 0) {
                vsapi->freeFrame(current);
                vsapi->freeFrame(reference);
                vsapi->freeFrame(currentSource);
                vsapi->freeFrame(referenceSource);
                vsapi->freeFrame(dst);
                vsapi->setFilterError("RIFEMVApprox: failed to export motion vectors", frameCtx);
                return nullptr;
            }

            if (!d->mvConfig.useChroma) {
                const auto lumaStartNs = d->perfStats ? monotonicNowNs() : 0;
                const auto sourceWidth = vsapi->getFrameWidth(currentSource, 0);
                const auto sourceHeight = vsapi->getFrameHeight(currentSource, 0);
                const auto sourceStride = static_cast<int>(vsapi->getStride(currentSource, 0) / vsapi->getVideoFrameFormat(currentSource)->bytesPerSample);
                const auto maxSample = static_cast<double>((1ULL << d->mvConfig.bits) - 1ULL);
                currentLumaPlane = getOrCreateFrameLumaPlane(d->lumaCache, currentSource, n, sourceWidth, sourceHeight, sourceStride, maxSample, vsapi);
                referenceLumaPlane = getOrCreateFrameLumaPlane(d->lumaCache, referenceSource, n + 1, sourceWidth, sourceHeight, sourceStride, maxSample, vsapi);
                currentLumaCache = currentLumaPlane.get();
                referenceLumaCache = referenceLumaPlane.get();
                if (d->perfStats)
                    accumulatePerfStat(d->perf->lumaBuildNs, monotonicNowNs() - lumaStartNs);
            }

            const auto vectorPackStartNs = d->perfStats ? monotonicNowNs() : 0;
            buildMotionVectorBlobsFromConfig(currentSource, referenceSource, scratch.flow.data(), width, height, true, d->mvConfig, vsapi,
                                             currentLumaCache, referenceLumaCache, backwardVectors, forwardVectors,
                                             backwardBlob, forwardBlob,
                                             needStats ? &backwardStats : nullptr,
                                             needStats ? &forwardStats : nullptr,
                                             d->sadStats, d->motionStats);
            if (d->perfStats)
                accumulatePerfStat(d->perf->vectorPackNs, monotonicNowNs() - vectorPackStartNs);
            const auto displacementBuildStartNs = d->perfStats ? monotonicNowNs() : 0;
            buildDisplacementFromFlow(scratch.flow.data(), width, height, 0, backwardDisplacement);
            buildDisplacementFromFlow(scratch.flow.data(), width, height, 2, forwardDisplacement);
            if (d->perfStats)
                accumulatePerfStat(d->perf->displacementBuildNs, monotonicNowNs() - displacementBuildStartNs);
        } else {
            buildMotionVectorBlobFromConfig(nullptr, nullptr, nullptr, 0, 0, false, d->mvConfig, true,
                                            backwardVectors, backwardBlob, nullptr, nullptr, nullptr,
                                            needStats ? &backwardStats : nullptr, d->sadStats, d->motionStats);
            buildMotionVectorBlobFromConfig(nullptr, nullptr, nullptr, 0, 0, false, d->mvConfig, false,
                                            forwardVectors, forwardBlob, nullptr, nullptr, nullptr,
                                            needStats ? &forwardStats : nullptr, d->sadStats, d->motionStats);
            backwardDisplacement.assign(planeSize * 2, 0.0f);
            forwardDisplacement.assign(planeSize * 2, 0.0f);
        }

        vsapi->mapSetData(props, RIFEMVBackwardVectorsInternalKey, backwardBlob.data(), static_cast<int>(backwardBlob.size()), dtBinary, maReplace);
        vsapi->mapSetData(props, RIFEMVForwardVectorsInternalKey, forwardBlob.data(), static_cast<int>(forwardBlob.size()), dtBinary, maReplace);
        if (needStats) {
            setMotionVectorInternalFrameStats(props, backwardStats, d->sadStats, d->motionStats, true, vsapi);
            setMotionVectorInternalFrameStats(props, forwardStats, d->sadStats, d->motionStats, false, vsapi);
        }
        vsapi->mapSetData(props, RIFEMVBackwardDisplacementInternalKey,
                          reinterpret_cast<const char*>(backwardDisplacement.data()),
                          static_cast<int>(backwardDisplacement.size() * sizeof(float)), dtBinary, maReplace);
        vsapi->mapSetData(props, RIFEMVForwardDisplacementInternalKey,
                          reinterpret_cast<const char*>(forwardDisplacement.data()),
                          static_cast<int>(forwardDisplacement.size() * sizeof(float)), dtBinary, maReplace);

        vsapi->freeFrame(current);
        vsapi->freeFrame(reference);
        vsapi->freeFrame(currentSource);
        vsapi->freeFrame(referenceSource);
        if (d->perfStats) {
            accumulatePerfStat(d->perf->pairFrames, 1);
            accumulatePerfStat(d->perf->pairTotalNs, monotonicNowNs() - pairStartNs);
        }
        return dst;
    }

    return nullptr;
}

static const VSFrame* VS_CC rifeMVApproxOutputGetFrame(int n, int activationReason, void* instanceData,
                                                       [[maybe_unused]] void** frameData,
                                                       VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEMVApproxOutputData*>(instanceData) };
    const auto delta = d->analysisData.nDeltaFrame;
    const auto valid = d->backward ? n + delta < d->vi.numFrames : n >= delta;
    const auto createInvalidFrame = [&]() {
        return createMotionVectorFrame(d->vi, d->analysisData, d->invalidBlob.data(), static_cast<int>(d->invalidBlob.size()), d->invalidStats,
                                       d->absSadClipRange, d->renderSadMask, d->sadStats, d->motionStats, d->perf.get(), core, vsapi);
    };

    if (activationReason == arInitial) {
        if (!valid)
            return createInvalidFrame();

        for (auto i = 0; i < delta; i++) {
            const auto pairIndex = d->backward ? n + i : n - 1 - i;
            vsapi->requestFrameFilter(pairIndex, d->node, frameCtx);
        }

        if (delta > 1) {
            vsapi->requestFrameFilter(n, d->sourceNode, frameCtx);
            vsapi->requestFrameFilter(d->backward ? n + delta : n - delta, d->sourceNode, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const auto outputStartNs = d->perfStats ? monotonicNowNs() : 0;
        if (!valid) {
            auto dst = createInvalidFrame();
            if (d->perfStats) {
                accumulatePerfStat(d->perf->outputFrames, 1);
                accumulatePerfStat(d->perf->outputTotalNs, monotonicNowNs() - outputStartNs);
            }
            return dst;
        }

        std::vector<const VSFrame*> pairFrames(delta);
        const VSFrame* current{};
        const VSFrame* reference{};
        const auto cleanup = [&]() {
            for (const auto* pairFrame : pairFrames) {
                if (pairFrame)
                    vsapi->freeFrame(pairFrame);
            }
            if (current)
                vsapi->freeFrame(current);
            if (reference)
                vsapi->freeFrame(reference);
        };

        for (auto i = 0; i < delta; i++) {
            const auto pairIndex = d->backward ? n + i : n - 1 - i;
            pairFrames[i] = vsapi->getFrameFilter(pairIndex, d->node, frameCtx);
        }

        if (delta == 1) {
            const auto props = vsapi->getFramePropertiesRO(pairFrames[0]);
            const auto blobKey = d->backward ? RIFEMVBackwardVectorsInternalKey : RIFEMVForwardVectorsInternalKey;
            int err{};
            const auto* vectorBlob = vsapi->mapGetData(props, blobKey, 0, &err);
            const auto vectorBlobSize = err ? 0 : vsapi->mapGetDataSize(props, blobKey, 0, nullptr);
            if (err || !vectorBlob || vectorBlobSize <= 0) {
                cleanup();
                vsapi->setFilterError("RIFEMVApprox: missing internal vector data", frameCtx);
                return nullptr;
            }

            MotionVectorFrameStats stats{};
            if (d->sadStats || d->motionStats)
                stats = getMotionVectorInternalFrameStats(props, d->backward, d->sadStats, d->motionStats, vsapi);
            auto dst = createMotionVectorFrame(d->vi, d->analysisData, vectorBlob, vectorBlobSize, stats,
                                               d->absSadClipRange, d->renderSadMask, d->sadStats, d->motionStats, d->perf.get(), core, vsapi);
            cleanup();
            if (d->perfStats) {
                accumulatePerfStat(d->perf->outputFrames, 1);
                accumulatePerfStat(d->perf->outputTotalNs, monotonicNowNs() - outputStartNs);
            }
            return dst;
        }

        current = vsapi->getFrameFilter(n, d->sourceNode, frameCtx);
        reference = vsapi->getFrameFilter(d->backward ? n + delta : n - delta, d->sourceNode, frameCtx);
        const auto width = d->mvConfig.inferenceWidth;
        const auto height = d->mvConfig.inferenceHeight;
        const auto displacementKey = d->backward ? RIFEMVBackwardDisplacementInternalKey : RIFEMVForwardDisplacementInternalKey;
        std::vector<const float*> displacementXs(delta);
        std::vector<const float*> displacementYs(delta);

        for (auto i = 0; i < delta; i++) {
            if (!getDisplacementPlanes(pairFrames[i], displacementKey, width, height,
                                       displacementXs[i], displacementYs[i], vsapi)) {
                cleanup();
                vsapi->setFilterError("RIFEMVApprox: missing internal displacement data", frameCtx);
                return nullptr;
            }
        }

        auto& scratch = getMotionVectorScratchBuffers();
        const auto composeStartNs = d->perfStats ? monotonicNowNs() : 0;
        composeDisplacementSequence(displacementXs, displacementYs, width, height, scratch.composedX, scratch.composedY);
        if (d->perfStats)
            accumulatePerfStat(d->perf->composeNs, monotonicNowNs() - composeStartNs);
        const auto vectorPackStartNs = d->perfStats ? monotonicNowNs() : 0;
        MotionVectorFrameStats stats{};
        const auto needStats = d->sadStats || d->motionStats;
        auto& vectors = d->backward ? scratch.backwardVectors : scratch.forwardVectors;
        auto& vectorBlob = d->backward ? scratch.backwardBlob : scratch.forwardBlob;
        buildMotionVectorBlobFromDisplacement(current, reference,
                                              scratch.composedX.data(), scratch.composedY.data(), width, height, true,
                                              d->mvConfig, d->backward, vectors, vectorBlob, vsapi, nullptr, nullptr,
                                              needStats ? &stats : nullptr, d->sadStats, d->motionStats);
        if (d->perfStats)
            accumulatePerfStat(d->perf->vectorPackNs, monotonicNowNs() - vectorPackStartNs);

        auto dst = createMotionVectorFrame(d->vi, d->analysisData, vectorBlob.data(), static_cast<int>(vectorBlob.size()), stats,
                                           d->absSadClipRange, d->renderSadMask, d->sadStats, d->motionStats, d->perf.get(), core, vsapi);
        cleanup();
        if (d->perfStats) {
            accumulatePerfStat(d->perf->outputFrames, 1);
            accumulatePerfStat(d->perf->outputTotalNs, monotonicNowNs() - outputStartNs);
        }
        return dst;
    }

    return nullptr;
}

static void VS_CC rifeMVApproxPairFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<RIFEMVApproxPairData*>(instanceData) };
    if (d->perfStats && d->perf)
        printMotionVectorPerfSummary(*d->perf, d->perfLabel);
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->sourceNode);
    delete d;

    if (--numGPUInstances == 0)
        ncnn::destroy_gpu_instance();
}

static void VS_CC rifeMVApproxOutputFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<RIFEMVApproxOutputData*>(instanceData) };
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->sourceNode);
    delete d;
}

static void VS_CC rifeMVPairFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<RIFEMVPairData*>(instanceData) };
    if (d->perfStats && d->perf)
        printMotionVectorPerfSummary(*d->perf, d->perfLabel);
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->sourceNode);
    delete d;

    if (--numGPUInstances == 0)
        ncnn::destroy_gpu_instance();
}

static void VS_CC rifeMVOutputFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<RIFEMVOutputData*>(instanceData) };
    vsapi->freeNode(d->node);
    delete d;
}

static void VS_CC rifeMVCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto pairData{ std::make_unique<RIFEMVPairData>() };
    VSNode* pairNode{};
    VSNode* backwardNode{};
    VSNode* forwardNode{};
    bool hasGPUInstance{};
    int mvAbsSADClipRange{};
    bool mvRenderSadMask{ true };
    bool mvSadStats{};
    bool mvMotionStats{};

    try {
        pairData->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        const auto sharedLumaSourceIdentity = reinterpret_cast<uintptr_t>(pairData->node);
        pairData->vi = *vsapi->getVideoInfo(pairData->node);
        const auto sourceVi = pairData->vi;
        bool sourceConverted{};
        int err;

        if (ncnn::create_gpu_instance())
            throw "failed to create GPU instance";
        ++numGPUInstances;
        hasGPUInstance = true;

        auto model_path{ vsapi->mapGetData(in, "model_path", 0, &err) };
        std::string modelPath{ err ? "" : model_path };

        auto gpuId{ vsapi->mapGetIntSaturated(in, "gpu_id", 0, &err) };
        if (err)
            gpuId = ncnn::get_default_gpu_index();

        auto gpuThread{ vsapi->mapGetIntSaturated(in, "gpu_thread", 0, &err) };
        if (err)
            gpuThread = 2;
        auto sharedFlowInFlight{ vsapi->mapGetIntSaturated(in, "shared_flow_inflight", 0, &err) };
        const auto sharedFlowInFlightSpecified = !err;
        auto sharedLumaCacheEnabled = !!vsapi->mapGetInt(in, "shared_luma_cache", 0, &err);
        if (err)
            sharedLumaCacheEnabled = true;
        auto sharedPackedCacheEnabled = !!vsapi->mapGetInt(in, "shared_packed_cache", 0, &err);
        if (err)
            sharedPackedCacheEnabled = true;
        auto packedCacheMiB{ vsapi->mapGetIntSaturated(in, "packed_cache_mib", 0, &err) };
        if (err)
            packedCacheMiB = 256;

        auto flowScale{ static_cast<float>(vsapi->mapGetFloat(in, "flow_scale", 0, &err)) };
        if (err)
            flowScale = 1.f;
        auto resScale{ static_cast<float>(vsapi->mapGetFloat(in, "res_scale", 0, &err)) };
        if (err)
            resScale = 1.f;
        FlowResizeMode flowResizeMode{ FlowResizeMode::Auto };
        const auto cpuFlowResize{ vsapi->mapGetIntSaturated(in, "cpu_flow_resize", 0, &err) };
        if (!err)
            flowResizeMode = cpuFlowResize ? FlowResizeMode::ForceCPU : FlowResizeMode::ForceGPU;
        const auto hasGpuMode = vsapi->mapNumElements(in, "gpu_mode") > 0;
        const auto gpuMode = vsapi->mapGetIntSaturated(in, "gpu_mode", 0, &err);
        if (hasGpuMode && err)
            throw "gpu_mode must be an integer";
        pairData->mvExportBackend = parseMotionVectorExportBackend(hasGpuMode ? static_cast<int>(gpuMode) : 0);
        const auto matrixInValue = vsapi->mapGetData(in, "matrix_in_s", 0, &err);
        const auto* matrixIn = err ? "709" : matrixInValue;
        const auto rangeInValue = vsapi->mapGetData(in, "range_in_s", 0, &err);
        const auto* rangeIn = err ? "full" : rangeInValue;
        mvSadStats = !!vsapi->mapGetInt(in, "sad_stats", 0, &err);
        mvMotionStats = !!vsapi->mapGetInt(in, "motion_stats", 0, &err);
        const auto perfStats = !!vsapi->mapGetInt(in, "perf_stats", 0, &err);
        auto mvBlockSizeX{ vsapi->mapGetIntSaturated(in, "blksize_x", 0, &err) };
        if (err)
            mvBlockSizeX = 16;
        auto mvBlockSizeY{ vsapi->mapGetIntSaturated(in, "blksize_y", 0, &err) };
        if (err)
            mvBlockSizeY = mvBlockSizeX;
        auto mvOverlapX{ vsapi->mapGetIntSaturated(in, "overlap_x", 0, &err) };
        if (err)
            mvOverlapX = mvBlockSizeX / 2;
        auto mvOverlapY{ vsapi->mapGetIntSaturated(in, "overlap_y", 0, &err) };
        if (err)
            mvOverlapY = mvBlockSizeY / 2;
        auto mvPel{ vsapi->mapGetIntSaturated(in, "pel", 0, &err) };
        if (err)
            mvPel = 1;
        auto mvDelta{ vsapi->mapGetIntSaturated(in, "delta", 0, &err) };
        if (err)
            mvDelta = 1;
        auto mvBits{ vsapi->mapGetIntSaturated(in, "bits", 0, &err) };
        if (err)
            mvBits = 8;
        mvAbsSADClipRange = vsapi->mapGetIntSaturated(in, "abs_sad_clip_range", 0, &err);
        if (err)
            mvAbsSADClipRange = 0;
        mvRenderSadMask = !!vsapi->mapGetInt(in, "render_sad_mask", 0, &err);
        if (err)
            mvRenderSadMask = true;
        auto mvSadMultiplier{ vsapi->mapGetFloat(in, "sad_multiplier", 0, &err) };
        if (err)
            mvSadMultiplier = 1.0;
        auto mvHPadding{ vsapi->mapGetIntSaturated(in, "hpad", 0, &err) };
        if (err)
            mvHPadding = 0;
        auto mvVPadding{ vsapi->mapGetIntSaturated(in, "vpad", 0, &err) };
        if (err)
            mvVPadding = 0;
        auto mvBlockReduce{ vsapi->mapGetIntSaturated(in, "block_reduce", 0, &err) };
        if (err)
            mvBlockReduce = MVBlockReduceAverage;
        const auto mvUseChroma = !!vsapi->mapGetInt(in, "chroma", 0, &err);

        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count())
            throw "invalid GPU device";

        const auto queueCount = std::max(1, static_cast<int>(ncnn::get_gpu_info(gpuId).compute_queue_count()));
        if (static_cast<uint32_t>(gpuThread) > static_cast<uint32_t>(queueCount))
            std::cerr << "Warning: gpu_thread is recommended to be between 1 and " << queueCount << " (inclusive)" << std::endl;
        if (!sharedFlowInFlightSpecified)
            sharedFlowInFlight = queueCount;
        if (sharedFlowInFlight < 1)
            throw "shared_flow_inflight must be greater than 0";
        if (sharedFlowInFlight > queueCount)
            std::cerr << "Warning: shared_flow_inflight is recommended to be between 1 and " << queueCount << " (inclusive)" << std::endl;

        if (gpuThread < 1)
            throw "gpu_thread must be greater than 0";

        validateAndNormalizeFlowScale(flowScale);
        validateResScale(resScale);
        validateSadMultiplier(mvSadMultiplier);
        validateAbsSADClipRange(mvAbsSADClipRange);

        const auto resolvedModel = resolveRIFEModel(modelPath);
        if (isEarlyUnsupportedRIFEV4Model(resolvedModel.modelPath))
            throw RIFEMVUnsupportedEarlyV4Error;

        if (!supportsMotionVectorExport(resolvedModel))
            throw RIFEMVModelRequirementError;

        if (mvBlockSizeX < 1)
            throw "blksize_x must be at least 1";

        if (mvBlockSizeY < 1)
            throw "blksize_y must be at least 1";

        if (mvOverlapX < 0 || mvOverlapX >= mvBlockSizeX)
            throw "overlap_x must be between 0 and blksize_x - 1";

        if (mvOverlapY < 0 || mvOverlapY >= mvBlockSizeY)
            throw "overlap_y must be between 0 and blksize_y - 1";

        if (mvPel < 1)
            throw "pel must be at least 1";

        if (mvDelta < 1)
            throw "delta must be at least 1";

        if (mvBits < 1 || mvBits > 16)
            throw "bits must be between 1 and 16 (inclusive)";

        if (mvHPadding < 0 || mvVPadding < 0)
            throw "hpad and vpad must be non-negative";

        if (mvBlockReduce != MVBlockReduceCenter && mvBlockReduce != MVBlockReduceAverage)
            throw "block_reduce must be 0 (center) or 1 (average)";
        if (packedCacheMiB < 1)
            throw "packed_cache_mib must be greater than 0";

        const auto inferenceWidth = computeInferenceDimension(sourceVi.width, resScale, "width");
        const auto inferenceHeight = computeInferenceDimension(sourceVi.height, resScale, "height");
        const auto mvInternalGeometry = createMotionVectorInternalGeometry(sourceVi.width, sourceVi.height,
                                                                           inferenceWidth, inferenceHeight,
                                                                           mvBlockSizeX, mvBlockSizeY,
                                                                           mvOverlapX, mvOverlapY,
                                                                           mvHPadding, mvVPadding);
        const auto clipSet = buildMotionVectorClipSet(in, pairData->node, sourceVi, inferenceWidth, inferenceHeight, core, vsapi);
        vsapi->freeNode(pairData->node);
        pairData->node = clipSet.inferenceNode;
        pairData->sourceNode = clipSet.sourceNode;
        sourceConverted = clipSet.convertedFromYUV;

        const VSVideoInfo* metadataVi = sourceConverted ? &sourceVi : nullptr;
        pairData->mvConfig = createMotionVectorConfig(pairData->vi, metadataVi, mvInternalGeometry, mvUseChroma, mvBlockSizeX, mvBlockSizeY,
                                                      mvOverlapX, mvOverlapY, mvPel, mvDelta,
                                  mvBits, mvHPadding, mvVPadding, mvBlockReduce, mvSadMultiplier);
        if (pairData->mvExportBackend == MotionVectorExportBackend::GpuFull)
            validateGpuFullMotionVectorBackend(pairData->mvConfig);
        printMotionVectorInvocation("RIFEMV", gpuId, gpuThread, sharedFlowInFlight, sharedLumaCacheEnabled, flowScale, flowResizeMode,
                                    pairData->mvExportBackend, sharedPackedCacheEnabled, packedCacheMiB, mvSadStats, mvMotionStats, perfStats,
                                    pairData->mvConfig, resScale, inferenceWidth, inferenceHeight,
                                    mvAbsSADClipRange, mvRenderSadMask, matrixIn, rangeIn, true);

        if (!vsapi->getVideoFormatByID(&pairData->vi.format, pfGray8, core))
            throw "failed to create RIFEMV output format";

        const auto localFlowInFlight = sharedFlowInFlightSpecified ? std::max(gpuThread, sharedFlowInFlight) : gpuThread;
        pairData->semaphore = std::make_unique<std::counting_semaphore<>>(localFlowInFlight);
        pairData->sharedFlowSemaphore = acquireSharedFlowSemaphore(gpuId, sharedFlowInFlight);
        if (sharedLumaCacheEnabled) {
            SharedMotionVectorLumaCacheKey key{};
            key.sourceIdentity = sharedLumaSourceIdentity;
            key.width = clipSet.sourceVi.width;
            key.height = clipSet.sourceVi.height;
            key.bits = mvBits;
            key.convertedFromYUV = sourceConverted;
            if (sourceConverted) {
                key.matrixIn = matrixIn;
                key.rangeIn = rangeIn;
            }
            pairData->lumaCache = acquireSharedLumaCache(key, 16);
        } else {
            pairData->lumaCache = createMotionVectorLumaCache(4);
        }
        const auto packedCacheMaxEntries = computePackedCacheMaxEntries(clipSet.inferenceVi.width, clipSet.inferenceVi.height, packedCacheMiB);
        if (sharedPackedCacheEnabled) {
            SharedMotionVectorPackedCacheKey key{};
            key.sourceIdentity = sharedLumaSourceIdentity;
            key.inferenceWidth = clipSet.inferenceVi.width;
            key.inferenceHeight = clipSet.inferenceVi.height;
            key.convertedFromYUV = sourceConverted;
            if (sourceConverted) {
                key.matrixIn = matrixIn;
                key.rangeIn = rangeIn;
            }
            pairData->packedCache = acquireSharedPackedCache(key, packedCacheMaxEntries);
        } else {
            pairData->packedCache = createMotionVectorPackedCache(packedCacheMaxEntries);
        }
        pairData->sadStats = mvSadStats;
        pairData->motionStats = mvMotionStats;
        pairData->perfStats = perfStats;
        if (pairData->perfStats) {
            pairData->perf = std::make_shared<MotionVectorPerfStats>();
            pairData->perfLabel = "RIFEMV(delta=" + std::to_string(pairData->mvConfig.delta) + ")";
        }
        pairData->rife = std::make_unique<RIFE>(gpuId, flowScale, 1, resolvedModel.rifeV2, resolvedModel.rifeV4,
                                                resolvedModel.padding, flowResizeMode, resolvedModel.disableVulkanFp16,
                                                pairData->mvExportBackend == MotionVectorExportBackend::GpuFull);
        loadRIFEModel(*pairData->rife, resolvedModel.modelPath);
    } catch (const std::exception& error) {
        vsapi->mapSetError(out, ("RIFEMV: "s + error.what()).c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);

        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    } catch (const char* error) {
        vsapi->mapSetError(out, ("RIFEMV: "s + error).c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);

        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }

    const auto outputVi = pairData->vi;
    const auto mvConfig = pairData->mvConfig;
    const auto mvPerfStatsEnabled = pairData->perfStats;
    const auto mvPerf = pairData->perf;
    VSFilterDependency pairDeps[]{ { pairData->node, rpGeneral }, { pairData->sourceNode, rpGeneral } };
    pairNode = vsapi->createVideoFilter2("RIFEMVPair", &pairData->vi, rifeMVPairGetFrame, rifeMVPairFree, fmParallel,
                                         pairDeps, 2, pairData.get(), core);
    if (!pairNode) {
        vsapi->mapSetError(out, "RIFEMV: failed to create internal pair filter");
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);
        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }
    pairData.release();

    auto backwardData = std::make_unique<RIFEMVOutputData>();
    backwardData->node = vsapi->addNodeRef(pairNode);
    backwardData->vi = outputVi;
    backwardData->analysisData = mvConfig.backwardAnalysisData;
    backwardData->invalidBlob = buildInvalidMotionVectorBlob(mvConfig, true,
                                                             (mvSadStats || mvMotionStats) ? &backwardData->invalidStats : nullptr,
                                                             mvSadStats, mvMotionStats);
    backwardData->absSadClipRange = mvAbsSADClipRange;
    backwardData->backward = true;
    backwardData->renderSadMask = mvRenderSadMask;
    backwardData->sadStats = mvSadStats;
    backwardData->motionStats = mvMotionStats;
    backwardData->perfStats = mvPerfStatsEnabled;
    backwardData->perf = mvPerf;
    VSFilterDependency backwardDeps[]{ { backwardData->node, rpGeneral } };
    backwardNode = vsapi->createVideoFilter2("RIFEMVBackward", &backwardData->vi, rifeMVOutputGetFrame, rifeMVOutputFree,
                                             fmParallel, backwardDeps, 1, backwardData.get(), core);
    if (!backwardNode) {
        vsapi->mapSetError(out, "RIFEMV: failed to create backward output filter");
        vsapi->freeNode(backwardData->node);
        vsapi->freeNode(pairNode);
        return;
    }
    backwardData.release();

    auto forwardData = std::make_unique<RIFEMVOutputData>();
    forwardData->node = vsapi->addNodeRef(pairNode);
    forwardData->vi = outputVi;
    forwardData->analysisData = mvConfig.forwardAnalysisData;
    forwardData->invalidBlob = buildInvalidMotionVectorBlob(mvConfig, false,
                                                            (mvSadStats || mvMotionStats) ? &forwardData->invalidStats : nullptr,
                                                            mvSadStats, mvMotionStats);
    forwardData->absSadClipRange = mvAbsSADClipRange;
    forwardData->backward = false;
    forwardData->renderSadMask = mvRenderSadMask;
    forwardData->sadStats = mvSadStats;
    forwardData->motionStats = mvMotionStats;
    forwardData->perfStats = mvPerfStatsEnabled;
    forwardData->perf = mvPerf;
    VSFilterDependency forwardDeps[]{ { forwardData->node, rpGeneral } };
    forwardNode = vsapi->createVideoFilter2("RIFEMVForward", &forwardData->vi, rifeMVOutputGetFrame, rifeMVOutputFree,
                                            fmParallel, forwardDeps, 1, forwardData.get(), core);
    if (!forwardNode) {
        vsapi->mapSetError(out, "RIFEMV: failed to create forward output filter");
        vsapi->freeNode(forwardData->node);
        vsapi->freeNode(backwardNode);
        vsapi->freeNode(pairNode);
        return;
    }
    forwardData.release();

    vsapi->freeNode(pairNode);
    vsapi->mapConsumeNode(out, "clip", backwardNode, maAppend);
    vsapi->mapConsumeNode(out, "clip", forwardNode, maAppend);
}

static void rifeMVApproxCreateImpl(const VSMap* in, VSMap* out, VSCore* core, const VSAPI* vsapi,
                                   const int maxDelta, const char* functionName) {
    auto pairData{ std::make_unique<RIFEMVApproxPairData>() };
    VSNode* sourceNode{};
    VSNode* pairNode{};
    bool hasGPUInstance{};
    std::vector<MotionVectorConfig> outputConfigs(maxDelta + 1);
    std::vector<VSNode*> outputNodes;
    int mvAbsSADClipRange{};
    bool mvRenderSadMask{ true };
    bool mvSadStats{};
    bool mvMotionStats{};

    try {
        pairData->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        const auto sharedLumaSourceIdentity = reinterpret_cast<uintptr_t>(pairData->node);
        pairData->vi = *vsapi->getVideoInfo(pairData->node);
        const auto sourceVi = pairData->vi;
        bool sourceConverted{};
        int err;

        if (ncnn::create_gpu_instance())
            throw "failed to create GPU instance";
        ++numGPUInstances;
        hasGPUInstance = true;

        auto model_path{ vsapi->mapGetData(in, "model_path", 0, &err) };
        std::string modelPath{ err ? "" : model_path };

        auto gpuId{ vsapi->mapGetIntSaturated(in, "gpu_id", 0, &err) };
        if (err)
            gpuId = ncnn::get_default_gpu_index();

        auto gpuThread{ vsapi->mapGetIntSaturated(in, "gpu_thread", 0, &err) };
        if (err)
            gpuThread = 2;
        auto sharedFlowInFlight{ vsapi->mapGetIntSaturated(in, "shared_flow_inflight", 0, &err) };
        const auto sharedFlowInFlightSpecified = !err;
        auto sharedLumaCacheEnabled = !!vsapi->mapGetInt(in, "shared_luma_cache", 0, &err);
        if (err)
            sharedLumaCacheEnabled = true;
        auto sharedPackedCacheEnabled = !!vsapi->mapGetInt(in, "shared_packed_cache", 0, &err);
        if (err)
            sharedPackedCacheEnabled = true;
        auto packedCacheMiB{ vsapi->mapGetIntSaturated(in, "packed_cache_mib", 0, &err) };
        if (err)
            packedCacheMiB = 256;
        const auto perfStats = !!vsapi->mapGetInt(in, "perf_stats", 0, &err);

        auto flowScale{ static_cast<float>(vsapi->mapGetFloat(in, "flow_scale", 0, &err)) };
        if (err)
            flowScale = 1.f;
        auto resScale{ static_cast<float>(vsapi->mapGetFloat(in, "res_scale", 0, &err)) };
        if (err)
            resScale = 1.f;
        FlowResizeMode flowResizeMode{ FlowResizeMode::Auto };
        const auto cpuFlowResize{ vsapi->mapGetIntSaturated(in, "cpu_flow_resize", 0, &err) };
        if (!err)
            flowResizeMode = cpuFlowResize ? FlowResizeMode::ForceCPU : FlowResizeMode::ForceGPU;
        const auto matrixInValue = vsapi->mapGetData(in, "matrix_in_s", 0, &err);
        const auto* matrixIn = err ? "709" : matrixInValue;
        const auto rangeInValue = vsapi->mapGetData(in, "range_in_s", 0, &err);
        const auto* rangeIn = err ? "full" : rangeInValue;
        mvSadStats = !!vsapi->mapGetInt(in, "sad_stats", 0, &err);
        mvMotionStats = !!vsapi->mapGetInt(in, "motion_stats", 0, &err);
        auto mvBlockSizeX{ vsapi->mapGetIntSaturated(in, "blksize_x", 0, &err) };
        if (err)
            mvBlockSizeX = 16;
        auto mvBlockSizeY{ vsapi->mapGetIntSaturated(in, "blksize_y", 0, &err) };
        if (err)
            mvBlockSizeY = mvBlockSizeX;
        auto mvOverlapX{ vsapi->mapGetIntSaturated(in, "overlap_x", 0, &err) };
        if (err)
            mvOverlapX = mvBlockSizeX / 2;
        auto mvOverlapY{ vsapi->mapGetIntSaturated(in, "overlap_y", 0, &err) };
        if (err)
            mvOverlapY = mvBlockSizeY / 2;
        auto mvPel{ vsapi->mapGetIntSaturated(in, "pel", 0, &err) };
        if (err)
            mvPel = 1;
        auto mvBits{ vsapi->mapGetIntSaturated(in, "bits", 0, &err) };
        if (err)
            mvBits = 8;
        mvAbsSADClipRange = vsapi->mapGetIntSaturated(in, "abs_sad_clip_range", 0, &err);
        if (err)
            mvAbsSADClipRange = 0;
        mvRenderSadMask = !!vsapi->mapGetInt(in, "render_sad_mask", 0, &err);
        if (err)
            mvRenderSadMask = true;
        auto mvSadMultiplier{ vsapi->mapGetFloat(in, "sad_multiplier", 0, &err) };
        if (err)
            mvSadMultiplier = 1.0;
        auto mvHPadding{ vsapi->mapGetIntSaturated(in, "hpad", 0, &err) };
        if (err)
            mvHPadding = 0;
        auto mvVPadding{ vsapi->mapGetIntSaturated(in, "vpad", 0, &err) };
        if (err)
            mvVPadding = 0;
        auto mvBlockReduce{ vsapi->mapGetIntSaturated(in, "block_reduce", 0, &err) };
        if (err)
            mvBlockReduce = MVBlockReduceAverage;
        const auto mvUseChroma = !!vsapi->mapGetInt(in, "chroma", 0, &err);

        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count())
            throw "invalid GPU device";

        const auto queueCount = std::max(1, static_cast<int>(ncnn::get_gpu_info(gpuId).compute_queue_count()));
        if (static_cast<uint32_t>(gpuThread) > static_cast<uint32_t>(queueCount))
            std::cerr << "Warning: gpu_thread is recommended to be between 1 and " << queueCount << " (inclusive)" << std::endl;
        if (!sharedFlowInFlightSpecified)
            sharedFlowInFlight = queueCount;
        if (sharedFlowInFlight < 1)
            throw "shared_flow_inflight must be greater than 0";
        if (sharedFlowInFlight > queueCount)
            std::cerr << "Warning: shared_flow_inflight is recommended to be between 1 and " << queueCount << " (inclusive)" << std::endl;

        if (gpuThread < 1)
            throw "gpu_thread must be greater than 0";

        validateAndNormalizeFlowScale(flowScale);
        validateResScale(resScale);
        validateSadMultiplier(mvSadMultiplier);
        validateAbsSADClipRange(mvAbsSADClipRange);

        const auto resolvedModel = resolveRIFEModel(modelPath);
        if (isEarlyUnsupportedRIFEV4Model(resolvedModel.modelPath))
            throw RIFEMVUnsupportedEarlyV4Error;

        if (!supportsMotionVectorExport(resolvedModel))
            throw RIFEMVModelRequirementError;

        if (mvBlockSizeX < 1)
            throw "blksize_x must be at least 1";

        if (mvBlockSizeY < 1)
            throw "blksize_y must be at least 1";

        if (mvOverlapX < 0 || mvOverlapX >= mvBlockSizeX)
            throw "overlap_x must be between 0 and blksize_x - 1";

        if (mvOverlapY < 0 || mvOverlapY >= mvBlockSizeY)
            throw "overlap_y must be between 0 and blksize_y - 1";

        if (mvPel < 1)
            throw "pel must be at least 1";

        if (mvBits < 1 || mvBits > 16)
            throw "bits must be between 1 and 16 (inclusive)";

        if (mvHPadding < 0 || mvVPadding < 0)
            throw "hpad and vpad must be non-negative";

        if (mvBlockReduce != MVBlockReduceCenter && mvBlockReduce != MVBlockReduceAverage)
            throw "block_reduce must be 0 (center) or 1 (average)";
        if (packedCacheMiB < 1)
            throw "packed_cache_mib must be greater than 0";

        const auto inferenceWidth = computeInferenceDimension(sourceVi.width, resScale, "width");
        const auto inferenceHeight = computeInferenceDimension(sourceVi.height, resScale, "height");
        const auto mvInternalGeometry = createMotionVectorInternalGeometry(sourceVi.width, sourceVi.height,
                                                                           inferenceWidth, inferenceHeight,
                                                                           mvBlockSizeX, mvBlockSizeY,
                                                                           mvOverlapX, mvOverlapY,
                                                                           mvHPadding, mvVPadding);
        const auto clipSet = buildMotionVectorClipSet(in, pairData->node, sourceVi, inferenceWidth, inferenceHeight, core, vsapi);
        vsapi->freeNode(pairData->node);
        pairData->node = clipSet.inferenceNode;
        pairData->sourceNode = clipSet.sourceNode;
        sourceConverted = clipSet.convertedFromYUV;

        const VSVideoInfo* metadataVi = sourceConverted ? &sourceVi : nullptr;
        pairData->mvConfig = createMotionVectorConfig(pairData->vi, metadataVi, mvInternalGeometry,
                                                      mvUseChroma, mvBlockSizeX, mvBlockSizeY,
                                                      mvOverlapX, mvOverlapY, mvPel, 1,
                                                      mvBits, mvHPadding, mvVPadding, mvBlockReduce, mvSadMultiplier);
        for (auto delta = 1; delta <= maxDelta; delta++) {
            outputConfigs[delta] = createMotionVectorConfig(pairData->vi, metadataVi, mvInternalGeometry,
                                                            mvUseChroma, mvBlockSizeX, mvBlockSizeY,
                                                            mvOverlapX, mvOverlapY, mvPel, delta,
                                                            mvBits, mvHPadding, mvVPadding, mvBlockReduce, mvSadMultiplier);
        }
        printMotionVectorInvocation(functionName, gpuId, gpuThread, sharedFlowInFlight, sharedLumaCacheEnabled, flowScale, flowResizeMode,
                                    MotionVectorExportBackend::Cpu, sharedPackedCacheEnabled, packedCacheMiB, mvSadStats, mvMotionStats, perfStats,
                                    pairData->mvConfig, resScale, inferenceWidth, inferenceHeight,
                                    mvAbsSADClipRange, mvRenderSadMask, matrixIn, rangeIn, false);

        if (!vsapi->getVideoFormatByID(&pairData->vi.format, pfGray8, core))
            throw "failed to create output format";

        sourceNode = vsapi->addNodeRef(pairData->sourceNode);
        const auto localFlowInFlight = sharedFlowInFlightSpecified ? std::max(gpuThread, sharedFlowInFlight) : gpuThread;
        pairData->semaphore = std::make_unique<std::counting_semaphore<>>(localFlowInFlight);
        pairData->sharedFlowSemaphore = acquireSharedFlowSemaphore(gpuId, sharedFlowInFlight);
        if (sharedLumaCacheEnabled) {
            SharedMotionVectorLumaCacheKey key{};
            key.sourceIdentity = sharedLumaSourceIdentity;
            key.width = clipSet.sourceVi.width;
            key.height = clipSet.sourceVi.height;
            key.bits = mvBits;
            key.convertedFromYUV = sourceConverted;
            if (sourceConverted) {
                key.matrixIn = matrixIn;
                key.rangeIn = rangeIn;
            }
            pairData->lumaCache = acquireSharedLumaCache(key, 16);
        } else {
            pairData->lumaCache = createMotionVectorLumaCache(4);
        }
        const auto packedCacheMaxEntries = computePackedCacheMaxEntries(clipSet.inferenceVi.width, clipSet.inferenceVi.height, packedCacheMiB);
        if (sharedPackedCacheEnabled) {
            SharedMotionVectorPackedCacheKey key{};
            key.sourceIdentity = sharedLumaSourceIdentity;
            key.inferenceWidth = clipSet.inferenceVi.width;
            key.inferenceHeight = clipSet.inferenceVi.height;
            key.convertedFromYUV = sourceConverted;
            if (sourceConverted) {
                key.matrixIn = matrixIn;
                key.rangeIn = rangeIn;
            }
            pairData->packedCache = acquireSharedPackedCache(key, packedCacheMaxEntries);
        } else {
            pairData->packedCache = createMotionVectorPackedCache(packedCacheMaxEntries);
        }
        pairData->sadStats = mvSadStats;
        pairData->motionStats = mvMotionStats;
        pairData->perfStats = perfStats;
        if (pairData->perfStats) {
            pairData->perf = std::make_shared<MotionVectorPerfStats>();
            pairData->perfLabel = std::string(functionName);
        }
        pairData->rife = std::make_unique<RIFE>(gpuId, flowScale, 1, resolvedModel.rifeV2, resolvedModel.rifeV4,
                                                resolvedModel.padding, flowResizeMode, resolvedModel.disableVulkanFp16);
        loadRIFEModel(*pairData->rife, resolvedModel.modelPath);
    } catch (const std::exception& error) {
        vsapi->mapSetError(out, (std::string(functionName) + ": " + error.what()).c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);
        vsapi->freeNode(sourceNode);

        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    } catch (const char* error) {
        vsapi->mapSetError(out, (std::string(functionName) + ": " + error).c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);
        vsapi->freeNode(sourceNode);

        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }

    const auto outputVi = pairData->vi;
    VSFilterDependency pairDeps[]{ { pairData->node, rpGeneral }, { pairData->sourceNode, rpGeneral } };
    pairNode = vsapi->createVideoFilter2("RIFEMVApproxPair", &pairData->vi, rifeMVApproxPairGetFrame,
                                         rifeMVApproxPairFree, fmParallel, pairDeps, 2, pairData.get(), core);
    if (!pairNode) {
        vsapi->mapSetError(out, (std::string(functionName) + ": failed to create internal pair filter").c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);
        vsapi->freeNode(sourceNode);
        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }
    const auto approxPerfStats = pairData->perfStats;
    const auto approxPerf = pairData->perf;
    pairData.release();

    const auto createOutputNode = [&](const MotionVectorConfig& mvConfig, const bool backward) {
        auto outputData{ std::make_unique<RIFEMVApproxOutputData>() };
        outputData->node = vsapi->addNodeRef(pairNode);
        outputData->sourceNode = vsapi->addNodeRef(sourceNode);
        outputData->vi = outputVi;
        outputData->mvConfig = mvConfig;
        outputData->analysisData = backward ? mvConfig.backwardAnalysisData : mvConfig.forwardAnalysisData;
        outputData->invalidBlob = buildInvalidMotionVectorBlob(mvConfig, backward,
                                                               (mvSadStats || mvMotionStats) ? &outputData->invalidStats : nullptr,
                                                               mvSadStats, mvMotionStats);
        outputData->absSadClipRange = mvAbsSADClipRange;
        outputData->backward = backward;
        outputData->renderSadMask = mvRenderSadMask;
        outputData->sadStats = mvSadStats;
        outputData->motionStats = mvMotionStats;
        outputData->perfStats = approxPerfStats;
        outputData->perf = approxPerf;
        VSFilterDependency deps[]{ { outputData->node, rpGeneral }, { outputData->sourceNode, rpGeneral } };
        auto node = vsapi->createVideoFilter2(backward ? "RIFEMVApproxBackward" : "RIFEMVApproxForward",
                                              &outputData->vi, rifeMVApproxOutputGetFrame, rifeMVApproxOutputFree,
                                              fmParallel, deps, 2, outputData.get(), core);
        if (!node) {
            vsapi->freeNode(outputData->node);
            vsapi->freeNode(outputData->sourceNode);
            return static_cast<VSNode*>(nullptr);
        }

        outputData.release();
        return node;
    };

    for (auto delta = 1; delta <= maxDelta; delta++) {
        auto backwardNode = createOutputNode(outputConfigs[delta], true);
        if (!backwardNode) {
            vsapi->mapSetError(out, (std::string(functionName) + ": failed to create backward output filter").c_str());
            for (auto* node : outputNodes)
                vsapi->freeNode(node);
            vsapi->freeNode(pairNode);
            vsapi->freeNode(sourceNode);
            return;
        }
        outputNodes.push_back(backwardNode);

        auto forwardNode = createOutputNode(outputConfigs[delta], false);
        if (!forwardNode) {
            vsapi->mapSetError(out, (std::string(functionName) + ": failed to create forward output filter").c_str());
            for (auto* node : outputNodes)
                vsapi->freeNode(node);
            vsapi->freeNode(pairNode);
            vsapi->freeNode(sourceNode);
            return;
        }
        outputNodes.push_back(forwardNode);
    }

    vsapi->freeNode(pairNode);
    vsapi->freeNode(sourceNode);
    for (auto* node : outputNodes)
        vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

static void VS_CC rifeMVApprox2Create(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData,
                                      VSCore* core, const VSAPI* vsapi) {
    rifeMVApproxCreateImpl(in, out, core, vsapi, 2, "RIFEMVApprox2");
}

static void VS_CC rifeMVApprox3Create(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData,
                                      VSCore* core, const VSAPI* vsapi) {
    rifeMVApproxCreateImpl(in, out, core, vsapi, 3, "RIFEMVApprox3");
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.nmkd.rmv", "rmv", "RIFE motion-vector export plugin for MVTools workflows",
                         VS_MAKE_VERSION(9, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    vspapi->registerFunction("RIFEMV",
                             "clip:vnode;"
                             "model_path:data;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "shared_flow_inflight:int:opt;"
                             "shared_luma_cache:int:opt;"
                             "shared_packed_cache:int:opt;"
                             "packed_cache_mib:int:opt;"
                             "flow_scale:float:opt;"
                             "res_scale:float:opt;"
                             "cpu_flow_resize:int:opt;"
                             "gpu_mode:int:opt;"
                             "perf_stats:int:opt;"
                             "blksize_x:int:opt;"
                             "blksize_y:int:opt;"
                             "overlap_x:int:opt;"
                             "overlap_y:int:opt;"
                             "pel:int:opt;"
                             "delta:int:opt;"
                             "bits:int:opt;"
                             "abs_sad_clip_range:int:opt;"
                             "render_sad_mask:int:opt;"
                             "sad_stats:int:opt;"
                             "motion_stats:int:opt;"
                             "sad_multiplier:float:opt;"
                             "matrix_in_s:data:opt;"
                             "range_in_s:data:opt;"
                             "hpad:int:opt;"
                             "vpad:int:opt;"
                             "block_reduce:int:opt;"
                             "chroma:int:opt;",
                             "clip:vnode[];",
                             rifeMVCreate, nullptr, plugin);

    vspapi->registerFunction("RIFEMVApprox2",
                             "clip:vnode;"
                             "model_path:data;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "shared_flow_inflight:int:opt;"
                             "shared_luma_cache:int:opt;"
                             "shared_packed_cache:int:opt;"
                             "packed_cache_mib:int:opt;"
                             "flow_scale:float:opt;"
                             "res_scale:float:opt;"
                             "cpu_flow_resize:int:opt;"
                             "perf_stats:int:opt;"
                             "blksize_x:int:opt;"
                             "blksize_y:int:opt;"
                             "overlap_x:int:opt;"
                             "overlap_y:int:opt;"
                             "pel:int:opt;"
                             "bits:int:opt;"
                             "abs_sad_clip_range:int:opt;"
                             "render_sad_mask:int:opt;"
                             "sad_stats:int:opt;"
                             "motion_stats:int:opt;"
                             "sad_multiplier:float:opt;"
                             "matrix_in_s:data:opt;"
                             "range_in_s:data:opt;"
                             "hpad:int:opt;"
                             "vpad:int:opt;"
                             "block_reduce:int:opt;"
                             "chroma:int:opt;",
                             "clip:vnode[];",
                             rifeMVApprox2Create, nullptr, plugin);

    vspapi->registerFunction("RIFEMVApprox3",
                             "clip:vnode;"
                             "model_path:data;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "shared_flow_inflight:int:opt;"
                             "shared_luma_cache:int:opt;"
                             "shared_packed_cache:int:opt;"
                             "packed_cache_mib:int:opt;"
                             "flow_scale:float:opt;"
                             "res_scale:float:opt;"
                             "cpu_flow_resize:int:opt;"
                             "perf_stats:int:opt;"
                             "blksize_x:int:opt;"
                             "blksize_y:int:opt;"
                             "overlap_x:int:opt;"
                             "overlap_y:int:opt;"
                             "pel:int:opt;"
                             "bits:int:opt;"
                             "abs_sad_clip_range:int:opt;"
                             "render_sad_mask:int:opt;"
                             "sad_stats:int:opt;"
                             "motion_stats:int:opt;"
                             "sad_multiplier:float:opt;"
                             "matrix_in_s:data:opt;"
                             "range_in_s:data:opt;"
                             "hpad:int:opt;"
                             "vpad:int:opt;"
                             "block_reduce:int:opt;"
                             "chroma:int:opt;",
                             "clip:vnode[];",
                             rifeMVApprox3Create, nullptr, plugin);
}
