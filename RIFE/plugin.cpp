// Copyright (c) 2021-2022 HolyWu
// SPDX-License-Identifier: MIT

#include <atomic>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
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
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define RIFEMV_HAS_SSE2 1
#else
#define RIFEMV_HAS_SSE2 0
#endif
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
    std::atomic<int64_t> gpuUploadNs{ 0 };
    std::atomic<int64_t> gpuPreprocNs{ 0 };
    std::atomic<int64_t> gpuInferenceNs{ 0 };
    std::atomic<int64_t> gpuFlowResizeNs{ 0 };
    std::atomic<int64_t> gpuFlowReduceNs{ 0 };
    std::atomic<int64_t> gpuFlowVectorNs{ 0 };
    std::atomic<int64_t> gpuReadbackNs{ 0 };
    std::atomic<int64_t> gpuTotalNs{ 0 };
    std::atomic<int64_t> gpuInputCacheHitCount{ 0 };
    std::atomic<int64_t> gpuInputCacheMissCount{ 0 };
    std::atomic<int64_t> gpuInputCacheWaitNs{ 0 };
    std::atomic<int64_t> packedCacheHitCount{ 0 };
    std::atomic<int64_t> packedCacheMissCount{ 0 };
    std::atomic<int64_t> packedBuildNs{ 0 };
    std::atomic<int64_t> packedWaitNs{ 0 };
    std::atomic<int64_t> lumaBuildNs{ 0 };
    std::atomic<int64_t> vectorPackNs{ 0 };
    std::atomic<int64_t> pairCarrierNs{ 0 };
    std::atomic<int64_t> pairPropertyNs{ 0 };
    std::atomic<int64_t> sadPackedBuildNs{ 0 };
    std::atomic<int64_t> outputFrameAllocNs{ 0 };
    std::atomic<int64_t> outputPropertyNs{ 0 };
    std::atomic<int64_t> degrainSadRecordNs{ 0 };
    std::atomic<int64_t> degrainCenteredRecordNs{ 0 };
    std::atomic<int64_t> degrainAccumulateRecordNs{ 0 };
    std::atomic<int64_t> degrainOutputRecordNs{ 0 };
    std::atomic<int64_t> degrainStatsRecordNs{ 0 };
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
constexpr auto MVToolsSuperHeightKey = "Super_height";
constexpr auto MVToolsSuperHPadKey = "Super_hpad";
constexpr auto MVToolsSuperVPadKey = "Super_vpad";
constexpr auto MVToolsSuperPelKey = "Super_pel";
constexpr auto MVToolsSuperModeYUVKey = "Super_modeyuv";
constexpr auto MVToolsSuperLevelsKey = "Super_levels";
constexpr auto RIFEMVBackwardVectorsInternalKey = "_RMVBackwardVectors";
constexpr auto RIFEMVForwardVectorsInternalKey = "_RMVForwardVectors";
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
constexpr auto RIFEDegrainAvgSadKey = "RMVD_AvgSad";
constexpr auto RIFEDegrainMaxSadKey = "RMVD_MaxSad";
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
    std::vector<RIFEReducedFlowBlock> reducedFlow;
    std::vector<RIFEGpuMotionVector> gpuVectors;
    std::vector<RIFEGpuPackedMotionVector> gpuPackedVectors;
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

enum class MotionVectorExportBackend : uint8_t {
    Cpu,
    GpuFlowReduce,
    GpuFull,
    GpuFullPacked,
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
    bool weightedSad;
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
    float sadY;
    float sadUV;
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

struct MotionVectorColorConversionOptions final {
    std::string matrixIn;
    std::string rangeIn;
    bool matrixInSpecified;
    bool rangeInSpecified;
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
static_assert(sizeof(RIFEGpuPackedMotionVector) == 8);
static_assert(sizeof(MVAnalysisData) == 84);
static_assert(offsetof(MVToolsVector, x) == offsetof(RIFEGpuMotionVector, x));
static_assert(offsetof(MVToolsVector, y) == offsetof(RIFEGpuMotionVector, y));
static_assert(offsetof(MVToolsVector, sad) == offsetof(RIFEGpuMotionVector, rawSad));
static_assert(offsetof(RIFEGpuMotionVector, reserved) == offsetof(MVToolsVector, sad) + sizeof(uint32_t));

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
    const auto gpuUploadNs = stats.gpuUploadNs.load(std::memory_order_relaxed);
    const auto gpuPreprocNs = stats.gpuPreprocNs.load(std::memory_order_relaxed);
    const auto gpuInferenceNs = stats.gpuInferenceNs.load(std::memory_order_relaxed);
    const auto gpuFlowResizeNs = stats.gpuFlowResizeNs.load(std::memory_order_relaxed);
    const auto gpuFlowReduceNs = stats.gpuFlowReduceNs.load(std::memory_order_relaxed);
    const auto gpuFlowVectorNs = stats.gpuFlowVectorNs.load(std::memory_order_relaxed);
    const auto gpuReadbackNs = stats.gpuReadbackNs.load(std::memory_order_relaxed);
    const auto gpuTotalNs = stats.gpuTotalNs.load(std::memory_order_relaxed);
    const auto gpuInputCacheHitCount = stats.gpuInputCacheHitCount.load(std::memory_order_relaxed);
    const auto gpuInputCacheMissCount = stats.gpuInputCacheMissCount.load(std::memory_order_relaxed);
    const auto gpuInputCacheWaitNs = stats.gpuInputCacheWaitNs.load(std::memory_order_relaxed);
    const auto packedCacheHitCount = stats.packedCacheHitCount.load(std::memory_order_relaxed);
    const auto packedCacheMissCount = stats.packedCacheMissCount.load(std::memory_order_relaxed);
    const auto packedBuildNs = stats.packedBuildNs.load(std::memory_order_relaxed);
    const auto packedWaitNs = stats.packedWaitNs.load(std::memory_order_relaxed);
    const auto lumaBuildNs = stats.lumaBuildNs.load(std::memory_order_relaxed);
    const auto vectorPackNs = stats.vectorPackNs.load(std::memory_order_relaxed);
    const auto pairCarrierNs = stats.pairCarrierNs.load(std::memory_order_relaxed);
    const auto pairPropertyNs = stats.pairPropertyNs.load(std::memory_order_relaxed);
    const auto sadPackedBuildNs = stats.sadPackedBuildNs.load(std::memory_order_relaxed);
    const auto outputFrameAllocNs = stats.outputFrameAllocNs.load(std::memory_order_relaxed);
    const auto outputPropertyNs = stats.outputPropertyNs.load(std::memory_order_relaxed);
    const auto degrainSadRecordNs = stats.degrainSadRecordNs.load(std::memory_order_relaxed);
    const auto degrainCenteredRecordNs = stats.degrainCenteredRecordNs.load(std::memory_order_relaxed);
    const auto degrainAccumulateRecordNs = stats.degrainAccumulateRecordNs.load(std::memory_order_relaxed);
    const auto degrainOutputRecordNs = stats.degrainOutputRecordNs.load(std::memory_order_relaxed);
    const auto degrainStatsRecordNs = stats.degrainStatsRecordNs.load(std::memory_order_relaxed);
    const auto rifeProcessAccountedNs = flowSetupNs + flowCpuPrepNs + flowCommandRecordNs + flowSubmitWaitNs +
                                        flowReadbackInvalidateNs + flowReadbackMapNs + flowUnpackNs +
                                        flowExportDirectNs + flowExportResizeNs + flowCleanupNs + gpuInputCacheWaitNs;
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
    std::cerr << "  gpu_total_ms=" << nsToMs(gpuTotalNs)
              << " gpu_upload_ms=" << nsToMs(gpuUploadNs)
              << " gpu_preproc_ms=" << nsToMs(gpuPreprocNs)
              << " gpu_inference_ms=" << nsToMs(gpuInferenceNs)
              << " gpu_flow_resize_ms=" << nsToMs(gpuFlowResizeNs)
              << " gpu_flow_reduce_ms=" << nsToMs(gpuFlowReduceNs)
              << " gpu_flow_vector_ms=" << nsToMs(gpuFlowVectorNs)
              << " gpu_readback_ms=" << nsToMs(gpuReadbackNs) << '\n';
    std::cerr << "  gpu_total_avg_ms=" << (flowCalls > 0 ? nsToMs(gpuTotalNs) / flowCalls : 0.0)
              << " gpu_upload_avg_ms=" << (flowCalls > 0 ? nsToMs(gpuUploadNs) / flowCalls : 0.0)
              << " gpu_preproc_avg_ms=" << (flowCalls > 0 ? nsToMs(gpuPreprocNs) / flowCalls : 0.0)
              << " gpu_inference_avg_ms=" << (flowCalls > 0 ? nsToMs(gpuInferenceNs) / flowCalls : 0.0)
              << " gpu_flow_resize_avg_ms=" << (flowCalls > 0 ? nsToMs(gpuFlowResizeNs) / flowCalls : 0.0)
              << " gpu_flow_reduce_avg_ms=" << (flowCalls > 0 ? nsToMs(gpuFlowReduceNs) / flowCalls : 0.0)
              << " gpu_flow_vector_avg_ms=" << (flowCalls > 0 ? nsToMs(gpuFlowVectorNs) / flowCalls : 0.0)
              << " gpu_readback_avg_ms=" << (flowCalls > 0 ? nsToMs(gpuReadbackNs) / flowCalls : 0.0)
              << " gpu_input_cache_hits=" << gpuInputCacheHitCount
              << " gpu_input_cache_misses=" << gpuInputCacheMissCount
              << " gpu_input_cache_wait_ms=" << nsToMs(gpuInputCacheWaitNs) << '\n';
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
              << " pair_carrier_ms=" << nsToMs(pairCarrierNs)
              << " pair_property_ms=" << nsToMs(pairPropertyNs)
              << " sad_packed_build_ms=" << nsToMs(sadPackedBuildNs)
              << " output_frame_alloc_ms=" << nsToMs(outputFrameAllocNs)
              << " output_property_ms=" << nsToMs(outputPropertyNs) << std::endl;
    if (degrainSadRecordNs || degrainAccumulateRecordNs || degrainOutputRecordNs) {
        std::cerr << "  degrain_sad_record_ms=" << nsToMs(degrainSadRecordNs)
                  << " degrain_centered_record_ms=" << nsToMs(degrainCenteredRecordNs)
                  << " degrain_accumulate_record_ms=" << nsToMs(degrainAccumulateRecordNs)
                  << " degrain_output_record_ms=" << nsToMs(degrainOutputRecordNs)
                  << " degrain_stats_record_ms=" << nsToMs(degrainStatsRecordNs) << std::endl;
    }
    std::cerr << "  local_wait_avg_ms=" << (flowCalls > 0 ? nsToMs(localSemaphoreWaitNs) / flowCalls : 0.0)
              << " shared_wait_avg_ms=" << (flowCalls > 0 ? nsToMs(sharedSemaphoreWaitNs) / flowCalls : 0.0) << std::endl;
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
    case MotionVectorExportBackend::GpuFullPacked:
        return "gpu_full_packed";
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
    case 3:
        return MotionVectorExportBackend::GpuFullPacked;
    default:
        throw "gpu_mode must be 0 (cpu), 1 (gpu_flow_reduce), 2 (gpu_full), or 3 (gpu_full_packed)";
    }
}

static MotionVectorColorConversionOptions readMotionVectorColorConversionOptions(const VSMap* in, const VSAPI* vsapi) {
    MotionVectorColorConversionOptions options{};
    int err{};

    options.matrixInSpecified = vsapi->mapNumElements(in, "matrix_in_s") > 0;
    if (options.matrixInSpecified) {
        const auto* value = vsapi->mapGetData(in, "matrix_in_s", 0, &err);
        if (err || !value)
            throw "matrix_in_s must be a string";
        options.matrixIn = value;
    }

    options.rangeInSpecified = vsapi->mapNumElements(in, "range_in_s") > 0;
    if (options.rangeInSpecified) {
        const auto* value = vsapi->mapGetData(in, "range_in_s", 0, &err);
        if (err || !value)
            throw "range_in_s must be a string";
        options.rangeIn = value;
    }

    return options;
}

static void setMotionVectorConversionCacheKey(SharedMotionVectorLumaCacheKey& key,
                                              const MotionVectorColorConversionOptions& options) {
    key.matrixIn = options.matrixInSpecified ? options.matrixIn : "<frame-props>";
    key.rangeIn = options.rangeInSpecified ? options.rangeIn : "<frame-props>";
}

static void setMotionVectorConversionCacheKey(SharedMotionVectorPackedCacheKey& key,
                                              const MotionVectorColorConversionOptions& options) {
    key.matrixIn = options.matrixInSpecified ? options.matrixIn : "<frame-props>";
    key.rangeIn = options.rangeInSpecified ? options.rangeIn : "<frame-props>";
}

static void printMotionVectorInvocation(const char* const functionName, const int gpuId, const int gpuThread,
                                        const int sharedFlowInFlight, const bool sharedLumaCache, const float flowScale,
                                        const FlowResizeMode flowResizeMode, const MotionVectorExportBackend mvExportBackend,
                                        const bool sharedPackedCache, const int packedCacheMiB,
                                        const bool sadStats, const bool motionStats, const bool perfStats,
                                        const MotionVectorConfig& config, const float resScale,
                                        const int inferenceWidth, const int inferenceHeight,
                                        const MotionVectorColorConversionOptions& conversionOptions,
                                        const bool hasSadClip,
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
            << " matrix_in_s=" << (conversionOptions.matrixInSpecified ? conversionOptions.matrixIn : "None")
            << " range_in_s=" << (conversionOptions.rangeInSpecified ? conversionOptions.rangeIn : "None")
            << " sad_clip=" << hasSadClip
            << " hpad=" << config.hPadding
            << " vpad=" << config.vPadding
            << " block_reduce=" << config.blockReduce
            << " chroma=" << config.useChroma
            << " weighted_sad=" << config.weightedSad
            << " sad_y=" << config.sadY
            << " sad_uv=" << config.sadUV
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
        const auto buildStartNs = perf ? monotonicNowNs() : 0;
        buildPackedInferenceFrame(frame, width, height, vsapi, *packed);
        if (perf)
            accumulatePerfStat(perf->packedBuildNs, monotonicNowNs() - buildStartNs);
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
                                                 const ncnn::Mat* const sadSrc0Packed, const ncnn::Mat* const sadSrc1Packed,
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
    const auto status = sadSrc0Packed && sadSrc1Packed ?
        rife->process_motion_vectors_gpu(src0Packed, src1Packed, *sadSrc0Packed, *sadSrc1Packed, vectorsOut, vectorConfig, flowPerf) :
        rife->process_motion_vectors_gpu(src0Packed, src1Packed, vectorsOut, vectorConfig, flowPerf);
    if (rifeProcessWallNs)
        *rifeProcessWallNs = monotonicNowNs() - rifeProcessStartNs;

    if (sharedSemaphore)
        sharedSemaphore->release();
    localSemaphore->release();
    return status;
}

static int processGpuPackedMotionVectorsWithSemaphores(const RIFE* const rife,
                                                       std::counting_semaphore<>* const localSemaphore,
                                                       std::counting_semaphore<>* const sharedSemaphore,
                                                       const ncnn::Mat& src0Packed, const ncnn::Mat& src1Packed,
                                                       const ncnn::Mat* const sadSrc0Packed, const ncnn::Mat* const sadSrc1Packed,
                                                       RIFEGpuPackedMotionVector* vectorsOut, const RIFEGpuMotionVectorConfig& vectorConfig,
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
    const auto status = sadSrc0Packed && sadSrc1Packed ?
        rife->process_motion_vectors_gpu_packed(src0Packed, src1Packed, *sadSrc0Packed, *sadSrc1Packed, vectorsOut, vectorConfig, flowPerf) :
        rife->process_motion_vectors_gpu_packed(src0Packed, src1Packed, vectorsOut, vectorConfig, flowPerf);
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
                                                   const bool weightedSad = false, const float sadY = 1.f, const float sadUV = 1.f) {
    MotionVectorConfig config{};
    config.useChroma = useChroma;
    config.weightedSad = weightedSad;
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
    config.sadY = sadY;
    config.sadUV = sadUV;

    const auto invalidSad = static_cast<int64_t>(blockSizeX) * blockSizeY * (1LL << bits);
    config.invalidSad = invalidSad;

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

static void validateWeightedSAD(const bool weightedSad, const float sadY, const float sadUV,
                                const bool useChroma, const MotionVectorExportBackend mvExportBackend) {
    if (!std::isfinite(sadY) || sadY < 0.f)
        throw "sad_y must be finite and greater than or equal to 0";
    if (!std::isfinite(sadUV) || sadUV < 0.f)
        throw "sad_uv must be finite and greater than or equal to 0";
    if (!weightedSad)
        return;
    if (!useChroma)
        throw "sad_y and sad_uv require chroma=1";
    if (mvExportBackend != MotionVectorExportBackend::GpuFull && mvExportBackend != MotionVectorExportBackend::GpuFullPacked)
        throw "sad_y and sad_uv currently require gpu_mode=2 or gpu_mode=3";
}

static void validateGpuFullMotionVectorBackend(const MotionVectorConfig& config, const int gpuMode) {
    if (config.inferenceWidth != config.backwardAnalysisData.nWidth ||
        config.inferenceHeight != config.backwardAnalysisData.nHeight)
        throw std::runtime_error("gpu_mode=" + std::to_string(gpuMode) + " currently requires inference dimensions to match the source dimensions");

    const auto maxSample = (1ULL << config.bits) - 1ULL;
    const auto sadScale = config.weightedSad ? static_cast<double>(config.sadY) + 2.0 * static_cast<double>(config.sadUV) : (config.useChroma ? 3.0 : 1.0);
    const auto maxRawSad = static_cast<long double>(config.blockSizeX) * config.blockSizeY * maxSample * sadScale;
    if (maxRawSad > static_cast<long double>(std::numeric_limits<uint32_t>::max()))
        throw std::runtime_error("gpu_mode=" + std::to_string(gpuMode) + " raw SAD exceeds 32-bit storage for this block size and bit depth");
}

static void validateGpuFullPackedMotionVectorBackend(const MotionVectorConfig& config) {
    // Validate the complete clamped output range before the shader narrows vector components.
    const auto validateAxis = [&](const char* const axis, const int size, const int blockSize, const int step, const int padding, const int blockCount) {
        const auto firstBlockCoord = -static_cast<int64_t>(padding);
        const auto lastBlockCoord = static_cast<int64_t>(blockCount - 1) * step - padding;
        const auto componentMin = [&](const int64_t blockCoord) {
            return (-static_cast<int64_t>(padding) - blockCoord) * config.pel;
        };
        const auto componentMax = [&](const int64_t blockCoord) {
            return (static_cast<int64_t>(size) - blockSize + padding - blockCoord) * config.pel;
        };
        const auto minimum = std::min(componentMin(firstBlockCoord), componentMin(lastBlockCoord));
        const auto maximum = std::max(componentMax(firstBlockCoord), componentMax(lastBlockCoord));
        if (minimum < std::numeric_limits<int16_t>::min() || maximum > std::numeric_limits<int16_t>::max()) {
            throw std::runtime_error("gpu_mode=3 " + std::string(axis) + " vector range [" + std::to_string(minimum) + ", " +
                                     std::to_string(maximum) + "] exceeds signed 16-bit storage");
        }
    };

    validateAxis("X", config.backwardAnalysisData.nWidth, config.blockSizeX, config.stepX, config.hPadding, config.blkX);
    validateAxis("Y", config.backwardAnalysisData.nHeight, config.blockSizeY, config.stepY, config.vPadding, config.blkY);
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

struct MotionVectorPreparedClip final {
    VSNode* node;
    VSVideoInfo vi;
    bool convertedFromYUV;
};

struct MotionVectorClipSet final {
    VSNode* sourceNode;
    VSVideoInfo sourceVi;
    VSNode* inferenceNode;
    VSVideoInfo inferenceVi;
    bool sourceConvertedFromYUV;
    bool inferenceConvertedFromYUV;
    bool sourceMatchesInference;
};

struct MotionVectorColorPropOverrideData final {
    VSNode* node;
    VSVideoInfo vi;
    bool clearMatrix;
    bool clearRange;
    bool normalizeRange;
};

static bool isRGBSVideoFormat(const VSVideoInfo& vi) noexcept {
    return vsh::isConstantVideoFormat(&vi) &&
           vi.format.colorFamily == cfRGB &&
           vi.format.sampleType == stFloat &&
           vi.format.bitsPerSample == 32;
}

static const VSFrame* VS_CC motionVectorColorPropOverrideGetFrame(int n, int activationReason, void* instanceData,
                                                                  [[maybe_unused]] void** frameData,
                                                                  VSFrameContext* frameCtx, VSCore* core,
                                                                  const VSAPI* vsapi) {
    auto d{ static_cast<const MotionVectorColorPropOverrideData*>(instanceData) };
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const auto* src = vsapi->getFrameFilter(n, d->node, frameCtx);
        std::array<const VSFrame*, 4> planeSrc{};
        std::array<int, 4> planes{};
        for (auto plane = 0; plane < d->vi.format.numPlanes; plane++) {
            planeSrc[plane] = src;
            planes[plane] = plane;
        }

        auto dst = vsapi->newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, planeSrc.data(), planes.data(), src, core);
        if (!dst) {
            vsapi->freeFrame(src);
            vsapi->setFilterError("RIFEMVColorProps: failed to copy frame", frameCtx);
            return nullptr;
        }

        auto props = vsapi->getFramePropertiesRW(dst);
        if (d->clearMatrix)
            vsapi->mapDeleteKey(props, "_Matrix");
        if (d->clearRange) {
            vsapi->mapDeleteKey(props, "_Range");
            vsapi->mapDeleteKey(props, "_ColorRange");
        } else if (d->normalizeRange) {
            int err{};
            if (vsapi->mapNumElements(props, "_Range") > 0) {
                const auto range = vsapi->mapGetInt(props, "_Range", 0, &err);
                if (!err && (range == 0 || range == 1))
                    vsapi->mapSetInt(props, "_Range", range, maReplace);
            } else if (vsapi->mapNumElements(props, "_ColorRange") > 0) {
                const auto colorRange = vsapi->mapGetInt(props, "_ColorRange", 0, &err);
                if (!err && (colorRange == 0 || colorRange == 1))
                    vsapi->mapSetInt(props, "_Range", colorRange == 0 ? 1 : 0, maReplace);
            }
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC motionVectorColorPropOverrideFree(void* instanceData, [[maybe_unused]] VSCore* core,
                                                    const VSAPI* vsapi) {
    auto d{ static_cast<MotionVectorColorPropOverrideData*>(instanceData) };
    vsapi->freeNode(d->node);
    delete d;
}

static VSNode* createMotionVectorColorPropOverrideClip(VSNode* sourceNode, const VSVideoInfo& sourceVi,
                                                       const bool clearMatrix, const bool clearRange,
                                                       const bool normalizeRange,
                                                       VSCore* core, const VSAPI* vsapi) {
    auto data = std::make_unique<MotionVectorColorPropOverrideData>();
    data->node = vsapi->addNodeRef(sourceNode);
    data->vi = sourceVi;
    data->clearMatrix = clearMatrix;
    data->clearRange = clearRange;
    data->normalizeRange = normalizeRange;

    VSFilterDependency deps[]{ { data->node, rpGeneral } };
    auto node = vsapi->createVideoFilter2("RIFEMVColorProps", &data->vi, motionVectorColorPropOverrideGetFrame,
                                          motionVectorColorPropOverrideFree, fmParallel, deps, 1, data.get(), core);
    if (!node) {
        vsapi->freeNode(data->node);
        throw "failed to create internal color-property override filter";
    }

    data.release();
    return node;
}

static MotionVectorPreparedClip buildMotionVectorRGBSSourceClip(VSNode* sourceNode, const VSVideoInfo& sourceVi,
                                                                const MotionVectorColorConversionOptions& conversionOptions,
                                                                const char* const clipName,
                                                                VSCore* core, const VSAPI* vsapi) {
    if (!vsh::isConstantVideoFormat(&sourceVi))
        throw std::runtime_error(std::string(clipName) + " must have a constant format");

    if (isRGBSVideoFormat(sourceVi))
        return { vsapi->addNodeRef(sourceNode), sourceVi, false };

    if (sourceVi.format.colorFamily != cfYUV)
        throw std::runtime_error(std::string(clipName) + " must be a constant RGBS clip or a constant YUV clip");

    auto resizePlugin = vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core);
    if (!resizePlugin)
        throw "resize plugin is required for internal YUV->RGBS conversion";

    auto overrideNode = createMotionVectorColorPropOverrideClip(sourceNode, sourceVi,
                                                               conversionOptions.matrixInSpecified,
                                                               conversionOptions.rangeInSpecified,
                                                               !conversionOptions.rangeInSpecified,
                                                               core, vsapi);
    auto resizeInput = overrideNode;

    auto args = vsapi->createMap();
    vsapi->mapSetNode(args, "clip", resizeInput, maReplace);
    vsapi->mapSetInt(args, "format", pfRGBS, maReplace);
    if (conversionOptions.matrixInSpecified)
        vsapi->mapSetData(args, "matrix_in_s", conversionOptions.matrixIn.c_str(), -1, dtUtf8, maReplace);
    if (conversionOptions.rangeInSpecified)
        vsapi->mapSetData(args, "range_in_s", conversionOptions.rangeIn.c_str(), -1, dtUtf8, maReplace);

    auto ret = vsapi->invoke(resizePlugin, "Bicubic", args);
    if (const auto* invokeError = vsapi->mapGetError(ret)) {
        const auto errorMessage = std::string("failed to convert clip to RGBS: ") + invokeError;
        vsapi->freeNode(overrideNode);
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        throw std::runtime_error(errorMessage);
    }

    int err{};
    auto rgbNode = vsapi->mapGetNode(ret, "clip", 0, &err);
    if (err || !rgbNode) {
        vsapi->freeNode(overrideNode);
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        throw "resize.Bicubic did not return a clip";
    }

    const auto rgbVi = *vsapi->getVideoInfo(rgbNode);
    if (!isRGBSVideoFormat(rgbVi)) {
        vsapi->freeNode(rgbNode);
        vsapi->freeNode(overrideNode);
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        throw "internal YUV->RGBS conversion did not produce a constant RGBS clip";
    }

    vsapi->freeNode(overrideNode);
    vsapi->freeMap(args);
    vsapi->freeMap(ret);
    return { rgbNode, rgbVi, true };
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

static void validateMotionVectorSadClip(const VSVideoInfo& sourceVi, const VSVideoInfo& sadVi) {
    if (sadVi.width != sourceVi.width || sadVi.height != sourceVi.height)
        throw "sad_clip must have the same width and height as clip";
    if (sadVi.numFrames != sourceVi.numFrames)
        throw "sad_clip must have the same frame count as clip";
}

static MotionVectorClipSet buildMotionVectorClipSet(VSNode* sourceNode, const VSVideoInfo& sourceVi,
                                                    VSNode* sadSourceNode, const VSVideoInfo* const sadSourceVi,
                                                    const MotionVectorColorConversionOptions& conversionOptions,
                                                    const int inferenceWidth, const int inferenceHeight,
                                                    VSCore* core, const VSAPI* vsapi) {
    MotionVectorClipSet clips{};
    MotionVectorPreparedClip motionSource{};

    try {
        motionSource = buildMotionVectorRGBSSourceClip(sourceNode, sourceVi, conversionOptions, "clip", core, vsapi);
        clips.inferenceConvertedFromYUV = motionSource.convertedFromYUV;

        if (inferenceWidth == motionSource.vi.width &&
            inferenceHeight == motionSource.vi.height) {
            clips.inferenceNode = vsapi->addNodeRef(motionSource.node);
            clips.inferenceVi = motionSource.vi;
        } else {
            clips.inferenceNode = resizeMotionVectorClip(motionSource.node, inferenceWidth, inferenceHeight, core, vsapi);
            clips.inferenceVi = *vsapi->getVideoInfo(clips.inferenceNode);
        }

        if (sadSourceNode) {
            const auto sadSource = buildMotionVectorRGBSSourceClip(sadSourceNode, *sadSourceVi, conversionOptions, "sad_clip", core, vsapi);
            clips.sourceNode = sadSource.node;
            clips.sourceVi = sadSource.vi;
            clips.sourceConvertedFromYUV = sadSource.convertedFromYUV;
            clips.sourceMatchesInference = false;
        } else {
            clips.sourceNode = vsapi->addNodeRef(motionSource.node);
            clips.sourceVi = motionSource.vi;
            clips.sourceConvertedFromYUV = motionSource.convertedFromYUV;
            clips.sourceMatchesInference = inferenceWidth == motionSource.vi.width && inferenceHeight == motionSource.vi.height;
        }

        vsapi->freeNode(motionSource.node);
    } catch (...) {
        vsapi->freeNode(motionSource.node);
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
    bool sourceMatchesInference;
    std::shared_ptr<MotionVectorPerfStats> perf;
    std::string perfLabel;
};

struct RIFEMVOutputData final {
    VSNode* node;
    VSVideoInfo vi;
    MVAnalysisData analysisData;
    std::vector<char> invalidBlob;
    MotionVectorFrameStats invalidStats;
    bool backward;
    bool sadStats;
    bool motionStats;
    bool perfStats;
    std::shared_ptr<MotionVectorPerfStats> perf;
};

struct RIFEDegrainData final {
    VSNode* inferenceNode;
    VSNode* sourceNode;
    VSVideoInfo vi;
    int radius;
    RIFEDegrainConfig config;
    std::unique_ptr<RIFE> rife;
    std::unique_ptr<std::counting_semaphore<>> semaphore;
    std::shared_ptr<std::counting_semaphore<>> sharedFlowSemaphore;
    std::shared_ptr<MotionVectorPackedCache> packedCache;
    bool sadStats;
    bool perfStats;
    std::shared_ptr<MotionVectorPerfStats> perf;
};

enum class CropGridMode : uint8_t {
    Vector,
    Super,
};

struct MVToolsSuperData final {
    int height;
    int hpad;
    int vpad;
    int pel;
    int modeYUV;
    int levels;
};

struct CropGridData final {
    VSNode* node{};
    VSVideoInfo vi;
    CropGridMode mode{ CropGridMode::Vector };
    MVToolsSuperData super{};
    int left;
    int right;
    int top;
    int bottom;
    int cropLeftPx;
    int cropRightPx;
    int cropTopPx;
    int cropBottomPx;
    int sourceWidth;
    int sourceHeight;
    int outputSourceWidth;
    int outputSourceHeight;
    int stepX;
    int stepY;
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

        return sad;
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

    return sad;
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

static char* prepareMotionVectorBlob(const size_t vectorCount, const bool valid, std::vector<char>& blob) {
    const auto planeSize = static_cast<MVArraySizeType>(sizeof(MVArraySizeType) + vectorCount * sizeof(MVToolsVector));
    const auto groupSize = static_cast<MVArraySizeType>(sizeof(MVArraySizeType) * 2 + planeSize);
    blob.resize(groupSize);
    auto* output = blob.data();
    const auto writeScalar = [&](const auto value) {
        std::memcpy(output, &value, sizeof(value));
        output += sizeof(value);
    };

    writeScalar(groupSize);
    writeScalar(valid ? MVArraySizeType{ 1 } : MVArraySizeType{ 0 });
    writeScalar(planeSize);
    return output;
}

template <typename VectorReader>
static void packMotionVectorBlobDirect(const size_t vectorCount, const bool valid, const int64_t invalidSad,
                                       std::vector<char>& blob, VectorReader&& readVector) {
    auto* output = prepareMotionVectorBlob(vectorCount, valid, blob);
    for (size_t i = 0; i < vectorCount; i++) {
        const auto vector = valid ? readVector(i) : MVToolsVector{ 0, 0, invalidSad };
        std::memcpy(output, &vector, sizeof(vector));
        output += sizeof(vector);
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
    vectorConfig.weightedSad = config.weightedSad ? 1 : 0;
    vectorConfig.motionScaleX = config.motionScaleX;
    vectorConfig.motionScaleY = config.motionScaleY;
    vectorConfig.sadY = config.sadY;
    vectorConfig.sadUV = config.sadUV;
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
    if (!includeSadStats && !includeMotionStats) {
        const auto* backwardGpuVectors = gpuVectors;
        const auto* forwardGpuVectors = gpuVectors + vectorCount;
        const auto packDirection = [&](const RIFEGpuMotionVector* source, std::vector<char>& blob) {
            if (!valid) {
                packMotionVectorBlobDirect(vectorCount, false, config.invalidSad, blob,
                                           [](const size_t) { return MVToolsVector{}; });
                return;
            }

            // gpu_full writes x/y, the low SAD word, and a zero high SAD word, matching MVToolsVector exactly.
            auto* output = prepareMotionVectorBlob(vectorCount, true, blob);
            std::memcpy(output, source, vectorCount * sizeof(MVToolsVector));
        };
        packDirection(backwardGpuVectors, backwardBlob);
        packDirection(forwardGpuVectors, forwardBlob);
        return;
    }

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

    const auto* backwardGpuVectors = gpuVectors;
    const auto* forwardGpuVectors = gpuVectors + vectorCount;
    for (size_t i = 0; i < vectorCount; i++) {
        backwardVectors[i] = { backwardGpuVectors[i].x, backwardGpuVectors[i].y, static_cast<int64_t>(backwardGpuVectors[i].rawSad) };
        forwardVectors[i] = { forwardGpuVectors[i].x, forwardGpuVectors[i].y, static_cast<int64_t>(forwardGpuVectors[i].rawSad) };
    }

    packMotionVectorBlob(backwardVectors, true, getMotionVectorAnalysisData(config, true), backwardBlob, backwardStats, includeSadStats, includeMotionStats);
    packMotionVectorBlob(forwardVectors, true, getMotionVectorAnalysisData(config, false), forwardBlob, forwardStats, includeSadStats, includeMotionStats);
}

static int decodePackedMotionVectorComponent(const uint32_t value) noexcept {
    // Decode two's-complement bits without relying on implementation-defined unsigned-to-signed narrowing.
    const auto bits = value & 0xffffu;
    return (bits & 0x8000u) != 0 ? static_cast<int>(bits) - 0x10000 : static_cast<int>(bits);
}

static void packValidGpuPackedMotionVectorBlob(const RIFEGpuPackedMotionVector* const source, const size_t vectorCount, std::vector<char>& blob) {
    auto* output = prepareMotionVectorBlob(vectorCount, true, blob);
    size_t i{};
#if RIFEMV_HAS_SSE2
    // Expand two packed vectors into two byte-compatible MVTools records per iteration.
    const auto sadWordMask = _mm_set_epi32(0, -1, 0, -1);
    for (; i + 1 < vectorCount; i += 2) {
        const auto input = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + i));
        const auto packedXY = _mm_shuffle_epi32(input, _MM_SHUFFLE(2, 0, 2, 0));
        const auto signedXY = _mm_unpacklo_epi16(packedXY, _mm_srai_epi16(packedXY, 15));
        const auto sadWords = _mm_and_si128(_mm_shuffle_epi32(input, _MM_SHUFFLE(3, 3, 1, 1)), sadWordMask);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i * sizeof(MVToolsVector)), _mm_unpacklo_epi64(signedXY, sadWords));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + (i + 1) * sizeof(MVToolsVector)), _mm_unpackhi_epi64(signedXY, sadWords));
    }
#endif
    for (; i < vectorCount; i++) {
        const auto vector = MVToolsVector{
            decodePackedMotionVectorComponent(source[i].packedXY),
            decodePackedMotionVectorComponent(source[i].packedXY >> 16),
            static_cast<int64_t>(source[i].rawSad)
        };
        std::memcpy(output + i * sizeof(MVToolsVector), &vector, sizeof(vector));
    }
}

static void buildMotionVectorBlobsFromGpuPackedVectors(const RIFEGpuPackedMotionVector* gpuVectors,
                                                       const bool valid, const MotionVectorConfig& config,
                                                       std::vector<MVToolsVector>& backwardVectors,
                                                       std::vector<MVToolsVector>& forwardVectors,
                                                       std::vector<char>& backwardBlob, std::vector<char>& forwardBlob,
                                                       MotionVectorFrameStats* const backwardStats = nullptr,
                                                       MotionVectorFrameStats* const forwardStats = nullptr,
                                                       const bool includeSadStats = true,
                                                       const bool includeMotionStats = true) {
    const auto vectorCount = static_cast<size_t>(config.blkX) * config.blkY;
    const auto unpackVector = [](const RIFEGpuPackedMotionVector& vector) {
        return MVToolsVector{
            decodePackedMotionVectorComponent(vector.packedXY),
            decodePackedMotionVectorComponent(vector.packedXY >> 16),
            static_cast<int64_t>(vector.rawSad)
        };
    };
    if (!includeSadStats && !includeMotionStats) {
        const auto* backwardGpuVectors = gpuVectors;
        const auto* forwardGpuVectors = gpuVectors + vectorCount;
        const auto packDirection = [&](const RIFEGpuPackedMotionVector* source, std::vector<char>& blob) {
            if (valid) {
                packValidGpuPackedMotionVectorBlob(source, vectorCount, blob);
                return;
            }
            packMotionVectorBlobDirect(vectorCount, false, config.invalidSad, blob,
                                       [](const size_t) { return MVToolsVector{}; });
        };
        packDirection(backwardGpuVectors, backwardBlob);
        packDirection(forwardGpuVectors, forwardBlob);
        return;
    }

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

    const auto* backwardGpuVectors = gpuVectors;
    const auto* forwardGpuVectors = gpuVectors + vectorCount;
    for (size_t i = 0; i < vectorCount; i++) {
        backwardVectors[i] = unpackVector(backwardGpuVectors[i]);
        forwardVectors[i] = unpackVector(forwardGpuVectors[i]);
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

static void zeroMotionVectorFrame(VSFrame* frame, const VSVideoInfo& vi, const VSAPI* vsapi);

static VSFrame* createMotionVectorFrame(const VSVideoInfo& vi, const MVAnalysisData& analysisData,
                                        const char* vectorBlob, const int vectorBlobSize,
                                        const MotionVectorFrameStats& stats,
                                        const bool includeSadStats, const bool includeMotionStats,
                                        MotionVectorPerfStats* const perf,
                                        VSCore* core, const VSAPI* vsapi) {
    const auto frameAllocStartNs = perf ? monotonicNowNs() : 0;
    auto dst = vsapi->newVideoFrame(&vi.format, vi.width, vi.height, nullptr, core);
    if (perf)
        accumulatePerfStat(perf->outputFrameAllocNs, monotonicNowNs() - frameAllocStartNs);
    zeroMotionVectorFrame(dst, vi, vsapi);
    auto props = vsapi->getFramePropertiesRW(dst);
    const auto propertyStartNs = perf ? monotonicNowNs() : 0;
    setMotionVectorProperties(props, analysisData, vectorBlob, vectorBlobSize, stats, includeSadStats, includeMotionStats, vsapi);
    if (perf)
        accumulatePerfStat(perf->outputPropertyNs, monotonicNowNs() - propertyStartNs);
    return dst;
}

static void zeroMotionVectorFrame(VSFrame* frame, const VSVideoInfo& vi, const VSAPI* vsapi) {
    auto* dstp = vsapi->getWritePtr(frame, 0);
    const auto dstStride = vsapi->getStride(frame, 0);
    for (auto y = 0; y < vi.height; y++)
        std::memset(dstp + static_cast<size_t>(y) * dstStride, 0, vi.width * vi.format.bytesPerSample);
}

static bool readMVAnalysisData(const VSMap* props, MVAnalysisData& analysisData, const VSAPI* vsapi) noexcept {
    int err{};
    const auto* data = vsapi->mapGetData(props, MVToolsAnalysisDataKey, 0, &err);
    if (err || vsapi->mapGetDataSize(props, MVToolsAnalysisDataKey, 0, nullptr) != sizeof(MVAnalysisData))
        return false;

    std::memcpy(&analysisData, data, sizeof(analysisData));
    return true;
}

static bool readMVToolsSuperData(const VSMap* props, MVToolsSuperData& super, const VSAPI* vsapi) noexcept {
    int err[6]{};
    super.height = vsapi->mapGetIntSaturated(props, MVToolsSuperHeightKey, 0, &err[0]);
    super.hpad = vsapi->mapGetIntSaturated(props, MVToolsSuperHPadKey, 0, &err[1]);
    super.vpad = vsapi->mapGetIntSaturated(props, MVToolsSuperVPadKey, 0, &err[2]);
    super.pel = vsapi->mapGetIntSaturated(props, MVToolsSuperPelKey, 0, &err[3]);
    super.modeYUV = vsapi->mapGetIntSaturated(props, MVToolsSuperModeYUVKey, 0, &err[4]);
    super.levels = vsapi->mapGetIntSaturated(props, MVToolsSuperLevelsKey, 0, &err[5]);
    for (const auto e : err) {
        if (e)
            return false;
    }
    return true;
}

static int readMotionVectorBlobValidity(const char* vectorBlob, const int vectorBlobSize) noexcept {
    if (!vectorBlob || vectorBlobSize < static_cast<int>(sizeof(MVArraySizeType) * 2))
        return 0;

    MVArraySizeType valid{};
    std::memcpy(&valid, vectorBlob + sizeof(MVArraySizeType), sizeof(valid));
    return valid != 0;
}

static bool hasPublicSadStats(const VSMap* props, const VSAPI* vsapi) noexcept {
    return vsapi->mapGetType(props, RIFEMVAvgSadKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVMaxSadKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVMinSadKey) != ptUnset;
}

static bool hasPublicMotionStats(const VSMap* props, const VSAPI* vsapi) noexcept {
    return vsapi->mapGetType(props, RIFEMVAvgAbsDxKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVAvgAbsDyKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVAvgAbsMotionKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVPanAmountKey) != ptUnset;
}

static void deleteMotionVectorPublicFrameStats(VSMap* props, const VSAPI* vsapi) {
    vsapi->mapDeleteKey(props, RIFEMVAvgSadKey);
    vsapi->mapDeleteKey(props, RIFEMVMaxSadKey);
    vsapi->mapDeleteKey(props, RIFEMVMinSadKey);
    vsapi->mapDeleteKey(props, RIFEMVAvgAbsDxKey);
    vsapi->mapDeleteKey(props, RIFEMVAvgAbsDyKey);
    vsapi->mapDeleteKey(props, RIFEMVAvgAbsMotionKey);
    vsapi->mapDeleteKey(props, RIFEMVPanAmountKey);
}

static int cropGridPlaneHeightLuma(const int srcHeight, const int level, const int yRatioUV, const int vpad) noexcept {
    auto height = srcHeight;
    for (auto i = 1; i <= level; i++)
        height = vpad >= yRatioUV ? ((height / yRatioUV + 1) / 2) * yRatioUV : ((height / yRatioUV) / 2) * yRatioUV;
    return height;
}

static int cropGridPlaneWidthLuma(const int srcWidth, const int level, const int xRatioUV, const int hpad) noexcept {
    auto width = srcWidth;
    for (auto i = 1; i <= level; i++)
        width = hpad >= xRatioUV ? ((width / xRatioUV + 1) / 2) * xRatioUV : ((width / xRatioUV) / 2) * xRatioUV;
    return width;
}

static ptrdiff_t cropGridPlaneSuperOffset(const bool chroma, const int srcHeight, const int level, const int pel,
                                          const int vpad, const ptrdiff_t planePitch, const int yRatioUV) noexcept {
    if (level == 0)
        return 0;

    auto height = srcHeight;
    auto offset = static_cast<ptrdiff_t>(pel) * pel * planePitch * (srcHeight + vpad * 2);
    for (auto i = 1; i < level; i++) {
        height = chroma ? cropGridPlaneHeightLuma(srcHeight * yRatioUV, i, yRatioUV, vpad * yRatioUV) / yRatioUV :
                          cropGridPlaneHeightLuma(srcHeight, i, yRatioUV, vpad);
        offset += planePitch * (height + vpad * 2);
    }
    return offset;
}

static int cropGridSuperHeight(const int srcHeight, const int levels, const int pel,
                               const int vpad, const int superWidth, const int yRatioUV) noexcept {
    auto height = static_cast<int>(cropGridPlaneSuperOffset(false, srcHeight, levels, pel, vpad, superWidth, yRatioUV) / superWidth);
    if (yRatioUV == 2 && (height & 1))
        height++;
    return height;
}

static int cropGridSuperWidth(const int srcWidth, const int hpad, const int xRatioUV) noexcept {
    auto width = srcWidth + hpad * 2;
    if (xRatioUV == 2 && (width & 1))
        width++;
    return width;
}

static void validateCropSubsamplingAlignment(const VSVideoInfo& vi, const int cropLeftPx, const int cropRightPx,
                                             const int cropTopPx, const int cropBottomPx) {
    const auto xRatioUV = 1 << vi.format.subSamplingW;
    const auto yRatioUV = 1 << vi.format.subSamplingH;
    if (vi.format.numPlanes > 1 &&
        (cropLeftPx % xRatioUV || cropRightPx % xRatioUV || cropTopPx % yRatioUV || cropBottomPx % yRatioUV))
        throw "crop values produce pixel crops that are not compatible with the clip's chroma subsampling";
}

static void copyCroppedVideoFramePixels(const VSFrame* src, VSFrame* dst, const int cropLeftPx, const int cropTopPx,
                                        const VSAPI* vsapi) {
    const auto* format = vsapi->getVideoFrameFormat(src);
    const auto bytesPerSample = format->bytesPerSample;
    for (auto plane = 0; plane < format->numPlanes; plane++) {
        const auto planeLeft = plane == 0 ? cropLeftPx : cropLeftPx >> format->subSamplingW;
        const auto planeTop = plane == 0 ? cropTopPx : cropTopPx >> format->subSamplingH;
        const auto width = vsapi->getFrameWidth(dst, plane);
        const auto height = vsapi->getFrameHeight(dst, plane);
        const auto* srcp = vsapi->getReadPtr(src, plane) + static_cast<size_t>(planeTop) * vsapi->getStride(src, plane) + static_cast<size_t>(planeLeft) * bytesPerSample;
        auto* dstp = vsapi->getWritePtr(dst, plane);
        const auto srcStride = vsapi->getStride(src, plane);
        const auto dstStride = vsapi->getStride(dst, plane);
        const auto rowBytes = static_cast<size_t>(width) * bytesPerSample;
        for (auto y = 0; y < height; y++)
            std::memcpy(dstp + static_cast<size_t>(y) * dstStride, srcp + static_cast<size_t>(y) * srcStride, rowBytes);
    }
}

static void zeroVideoFrame(VSFrame* frame, const VSAPI* vsapi) {
    const auto* format = vsapi->getVideoFrameFormat(frame);
    for (auto plane = 0; plane < format->numPlanes; plane++) {
        auto* dstp = vsapi->getWritePtr(frame, plane);
        const auto stride = vsapi->getStride(frame, plane);
        const auto height = vsapi->getFrameHeight(frame, plane);
        for (auto y = 0; y < height; y++)
            std::memset(dstp + static_cast<size_t>(y) * stride, 0, stride);
    }
}

static void padCroppedSuperPlane(uint8_t* plane, const ptrdiff_t stride, const int width, const int height,
                                 const int hpad, const int vpad, const int bytesPerSample) {
    if (width <= 0 || height <= 0)
        return;

    for (auto y = 0; y < height; y++) {
        auto* row = plane + static_cast<size_t>(vpad + y) * stride;
        auto* first = row + static_cast<size_t>(hpad) * bytesPerSample;
        auto* last = first + static_cast<size_t>(width - 1) * bytesPerSample;
        for (auto x = 0; x < hpad; x++)
            std::memcpy(row + static_cast<size_t>(x) * bytesPerSample, first, bytesPerSample);
        for (auto x = 0; x < hpad; x++)
            std::memcpy(first + static_cast<size_t>(width + x) * bytesPerSample, last, bytesPerSample);
    }

    const auto rowBytes = static_cast<size_t>(width + hpad * 2) * bytesPerSample;
    const auto* firstContentRow = plane + static_cast<size_t>(vpad) * stride;
    for (auto y = 0; y < vpad; y++)
        std::memcpy(plane + static_cast<size_t>(y) * stride, firstContentRow, rowBytes);

    const auto* lastContentRow = plane + static_cast<size_t>(vpad + height - 1) * stride;
    for (auto y = 0; y < vpad; y++)
        std::memcpy(plane + static_cast<size_t>(vpad + height + y) * stride, lastContentRow, rowBytes);
}

static bool cropMotionVectorGrid(const std::vector<MVToolsVector>& srcVectors, const MVAnalysisData& srcAnalysisData,
                                 const int left, const int right, const int top, const int bottom,
                                 std::vector<MVToolsVector>& dstVectors, MVAnalysisData& dstAnalysisData) {
    if (srcAnalysisData.nLvCount != 1 || srcAnalysisData.nBlkX <= 0 || srcAnalysisData.nBlkY <= 0 ||
        left < 0 || right < 0 || top < 0 || bottom < 0 ||
        left + right >= srcAnalysisData.nBlkX || top + bottom >= srcAnalysisData.nBlkY)
        return false;

    const auto dstBlkX = srcAnalysisData.nBlkX - left - right;
    const auto dstBlkY = srcAnalysisData.nBlkY - top - bottom;
    dstAnalysisData = srcAnalysisData;
    const auto stepX = srcAnalysisData.nBlkSizeX - srcAnalysisData.nOverlapX;
    const auto stepY = srcAnalysisData.nBlkSizeY - srcAnalysisData.nOverlapY;
    dstAnalysisData.nWidth -= (left + right) * stepX;
    dstAnalysisData.nHeight -= (top + bottom) * stepY;
    dstAnalysisData.nBlkX = dstBlkX;
    dstAnalysisData.nBlkY = dstBlkY;
    if (dstAnalysisData.nWidth <= 0 || dstAnalysisData.nHeight <= 0)
        return false;

    dstVectors.resize(static_cast<size_t>(dstBlkX) * dstBlkY);
    for (auto y = 0; y < dstBlkY; y++) {
        const auto* srcRow = srcVectors.data() + static_cast<size_t>(top + y) * srcAnalysisData.nBlkX + left;
        auto* dstRow = dstVectors.data() + static_cast<size_t>(y) * dstBlkX;
        std::memcpy(dstRow, srcRow, static_cast<size_t>(dstBlkX) * sizeof(MVToolsVector));

        for (auto x = 0; x < dstBlkX; x++) {
            auto& vector = dstRow[x];
            const auto blockX = x * stepX - dstAnalysisData.nHPadding;
            const auto blockY = y * stepY - dstAnalysisData.nVPadding;
            vector.x = clampMotionVectorComponent(vector.x, dstAnalysisData.nPel, blockX, dstAnalysisData.nBlkSizeX,
                                                  dstAnalysisData.nWidth, dstAnalysisData.nHPadding);
            vector.y = clampMotionVectorComponent(vector.y, dstAnalysisData.nPel, blockY, dstAnalysisData.nBlkSizeY,
                                                  dstAnalysisData.nHeight, dstAnalysisData.nVPadding);
        }
    }

    return true;
}

static void setMVToolsSuperProperties(VSMap* props, const MVToolsSuperData& super, const VSAPI* vsapi) {
    vsapi->mapSetInt(props, MVToolsSuperHeightKey, super.height, maReplace);
    vsapi->mapSetInt(props, MVToolsSuperHPadKey, super.hpad, maReplace);
    vsapi->mapSetInt(props, MVToolsSuperVPadKey, super.vpad, maReplace);
    vsapi->mapSetInt(props, MVToolsSuperPelKey, super.pel, maReplace);
    vsapi->mapSetInt(props, MVToolsSuperModeYUVKey, super.modeYUV, maReplace);
    vsapi->mapSetInt(props, MVToolsSuperLevelsKey, super.levels, maReplace);
}

static void copyCroppedSuperFrame(const VSFrame* src, VSFrame* dst, const CropGridData& d, const VSAPI* vsapi) {
    const auto* format = vsapi->getVideoFrameFormat(src);
    const auto bytesPerSample = format->bytesPerSample;
    const auto xRatioUV = 1 << format->subSamplingW;
    const auto yRatioUV = 1 << format->subSamplingH;
    zeroVideoFrame(dst, vsapi);

    for (auto plane = 0; plane < format->numPlanes; plane++) {
        if (!(d.super.modeYUV & (1 << plane)))
            continue;

        const auto chroma = plane != 0;
        const auto planeXRatio = chroma ? xRatioUV : 1;
        const auto planeYRatio = chroma ? yRatioUV : 1;
        const auto planeHPad = d.super.hpad / planeXRatio;
        const auto planeVPad = d.super.vpad / planeYRatio;
        const auto srcPlaneHeight = d.sourceHeight / planeYRatio;
        const auto dstPlaneHeight = d.outputSourceHeight / planeYRatio;
        const auto srcStride = vsapi->getStride(src, plane);
        const auto dstStride = vsapi->getStride(dst, plane);
        const auto* srcPlane = vsapi->getReadPtr(src, plane);
        auto* dstPlane = vsapi->getWritePtr(dst, plane);

        for (auto level = 0; level < d.super.levels; level++) {
            const auto srcLevelLumaWidth = cropGridPlaneWidthLuma(d.sourceWidth, level, xRatioUV, d.super.hpad);
            const auto srcLevelLumaHeight = cropGridPlaneHeightLuma(d.sourceHeight, level, yRatioUV, d.super.vpad);
            const auto dstLevelLumaWidth = cropGridPlaneWidthLuma(d.outputSourceWidth, level, xRatioUV, d.super.hpad);
            const auto dstLevelLumaHeight = cropGridPlaneHeightLuma(d.outputSourceHeight, level, yRatioUV, d.super.vpad);
            const auto cropLeftLuma = cropGridPlaneWidthLuma(d.cropLeftPx, level, xRatioUV, d.super.hpad);
            const auto cropTopLuma = cropGridPlaneHeightLuma(d.cropTopPx, level, yRatioUV, d.super.vpad);
            const auto srcLevelWidth = srcLevelLumaWidth / planeXRatio;
            const auto srcLevelHeight = srcLevelLumaHeight / planeYRatio;
            const auto dstLevelWidth = dstLevelLumaWidth / planeXRatio;
            const auto dstLevelHeight = dstLevelLumaHeight / planeYRatio;
            const auto cropLeft = cropLeftLuma / planeXRatio;
            const auto cropTop = cropTopLuma / planeYRatio;

            if (cropLeft < 0 || cropTop < 0 || dstLevelWidth <= 0 || dstLevelHeight <= 0 ||
                cropLeft + dstLevelWidth > srcLevelWidth || cropTop + dstLevelHeight > srcLevelHeight)
                throw std::runtime_error("CropGrid: Super level crop is outside the source level");

            const auto srcOffset = cropGridPlaneSuperOffset(chroma, srcPlaneHeight, level, d.super.pel, planeVPad, srcStride, yRatioUV);
            const auto dstOffset = cropGridPlaneSuperOffset(chroma, dstPlaneHeight, level, d.super.pel, planeVPad, dstStride, yRatioUV);
            const auto sliceCount = level == 0 ? d.super.pel * d.super.pel : 1;
            const auto srcPaddedHeight = srcLevelHeight + planeVPad * 2;
            const auto dstPaddedHeight = dstLevelHeight + planeVPad * 2;
            const auto rowBytes = static_cast<size_t>(dstLevelWidth) * bytesPerSample;

            for (auto slice = 0; slice < sliceCount; slice++) {
                const auto* srcSlice = srcPlane + srcOffset + static_cast<size_t>(slice) * srcStride * srcPaddedHeight;
                auto* dstSlice = dstPlane + dstOffset + static_cast<size_t>(slice) * dstStride * dstPaddedHeight;
                const auto* srcCenter = srcSlice + static_cast<size_t>(planeVPad + cropTop) * srcStride + static_cast<size_t>(planeHPad + cropLeft) * bytesPerSample;
                auto* dstCenter = dstSlice + static_cast<size_t>(planeVPad) * dstStride + static_cast<size_t>(planeHPad) * bytesPerSample;

                for (auto y = 0; y < dstLevelHeight; y++)
                    std::memcpy(dstCenter + static_cast<size_t>(y) * dstStride, srcCenter + static_cast<size_t>(y) * srcStride, rowBytes);

                padCroppedSuperPlane(dstSlice, dstStride, dstLevelWidth, dstLevelHeight, planeHPad, planeVPad, bytesPerSample);
            }
        }
    }
}

static const VSFrame* VS_CC cropGridGetFrame(int n, int activationReason, void* instanceData,
                                             [[maybe_unused]] void** frameData,
                                             VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const CropGridData*>(instanceData) };

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto src = vsapi->getFrameFilter(n, d->node, frameCtx);

        try {
            if (d->mode == CropGridMode::Super) {
                auto dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, nullptr, core);
                copyCroppedSuperFrame(src, dst, *d, vsapi);
                if (n == 0) {
                    auto super = d->super;
                    super.height = d->outputSourceHeight;
                    setMVToolsSuperProperties(vsapi->getFramePropertiesRW(dst), super, vsapi);
                }
                vsapi->freeFrame(src);
                return dst;
            }

            MVAnalysisData analysisData{};
            const auto srcProps = vsapi->getFramePropertiesRO(src);
            if (!readMVAnalysisData(srcProps, analysisData, vsapi) || analysisData.nLvCount != 1) {
                vsapi->freeFrame(src);
                vsapi->setFilterError("CropGrid: input frame is missing single-level MVTools vector metadata", frameCtx);
                return nullptr;
            }
            if (analysisData.nBlkSizeX - analysisData.nOverlapX != d->stepX ||
                analysisData.nBlkSizeY - analysisData.nOverlapY != d->stepY ||
                analysisData.nWidth != d->sourceWidth || analysisData.nHeight != d->sourceHeight) {
                vsapi->freeFrame(src);
                vsapi->setFilterError("CropGrid: per-frame vector metadata does not match the first frame", frameCtx);
                return nullptr;
            }

            int err{};
            const auto* vectorBlob = vsapi->mapGetData(srcProps, MVToolsVectorsKey, 0, &err);
            const auto vectorBlobSize = err ? 0 : vsapi->mapGetDataSize(srcProps, MVToolsVectorsKey, 0, nullptr);
            std::vector<MVToolsVector> vectors;
            if (err || !unpackMotionVectorBlob(vectorBlob, vectorBlobSize, analysisData, vectors)) {
                vsapi->freeFrame(src);
                vsapi->setFilterError("CropGrid: failed to read MVTools vector blob", frameCtx);
                return nullptr;
            }

            MVAnalysisData croppedAnalysisData{};
            std::vector<MVToolsVector> croppedVectors;
            if (!cropMotionVectorGrid(vectors, analysisData, d->left, d->right, d->top, d->bottom, croppedVectors, croppedAnalysisData)) {
                vsapi->freeFrame(src);
                vsapi->setFilterError("CropGrid: vector crop is outside the source grid", frameCtx);
                return nullptr;
            }

            std::vector<char> croppedBlob;
            MotionVectorFrameStats stats{};
            const auto includeSadStats = hasPublicSadStats(srcProps, vsapi);
            const auto includeMotionStats = hasPublicMotionStats(srcProps, vsapi);
            packMotionVectorBlob(croppedVectors, readMotionVectorBlobValidity(vectorBlob, vectorBlobSize), croppedAnalysisData,
                                 croppedBlob, &stats, includeSadStats, includeMotionStats);

            auto dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
            copyCroppedVideoFramePixels(src, dst, d->cropLeftPx, d->cropTopPx, vsapi);
            auto dstProps = vsapi->getFramePropertiesRW(dst);
            deleteMotionVectorPublicFrameStats(dstProps, vsapi);
            setMotionVectorProperties(dstProps, croppedAnalysisData, croppedBlob.data(), static_cast<int>(croppedBlob.size()),
                                      stats, includeSadStats, includeMotionStats, vsapi);
            vsapi->freeFrame(src);
            return dst;
        } catch (const std::exception& error) {
            vsapi->freeFrame(src);
            vsapi->setFilterError(error.what(), frameCtx);
            return nullptr;
        }
    }

    return nullptr;
}

static void VS_CC cropGridFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<CropGridData*>(instanceData) };
    vsapi->freeNode(d->node);
    delete d;
}

static const VSFrame* VS_CC rifeMVPairGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData,
                                               VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEMVPairData*>(instanceData) };
    const auto delta = d->mvConfig.delta;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (!d->sourceMatchesInference)
            vsapi->requestFrameFilter(n, d->sourceNode, frameCtx);
        if (n + delta < d->vi.numFrames) {
            vsapi->requestFrameFilter(n + delta, d->node, frameCtx);
            if (!d->sourceMatchesInference)
                vsapi->requestFrameFilter(n + delta, d->sourceNode, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const auto pairStartNs = d->perfStats ? monotonicNowNs() : 0;
        auto current = vsapi->getFrameFilter(n, d->node, frameCtx);
        const auto* currentSource = d->sourceMatchesInference ? vsapi->addFrameRef(current) : vsapi->getFrameFilter(n, d->sourceNode, frameCtx);
        const VSFrame* reference = n + delta < d->vi.numFrames ? vsapi->getFrameFilter(n + delta, d->node, frameCtx) : nullptr;
        const VSFrame* referenceSource = n + delta < d->vi.numFrames ?
            (d->sourceMatchesInference ? vsapi->addFrameRef(reference) : vsapi->getFrameFilter(n + delta, d->sourceNode, frameCtx)) : nullptr;

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
            const auto useGpuFullPacked = d->mvExportBackend == MotionVectorExportBackend::GpuFullPacked;
            const auto useGpuVectorExport = useGpuFull || useGpuFullPacked;
            RIFEFlowReduceConfig reduceConfig{};
            RIFEGpuMotionVectorConfig gpuVectorConfig{};
            if (useGpuFlowReduce) {
                reduceConfig = createRIFEFlowReduceConfig(d->mvConfig);
                scratch.reducedFlow.resize(static_cast<size_t>(reduceConfig.blockCountX) * reduceConfig.blockCountY);
            } else if (useGpuVectorExport) {
                gpuVectorConfig = createRIFEGpuMotionVectorConfig(d->mvConfig);
                const auto vectorCount = static_cast<size_t>(gpuVectorConfig.blockCountX) * gpuVectorConfig.blockCountY * 2;
                if (useGpuFullPacked)
                    scratch.gpuPackedVectors.resize(vectorCount);
                else
                    scratch.gpuVectors.resize(vectorCount);
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
            std::shared_ptr<const ncnn::Mat> currentSadPacked;
            std::shared_ptr<const ncnn::Mat> referenceSadPacked;
            const ncnn::Mat* currentSadPackedPtr{};
            const ncnn::Mat* referenceSadPackedPtr{};
            if (useGpuVectorExport) {
                if (d->sourceMatchesInference) {
                    currentSadPackedPtr = currentPacked.get();
                    referenceSadPackedPtr = referencePacked.get();
                } else {
                    const auto sadPackedBuildStartNs = d->perfStats ? monotonicNowNs() : 0;
                    const auto sourceWidth = vsapi->getFrameWidth(currentSource, 0);
                    const auto sourceHeight = vsapi->getFrameHeight(currentSource, 0);
                    currentSadPacked = getOrCreatePackedInferenceFrame(nullptr, currentSource, n, sourceWidth, sourceHeight,
                                                                       d->perfStats ? d->perf.get() : nullptr, vsapi);
                    referenceSadPacked = getOrCreatePackedInferenceFrame(nullptr, referenceSource, n + delta, sourceWidth, sourceHeight,
                                                                         d->perfStats ? d->perf.get() : nullptr, vsapi);
                    currentSadPackedPtr = currentSadPacked.get();
                    referenceSadPackedPtr = referenceSadPacked.get();
                    if (d->perfStats)
                        accumulatePerfStat(d->perf->sadPackedBuildNs, monotonicNowNs() - sadPackedBuildStartNs);
                }
            }

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
                                                               currentSadPackedPtr, referenceSadPackedPtr,
                                                               scratch.gpuVectors.data(), gpuVectorConfig,
                                                               d->perfStats ? &semaphoreWaitNs : nullptr,
                                                               d->perfStats ? &localSemaphoreWaitNs : nullptr,
                                                               d->perfStats ? &sharedSemaphoreWaitNs : nullptr,
                                                               d->perfStats ? &rifeProcessWallNs : nullptr,
                                                               d->perfStats ? &flowPerf : nullptr);
            } else if (useGpuFullPacked) {
                status = processGpuPackedMotionVectorsWithSemaphores(d->rife.get(), d->semaphore.get(), d->sharedFlowSemaphore.get(),
                                                                     *currentPacked, *referencePacked,
                                                                     currentSadPackedPtr, referenceSadPackedPtr,
                                                                     scratch.gpuPackedVectors.data(), gpuVectorConfig,
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
                accumulatePerfStat(d->perf->gpuUploadNs, flowPerf.gpuUploadNs);
                accumulatePerfStat(d->perf->gpuPreprocNs, flowPerf.gpuPreprocNs);
                accumulatePerfStat(d->perf->gpuInferenceNs, flowPerf.gpuInferenceNs);
                accumulatePerfStat(d->perf->gpuFlowResizeNs, flowPerf.gpuFlowResizeNs);
                accumulatePerfStat(d->perf->gpuFlowReduceNs, flowPerf.gpuFlowReduceNs);
                accumulatePerfStat(d->perf->gpuFlowVectorNs, flowPerf.gpuFlowVectorNs);
                accumulatePerfStat(d->perf->gpuReadbackNs, flowPerf.gpuReadbackNs);
                accumulatePerfStat(d->perf->gpuTotalNs, flowPerf.gpuTotalNs);
                accumulatePerfStat(d->perf->gpuInputCacheHitCount, flowPerf.gpuInputCacheHits);
                accumulatePerfStat(d->perf->gpuInputCacheMissCount, flowPerf.gpuInputCacheMisses);
                accumulatePerfStat(d->perf->gpuInputCacheWaitNs, flowPerf.gpuInputCacheWaitNs);
            }
            if (status != 0) {
                vsapi->freeFrame(current);
                vsapi->freeFrame(reference);
                vsapi->freeFrame(currentSource);
                vsapi->freeFrame(referenceSource);
                vsapi->setFilterError("RIFEMV: failed to export motion vectors", frameCtx);
                return nullptr;
            }

            if (!useGpuVectorExport && !d->mvConfig.useChroma) {
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
            } else if (useGpuFullPacked) {
                buildMotionVectorBlobsFromGpuPackedVectors(scratch.gpuPackedVectors.data(), true, d->mvConfig,
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

        const auto pairCarrierStartNs = d->perfStats ? monotonicNowNs() : 0;
        auto dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, nullptr, core);
        zeroMotionVectorFrame(dst, d->vi, vsapi);
        if (d->perfStats)
            accumulatePerfStat(d->perf->pairCarrierNs, monotonicNowNs() - pairCarrierStartNs);

        auto props = vsapi->getFramePropertiesRW(dst);
        const auto pairPropertyStartNs = d->perfStats ? monotonicNowNs() : 0;
        setMotionVectorProperties(props, d->mvConfig.backwardAnalysisData, backwardBlob.data(), static_cast<int>(backwardBlob.size()),
                                  backwardStats, d->sadStats, d->motionStats, vsapi);
        vsapi->mapSetData(props, RIFEMVForwardVectorsInternalKey, forwardBlob.data(), static_cast<int>(forwardBlob.size()), dtBinary, maReplace);
        if (needStats)
            setMotionVectorInternalFrameStats(props, forwardStats, d->sadStats, d->motionStats, false, vsapi);
        if (d->perfStats)
            accumulatePerfStat(d->perf->pairPropertyNs, monotonicNowNs() - pairPropertyStartNs);

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
                                           d->sadStats, d->motionStats, d->perf.get(), core, vsapi);
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
                                           d->sadStats, d->motionStats, d->perf.get(), core, vsapi);

        vsapi->freeFrame(pairFrame);
        if (d->perfStats) {
            accumulatePerfStat(d->perf->outputFrames, 1);
            accumulatePerfStat(d->perf->outputTotalNs, monotonicNowNs() - outputStartNs);
        }
        return dst;
    }

    return nullptr;
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

// Convert native integer YUV planes to normalized float buffers for Vulkan processing.
static void buildDegrainNativePlanes(const VSFrame* frame, std::array<ncnn::Mat, 3>& planes, const VSAPI* vsapi) {
    const auto* format = vsapi->getVideoFrameFormat(frame);
    const auto maxSample = static_cast<float>((1 << format->bitsPerSample) - 1);
    for (int plane = 0; plane < 3; plane++) {
        const auto width = vsapi->getFrameWidth(frame, plane);
        const auto height = vsapi->getFrameHeight(frame, plane);
        const auto stride = vsapi->getStride(frame, plane);
        const auto* source = vsapi->getReadPtr(frame, plane);
        planes[plane].create(width, height, sizeof(float), 1);
        auto* destination = static_cast<float*>(planes[plane].data);
        for (int y = 0; y < height; y++) {
            const auto* row = source + static_cast<ptrdiff_t>(y) * stride;
            for (int x = 0; x < width; x++) {
                const auto sample = format->bytesPerSample == 1 ? row[x] : reinterpret_cast<const uint16_t*>(row)[x];
                destination[static_cast<size_t>(y) * width + x] = sample / maxSample;
            }
        }
    }
}

// Quantize normalized float results back to the source format without changing frame properties.
static void writeDegrainNativePlanes(VSFrame* frame, const std::array<ncnn::Mat, 3>& planes, const VSAPI* vsapi) {
    const auto* format = vsapi->getVideoFrameFormat(frame);
    const auto maxSample = (1 << format->bitsPerSample) - 1;
    for (int plane = 0; plane < 3; plane++) {
        const auto width = vsapi->getFrameWidth(frame, plane);
        const auto height = vsapi->getFrameHeight(frame, plane);
        const auto stride = vsapi->getStride(frame, plane);
        auto* destination = vsapi->getWritePtr(frame, plane);
        const auto* source = static_cast<const float*>(planes[plane].data);
        for (int y = 0; y < height; y++) {
            auto* row = destination + static_cast<ptrdiff_t>(y) * stride;
            for (int x = 0; x < width; x++) {
                const auto value = std::clamp(static_cast<int>(std::floor(source[static_cast<size_t>(y) * width + x] * maxSample + 0.5f)), 0, maxSample);
                if (format->bytesPerSample == 1)
                    row[x] = static_cast<uint8_t>(value);
                else
                    reinterpret_cast<uint16_t*>(row)[x] = static_cast<uint16_t>(value);
            }
        }
    }
}

static const VSFrame* VS_CC rifeDegrainGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData,
                                                VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<RIFEDegrainData*>(instanceData);
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->inferenceNode, frameCtx);
        vsapi->requestFrameFilter(n, d->sourceNode, frameCtx);
        for (int delta = 1; delta <= d->radius; delta++) {
            if (n - delta >= 0) {
                vsapi->requestFrameFilter(n - delta, d->inferenceNode, frameCtx);
                vsapi->requestFrameFilter(n - delta, d->sourceNode, frameCtx);
            }
            if (n + delta < d->vi.numFrames) {
                vsapi->requestFrameFilter(n + delta, d->inferenceNode, frameCtx);
                vsapi->requestFrameFilter(n + delta, d->sourceNode, frameCtx);
            }
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady)
        return nullptr;

    const auto frame_start_ns = d->perfStats ? monotonicNowNs() : 0;
    const VSFrame* currentInferenceFrame = vsapi->getFrameFilter(n, d->inferenceNode, frameCtx);
    const VSFrame* currentSourceFrame = vsapi->getFrameFilter(n, d->sourceNode, frameCtx);
    auto currentInference = getOrCreatePackedInferenceFrame(d->packedCache, currentInferenceFrame, n,
                                                            currentInferenceFrame ? vsapi->getFrameWidth(currentInferenceFrame, 0) : 0,
                                                            currentInferenceFrame ? vsapi->getFrameHeight(currentInferenceFrame, 0) : 0,
                                                            d->perfStats ? d->perf.get() : nullptr, vsapi);
    std::array<ncnn::Mat, 3> currentPlanes;
    buildDegrainNativePlanes(currentSourceFrame, currentPlanes, vsapi);
    std::array<const ncnn::Mat*, 3> currentPlanePointers{ &currentPlanes[0], &currentPlanes[1], &currentPlanes[2] };

    std::vector<const VSFrame*> referenceInferenceFrames;
    std::vector<const VSFrame*> referenceSourceFrames;
    std::vector<std::shared_ptr<const ncnn::Mat>> referenceInference;
    std::vector<std::array<ncnn::Mat, 3>> referencePlanes;
    std::vector<RIFEDegrainReference> references;
    std::vector<int> validSlots;
    referenceInferenceFrames.reserve(d->radius * 2);
    referenceSourceFrames.reserve(d->radius * 2);
    referenceInference.reserve(d->radius * 2);
    referencePlanes.reserve(d->radius * 2);
    references.reserve(d->radius * 2);
    for (int delta = 1; delta <= d->radius; delta++) {
        const int indices[2]{ n - delta, n + delta };
        for (int direction = 0; direction < 2; direction++) {
            const auto index = indices[direction];
            if (index < 0 || index >= d->vi.numFrames)
                continue;
            auto* inferenceFrame = vsapi->getFrameFilter(index, d->inferenceNode, frameCtx);
            auto* sourceFrame = vsapi->getFrameFilter(index, d->sourceNode, frameCtx);
            referenceInferenceFrames.push_back(inferenceFrame);
            referenceSourceFrames.push_back(sourceFrame);
            referenceInference.push_back(getOrCreatePackedInferenceFrame(d->packedCache, inferenceFrame, index,
                                                                          vsapi->getFrameWidth(inferenceFrame, 0),
                                                                          vsapi->getFrameHeight(inferenceFrame, 0),
                                                                          d->perfStats ? d->perf.get() : nullptr, vsapi));
            referencePlanes.emplace_back();
            buildDegrainNativePlanes(sourceFrame, referencePlanes.back(), vsapi);
            const auto& planes = referencePlanes.back();
            references.push_back({ referenceInference.back().get(), { &planes[0], &planes[1], &planes[2] } });
            validSlots.push_back((delta - 1) * 2 + direction);
        }
    }

    std::array<ncnn::Mat, 3> outputPlanes;
    std::array<ncnn::Mat*, 3> outputPlanePointers{ &outputPlanes[0], &outputPlanes[1], &outputPlanes[2] };
    std::vector<RIFEDegrainStats> stats;
    FlowPerfBreakdown flowPerf{};
    int64_t local_wait_ns{};
    int64_t shared_wait_ns{};
    int64_t process_ns{};
    int status{};
    if (references.empty()) {
        outputPlanes = currentPlanes;
    } else {
        const auto wait_start_ns = d->perfStats ? monotonicNowNs() : 0;
        d->semaphore->acquire();
        local_wait_ns = d->perfStats ? monotonicNowNs() - wait_start_ns : 0;
        const auto shared_wait_start_ns = d->perfStats ? monotonicNowNs() : 0;
        d->sharedFlowSemaphore->acquire();
        shared_wait_ns = d->perfStats ? monotonicNowNs() - shared_wait_start_ns : 0;
        const auto process_start_ns = d->perfStats ? monotonicNowNs() : 0;
        status = d->rife->process_degrain(*currentInference, currentPlanePointers, references, d->config,
                                          outputPlanePointers, d->sadStats ? &stats : nullptr,
                                          d->perfStats ? &flowPerf : nullptr);
        process_ns = d->perfStats ? monotonicNowNs() - process_start_ns : 0;
        d->sharedFlowSemaphore->release();
        d->semaphore->release();
    }

    VSFrame* output{};
    if (status == 0) {
        output = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, currentSourceFrame, core);
        writeDegrainNativePlanes(output, outputPlanes, vsapi);
        if (d->sadStats) {
            std::vector<int64_t> average(d->radius * 2, -1);
            std::vector<int64_t> maximum(d->radius * 2, -1);
            for (size_t i = 0; i < stats.size(); i++) {
                average[validSlots[i]] = static_cast<int64_t>(std::floor(stats[i].averageSad + 0.5f));
                maximum[validSlots[i]] = static_cast<int64_t>(std::floor(stats[i].maximumSad + 0.5f));
            }
            auto* props = vsapi->getFramePropertiesRW(output);
            for (int i = 0; i < d->radius * 2; i++) {
                vsapi->mapSetInt(props, RIFEDegrainAvgSadKey, average[i], i == 0 ? maReplace : maAppend);
                vsapi->mapSetInt(props, RIFEDegrainMaxSadKey, maximum[i], i == 0 ? maReplace : maAppend);
            }
        }
    } else {
        vsapi->setFilterError("RIFEDegrain: GPU denoising failed", frameCtx);
    }

    vsapi->freeFrame(currentInferenceFrame);
    vsapi->freeFrame(currentSourceFrame);
    for (const auto* frame : referenceInferenceFrames)
        vsapi->freeFrame(frame);
    for (const auto* frame : referenceSourceFrames)
        vsapi->freeFrame(frame);
    if (d->perfStats) {
        accumulatePerfStat(d->perf->pairFrames, 1);
        accumulatePerfStat(d->perf->flowCalls, static_cast<int64_t>(references.size()));
        accumulatePerfStat(d->perf->pairTotalNs, monotonicNowNs() - frame_start_ns);
        accumulatePerfStat(d->perf->localSemaphoreWaitNs, local_wait_ns);
        accumulatePerfStat(d->perf->sharedSemaphoreWaitNs, shared_wait_ns);
        accumulatePerfStat(d->perf->semaphoreWaitNs, local_wait_ns + shared_wait_ns);
        accumulatePerfStat(d->perf->processFlowNs, process_ns);
        accumulatePerfStat(d->perf->rifeProcessWallNs, process_ns);
        accumulatePerfStat(d->perf->flowSetupNs, flowPerf.setupNs);
        accumulatePerfStat(d->perf->flowCommandRecordNs, flowPerf.commandRecordNs);
        accumulatePerfStat(d->perf->flowInferenceRecordNs, flowPerf.inferenceRecordNs);
        accumulatePerfStat(d->perf->flowSubmitWaitNs, flowPerf.submitWaitNs);
        accumulatePerfStat(d->perf->flowCleanupNs, flowPerf.cleanupNs);
        accumulatePerfStat(d->perf->flowReadbackBytes, flowPerf.readbackBytes);
        accumulatePerfStat(d->perf->degrainSadRecordNs, flowPerf.degrainSadRecordNs);
        accumulatePerfStat(d->perf->degrainCenteredRecordNs, flowPerf.degrainCenteredRecordNs);
        accumulatePerfStat(d->perf->degrainAccumulateRecordNs, flowPerf.degrainAccumulateRecordNs);
        accumulatePerfStat(d->perf->degrainOutputRecordNs, flowPerf.degrainOutputRecordNs);
        accumulatePerfStat(d->perf->degrainStatsRecordNs, flowPerf.degrainStatsRecordNs);
    }
    return output;
}

static void VS_CC rifeDegrainFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<RIFEDegrainData*>(instanceData);
    if (d->perfStats && d->perf)
        printMotionVectorPerfSummary(*d->perf, "RIFEDegrain(radius=" + std::to_string(d->radius) + ")");
    vsapi->freeNode(d->inferenceNode);
    vsapi->freeNode(d->sourceNode);
    delete d;
    if (--numGPUInstances == 0)
        ncnn::destroy_gpu_instance();
}

static void validateCropGridAnalysisData(const MVAnalysisData& analysisData) {
    if (analysisData.nLvCount != 1)
        throw "only single-level MVTools vector clips are supported";
    if (analysisData.nBlkSizeX <= 0 || analysisData.nBlkSizeY <= 0)
        throw "invalid MVTools block size metadata";
    if (analysisData.nOverlapX < 0 || analysisData.nOverlapX >= analysisData.nBlkSizeX ||
        analysisData.nOverlapY < 0 || analysisData.nOverlapY >= analysisData.nBlkSizeY)
        throw "invalid MVTools overlap metadata";
    if (analysisData.nBlkX <= 0 || analysisData.nBlkY <= 0 || analysisData.nWidth <= 0 || analysisData.nHeight <= 0)
        throw "invalid MVTools vector grid metadata";
}

static void configureCropGridData(CropGridData& data, const MVAnalysisData& analysisData) {
    validateCropGridAnalysisData(analysisData);
    data.stepX = analysisData.nBlkSizeX - analysisData.nOverlapX;
    data.stepY = analysisData.nBlkSizeY - analysisData.nOverlapY;
    if (data.stepX <= 0 || data.stepY <= 0)
        throw "invalid MVTools block step metadata";
    if (data.left + data.right >= analysisData.nBlkX || data.top + data.bottom >= analysisData.nBlkY)
        throw "crop removes the entire vector grid";

    data.cropLeftPx = data.left * data.stepX;
    data.cropRightPx = data.right * data.stepX;
    data.cropTopPx = data.top * data.stepY;
    data.cropBottomPx = data.bottom * data.stepY;
    data.sourceWidth = analysisData.nWidth;
    data.sourceHeight = analysisData.nHeight;
    data.outputSourceWidth = data.sourceWidth - data.cropLeftPx - data.cropRightPx;
    data.outputSourceHeight = data.sourceHeight - data.cropTopPx - data.cropBottomPx;
    if (data.outputSourceWidth <= 0 || data.outputSourceHeight <= 0)
        throw "crop removes the entire clip";
}

static void configureCropGridStepData(CropGridData& data, const MVAnalysisData& analysisData) {
    validateCropGridAnalysisData(analysisData);
    data.stepX = analysisData.nBlkSizeX - analysisData.nOverlapX;
    data.stepY = analysisData.nBlkSizeY - analysisData.nOverlapY;
    if (data.stepX <= 0 || data.stepY <= 0)
        throw "invalid MVTools block step metadata";

    data.cropLeftPx = data.left * data.stepX;
    data.cropRightPx = data.right * data.stepX;
    data.cropTopPx = data.top * data.stepY;
    data.cropBottomPx = data.bottom * data.stepY;
}

static void VS_CC cropGridCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto data{ std::make_unique<CropGridData>() };
    VSNode* vectorsNode{};
    const VSFrame* firstFrame{};
    const VSFrame* vectorFrame{};

    try {
        int err{};
        data->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        data->vi = *vsapi->getVideoInfo(data->node);
        if (!vsh::isConstantVideoFormat(&data->vi))
            throw "clip must have constant dimensions and format";

        data->left = vsapi->mapGetIntSaturated(in, "left", 0, &err);
        if (err)
            data->left = 0;
        data->right = vsapi->mapGetIntSaturated(in, "right", 0, &err);
        if (err)
            data->right = 0;
        data->top = vsapi->mapGetIntSaturated(in, "top", 0, &err);
        if (err)
            data->top = 0;
        data->bottom = vsapi->mapGetIntSaturated(in, "bottom", 0, &err);
        if (err)
            data->bottom = 0;
        if (data->left < 0 || data->right < 0 || data->top < 0 || data->bottom < 0)
            throw "crop values must be non-negative";

        vectorsNode = vsapi->mapGetNode(in, "vectors", 0, &err);
        data->mode = err ? CropGridMode::Vector : CropGridMode::Super;

        std::array<char, 1024> errorMsg{};
        firstFrame = vsapi->getFrame(0, data->node, errorMsg.data(), static_cast<int>(errorMsg.size()));
        if (!firstFrame)
            throw std::runtime_error(std::string("failed to retrieve first clip frame: ") + errorMsg.data());

        if (data->mode == CropGridMode::Vector) {
            MVAnalysisData analysisData{};
            if (!readMVAnalysisData(vsapi->getFramePropertiesRO(firstFrame), analysisData, vsapi))
                throw "clip is not a single-level MVTools vector clip; pass vectors= when cropping an mv.Super clip";
            configureCropGridData(*data, analysisData);
            if (data->vi.width != analysisData.nWidth || data->vi.height != analysisData.nHeight)
                throw "vector carrier dimensions do not match MVTools vector metadata";
            validateCropSubsamplingAlignment(data->vi, data->cropLeftPx, data->cropRightPx, data->cropTopPx, data->cropBottomPx);
            data->vi.width = data->outputSourceWidth;
            data->vi.height = data->outputSourceHeight;
        } else {
            vectorFrame = vsapi->getFrame(0, vectorsNode, errorMsg.data(), static_cast<int>(errorMsg.size()));
            if (!vectorFrame)
                throw std::runtime_error(std::string("failed to retrieve first vectors frame: ") + errorMsg.data());

            MVAnalysisData analysisData{};
            if (!readMVAnalysisData(vsapi->getFramePropertiesRO(vectorFrame), analysisData, vsapi))
                throw "vectors clip is missing MVTools vector metadata";
            configureCropGridStepData(*data, analysisData);
            if (!readMVToolsSuperData(vsapi->getFramePropertiesRO(firstFrame), data->super, vsapi))
                throw "clip is not an mv.Super clip or is missing Super metadata on frame 0";
            const auto superSourceWidth = data->vi.width - data->super.hpad * 2;
            const auto superSourceHeight = data->super.height;
            const auto vectorsMatchOriginal = superSourceWidth == analysisData.nWidth && superSourceHeight == analysisData.nHeight;
            const auto vectorsMatchCropped = superSourceWidth == analysisData.nWidth + data->cropLeftPx + data->cropRightPx &&
                                             superSourceHeight == analysisData.nHeight + data->cropTopPx + data->cropBottomPx;
            if (!vectorsMatchOriginal && !vectorsMatchCropped) {
                const auto expectedCroppedWidth = analysisData.nWidth + data->cropLeftPx + data->cropRightPx;
                const auto expectedCroppedHeight = analysisData.nHeight + data->cropTopPx + data->cropBottomPx;
                throw std::runtime_error("Super clip dimensions do not match vector metadata: super carrier=" +
                                         std::to_string(data->vi.width) + "x" + std::to_string(data->vi.height) +
                                         ", super source=" + std::to_string(superSourceWidth) + "x" + std::to_string(superSourceHeight) +
                                         ", vector source=" + std::to_string(analysisData.nWidth) + "x" + std::to_string(analysisData.nHeight) +
                                         ", expected super source=" + std::to_string(analysisData.nWidth) + "x" + std::to_string(analysisData.nHeight) +
                                         " or " + std::to_string(expectedCroppedWidth) + "x" + std::to_string(expectedCroppedHeight) +
                                         " from crop L/R/T/B=" + std::to_string(data->cropLeftPx) + "/" + std::to_string(data->cropRightPx) +
                                         "/" + std::to_string(data->cropTopPx) + "/" + std::to_string(data->cropBottomPx));
            }
            if (data->super.pel != analysisData.nPel)
                throw "Super pel does not match vector metadata";
            if (data->super.hpad != analysisData.nHPadding || data->super.vpad != analysisData.nVPadding)
                throw "Super padding does not match vector metadata";
            if (data->super.levels <= 0)
                throw "Super clip has invalid level metadata";

            data->sourceWidth = superSourceWidth;
            data->sourceHeight = superSourceHeight;
            data->outputSourceWidth = data->sourceWidth - data->cropLeftPx - data->cropRightPx;
            data->outputSourceHeight = data->sourceHeight - data->cropTopPx - data->cropBottomPx;
            if (data->outputSourceWidth <= 0 || data->outputSourceHeight <= 0)
                throw "crop removes the entire clip";

            validateCropSubsamplingAlignment(data->vi, data->cropLeftPx, data->cropRightPx, data->cropTopPx, data->cropBottomPx);
            const auto xRatioUV = 1 << data->vi.format.subSamplingW;
            const auto yRatioUV = 1 << data->vi.format.subSamplingH;
            data->vi.width = cropGridSuperWidth(data->outputSourceWidth, data->super.hpad, xRatioUV);
            data->vi.height = cropGridSuperHeight(data->outputSourceHeight, data->super.levels, data->super.pel,
                                                  data->super.vpad, data->vi.width, yRatioUV);
        }

        vsapi->freeFrame(firstFrame);
        firstFrame = nullptr;
        vsapi->freeFrame(vectorFrame);
        vectorFrame = nullptr;
        vsapi->freeNode(vectorsNode);
        vectorsNode = nullptr;

        VSFilterDependency deps[]{ { data->node, rpStrictSpatial } };
        vsapi->createVideoFilter(out, "CropGrid", &data->vi, cropGridGetFrame, cropGridFree,
                                 fmParallel, deps, 1, data.get(), core);
        data.release();
    } catch (const std::exception& error) {
        vsapi->mapSetError(out, ("CropGrid: "s + error.what()).c_str());
        vsapi->freeFrame(firstFrame);
        vsapi->freeFrame(vectorFrame);
        vsapi->freeNode(vectorsNode);
        vsapi->freeNode(data->node);
    } catch (const char* error) {
        vsapi->mapSetError(out, ("CropGrid: "s + error).c_str());
        vsapi->freeFrame(firstFrame);
        vsapi->freeFrame(vectorFrame);
        vsapi->freeNode(vectorsNode);
        vsapi->freeNode(data->node);
    }
}

static void VS_CC rifeMVCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto pairData{ std::make_unique<RIFEMVPairData>() };
    VSNode* sadInputNode{};
    VSNode* pairNode{};
    VSNode* forwardNode{};
    bool hasGPUInstance{};
    bool mvSadStats{};
    bool mvMotionStats{};

    try {
        pairData->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        const auto sharedPackedSourceIdentity = reinterpret_cast<uintptr_t>(pairData->node);
        pairData->vi = *vsapi->getVideoInfo(pairData->node);
        const auto sourceVi = pairData->vi;
        int err;
        int sadErr{};
        sadInputNode = vsapi->mapGetNode(in, "sad_clip", 0, &sadErr);
        const auto hasSadClip = !sadErr;
        VSVideoInfo sadInputVi{};
        if (hasSadClip) {
            sadInputVi = *vsapi->getVideoInfo(sadInputNode);
            validateMotionVectorSadClip(sourceVi, sadInputVi);
        }
        const auto sharedLumaSourceIdentity = reinterpret_cast<uintptr_t>(hasSadClip ? sadInputNode : pairData->node);

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
        const auto conversionOptions = readMotionVectorColorConversionOptions(in, vsapi);
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
        const auto hasSadY = vsapi->mapNumElements(in, "sad_y") > 0;
        const auto hasSadUV = vsapi->mapNumElements(in, "sad_uv") > 0;
        const auto mvWeightedSad = hasSadY || hasSadUV;
        auto mvSadY{ static_cast<float>(vsapi->mapGetFloat(in, "sad_y", 0, &err)) };
        if (hasSadY && err)
            throw "sad_y must be a float";
        if (err)
            mvSadY = 1.f;
        auto mvSadUV{ static_cast<float>(vsapi->mapGetFloat(in, "sad_uv", 0, &err)) };
        if (hasSadUV && err)
            throw "sad_uv must be a float";
        if (err)
            mvSadUV = 1.f;

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
        validateWeightedSAD(mvWeightedSad, mvSadY, mvSadUV, mvUseChroma, pairData->mvExportBackend);

        const auto inferenceWidth = computeInferenceDimension(sourceVi.width, resScale, "width");
        const auto inferenceHeight = computeInferenceDimension(sourceVi.height, resScale, "height");
        const auto mvInternalGeometry = createMotionVectorInternalGeometry(sourceVi.width, sourceVi.height,
                                                                           inferenceWidth, inferenceHeight,
                                                                           mvBlockSizeX, mvBlockSizeY,
                                                                           mvOverlapX, mvOverlapY,
                                                                           mvHPadding, mvVPadding);
        const auto clipSet = buildMotionVectorClipSet(pairData->node, sourceVi,
                                                      hasSadClip ? sadInputNode : nullptr,
                                                      hasSadClip ? &sadInputVi : nullptr,
                                                      conversionOptions, inferenceWidth, inferenceHeight, core, vsapi);
        vsapi->freeNode(pairData->node);
        pairData->node = clipSet.inferenceNode;
        pairData->sourceNode = clipSet.sourceNode;
        pairData->sourceMatchesInference = clipSet.sourceMatchesInference;
        vsapi->freeNode(sadInputNode);
        sadInputNode = nullptr;

        const VSVideoInfo* metadataVi = clipSet.inferenceConvertedFromYUV ? &sourceVi : nullptr;
        pairData->mvConfig = createMotionVectorConfig(pairData->vi, metadataVi, mvInternalGeometry, mvUseChroma, mvBlockSizeX, mvBlockSizeY,
                                                      mvOverlapX, mvOverlapY, mvPel, mvDelta,
                                                      mvBits, mvHPadding, mvVPadding, mvBlockReduce, mvWeightedSad, mvSadY, mvSadUV);
        const auto useGpuFull = pairData->mvExportBackend == MotionVectorExportBackend::GpuFull;
        const auto useGpuFullPacked = pairData->mvExportBackend == MotionVectorExportBackend::GpuFullPacked;
        if (useGpuFull || useGpuFullPacked)
            validateGpuFullMotionVectorBackend(pairData->mvConfig, static_cast<int>(pairData->mvExportBackend));
        if (useGpuFullPacked)
            validateGpuFullPackedMotionVectorBackend(pairData->mvConfig);
        printMotionVectorInvocation("RIFEMV", gpuId, gpuThread, sharedFlowInFlight, sharedLumaCacheEnabled, flowScale, flowResizeMode,
                                    pairData->mvExportBackend, sharedPackedCacheEnabled, packedCacheMiB, mvSadStats, mvMotionStats, perfStats,
                                    pairData->mvConfig, resScale, inferenceWidth, inferenceHeight,
                                    conversionOptions, hasSadClip, true);

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
            key.convertedFromYUV = clipSet.sourceConvertedFromYUV;
            if (clipSet.sourceConvertedFromYUV)
                setMotionVectorConversionCacheKey(key, conversionOptions);
            pairData->lumaCache = acquireSharedLumaCache(key, 16);
        } else {
            pairData->lumaCache = createMotionVectorLumaCache(4);
        }
        const auto packedCacheMaxEntries = computePackedCacheMaxEntries(clipSet.inferenceVi.width, clipSet.inferenceVi.height, packedCacheMiB);
        if (sharedPackedCacheEnabled) {
            SharedMotionVectorPackedCacheKey key{};
            key.sourceIdentity = sharedPackedSourceIdentity;
            key.inferenceWidth = clipSet.inferenceVi.width;
            key.inferenceHeight = clipSet.inferenceVi.height;
            key.convertedFromYUV = clipSet.inferenceConvertedFromYUV;
            if (clipSet.inferenceConvertedFromYUV)
                setMotionVectorConversionCacheKey(key, conversionOptions);
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
                                                useGpuFull || useGpuFullPacked);
        loadRIFEModel(*pairData->rife, resolvedModel.modelPath);
    } catch (const std::exception& error) {
        vsapi->mapSetError(out, ("RIFEMV: "s + error.what()).c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);
        vsapi->freeNode(sadInputNode);

        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    } catch (const char* error) {
        vsapi->mapSetError(out, ("RIFEMV: "s + error).c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);
        vsapi->freeNode(sadInputNode);

        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }

    const auto outputVi = pairData->vi;
    const auto mvConfig = pairData->mvConfig;
    const auto mvPerfStatsEnabled = pairData->perfStats;
    const auto mvPerf = pairData->perf;
    VSFilterDependency pairDeps[]{ { pairData->node, rpGeneral }, { pairData->sourceNode, rpGeneral } };
    const auto pairDependencyCount = pairData->sourceMatchesInference ? 1 : 2;
    pairNode = vsapi->createVideoFilter2("RIFEMVPair", &pairData->vi, rifeMVPairGetFrame, rifeMVPairFree, fmParallel,
                                         pairDeps, pairDependencyCount, pairData.get(), core);
    if (!pairNode) {
        vsapi->mapSetError(out, "RIFEMV: failed to create internal pair filter");
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);
        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }
    pairData.release();

    auto forwardData = std::make_unique<RIFEMVOutputData>();
    forwardData->node = vsapi->addNodeRef(pairNode);
    forwardData->vi = outputVi;
    forwardData->analysisData = mvConfig.forwardAnalysisData;
    forwardData->invalidBlob = buildInvalidMotionVectorBlob(mvConfig, false,
                                                            (mvSadStats || mvMotionStats) ? &forwardData->invalidStats : nullptr,
                                                            mvSadStats, mvMotionStats);
    forwardData->backward = false;
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
        vsapi->freeNode(pairNode);
        return;
    }
    forwardData.release();

    vsapi->mapConsumeNode(out, "clip", pairNode, maAppend);
    vsapi->mapConsumeNode(out, "clip", forwardNode, maAppend);
}

static void VS_CC rifeDegrainCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto data = std::make_unique<RIFEDegrainData>();
    VSNode* inputNode{};
    VSNode* flowInputNode{};
    bool hasGPUInstance{};
    try {
        int err{};
        inputNode = vsapi->mapGetNode(in, "clip", 0, nullptr);
        data->vi = *vsapi->getVideoInfo(inputNode);
        if (!vsh::isConstantVideoFormat(&data->vi) || data->vi.format.colorFamily != cfYUV ||
            data->vi.format.sampleType != stInteger || data->vi.format.bitsPerSample < 8 || data->vi.format.bitsPerSample > 16 ||
            data->vi.format.numPlanes != 3 || data->vi.format.subSamplingW > 1 || data->vi.format.subSamplingH > 1)
            throw "clip must be constant-format 8-16 bit integer YUV420, YUV422, YUV440, or YUV444";
        flowInputNode = vsapi->mapGetNode(in, "flow_clip", 0, &err);
        if (err)
            flowInputNode = vsapi->addNodeRef(inputNode);
        const auto flowInputVi = *vsapi->getVideoInfo(flowInputNode);
        if (!vsh::isConstantVideoFormat(&flowInputVi) || flowInputVi.numFrames != data->vi.numFrames)
            throw "flow_clip must have a constant format and the same frame count as clip";
        if (static_cast<int64_t>(flowInputVi.width) * data->vi.height != static_cast<int64_t>(data->vi.width) * flowInputVi.height)
            throw "flow_clip must have the same display aspect ratio as clip";
        const auto sourceIdentity = reinterpret_cast<uintptr_t>(flowInputNode);
        if (ncnn::create_gpu_instance())
            throw "failed to create GPU instance";
        ++numGPUInstances;
        hasGPUInstance = true;

        const auto modelPathValue = vsapi->mapGetData(in, "model_path", 0, &err);
        const std::string modelPath = err ? "" : modelPathValue;
        auto gpuId = vsapi->mapGetIntSaturated(in, "gpu_id", 0, &err);
        if (err)
            gpuId = ncnn::get_default_gpu_index();
        auto gpuThread = vsapi->mapGetIntSaturated(in, "gpu_thread", 0, &err);
        if (err)
            gpuThread = 2;
        auto sharedFlowInFlight = vsapi->mapGetIntSaturated(in, "shared_flow_inflight", 0, &err);
        const auto sharedFlowSpecified = !err;
        auto sharedPackedCache = !!vsapi->mapGetInt(in, "shared_packed_cache", 0, &err);
        if (err)
            sharedPackedCache = true;
        auto packedCacheMiB = vsapi->mapGetIntSaturated(in, "packed_cache_mib", 0, &err);
        if (err)
            packedCacheMiB = 256;
        auto flowScale = static_cast<float>(vsapi->mapGetFloat(in, "flow_scale", 0, &err));
        if (err)
            flowScale = 1.f;
        auto resScale = static_cast<float>(vsapi->mapGetFloat(in, "res_scale", 0, &err));
        if (err)
            resScale = 1.f;
        data->radius = vsapi->mapGetIntSaturated(in, "radius", 0, &err);
        if (err)
            data->radius = 1;
        data->config.thSad = static_cast<float>(vsapi->mapGetFloat(in, "thsad", 0, &err));
        if (err)
            data->config.thSad = 400.f;
        data->config.thSadC = static_cast<float>(vsapi->mapGetFloat(in, "thsadc", 0, &err));
        if (err)
            data->config.thSadC = data->config.thSad;
        data->config.sadCenter = !!vsapi->mapGetInt(in, "sad_center", 0, &err);
        if (err)
            data->config.sadCenter = false;
        data->config.sadCenterFloor = static_cast<float>(vsapi->mapGetFloat(in, "sad_center_floor", 0, &err));
        if (err)
            data->config.sadCenterFloor = 0.f;
        data->config.limit = static_cast<float>(vsapi->mapGetFloat(in, "limit", 0, &err));
        if (err)
            data->config.limit = 255.f;
        data->config.limitC = static_cast<float>(vsapi->mapGetFloat(in, "limitc", 0, &err));
        if (err)
            data->config.limitC = data->config.limit;
        data->config.flowConsistency = static_cast<float>(vsapi->mapGetFloat(in, "flow_consistency", 0, &err));
        if (err)
            data->config.flowConsistency = 1.5f;
        data->sadStats = !!vsapi->mapGetInt(in, "sad_stats", 0, &err);
        if (err)
            data->sadStats = false;
        data->perfStats = !!vsapi->mapGetInt(in, "perf_stats", 0, &err);
        if (err)
            data->perfStats = false;
        const auto conversionOptions = readMotionVectorColorConversionOptions(in, vsapi);

        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count())
            throw "invalid GPU device";
        if (gpuThread < 1)
            throw "gpu_thread must be greater than 0";
        const auto queueCount = std::max(1, static_cast<int>(ncnn::get_gpu_info(gpuId).compute_queue_count()));
        if (!sharedFlowSpecified)
            sharedFlowInFlight = queueCount;
        if (sharedFlowInFlight < 1)
            throw "shared_flow_inflight must be greater than 0";
        if (packedCacheMiB < 1)
            throw "packed_cache_mib must be greater than 0";
        if (data->radius < 1 || data->radius > 3)
            throw "radius must be between 1 and 3 (inclusive)";
        if (!std::isfinite(data->config.thSad) || data->config.thSad < 0.f || data->config.thSad > 16320.f ||
            !std::isfinite(data->config.thSadC) || data->config.thSadC < 0.f || data->config.thSadC > 16320.f)
            throw "thsad and thsadc must be finite and between 0 and 16320 (inclusive)";
        if (!std::isfinite(data->config.limit) || data->config.limit < 0.f || data->config.limit > 255.f ||
            !std::isfinite(data->config.limitC) || data->config.limitC < 0.f || data->config.limitC > 255.f)
            throw "limit and limitc must be finite and between 0 and 255 (inclusive)";
        if (!std::isfinite(data->config.flowConsistency) || data->config.flowConsistency < 0.f || data->config.flowConsistency > 32.f)
            throw "flow_consistency must be finite and between 0 and 32 source pixels (inclusive)";
        if (!std::isfinite(data->config.sadCenterFloor) || data->config.sadCenterFloor < 0.f || data->config.sadCenterFloor > 1.f)
            throw "sad_center_floor must be finite and between 0 and 1 (inclusive)";
        if (!data->config.sadCenter && data->config.sadCenterFloor > 0.f)
            throw "sad_center_floor requires sad_center=1";
        validateAndNormalizeFlowScale(flowScale);
        validateResScale(resScale);
        const auto resolvedModel = resolveRIFEModel(modelPath);
        if (isEarlyUnsupportedRIFEV4Model(resolvedModel.modelPath))
            throw RIFEMVUnsupportedEarlyV4Error;
        if (!supportsMotionVectorExport(resolvedModel))
            throw RIFEMVModelRequirementError;

        const auto inferenceWidth = computeInferenceDimension(flowInputVi.width, resScale, "width");
        const auto inferenceHeight = computeInferenceDimension(flowInputVi.height, resScale, "height");
        const auto clipSet = buildMotionVectorClipSet(flowInputNode, flowInputVi, nullptr, nullptr, conversionOptions,
                                                      inferenceWidth, inferenceHeight, core, vsapi);
        auto* nativeSourceNode = vsapi->addNodeRef(inputNode);
        vsapi->freeNode(inputNode);
        inputNode = nullptr;
        vsapi->freeNode(flowInputNode);
        flowInputNode = nullptr;
        data->inferenceNode = clipSet.inferenceNode;
        vsapi->freeNode(clipSet.sourceNode);
        data->sourceNode = nativeSourceNode;
        data->config.lumaWidth = data->vi.width;
        data->config.lumaHeight = data->vi.height;
        data->config.subSamplingW = data->vi.format.subSamplingW;
        data->config.subSamplingH = data->vi.format.subSamplingH;
        data->config.motionScaleX = static_cast<float>(data->vi.width) / inferenceWidth;
        data->config.motionScaleY = static_cast<float>(data->vi.height) / inferenceHeight;
        data->config.sadStats = data->sadStats;

        const auto localFlowInFlight = sharedFlowSpecified ? std::max(gpuThread, sharedFlowInFlight) : gpuThread;
        data->semaphore = std::make_unique<std::counting_semaphore<>>(localFlowInFlight);
        data->sharedFlowSemaphore = acquireSharedFlowSemaphore(gpuId, sharedFlowInFlight);
        const auto packedCacheMaxEntries = computePackedCacheMaxEntries(inferenceWidth, inferenceHeight, packedCacheMiB);
        if (sharedPackedCache) {
            SharedMotionVectorPackedCacheKey key{};
            key.sourceIdentity = sourceIdentity;
            key.inferenceWidth = inferenceWidth;
            key.inferenceHeight = inferenceHeight;
            key.convertedFromYUV = clipSet.inferenceConvertedFromYUV;
            if (clipSet.inferenceConvertedFromYUV)
                setMotionVectorConversionCacheKey(key, conversionOptions);
            data->packedCache = acquireSharedPackedCache(key, packedCacheMaxEntries);
        } else {
            data->packedCache = createMotionVectorPackedCache(packedCacheMaxEntries);
        }
        if (data->perfStats)
            data->perf = std::make_shared<MotionVectorPerfStats>();
        data->rife = std::make_unique<RIFE>(gpuId, flowScale, 1, resolvedModel.rifeV2, resolvedModel.rifeV4,
                                            resolvedModel.padding, FlowResizeMode::ForceGPU,
                                            resolvedModel.disableVulkanFp16, false, true);
        loadRIFEModel(*data->rife, resolvedModel.modelPath);
        std::cerr << "[rmv] RIFEDegrain parameters: gpu_id=" << gpuId << " radius=" << data->radius
                  << " flow_scale=" << flowScale << " res_scale=" << resScale
                  << " flow_input=" << flowInputVi.width << 'x' << flowInputVi.height
                  << " inference=" << inferenceWidth << 'x' << inferenceHeight
                  << " sad_center=" << data->config.sadCenter << " flow_consistency=" << data->config.flowConsistency
                  << " sad_stats=" << data->sadStats << std::endl;

        VSFilterDependency deps[]{ { data->inferenceNode, rpGeneral }, { data->sourceNode, rpGeneral } };
        vsapi->createVideoFilter(out, "RIFEDegrain", &data->vi, rifeDegrainGetFrame, rifeDegrainFree,
                                 fmParallel, deps, 2, data.get(), core);
        data.release();
    } catch (const std::exception& error) {
        vsapi->mapSetError(out, ("RIFEDegrain: "s + error.what()).c_str());
        vsapi->freeNode(inputNode);
        vsapi->freeNode(flowInputNode);
        vsapi->freeNode(data->inferenceNode);
        vsapi->freeNode(data->sourceNode);
        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
    } catch (const char* error) {
        vsapi->mapSetError(out, ("RIFEDegrain: "s + error).c_str());
        vsapi->freeNode(inputNode);
        vsapi->freeNode(flowInputNode);
        vsapi->freeNode(data->inferenceNode);
        vsapi->freeNode(data->sourceNode);
        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
    }
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.nmkd.rmv", "rmv", "RIFE motion-vector export and dense temporal denoising plugin",
                         VS_MAKE_VERSION(9, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    vspapi->registerFunction("CropGrid",
                             "clip:vnode;"
                             "left:int:opt;"
                             "right:int:opt;"
                             "top:int:opt;"
                             "bottom:int:opt;"
                             "vectors:vnode:opt;",
                             "clip:vnode;",
                             cropGridCreate, nullptr, plugin);

    vspapi->registerFunction("RIFEMV",
                             "clip:vnode;"
                             "model_path:data;"
                             "sad_clip:vnode:opt;"
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
                             "sad_y:float:opt;"
                             "sad_uv:float:opt;"
                             "sad_stats:int:opt;"
                             "motion_stats:int:opt;"
                             "matrix_in_s:data:opt;"
                             "range_in_s:data:opt;"
                             "hpad:int:opt;"
                             "vpad:int:opt;"
                             "block_reduce:int:opt;"
                             "chroma:int:opt;",
                             "clip:vnode[];",
                             rifeMVCreate, nullptr, plugin);
    vspapi->registerFunction("RIFEDegrain",
                             "clip:vnode;"
                             "flow_clip:vnode:opt;"
                             "model_path:data;"
                             "radius:int:opt;"
                             "thsad:float:opt;"
                             "thsadc:float:opt;"
                             "sad_center:int:opt;"
                             "sad_center_floor:float:opt;"
                             "limit:float:opt;"
                             "limitc:float:opt;"
                             "flow_consistency:float:opt;"
                             "sad_stats:int:opt;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "shared_flow_inflight:int:opt;"
                             "shared_packed_cache:int:opt;"
                             "packed_cache_mib:int:opt;"
                             "flow_scale:float:opt;"
                             "res_scale:float:opt;"
                             "perf_stats:int:opt;"
                             "matrix_in_s:data:opt;"
                             "range_in_s:data:opt;",
                             "clip:vnode;",
                             rifeDegrainCreate, nullptr, plugin);
}
