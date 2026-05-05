/*
    MIT License

    Copyright (c) 2021-2022 HolyWu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
    std::atomic<int64_t> flowCpuPrepNs{ 0 };
    std::atomic<int64_t> flowCommandRecordNs{ 0 };
    std::atomic<int64_t> flowSubmitWaitNs{ 0 };
    std::atomic<int64_t> flowUnpackNs{ 0 };
    std::atomic<int64_t> flowExportDirectNs{ 0 };
    std::atomic<int64_t> flowExportResizeNs{ 0 };
    std::atomic<int64_t> lumaBuildNs{ 0 };
    std::atomic<int64_t> vectorPackNs{ 0 };
    std::atomic<int64_t> displacementBuildNs{ 0 };
    std::atomic<int64_t> composeNs{ 0 };
};

struct MotionVectorScratchBuffers final {
    std::vector<float> flow;
    std::vector<float> currentLuma;
    std::vector<float> referenceLuma;
    std::vector<float> backwardDisplacement;
    std::vector<float> forwardDisplacement;
    std::vector<float> composedX;
    std::vector<float> composedY;
};

namespace {

constexpr auto MVToolsAnalysisDataKey = "MVTools_MVAnalysisData";
constexpr auto MVToolsVectorsKey = "MVTools_vectors";
constexpr auto RIFEMVBackwardVectorsInternalKey = "_RMVBackwardVectors";
constexpr auto RIFEMVForwardVectorsInternalKey = "_RMVForwardVectors";
constexpr auto RIFEMVBackwardDisplacementInternalKey = "_RMVBackwardDisplacement";
constexpr auto RIFEMVForwardDisplacementInternalKey = "_RMVForwardDisplacement";
constexpr auto RIFEMVBackwardAvgSadRawInternalKey = "_RMVBackwardAvgSadRaw";
constexpr auto RIFEMVForwardAvgSadRawInternalKey = "_RMVForwardAvgSadRaw";
constexpr auto RIFEMVBackwardAvgSad8x8InternalKey = "_RMVBackwardAvgSad8x8";
constexpr auto RIFEMVForwardAvgSad8x8InternalKey = "_RMVForwardAvgSad8x8";
constexpr auto RIFEMVBackwardAvgSadHigh2PctInternalKey = "_RMVBackwardAvgSadHigh2Pct";
constexpr auto RIFEMVForwardAvgSadHigh2PctInternalKey = "_RMVForwardAvgSadHigh2Pct";
constexpr auto RIFEMVBackwardAvgSadHigh10PctInternalKey = "_RMVBackwardAvgSadHigh10Pct";
constexpr auto RIFEMVForwardAvgSadHigh10PctInternalKey = "_RMVForwardAvgSadHigh10Pct";
constexpr auto RIFEMVBackwardAvgSadHigh25PctInternalKey = "_RMVBackwardAvgSadHigh25Pct";
constexpr auto RIFEMVForwardAvgSadHigh25PctInternalKey = "_RMVForwardAvgSadHigh25Pct";
constexpr auto RIFEMVBackwardAvgSadLow2PctInternalKey = "_RMVBackwardAvgSadLow2Pct";
constexpr auto RIFEMVForwardAvgSadLow2PctInternalKey = "_RMVForwardAvgSadLow2Pct";
constexpr auto RIFEMVBackwardAvgSadLow10PctInternalKey = "_RMVBackwardAvgSadLow10Pct";
constexpr auto RIFEMVForwardAvgSadLow10PctInternalKey = "_RMVForwardAvgSadLow10Pct";
constexpr auto RIFEMVBackwardAvgSadLow25PctInternalKey = "_RMVBackwardAvgSadLow25Pct";
constexpr auto RIFEMVForwardAvgSadLow25PctInternalKey = "_RMVForwardAvgSadLow25Pct";
constexpr auto RIFEMVBackwardSadAvgDeviationInternalKey = "_RMVBackwardSadAvgDeviation";
constexpr auto RIFEMVForwardSadAvgDeviationInternalKey = "_RMVForwardSadAvgDeviation";
constexpr auto RIFEMVBackwardAvgAbsDxInternalKey = "_RMVBackwardAvgAbsDx";
constexpr auto RIFEMVForwardAvgAbsDxInternalKey = "_RMVForwardAvgAbsDx";
constexpr auto RIFEMVBackwardAvgAbsDyInternalKey = "_RMVBackwardAvgAbsDy";
constexpr auto RIFEMVForwardAvgAbsDyInternalKey = "_RMVForwardAvgAbsDy";
constexpr auto RIFEMVBackwardAvgAbsMotionInternalKey = "_RMVBackwardAvgAbsMotion";
constexpr auto RIFEMVForwardAvgAbsMotionInternalKey = "_RMVForwardAvgAbsMotion";
constexpr auto RIFEMVBackwardPanAmountInternalKey = "_RMVBackwardPanAmount";
constexpr auto RIFEMVForwardPanAmountInternalKey = "_RMVForwardPanAmount";
constexpr auto RIFEMVAvgSadKey = "RMV_AvgSad";
constexpr auto RIFEMVAvgSadNormKey = "RMV_AvgSadNorm";
constexpr auto RIFEMVAvgSadHigh2PctKey = "RMV_AvgSadHigh2Pct";
constexpr auto RIFEMVAvgSadHigh10PctKey = "RMV_AvgSadHigh10Pct";
constexpr auto RIFEMVAvgSadHigh25PctKey = "RMV_AvgSadHigh25Pct";
constexpr auto RIFEMVAvgSadLow2PctKey = "RMV_AvgSadLow2Pct";
constexpr auto RIFEMVAvgSadLow10PctKey = "RMV_AvgSadLow10Pct";
constexpr auto RIFEMVAvgSadLow25PctKey = "RMV_AvgSadLow25Pct";
constexpr auto RIFEMVSadAvgDeviationKey = "RMV_SadAvgDeviation";
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
    int64_t averageSadRaw;
    int64_t averageSad8x8;
    int64_t averageSadHigh2Pct;
    int64_t averageSadHigh10Pct;
    int64_t averageSadHigh25Pct;
    int64_t averageSadLow2Pct;
    int64_t averageSadLow10Pct;
    int64_t averageSadLow25Pct;
    int64_t sadAvgDeviation;
    double averageAbsDx;
    double averageAbsDy;
    double averageAbsMotion;
    double panAmount;
};

struct MotionVectorFrameStatsKeys final {
    const char* averageSadRaw;
    const char* averageSad8x8;
    const char* averageSadHigh2Pct;
    const char* averageSadHigh10Pct;
    const char* averageSadHigh25Pct;
    const char* averageSadLow2Pct;
    const char* averageSadLow10Pct;
    const char* averageSadLow25Pct;
    const char* sadAvgDeviation;
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
};

constexpr auto RIFEMVModelRequirementError =
    "motion-vector export requires the rife-v3.1/rife-v3.9 model or a rife-v4.2+ model";
constexpr auto RIFEMVUnsupportedEarlyV4Error =
    "legacy rife-v4, rife-v4.0, and rife-v4.1 are not supported for motion-vector export; use rife-v4.2 or newer";

static_assert(sizeof(MVArraySizeType) == 4);
static_assert(sizeof(MVToolsVector) == 16);
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

static int64_t roundPositiveAverageToInt64(const int64_t sum, const int64_t count) noexcept {
    if (count <= 0)
        return 0;

    return (sum + count / 2) / count;
}

static int64_t computePercentileSampleCount(const int64_t totalCount, const int64_t percent) noexcept {
    if (totalCount <= 0)
        return 0;

    return std::max<int64_t>(1, (totalCount * percent + 99) / 100);
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
    const auto flowCpuPrepNs = stats.flowCpuPrepNs.load(std::memory_order_relaxed);
    const auto flowCommandRecordNs = stats.flowCommandRecordNs.load(std::memory_order_relaxed);
    const auto flowSubmitWaitNs = stats.flowSubmitWaitNs.load(std::memory_order_relaxed);
    const auto flowUnpackNs = stats.flowUnpackNs.load(std::memory_order_relaxed);
    const auto flowExportDirectNs = stats.flowExportDirectNs.load(std::memory_order_relaxed);
    const auto flowExportResizeNs = stats.flowExportResizeNs.load(std::memory_order_relaxed);
    const auto lumaBuildNs = stats.lumaBuildNs.load(std::memory_order_relaxed);
    const auto vectorPackNs = stats.vectorPackNs.load(std::memory_order_relaxed);
    const auto displacementBuildNs = stats.displacementBuildNs.load(std::memory_order_relaxed);
    const auto composeNs = stats.composeNs.load(std::memory_order_relaxed);

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
              << " process_flow_avg_ms=" << (flowCalls > 0 ? nsToMs(processFlowNs) / flowCalls : 0.0) << '\n';
    std::cerr << "  flow_cpu_prep_ms=" << nsToMs(flowCpuPrepNs)
              << " flow_record_ms=" << nsToMs(flowCommandRecordNs)
              << " flow_submit_wait_ms=" << nsToMs(flowSubmitWaitNs)
              << " flow_unpack_ms=" << nsToMs(flowUnpackNs)
              << " flow_export_direct_ms=" << nsToMs(flowExportDirectNs)
              << " flow_export_resize_ms=" << nsToMs(flowExportResizeNs) << '\n';
    std::cerr << "  flow_cpu_prep_avg_ms=" << (flowCalls > 0 ? nsToMs(flowCpuPrepNs) / flowCalls : 0.0)
              << " flow_record_avg_ms=" << (flowCalls > 0 ? nsToMs(flowCommandRecordNs) / flowCalls : 0.0)
              << " flow_submit_wait_avg_ms=" << (flowCalls > 0 ? nsToMs(flowSubmitWaitNs) / flowCalls : 0.0)
              << " flow_unpack_avg_ms=" << (flowCalls > 0 ? nsToMs(flowUnpackNs) / flowCalls : 0.0)
              << " flow_export_direct_avg_ms=" << (flowCalls > 0 ? nsToMs(flowExportDirectNs) / flowCalls : 0.0)
              << " flow_export_resize_avg_ms=" << (flowCalls > 0 ? nsToMs(flowExportResizeNs) / flowCalls : 0.0) << '\n';
    std::cerr << "  semaphore_wait_ms=" << nsToMs(semaphoreWaitNs)
              << " local_wait_ms=" << nsToMs(localSemaphoreWaitNs)
              << " shared_wait_ms=" << nsToMs(sharedSemaphoreWaitNs)
              << " luma_build_ms=" << nsToMs(lumaBuildNs)
              << " vector_pack_ms=" << nsToMs(vectorPackNs)
              << " displacement_build_ms=" << nsToMs(displacementBuildNs)
              << " compose_ms=" << nsToMs(composeNs) << std::endl;
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

static void printMotionVectorInvocation(const char* const functionName, const int gpuId, const int gpuThread,
                                        const int sharedFlowInFlight, const float flowScale,
                                        const FlowResizeMode flowResizeMode, const bool perfStats,
                                        const MotionVectorConfig& config, const float resScale,
                                        const int inferenceWidth, const int inferenceHeight,
                                        const char* const matrixIn, const char* const rangeIn,
                                        const bool includeDelta) {
    std::ostringstream message;
    message << std::boolalpha
            << "[rmv] " << functionName << " parameters: gpu_id=" << gpuId
            << " gpu_thread=" << gpuThread
            << " shared_flow_inflight=" << sharedFlowInFlight
            << " flow_scale=" << flowScale
            << " res_scale=" << resScale
            << " cpu_flow_resize=" << flowResizeModeName(flowResizeMode)
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
                                                            const MVAnalysisData& analysisData) noexcept {
    MotionVectorFrameStats stats{};
    if (vectors.empty())
        return stats;

    int64_t sadRawSum{};
    int64_t sad8x8Sum{};
    int64_t absDxSum{};
    int64_t absDySum{};
    int64_t maxSad8x8{};
    double absMotionSum{};
    std::vector<int64_t> normalizedSads;
    std::vector<int> signedDxs;
    std::vector<int> signedDys;
    normalizedSads.reserve(vectors.size());
    signedDxs.reserve(vectors.size());
    signedDys.reserve(vectors.size());
    for (const auto& vector : vectors) {
        const auto sad8x8 = normalizeMotionVectorSadTo8x8(vector.sad, analysisData);
        sadRawSum += vector.sad;
        sad8x8Sum += sad8x8;
        absDxSum += std::llabs(static_cast<int64_t>(vector.x));
        absDySum += std::llabs(static_cast<int64_t>(vector.y));
        absMotionSum += std::hypot(static_cast<double>(vector.x), static_cast<double>(vector.y));
        maxSad8x8 = std::max(maxSad8x8, sad8x8);
        normalizedSads.push_back(sad8x8);
        signedDxs.push_back(vector.x);
        signedDys.push_back(vector.y);
    }

    const auto vectorCount = static_cast<int64_t>(vectors.size());
    stats.averageSadRaw = roundPositiveAverageToInt64(sadRawSum, vectorCount);
    stats.averageSad8x8 = roundPositiveAverageToInt64(sad8x8Sum, vectorCount);
    stats.averageAbsDx = static_cast<double>(absDxSum) / static_cast<double>(vectorCount);
    stats.averageAbsDy = static_cast<double>(absDySum) / static_cast<double>(vectorCount);
    stats.averageAbsMotion = absMotionSum / static_cast<double>(vectorCount);

    std::sort(normalizedSads.begin(), normalizedSads.end());
    const auto computeAverage = [&](const bool highest, const int64_t percent) {
        const auto sampleCount = computePercentileSampleCount(vectorCount, percent);
        int64_t subsetSum{};
        if (highest) {
            for (auto i = vectorCount - sampleCount; i < vectorCount; i++)
                subsetSum += normalizedSads[static_cast<size_t>(i)];
        } else {
            for (auto i = int64_t{}; i < sampleCount; i++)
                subsetSum += normalizedSads[static_cast<size_t>(i)];
        }

        return roundPositiveAverageToInt64(subsetSum, sampleCount);
    };

    stats.averageSadHigh2Pct = computeAverage(true, 2);
    stats.averageSadHigh10Pct = computeAverage(true, 10);
    stats.averageSadHigh25Pct = computeAverage(true, 25);
    stats.averageSadLow2Pct = computeAverage(false, 2);
    stats.averageSadLow10Pct = computeAverage(false, 10);
    stats.averageSadLow25Pct = computeAverage(false, 25);
    stats.sadAvgDeviation = std::llabs(maxSad8x8 - stats.averageSad8x8);

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
                                      const MotionVectorFrameStats& stats, const VSAPI* vsapi) {
    vsapi->mapSetData(props, MVToolsAnalysisDataKey, reinterpret_cast<const char*>(&analysisData), sizeof(analysisData), dtBinary, maReplace);
    vsapi->mapSetData(props, MVToolsVectorsKey, vectorBlob, vectorBlobSize, dtBinary, maReplace);
    vsapi->mapSetInt(props, RIFEMVAvgSadKey, stats.averageSad8x8, maReplace);
    vsapi->mapSetInt(props, RIFEMVAvgSadNormKey, stats.averageSadRaw, maReplace);
    vsapi->mapSetInt(props, RIFEMVAvgSadHigh2PctKey, stats.averageSadHigh2Pct, maReplace);
    vsapi->mapSetInt(props, RIFEMVAvgSadHigh10PctKey, stats.averageSadHigh10Pct, maReplace);
    vsapi->mapSetInt(props, RIFEMVAvgSadHigh25PctKey, stats.averageSadHigh25Pct, maReplace);
    vsapi->mapSetInt(props, RIFEMVAvgSadLow2PctKey, stats.averageSadLow2Pct, maReplace);
    vsapi->mapSetInt(props, RIFEMVAvgSadLow10PctKey, stats.averageSadLow10Pct, maReplace);
    vsapi->mapSetInt(props, RIFEMVAvgSadLow25PctKey, stats.averageSadLow25Pct, maReplace);
    vsapi->mapSetInt(props, RIFEMVSadAvgDeviationKey, stats.sadAvgDeviation, maReplace);
    vsapi->mapSetFloat(props, RIFEMVAvgAbsDxKey, stats.averageAbsDx, maReplace);
    vsapi->mapSetFloat(props, RIFEMVAvgAbsDyKey, stats.averageAbsDy, maReplace);
    vsapi->mapSetFloat(props, RIFEMVAvgAbsMotionKey, stats.averageAbsMotion, maReplace);
    vsapi->mapSetFloat(props, RIFEMVPanAmountKey, stats.panAmount, maReplace);
}

static MotionVectorFrameStatsKeys getMotionVectorInternalFrameStatsKeys(const bool backward) noexcept {
    if (backward) {
        return MotionVectorFrameStatsKeys{
            RIFEMVBackwardAvgSadRawInternalKey,
            RIFEMVBackwardAvgSad8x8InternalKey,
            RIFEMVBackwardAvgSadHigh2PctInternalKey,
            RIFEMVBackwardAvgSadHigh10PctInternalKey,
            RIFEMVBackwardAvgSadHigh25PctInternalKey,
            RIFEMVBackwardAvgSadLow2PctInternalKey,
            RIFEMVBackwardAvgSadLow10PctInternalKey,
            RIFEMVBackwardAvgSadLow25PctInternalKey,
            RIFEMVBackwardSadAvgDeviationInternalKey,
            RIFEMVBackwardAvgAbsDxInternalKey,
            RIFEMVBackwardAvgAbsDyInternalKey,
            RIFEMVBackwardAvgAbsMotionInternalKey,
            RIFEMVBackwardPanAmountInternalKey
        };
    }

    return MotionVectorFrameStatsKeys{
        RIFEMVForwardAvgSadRawInternalKey,
        RIFEMVForwardAvgSad8x8InternalKey,
        RIFEMVForwardAvgSadHigh2PctInternalKey,
        RIFEMVForwardAvgSadHigh10PctInternalKey,
        RIFEMVForwardAvgSadHigh25PctInternalKey,
        RIFEMVForwardAvgSadLow2PctInternalKey,
        RIFEMVForwardAvgSadLow10PctInternalKey,
        RIFEMVForwardAvgSadLow25PctInternalKey,
        RIFEMVForwardSadAvgDeviationInternalKey,
        RIFEMVForwardAvgAbsDxInternalKey,
        RIFEMVForwardAvgAbsDyInternalKey,
        RIFEMVForwardAvgAbsMotionInternalKey,
        RIFEMVForwardPanAmountInternalKey
    };
}

static void setMotionVectorInternalFrameStats(VSMap* props, const MotionVectorFrameStats& stats,
                                              const bool backward, const VSAPI* vsapi) {
    const auto keys = getMotionVectorInternalFrameStatsKeys(backward);
    vsapi->mapSetInt(props, keys.averageSadRaw, stats.averageSadRaw, maReplace);
    vsapi->mapSetInt(props, keys.averageSad8x8, stats.averageSad8x8, maReplace);
    vsapi->mapSetInt(props, keys.averageSadHigh2Pct, stats.averageSadHigh2Pct, maReplace);
    vsapi->mapSetInt(props, keys.averageSadHigh10Pct, stats.averageSadHigh10Pct, maReplace);
    vsapi->mapSetInt(props, keys.averageSadHigh25Pct, stats.averageSadHigh25Pct, maReplace);
    vsapi->mapSetInt(props, keys.averageSadLow2Pct, stats.averageSadLow2Pct, maReplace);
    vsapi->mapSetInt(props, keys.averageSadLow10Pct, stats.averageSadLow10Pct, maReplace);
    vsapi->mapSetInt(props, keys.averageSadLow25Pct, stats.averageSadLow25Pct, maReplace);
    vsapi->mapSetInt(props, keys.sadAvgDeviation, stats.sadAvgDeviation, maReplace);
    vsapi->mapSetFloat(props, keys.averageAbsDx, stats.averageAbsDx, maReplace);
    vsapi->mapSetFloat(props, keys.averageAbsDy, stats.averageAbsDy, maReplace);
    vsapi->mapSetFloat(props, keys.averageAbsMotion, stats.averageAbsMotion, maReplace);
    vsapi->mapSetFloat(props, keys.panAmount, stats.panAmount, maReplace);
}

static MotionVectorFrameStats getMotionVectorInternalFrameStats(const VSMap* props, const bool backward,
                                                                const VSAPI* vsapi) noexcept {
    const auto keys = getMotionVectorInternalFrameStatsKeys(backward);
    MotionVectorFrameStats stats{};
    stats.averageSadRaw = vsapi->mapGetInt(props, keys.averageSadRaw, 0, nullptr);
    stats.averageSad8x8 = vsapi->mapGetInt(props, keys.averageSad8x8, 0, nullptr);
    stats.averageSadHigh2Pct = vsapi->mapGetInt(props, keys.averageSadHigh2Pct, 0, nullptr);
    stats.averageSadHigh10Pct = vsapi->mapGetInt(props, keys.averageSadHigh10Pct, 0, nullptr);
    stats.averageSadHigh25Pct = vsapi->mapGetInt(props, keys.averageSadHigh25Pct, 0, nullptr);
    stats.averageSadLow2Pct = vsapi->mapGetInt(props, keys.averageSadLow2Pct, 0, nullptr);
    stats.averageSadLow10Pct = vsapi->mapGetInt(props, keys.averageSadLow10Pct, 0, nullptr);
    stats.averageSadLow25Pct = vsapi->mapGetInt(props, keys.averageSadLow25Pct, 0, nullptr);
    stats.sadAvgDeviation = vsapi->mapGetInt(props, keys.sadAvgDeviation, 0, nullptr);
    stats.averageAbsDx = vsapi->mapGetFloat(props, keys.averageAbsDx, 0, nullptr);
    stats.averageAbsDy = vsapi->mapGetFloat(props, keys.averageAbsDy, 0, nullptr);
    stats.averageAbsMotion = vsapi->mapGetFloat(props, keys.averageAbsMotion, 0, nullptr);
    stats.panAmount = vsapi->mapGetFloat(props, keys.panAmount, 0, nullptr);
    return stats;
}

static void deleteMotionVectorInternalFrameStats(VSMap* props, const VSAPI* vsapi) {
    for (const auto backward : { true, false }) {
        const auto keys = getMotionVectorInternalFrameStatsKeys(backward);
        vsapi->mapDeleteKey(props, keys.averageSadRaw);
        vsapi->mapDeleteKey(props, keys.averageSad8x8);
        vsapi->mapDeleteKey(props, keys.averageSadHigh2Pct);
        vsapi->mapDeleteKey(props, keys.averageSadHigh10Pct);
        vsapi->mapDeleteKey(props, keys.averageSadHigh25Pct);
        vsapi->mapDeleteKey(props, keys.averageSadLow2Pct);
        vsapi->mapDeleteKey(props, keys.averageSadLow10Pct);
        vsapi->mapDeleteKey(props, keys.averageSadLow25Pct);
        vsapi->mapDeleteKey(props, keys.sadAvgDeviation);
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

static int processFlowWithSemaphores(const RIFE* const rife,
                                     std::counting_semaphore<>* const localSemaphore,
                                     std::counting_semaphore<>* const sharedSemaphore,
                                     const float* src0R, const float* src0G, const float* src0B,
                                     const float* src1R, const float* src1G, const float* src1B,
                                     float* flowOut, const int width, const int height, const ptrdiff_t stride,
                                     int64_t* waitNs = nullptr,
                                     int64_t* localWaitNs = nullptr,
                                     int64_t* sharedWaitNs = nullptr,
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

    const auto status = rife->process_flow(src0R, src0G, src0B, src1R, src1G, src1B, flowOut, width, height, stride, flowPerf);

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
    bool backward;
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
    bool backward;
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

static std::vector<char> packMotionVectorBlob(const std::vector<MVToolsVector>& vectors, const bool valid,
                                              const MVAnalysisData& analysisData,
                                              MotionVectorFrameStats* const stats = nullptr) {
    const auto planeSize = static_cast<MVArraySizeType>(sizeof(MVArraySizeType) + vectors.size() * sizeof(MVToolsVector));
    const auto groupSize = static_cast<MVArraySizeType>(sizeof(MVArraySizeType) * 2 + planeSize);
    std::vector<char> blob(groupSize);
    size_t offset{};
    const auto writeScalar = [&](const auto value) {
        std::memcpy(blob.data() + offset, &value, sizeof(value));
        offset += sizeof(value);
    };

    writeScalar(groupSize);
    writeScalar(valid ? MVArraySizeType{ 1 } : MVArraySizeType{ 0 });
    writeScalar(planeSize);
    std::memcpy(blob.data() + offset, vectors.data(), vectors.size() * sizeof(MVToolsVector));
    if (stats)
        *stats = computeMotionVectorFrameStats(vectors, analysisData);

    return blob;
}

static std::vector<char> buildMVToolsVectorBlob(const VSFrame* current, const VSFrame* reference, const float* flow,
                                                const int flowWidth, const int flowHeight,
                                                const bool valid, const MotionVectorExportContext& ctx,
                                                const MVAnalysisData& analysisData,
                                                const VSAPI* vsapi,
                                                const std::vector<float>* currentLumaCache = nullptr,
                                                const std::vector<float>* referenceLumaCache = nullptr,
                                                MotionVectorFrameStats* const stats = nullptr) {
    const auto vectorCount = static_cast<size_t>(ctx.blkX) * ctx.blkY;
    std::vector<MVToolsVector> vectors(vectorCount);

    if (!valid) {
        for (auto& vector : vectors) {
            vector.x = 0;
            vector.y = 0;
            vector.sad = ctx.invalidSad;
        }

        return packMotionVectorBlob(vectors, false, analysisData, stats);
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

    return packMotionVectorBlob(vectors, true, analysisData, stats);
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

static std::vector<char> buildMotionVectorBlobFromConfig(const VSFrame* current, const VSFrame* reference, const float* flow,
                                                         const int flowWidth, const int flowHeight,
                                                         const bool valid, const MotionVectorConfig& config,
                                                         const bool backward, const VSAPI* vsapi,
                                                         const std::vector<float>* currentLumaCache = nullptr,
                                                         const std::vector<float>* referenceLumaCache = nullptr,
                                                         MotionVectorFrameStats* const stats = nullptr) {
    const auto ctx = createMotionVectorExportContext(config, backward);
    return buildMVToolsVectorBlob(current, reference, flow, flowWidth, flowHeight, valid, ctx,
                                  getMotionVectorAnalysisData(config, backward), vsapi,
                                  currentLumaCache, referenceLumaCache, stats);
}

static std::vector<char> buildInvalidMotionVectorBlob(const MotionVectorConfig& config, const bool backward,
                                                      MotionVectorFrameStats* const stats = nullptr) {
    return buildMotionVectorBlobFromConfig(nullptr, nullptr, nullptr, 0, 0, false, config, backward, nullptr, nullptr, nullptr, stats);
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

static std::vector<char> buildMotionVectorBlobFromDisplacement(const VSFrame* current, const VSFrame* reference,
                                                               const float* displacementX, const float* displacementY,
                                                               const int displacementWidth, const int displacementHeight,
                                                               const bool valid, const MotionVectorConfig& config,
                                                               const bool backward, const VSAPI* vsapi,
                                                               const std::vector<float>* currentLumaCache = nullptr,
                                                               const std::vector<float>* referenceLumaCache = nullptr,
                                                               MotionVectorFrameStats* const stats = nullptr) {
    const auto ctx = createMotionVectorExportContext(config, backward);
    const auto vectorCount = static_cast<size_t>(ctx.blkX) * ctx.blkY;
    std::vector<MVToolsVector> vectors(vectorCount);

    if (!valid) {
        for (auto& vector : vectors) {
            vector.x = 0;
            vector.y = 0;
            vector.sad = ctx.invalidSad;
        }
        return packMotionVectorBlob(vectors, false, getMotionVectorAnalysisData(config, backward), stats);
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

    return packMotionVectorBlob(vectors, true, getMotionVectorAnalysisData(config, backward), stats);
}

static void zeroMotionVectorFrame(VSFrame* frame, const VSVideoInfo& vi, const VSAPI* vsapi);

static VSFrame* createMotionVectorFrame(const VSVideoInfo& vi, const MVAnalysisData& analysisData,
                                        const char* vectorBlob, const int vectorBlobSize,
                                        const MotionVectorFrameStats& stats,
                                        VSCore* core, const VSAPI* vsapi) {
    auto dst = vsapi->newVideoFrame(&vi.format, vi.width, vi.height, nullptr, core);
    zeroMotionVectorFrame(dst, vi, vsapi);
    auto props = vsapi->getFramePropertiesRW(dst);
    setMotionVectorProperties(props, analysisData, vectorBlob, vectorBlobSize, stats, vsapi);
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

        std::vector<char> backwardBlob;
        std::vector<char> forwardBlob;
        MotionVectorFrameStats backwardStats{};
        MotionVectorFrameStats forwardStats{};
        if (reference) {
            const auto width = vsapi->getFrameWidth(current, 0);
            const auto height = vsapi->getFrameHeight(current, 0);
            const auto stride = vsapi->getStride(current, 0) / vsapi->getVideoFrameFormat(current)->bytesPerSample;
            auto& scratch = getMotionVectorScratchBuffers();
            const auto flowSize = static_cast<size_t>(width) * height * 4;
            scratch.flow.resize(flowSize);
            const std::vector<float>* currentLumaCache = nullptr;
            const std::vector<float>* referenceLumaCache = nullptr;
            const auto currentR = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 0));
            const auto currentG = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 1));
            const auto currentB = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 2));
            const auto referenceR = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 0));
            const auto referenceG = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 1));
            const auto referenceB = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 2));

            int64_t semaphoreWaitNs{};
            int64_t localSemaphoreWaitNs{};
            int64_t sharedSemaphoreWaitNs{};
            FlowPerfBreakdown flowPerf{};
            const auto processFlowStartNs = d->perfStats ? monotonicNowNs() : 0;
            const auto status = processFlowWithSemaphores(d->rife.get(), d->semaphore.get(), d->sharedFlowSemaphore.get(),
                                                          currentR, currentG, currentB, referenceR, referenceG, referenceB,
                                                          scratch.flow.data(), width, height, stride,
                                                          d->perfStats ? &semaphoreWaitNs : nullptr,
                                                          d->perfStats ? &localSemaphoreWaitNs : nullptr,
                                                          d->perfStats ? &sharedSemaphoreWaitNs : nullptr,
                                                          d->perfStats ? &flowPerf : nullptr);
            if (d->perfStats) {
                accumulatePerfStat(d->perf->semaphoreWaitNs, semaphoreWaitNs);
                accumulatePerfStat(d->perf->localSemaphoreWaitNs, localSemaphoreWaitNs);
                accumulatePerfStat(d->perf->sharedSemaphoreWaitNs, sharedSemaphoreWaitNs);
                accumulatePerfStat(d->perf->flowCalls, 1);
                accumulatePerfStat(d->perf->processFlowNs, monotonicNowNs() - processFlowStartNs);
                accumulatePerfStat(d->perf->flowCpuPrepNs, flowPerf.cpuPrepNs);
                accumulatePerfStat(d->perf->flowCommandRecordNs, flowPerf.commandRecordNs);
                accumulatePerfStat(d->perf->flowSubmitWaitNs, flowPerf.submitWaitNs);
                accumulatePerfStat(d->perf->flowUnpackNs, flowPerf.unpackNs);
                accumulatePerfStat(d->perf->flowExportDirectNs, flowPerf.exportDirectNs);
                accumulatePerfStat(d->perf->flowExportResizeNs, flowPerf.exportResizeNs);
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

            if (!d->mvConfig.useChroma) {
                const auto lumaStartNs = d->perfStats ? monotonicNowNs() : 0;
                const auto sourceWidth = vsapi->getFrameWidth(currentSource, 0);
                const auto sourceHeight = vsapi->getFrameHeight(currentSource, 0);
                const auto sourceStride = static_cast<int>(vsapi->getStride(currentSource, 0) / vsapi->getVideoFrameFormat(currentSource)->bytesPerSample);
                buildFrameLumaPlane(currentSource, sourceWidth, sourceHeight, sourceStride, scratch.currentLuma, static_cast<double>((1ULL << d->mvConfig.bits) - 1ULL), vsapi);
                buildFrameLumaPlane(referenceSource, sourceWidth, sourceHeight, sourceStride, scratch.referenceLuma, static_cast<double>((1ULL << d->mvConfig.bits) - 1ULL), vsapi);
                currentLumaCache = &scratch.currentLuma;
                referenceLumaCache = &scratch.referenceLuma;
                if (d->perfStats)
                    accumulatePerfStat(d->perf->lumaBuildNs, monotonicNowNs() - lumaStartNs);
            }

            const auto vectorPackStartNs = d->perfStats ? monotonicNowNs() : 0;
            backwardBlob = buildMotionVectorBlobFromConfig(currentSource, referenceSource, scratch.flow.data(), width, height, true, d->mvConfig, true, vsapi,
                                                           currentLumaCache, referenceLumaCache, &backwardStats);
            forwardBlob = buildMotionVectorBlobFromConfig(referenceSource, currentSource, scratch.flow.data(), width, height, true, d->mvConfig, false, vsapi,
                                                          referenceLumaCache, currentLumaCache, &forwardStats);
            if (d->perfStats)
                accumulatePerfStat(d->perf->vectorPackNs, monotonicNowNs() - vectorPackStartNs);
        } else {
            backwardBlob = buildInvalidMotionVectorBlob(d->mvConfig, true, &backwardStats);
            forwardBlob = buildInvalidMotionVectorBlob(d->mvConfig, false, &forwardStats);
        }

        vsapi->mapSetData(props, RIFEMVBackwardVectorsInternalKey, backwardBlob.data(), static_cast<int>(backwardBlob.size()), dtBinary, maReplace);
        vsapi->mapSetData(props, RIFEMVForwardVectorsInternalKey, forwardBlob.data(), static_cast<int>(forwardBlob.size()), dtBinary, maReplace);
        setMotionVectorInternalFrameStats(props, backwardStats, true, vsapi);
        setMotionVectorInternalFrameStats(props, forwardStats, false, vsapi);

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
            auto dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, nullptr, core);
            zeroMotionVectorFrame(dst, d->vi, vsapi);
            auto props = vsapi->getFramePropertiesRW(dst);
            setMotionVectorProperties(props, d->analysisData, d->invalidBlob.data(), static_cast<int>(d->invalidBlob.size()), d->invalidStats, vsapi);
            return dst;
        }
    } else if (activationReason == arAllFramesReady) {
        const auto outputStartNs = d->perfStats ? monotonicNowNs() : 0;
        const VSFrame* pairFrame{};
        if (pairIndex >= 0 && pairIndex < d->vi.numFrames)
            pairFrame = vsapi->getFrameFilter(pairIndex, d->node, frameCtx);

        VSFrame* dst{};
        const char* vectorBlob = nullptr;
        int vectorBlobSize{};
        auto stats = d->invalidStats;
        if (pairFrame) {
            const auto pairProps = vsapi->getFramePropertiesRO(pairFrame);
            const auto blobKey = d->backward ? RIFEMVBackwardVectorsInternalKey : RIFEMVForwardVectorsInternalKey;
            vectorBlob = vsapi->mapGetData(pairProps, blobKey, 0, nullptr);
            vectorBlobSize = vsapi->mapGetDataSize(pairProps, blobKey, 0, nullptr);
            stats = getMotionVectorInternalFrameStats(pairProps, d->backward, vsapi);
            dst = vsapi->copyFrame(pairFrame, core);
        } else {
            dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, nullptr, core);
            zeroMotionVectorFrame(dst, d->vi, vsapi);
            vectorBlob = d->invalidBlob.data();
            vectorBlobSize = static_cast<int>(d->invalidBlob.size());
        }

        auto props = vsapi->getFramePropertiesRW(dst);
        vsapi->mapDeleteKey(props, RIFEMVBackwardVectorsInternalKey);
        vsapi->mapDeleteKey(props, RIFEMVForwardVectorsInternalKey);
        deleteMotionVectorInternalFrameStats(props, vsapi);
        setMotionVectorProperties(props, d->analysisData, vectorBlob, vectorBlobSize, stats, vsapi);

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

        std::vector<char> backwardBlob;
        std::vector<char> forwardBlob;
        MotionVectorFrameStats backwardStats{};
        MotionVectorFrameStats forwardStats{};
        auto& backwardDisplacement = scratch.backwardDisplacement;
        auto& forwardDisplacement = scratch.forwardDisplacement;
        const auto planeSize = static_cast<size_t>(d->mvConfig.inferenceWidth) * d->mvConfig.inferenceHeight;

        if (reference) {
            const auto width = vsapi->getFrameWidth(current, 0);
            const auto height = vsapi->getFrameHeight(current, 0);
            const auto stride = vsapi->getStride(current, 0) / vsapi->getVideoFrameFormat(current)->bytesPerSample;
            const auto flowSize = static_cast<size_t>(width) * height * 4;
            scratch.flow.resize(flowSize);
            const std::vector<float>* currentLumaCache = nullptr;
            const std::vector<float>* referenceLumaCache = nullptr;
            const auto currentR = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 0));
            const auto currentG = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 1));
            const auto currentB = reinterpret_cast<const float*>(vsapi->getReadPtr(current, 2));
            const auto referenceR = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 0));
            const auto referenceG = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 1));
            const auto referenceB = reinterpret_cast<const float*>(vsapi->getReadPtr(reference, 2));

            int64_t semaphoreWaitNs{};
            int64_t localSemaphoreWaitNs{};
            int64_t sharedSemaphoreWaitNs{};
            FlowPerfBreakdown flowPerf{};
            const auto processFlowStartNs = d->perfStats ? monotonicNowNs() : 0;
            const auto status = processFlowWithSemaphores(d->rife.get(), d->semaphore.get(), d->sharedFlowSemaphore.get(),
                                                          currentR, currentG, currentB, referenceR, referenceG, referenceB,
                                                          scratch.flow.data(), width, height, stride,
                                                          d->perfStats ? &semaphoreWaitNs : nullptr,
                                                          d->perfStats ? &localSemaphoreWaitNs : nullptr,
                                                          d->perfStats ? &sharedSemaphoreWaitNs : nullptr,
                                                          d->perfStats ? &flowPerf : nullptr);
            if (d->perfStats) {
                accumulatePerfStat(d->perf->semaphoreWaitNs, semaphoreWaitNs);
                accumulatePerfStat(d->perf->localSemaphoreWaitNs, localSemaphoreWaitNs);
                accumulatePerfStat(d->perf->sharedSemaphoreWaitNs, sharedSemaphoreWaitNs);
                accumulatePerfStat(d->perf->flowCalls, 1);
                accumulatePerfStat(d->perf->processFlowNs, monotonicNowNs() - processFlowStartNs);
                accumulatePerfStat(d->perf->flowCpuPrepNs, flowPerf.cpuPrepNs);
                accumulatePerfStat(d->perf->flowCommandRecordNs, flowPerf.commandRecordNs);
                accumulatePerfStat(d->perf->flowSubmitWaitNs, flowPerf.submitWaitNs);
                accumulatePerfStat(d->perf->flowUnpackNs, flowPerf.unpackNs);
                accumulatePerfStat(d->perf->flowExportDirectNs, flowPerf.exportDirectNs);
                accumulatePerfStat(d->perf->flowExportResizeNs, flowPerf.exportResizeNs);
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
                buildFrameLumaPlane(currentSource, sourceWidth, sourceHeight, sourceStride, scratch.currentLuma, static_cast<double>((1ULL << d->mvConfig.bits) - 1ULL), vsapi);
                buildFrameLumaPlane(referenceSource, sourceWidth, sourceHeight, sourceStride, scratch.referenceLuma, static_cast<double>((1ULL << d->mvConfig.bits) - 1ULL), vsapi);
                currentLumaCache = &scratch.currentLuma;
                referenceLumaCache = &scratch.referenceLuma;
                if (d->perfStats)
                    accumulatePerfStat(d->perf->lumaBuildNs, monotonicNowNs() - lumaStartNs);
            }

            const auto vectorPackStartNs = d->perfStats ? monotonicNowNs() : 0;
            backwardBlob = buildMotionVectorBlobFromConfig(currentSource, referenceSource, scratch.flow.data(), width, height, true, d->mvConfig, true, vsapi,
                                                           currentLumaCache, referenceLumaCache, &backwardStats);
            forwardBlob = buildMotionVectorBlobFromConfig(referenceSource, currentSource, scratch.flow.data(), width, height, true, d->mvConfig, false, vsapi,
                                                          referenceLumaCache, currentLumaCache, &forwardStats);
            if (d->perfStats)
                accumulatePerfStat(d->perf->vectorPackNs, monotonicNowNs() - vectorPackStartNs);
            const auto displacementBuildStartNs = d->perfStats ? monotonicNowNs() : 0;
            buildDisplacementFromFlow(scratch.flow.data(), width, height, 0, backwardDisplacement);
            buildDisplacementFromFlow(scratch.flow.data(), width, height, 2, forwardDisplacement);
            if (d->perfStats)
                accumulatePerfStat(d->perf->displacementBuildNs, monotonicNowNs() - displacementBuildStartNs);
        } else {
            backwardBlob = buildInvalidMotionVectorBlob(d->mvConfig, true, &backwardStats);
            forwardBlob = buildInvalidMotionVectorBlob(d->mvConfig, false, &forwardStats);
            backwardDisplacement.assign(planeSize * 2, 0.0f);
            forwardDisplacement.assign(planeSize * 2, 0.0f);
        }

        vsapi->mapSetData(props, RIFEMVBackwardVectorsInternalKey, backwardBlob.data(), static_cast<int>(backwardBlob.size()), dtBinary, maReplace);
        vsapi->mapSetData(props, RIFEMVForwardVectorsInternalKey, forwardBlob.data(), static_cast<int>(forwardBlob.size()), dtBinary, maReplace);
        setMotionVectorInternalFrameStats(props, backwardStats, true, vsapi);
        setMotionVectorInternalFrameStats(props, forwardStats, false, vsapi);
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
        return createMotionVectorFrame(d->vi, d->analysisData, d->invalidBlob.data(), static_cast<int>(d->invalidBlob.size()), d->invalidStats, core, vsapi);
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

            const auto stats = getMotionVectorInternalFrameStats(props, d->backward, vsapi);
            auto dst = createMotionVectorFrame(d->vi, d->analysisData, vectorBlob, vectorBlobSize, stats, core, vsapi);
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
        const auto vectorBlob = buildMotionVectorBlobFromDisplacement(current, reference,
                                                                      scratch.composedX.data(), scratch.composedY.data(), width, height, true,
                                                                      d->mvConfig, d->backward, vsapi, nullptr, nullptr, &stats);
        if (d->perfStats)
            accumulatePerfStat(d->perf->vectorPackNs, monotonicNowNs() - vectorPackStartNs);

        auto dst = createMotionVectorFrame(d->vi, d->analysisData, vectorBlob.data(), static_cast<int>(vectorBlob.size()), stats, core, vsapi);
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
    VSNode* mvClip{};
    VSNode* pairNode{};
    VSNode* backwardNode{};
    VSNode* forwardNode{};
    bool hasGPUInstance{};

    try {
        pairData->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        pairData->vi = *vsapi->getVideoInfo(pairData->node);
        const auto sourceVi = pairData->vi;
        bool sourceConverted{};
        VSVideoInfo mvClipVi{};
        bool hasMVClip{};
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

        mvClip = vsapi->mapGetNode(in, "meta_clip", 0, &err);
        if (!err) {
            mvClipVi = *vsapi->getVideoInfo(mvClip);
            hasMVClip = true;
        }

        if (hasMVClip) {
            if (!vsh::isConstantVideoFormat(&mvClipVi))
                throw "meta_clip must have a constant format";

            if (mvClipVi.width != sourceVi.width || mvClipVi.height != sourceVi.height)
                throw "meta_clip dimensions must match clip";

        }

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

        const VSVideoInfo* metadataVi = hasMVClip ? &mvClipVi : (sourceConverted ? &sourceVi : nullptr);
        pairData->mvConfig = createMotionVectorConfig(pairData->vi, metadataVi, mvInternalGeometry, mvUseChroma, mvBlockSizeX, mvBlockSizeY,
                                                      mvOverlapX, mvOverlapY, mvPel, mvDelta,
                                  mvBits, mvHPadding, mvVPadding, mvBlockReduce, mvSadMultiplier);
        printMotionVectorInvocation("RIFEMV", gpuId, gpuThread, sharedFlowInFlight, flowScale, flowResizeMode,
                                    perfStats, pairData->mvConfig, resScale, inferenceWidth, inferenceHeight,
                                    matrixIn, rangeIn, true);

        if (!vsapi->getVideoFormatByID(&pairData->vi.format, pfGray8, core))
            throw "failed to create RIFEMV output format";

        if (mvClip) {
            vsapi->freeNode(mvClip);
            mvClip = nullptr;
        }

        const auto localFlowInFlight = sharedFlowInFlightSpecified ? std::max(gpuThread, sharedFlowInFlight) : gpuThread;
        pairData->semaphore = std::make_unique<std::counting_semaphore<>>(localFlowInFlight);
        pairData->sharedFlowSemaphore = acquireSharedFlowSemaphore(gpuId, sharedFlowInFlight);
        pairData->perfStats = perfStats;
        if (pairData->perfStats) {
            pairData->perf = std::make_shared<MotionVectorPerfStats>();
            pairData->perfLabel = "RIFEMV(delta=" + std::to_string(pairData->mvConfig.delta) + ")";
        }
        pairData->rife = std::make_unique<RIFE>(gpuId, flowScale, 1, resolvedModel.rifeV2, resolvedModel.rifeV4, resolvedModel.padding, flowResizeMode);
        loadRIFEModel(*pairData->rife, resolvedModel.modelPath);
    } catch (const std::exception& error) {
        vsapi->mapSetError(out, ("RIFEMV: "s + error.what()).c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);
        vsapi->freeNode(mvClip);

        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    } catch (const char* error) {
        vsapi->mapSetError(out, ("RIFEMV: "s + error).c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);
        vsapi->freeNode(mvClip);

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
    backwardData->invalidBlob = buildInvalidMotionVectorBlob(mvConfig, true, &backwardData->invalidStats);
    backwardData->backward = true;
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
    forwardData->invalidBlob = buildInvalidMotionVectorBlob(mvConfig, false, &forwardData->invalidStats);
    forwardData->backward = false;
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
    VSNode* mvClip{};
    VSNode* sourceNode{};
    VSNode* pairNode{};
    VSVideoInfo mvClipVi{};
    bool hasMVClip{};
    bool hasGPUInstance{};
    std::vector<MotionVectorConfig> outputConfigs(maxDelta + 1);
    std::vector<VSNode*> outputNodes;

    try {
        pairData->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
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

        mvClip = vsapi->mapGetNode(in, "meta_clip", 0, &err);
        if (!err) {
            mvClipVi = *vsapi->getVideoInfo(mvClip);
            hasMVClip = true;
        }

        if (hasMVClip) {
            if (!vsh::isConstantVideoFormat(&mvClipVi))
                throw "meta_clip must have a constant format";

            if (mvClipVi.width != sourceVi.width || mvClipVi.height != sourceVi.height)
                throw "meta_clip dimensions must match clip";

        }

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

        const VSVideoInfo* metadataVi = hasMVClip ? &mvClipVi : (sourceConverted ? &sourceVi : nullptr);
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
        printMotionVectorInvocation(functionName, gpuId, gpuThread, sharedFlowInFlight, flowScale, flowResizeMode,
                                    perfStats, pairData->mvConfig, resScale, inferenceWidth, inferenceHeight,
                                    matrixIn, rangeIn, false);

        if (!vsapi->getVideoFormatByID(&pairData->vi.format, pfGray8, core))
            throw "failed to create output format";

        if (mvClip) {
            vsapi->freeNode(mvClip);
            mvClip = nullptr;
        }

        sourceNode = vsapi->addNodeRef(pairData->sourceNode);
        const auto localFlowInFlight = sharedFlowInFlightSpecified ? std::max(gpuThread, sharedFlowInFlight) : gpuThread;
        pairData->semaphore = std::make_unique<std::counting_semaphore<>>(localFlowInFlight);
        pairData->sharedFlowSemaphore = acquireSharedFlowSemaphore(gpuId, sharedFlowInFlight);
        pairData->perfStats = perfStats;
        if (pairData->perfStats) {
            pairData->perf = std::make_shared<MotionVectorPerfStats>();
            pairData->perfLabel = std::string(functionName);
        }
        pairData->rife = std::make_unique<RIFE>(gpuId, flowScale, 1, resolvedModel.rifeV2, resolvedModel.rifeV4, resolvedModel.padding, flowResizeMode);
        loadRIFEModel(*pairData->rife, resolvedModel.modelPath);
    } catch (const std::exception& error) {
        vsapi->mapSetError(out, (std::string(functionName) + ": " + error.what()).c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);
        vsapi->freeNode(mvClip);
        vsapi->freeNode(sourceNode);

        if (hasGPUInstance && --numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    } catch (const char* error) {
        vsapi->mapSetError(out, (std::string(functionName) + ": " + error).c_str());
        vsapi->freeNode(pairData->node);
        vsapi->freeNode(pairData->sourceNode);
        vsapi->freeNode(mvClip);
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
        outputData->invalidBlob = buildInvalidMotionVectorBlob(mvConfig, backward, &outputData->invalidStats);
        outputData->backward = backward;
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
                             "flow_scale:float:opt;"
                             "res_scale:float:opt;"
                             "cpu_flow_resize:int:opt;"
                             "perf_stats:int:opt;"
                             "blksize_x:int:opt;"
                             "blksize_y:int:opt;"
                             "overlap_x:int:opt;"
                             "overlap_y:int:opt;"
                             "pel:int:opt;"
                             "delta:int:opt;"
                             "bits:int:opt;"
                             "sad_multiplier:float:opt;"
                             "meta_clip:vnode:opt;"
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
                             "sad_multiplier:float:opt;"
                             "meta_clip:vnode:opt;"
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
                             "sad_multiplier:float:opt;"
                             "meta_clip:vnode:opt;"
                             "matrix_in_s:data:opt;"
                             "range_in_s:data:opt;"
                             "hpad:int:opt;"
                             "vpad:int:opt;"
                             "block_reduce:int:opt;"
                             "chroma:int:opt;",
                             "clip:vnode[];",
                             rifeMVApprox3Create, nullptr, plugin);
}
