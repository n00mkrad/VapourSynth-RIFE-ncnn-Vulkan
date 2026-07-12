// Copyright (c) 2021-2022 HolyWu
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define SCALEGRID_HAS_SSE2 1
#else
#define SCALEGRID_HAS_SSE2 0
#endif

#include "MotionVectorGrid.h"
#include "ScaleGrid.h"
#include "VSHelper4.h"

using namespace std::literals;

namespace {

constexpr auto MVToolsAnalysisDataKey = "MVTools_MVAnalysisData";
constexpr auto MVToolsVectorsKey = "MVTools_vectors";
constexpr auto RIFEMVAvgSadKey = "RMV_AvgSad";
constexpr auto RIFEMVMaxSadKey = "RMV_MaxSad";
constexpr auto RIFEMVMinSadKey = "RMV_MinSad";
constexpr auto RIFEMVAvgAbsDxKey = "RMV_AvgAbsDx";
constexpr auto RIFEMVAvgAbsDyKey = "RMV_AvgAbsDy";
constexpr auto RIFEMVAvgAbsMotionKey = "RMV_AvgAbsMotion";
constexpr auto RIFEMVPanAmountKey = "RMV_PanAmount";
constexpr long double StructuralScaleTolerance = 1e-5L;

enum class FastGeometryMode {
    none,
    doubleScale,
    halfScale
};

enum class FastSadMode {
    generic,
    identity,
    quadruple,
    quarter
};

struct ComponentBounds final {
    int min;
    int max;
};

int scaleDoubleVector(const int value, const char* const name) {
    const auto scaled = static_cast<int64_t>(value) * 2;
    if (scaled < std::numeric_limits<int>::min() || scaled > std::numeric_limits<int>::max())
        throw std::runtime_error(std::string(name) + " vector scaling overflows integer storage");
    return static_cast<int>(scaled);
}

int scaleHalfVector(const int value) noexcept {
    const auto quotient = value / 2;
    const auto remainder = value % 2;
    return quotient + (remainder > 0 ? 1 : remainder < 0 ? -1 : 0);
}

int64_t scaleQuadrupleSad(const int64_t value) {
    constexpr auto minValue = std::numeric_limits<int64_t>::min();
    constexpr auto maxValue = std::numeric_limits<int64_t>::max();
    if (value > maxValue / 4 || value < minValue / 4)
        throw "SAD scaling overflows 64-bit integer storage";
    return value * 4;
}

int64_t scaleQuarterSad(const int64_t value) noexcept {
    const auto quotient = value / 4;
    const auto remainder = value % 4;
    return quotient + (remainder >= 2 ? 1 : remainder <= -2 ? -1 : 0);
}

int64_t scaleSadValue(const int64_t value, const double scale) {
    const auto scaled = static_cast<long double>(value) * static_cast<long double>(scale);
    const auto rounded = std::round(scaled);
    if (!std::isfinite(static_cast<double>(scaled)) ||
        rounded < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
        rounded > static_cast<long double>(std::numeric_limits<int64_t>::max()))
        throw "SAD scaling overflows 64-bit integer storage";
    return static_cast<int64_t>(std::llround(scaled));
}

class ScaleGrid final {
public:
    static void registerFunction(VSPlugin* plugin, const VSPLUGINAPI* vspapi);

private:
    VSNode* node{};
    VSVideoInfo sourceVi{};
    VSVideoInfo outputVi{};
    MVAnalysisData sourceAnalysisData{};
    MVAnalysisData scaledAnalysisData{};
    double scaleX{ 1.0 };
    double scaleY{ 1.0 };
    double sadScale{ 1.0 };
    bool scaleMeta{ true };
    bool scaleClip{ true };
    bool strict{ true };
    VSFrame* blankCarrier{};
    FastGeometryMode fastGeometryMode{ FastGeometryMode::none };
    FastSadMode fastSadMode{ FastSadMode::generic };
    std::vector<ComponentBounds> xBounds;
    std::vector<ComponentBounds> yBounds;

    static void VS_CC create(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi);
    static const VSFrame* VS_CC getFrame(int n, int activationReason, void* instanceData,
                                         void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi);
    static void VS_CC free(void* instanceData, VSCore* core, const VSAPI* vsapi);

    static bool readMVAnalysisData(const VSMap* props, MVAnalysisData& analysisData, const VSAPI* vsapi) noexcept;
    static bool readMotionVectorBlobRecords(const char* vectorBlob, int vectorBlobSize,
                                            const MVAnalysisData& analysisData, const char*& records,
                                            size_t& vectorCount);
    static bool readMotionVectorBlobValidity(const char* vectorBlob, int vectorBlobSize) noexcept;
    static bool hasPublicSadStats(const VSMap* props, const VSAPI* vsapi) noexcept;
    static bool hasPublicMotionStats(const VSMap* props, const VSAPI* vsapi) noexcept;
    static void deletePublicStats(VSMap* props, const VSAPI* vsapi);
    static void zeroFrame(VSFrame* frame, const VSAPI* vsapi);
    static void validateInputAnalysisData(const MVAnalysisData& analysisData);
    static int scaleStructuralValue(int value, double scale, const char* name, bool allowZero);
    static MVAnalysisData createScaledAnalysisData(const MVAnalysisData& source, double scaleX, double scaleY);
    static void validateScaledGrid(const MVAnalysisData& analysisData, bool strict);
    static int scaleVectorComponent(int value, double scale, const char* name);
    static int computeBlockCoordinate(int blockIndex, int step, int padding, const char* name);
    static ComponentBounds createComponentBounds(int blockIndex, int step, int padding,
                                                 int blockSize, int size, int pel, const char* name);
    void configureFastPath();
    void transformFast(const char* sourceRecords, size_t vectorCount, bool valid,
                       std::vector<char>& scaledBlob) const;
};

bool ScaleGrid::readMVAnalysisData(const VSMap* props, MVAnalysisData& analysisData, const VSAPI* vsapi) noexcept {
    int err{};
    const auto* data = vsapi->mapGetData(props, MVToolsAnalysisDataKey, 0, &err);
    if (err || vsapi->mapGetDataSize(props, MVToolsAnalysisDataKey, 0, nullptr) != sizeof(MVAnalysisData))
        return false;
    std::memcpy(&analysisData, data, sizeof(analysisData));
    return true;
}

bool ScaleGrid::readMotionVectorBlobRecords(const char* vectorBlob, const int vectorBlobSize,
                                            const MVAnalysisData& analysisData, const char*& records,
                                            size_t& vectorCount) {
    if (!vectorBlob || vectorBlobSize < static_cast<int>(sizeof(MVArraySizeType) * 3) ||
        analysisData.nBlkX <= 0 || analysisData.nBlkY <= 0)
        return false;
    vectorCount = static_cast<size_t>(analysisData.nBlkX) * analysisData.nBlkY;
    const auto expectedPlaneSize64 = static_cast<uint64_t>(sizeof(MVArraySizeType)) + static_cast<uint64_t>(vectorCount) * sizeof(MVToolsVector);
    const auto expectedGroupSize64 = static_cast<uint64_t>(sizeof(MVArraySizeType) * 2) + expectedPlaneSize64;
    if (expectedGroupSize64 > static_cast<uint64_t>(std::numeric_limits<MVArraySizeType>::max()))
        return false;
    const auto expectedPlaneSize = static_cast<MVArraySizeType>(expectedPlaneSize64);
    const auto expectedGroupSize = static_cast<MVArraySizeType>(expectedGroupSize64);
    MVArraySizeType groupSize{};
    MVArraySizeType planeSize{};
    std::memcpy(&groupSize, vectorBlob, sizeof(groupSize));
    std::memcpy(&planeSize, vectorBlob + sizeof(MVArraySizeType) * 2, sizeof(planeSize));
    if (groupSize < expectedGroupSize || groupSize > vectorBlobSize ||
        planeSize < expectedPlaneSize || planeSize > groupSize - static_cast<MVArraySizeType>(sizeof(MVArraySizeType) * 2) ||
        vectorBlobSize < expectedGroupSize)
        return false;
    records = vectorBlob + sizeof(MVArraySizeType) * 3;
    return true;
}

bool ScaleGrid::readMotionVectorBlobValidity(const char* vectorBlob, const int vectorBlobSize) noexcept {
    if (!vectorBlob || vectorBlobSize < static_cast<int>(sizeof(MVArraySizeType) * 2))
        return false;
    MVArraySizeType valid{};
    std::memcpy(&valid, vectorBlob + sizeof(MVArraySizeType), sizeof(valid));
    return valid != 0;
}

bool ScaleGrid::hasPublicSadStats(const VSMap* props, const VSAPI* vsapi) noexcept {
    return vsapi->mapGetType(props, RIFEMVAvgSadKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVMaxSadKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVMinSadKey) != ptUnset;
}

bool ScaleGrid::hasPublicMotionStats(const VSMap* props, const VSAPI* vsapi) noexcept {
    return vsapi->mapGetType(props, RIFEMVAvgAbsDxKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVAvgAbsDyKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVAvgAbsMotionKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVPanAmountKey) != ptUnset;
}

void ScaleGrid::deletePublicStats(VSMap* props, const VSAPI* vsapi) {
    vsapi->mapDeleteKey(props, RIFEMVAvgSadKey);
    vsapi->mapDeleteKey(props, RIFEMVMaxSadKey);
    vsapi->mapDeleteKey(props, RIFEMVMinSadKey);
    vsapi->mapDeleteKey(props, RIFEMVAvgAbsDxKey);
    vsapi->mapDeleteKey(props, RIFEMVAvgAbsDyKey);
    vsapi->mapDeleteKey(props, RIFEMVAvgAbsMotionKey);
    vsapi->mapDeleteKey(props, RIFEMVPanAmountKey);
}

void ScaleGrid::zeroFrame(VSFrame* frame, const VSAPI* vsapi) {
    const auto* format = vsapi->getVideoFrameFormat(frame);
    for (auto plane = 0; plane < format->numPlanes; plane++) {
        auto* dstp = vsapi->getWritePtr(frame, plane);
        const auto stride = vsapi->getStride(frame, plane);
        const auto height = vsapi->getFrameHeight(frame, plane);
        for (auto y = 0; y < height; y++)
            std::memset(dstp + static_cast<size_t>(y) * stride, 0, stride);
    }
}

void ScaleGrid::validateInputAnalysisData(const MVAnalysisData& analysisData) {
    if (analysisData.nLvCount != 1)
        throw "only single-level MVTools vector clips are supported";
    if (analysisData.nBlkSizeX <= 0 || analysisData.nBlkSizeY <= 0)
        throw "invalid MVTools block size metadata";
    if (analysisData.nOverlapX < 0 || analysisData.nOverlapX >= analysisData.nBlkSizeX ||
        analysisData.nOverlapY < 0 || analysisData.nOverlapY >= analysisData.nBlkSizeY)
        throw "invalid MVTools overlap metadata";
    if (analysisData.nBlkX <= 0 || analysisData.nBlkY <= 0 || analysisData.nWidth <= 0 || analysisData.nHeight <= 0)
        throw "invalid MVTools vector grid metadata";
    if (analysisData.nPel <= 0)
        throw "invalid MVTools pel metadata";
}

int ScaleGrid::scaleStructuralValue(const int value, const double scale, const char* const name, const bool allowZero) {
    // Structural fields may use approximate fractional scales only when they still resolve to integral geometry.
    const auto scaled = static_cast<long double>(value) * static_cast<long double>(scale);
    if (!std::isfinite(static_cast<double>(scaled)) || scaled > static_cast<long double>(std::numeric_limits<int>::max()))
        throw std::runtime_error(std::string(name) + " scaling overflows integer storage");
    const auto rounded = std::round(scaled);
    const auto tolerance = StructuralScaleTolerance * std::max(1.0L, std::abs(scaled));
    if (std::abs(scaled - rounded) > tolerance)
        throw std::runtime_error(std::string(name) + " does not scale to an integer");
    const auto result = static_cast<int>(rounded);
    if (result < 0 || (!allowZero && result == 0))
        throw std::runtime_error(std::string(name) + " must remain " + (allowZero ? "non-negative" : "positive") + " after scaling");
    return result;
}

MVAnalysisData ScaleGrid::createScaledAnalysisData(const MVAnalysisData& source, const double scaleX, const double scaleY) {
    // Keep vector cardinality and non-spatial MVTools semantics while moving the grid into target coordinates.
    auto scaled = source;
    scaled.nBlkSizeX = scaleStructuralValue(source.nBlkSizeX, scaleX, "block width", false);
    scaled.nBlkSizeY = scaleStructuralValue(source.nBlkSizeY, scaleY, "block height", false);
    scaled.nOverlapX = scaleStructuralValue(source.nOverlapX, scaleX, "horizontal overlap", true);
    scaled.nOverlapY = scaleStructuralValue(source.nOverlapY, scaleY, "vertical overlap", true);
    scaled.nWidth = scaleStructuralValue(source.nWidth, scaleX, "source width", false);
    scaled.nHeight = scaleStructuralValue(source.nHeight, scaleY, "source height", false);
    scaled.nHPadding = scaleStructuralValue(source.nHPadding, scaleX, "horizontal padding", true);
    scaled.nVPadding = scaleStructuralValue(source.nVPadding, scaleY, "vertical padding", true);
    return scaled;
}

void ScaleGrid::validateScaledGrid(const MVAnalysisData& analysisData, const bool strict) {
    const auto stepX = analysisData.nBlkSizeX - analysisData.nOverlapX;
    const auto stepY = analysisData.nBlkSizeY - analysisData.nOverlapY;
    if (stepX <= 0 || stepY <= 0)
        throw "scaled overlap must be smaller than the corresponding block size";
    if (!strict) {
        if (analysisData.nBlkSizeX < 2 || analysisData.nBlkSizeY < 2)
            throw "scaled block size must be at least 2x2 when strict=False";
        return;
    }
    static constexpr std::array<std::pair<int, int>, 12> validBlockSizes{{
        { 4, 4 }, { 8, 4 }, { 8, 8 }, { 16, 2 }, { 16, 8 }, { 16, 16 },
        { 32, 16 }, { 32, 32 }, { 64, 32 }, { 64, 64 }, { 128, 64 }, { 128, 128 }
    }};
    const auto blockSize = std::pair{ analysisData.nBlkSizeX, analysisData.nBlkSizeY };
    if (std::find(validBlockSizes.begin(), validBlockSizes.end(), blockSize) == validBlockSizes.end())
        throw "scaled block size is not supported by MVTools";
    if (analysisData.nOverlapX > analysisData.nBlkSizeX / 2 || analysisData.nOverlapY > analysisData.nBlkSizeY / 2)
        throw "scaled overlap exceeds half of the corresponding block size";
}

int ScaleGrid::scaleVectorComponent(const int value, const double scale, const char* const name) {
    const auto scaled = static_cast<long double>(value) * static_cast<long double>(scale);
    const auto rounded = std::round(scaled);
    if (!std::isfinite(static_cast<double>(scaled)) ||
        rounded < static_cast<long double>(std::numeric_limits<int>::min()) ||
        rounded > static_cast<long double>(std::numeric_limits<int>::max()))
        throw std::runtime_error(std::string(name) + " vector scaling overflows integer storage");
    return static_cast<int>(std::lround(scaled));
}

int ScaleGrid::computeBlockCoordinate(const int blockIndex, const int step, const int padding, const char* const name) {
    const auto coordinate = static_cast<int64_t>(blockIndex) * step - padding;
    if (coordinate < std::numeric_limits<int>::min() || coordinate > std::numeric_limits<int>::max())
        throw std::runtime_error(std::string(name) + " block coordinate overflows integer storage");
    return static_cast<int>(coordinate);
}

ComponentBounds ScaleGrid::createComponentBounds(const int blockIndex, const int step, const int padding,
                                                  const int blockSize, const int size, const int pel,
                                                  const char* const name) {
    const auto blockCoordinate = computeBlockCoordinate(blockIndex, step, padding, name);
    const auto minPixelDelta = -static_cast<int64_t>(padding) - blockCoordinate;
    const auto maxPixelDelta = static_cast<int64_t>(size) - blockSize + padding - blockCoordinate;
    const auto pelValue = static_cast<int64_t>(pel);
    const auto minVector = minPixelDelta * pelValue;
    const auto maxVector = maxPixelDelta * pelValue;
    if (minVector < std::numeric_limits<int>::min() || minVector > std::numeric_limits<int>::max() ||
        maxVector < std::numeric_limits<int>::min() || maxVector > std::numeric_limits<int>::max())
        throw std::runtime_error(std::string(name) + " vector clamp bounds overflow integer storage");
    return { static_cast<int>(minVector), static_cast<int>(maxVector) };
}

void ScaleGrid::configureFastPath() {
    if (scaleX == 2.0 && scaleY == 2.0)
        fastGeometryMode = FastGeometryMode::doubleScale;
    else if (scaleX == 0.5 && scaleY == 0.5)
        fastGeometryMode = FastGeometryMode::halfScale;
    else
        fastGeometryMode = FastGeometryMode::none;

    if (sadScale == 1.0)
        fastSadMode = FastSadMode::identity;
    else if (sadScale == 4.0)
        fastSadMode = FastSadMode::quadruple;
    else if (sadScale == 0.25)
        fastSadMode = FastSadMode::quarter;
    else
        fastSadMode = FastSadMode::generic;

    if (fastGeometryMode == FastGeometryMode::none)
        return;

    const auto stepX = scaledAnalysisData.nBlkSizeX - scaledAnalysisData.nOverlapX;
    const auto stepY = scaledAnalysisData.nBlkSizeY - scaledAnalysisData.nOverlapY;
    xBounds.resize(static_cast<size_t>(scaledAnalysisData.nBlkX));
    yBounds.resize(static_cast<size_t>(scaledAnalysisData.nBlkY));
    for (auto x = 0; x < scaledAnalysisData.nBlkX; x++)
        xBounds[x] = createComponentBounds(x, stepX, scaledAnalysisData.nHPadding,
                                           scaledAnalysisData.nBlkSizeX, scaledAnalysisData.nWidth,
                                           scaledAnalysisData.nPel, "horizontal");
    for (auto y = 0; y < scaledAnalysisData.nBlkY; y++)
        yBounds[y] = createComponentBounds(y, stepY, scaledAnalysisData.nVPadding,
                                           scaledAnalysisData.nBlkSizeY, scaledAnalysisData.nHeight,
                                           scaledAnalysisData.nPel, "vertical");
}

template<FastGeometryMode Geometry, FastSadMode Sad>
void transformFastRecords(const char* const sourceRecords, char* const outputRecords,
                          const MVAnalysisData& analysisData,
                          const std::vector<ComponentBounds>& xBounds,
                          const std::vector<ComponentBounds>& yBounds,
                          const double sadScale) {
    const auto blockCountX = static_cast<size_t>(analysisData.nBlkX);
    for (auto y = 0; y < analysisData.nBlkY; y++) {
        for (auto x = 0; x < analysisData.nBlkX; x++) {
            const auto index = static_cast<size_t>(y) * blockCountX + x;
            MVToolsVector vector{};
            std::memcpy(&vector, sourceRecords + index * sizeof(vector), sizeof(vector));
            if constexpr (Geometry == FastGeometryMode::doubleScale) {
                vector.x = scaleDoubleVector(vector.x, "horizontal");
                vector.y = scaleDoubleVector(vector.y, "vertical");
            } else {
                vector.x = scaleHalfVector(vector.x);
                vector.y = scaleHalfVector(vector.y);
            }
            if constexpr (Sad == FastSadMode::quadruple)
                vector.sad = scaleQuadrupleSad(vector.sad);
            else if constexpr (Sad == FastSadMode::quarter)
                vector.sad = scaleQuarterSad(vector.sad);
            else if constexpr (Sad == FastSadMode::generic)
                vector.sad = scaleSadValue(vector.sad, sadScale);
            vector.x = std::clamp(vector.x, xBounds[x].min, xBounds[x].max);
            vector.y = std::clamp(vector.y, yBounds[y].min, yBounds[y].max);
            std::memcpy(outputRecords + index * sizeof(vector), &vector, sizeof(vector));
        }
    }
}

#if SCALEGRID_HAS_SSE2
inline __m128i clampFastComponents(const __m128i value, const ComponentBounds xBounds,
                                   const ComponentBounds yBounds) noexcept {
    const auto minValues = _mm_set_epi32(0, 0, yBounds.min, xBounds.min);
    const auto maxValues = _mm_set_epi32(0, 0, yBounds.max, xBounds.max);
    const auto belowMin = _mm_cmpgt_epi32(minValues, value);
    const auto clampedMin = _mm_or_si128(_mm_and_si128(belowMin, minValues), _mm_andnot_si128(belowMin, value));
    const auto aboveMax = _mm_cmpgt_epi32(clampedMin, maxValues);
    return _mm_or_si128(_mm_and_si128(aboveMax, maxValues), _mm_andnot_si128(aboveMax, clampedMin));
}

template<FastGeometryMode Geometry, FastSadMode Sad>
void transformFastRecordsSSE2(const char* const sourceRecords, char* const outputRecords,
                              const MVAnalysisData& analysisData,
                              const std::vector<ComponentBounds>& xBounds,
                              const std::vector<ComponentBounds>& yBounds) {
    const auto blockCountX = static_cast<size_t>(analysisData.nBlkX);
    const auto recordSize = sizeof(MVToolsVector);
    const auto zero = _mm_setzero_si128();
    const auto one = _mm_set1_epi32(1);
    const auto maxInput = _mm_set1_epi32(std::numeric_limits<int>::max() / 2);
    const auto minInput = _mm_set1_epi32(std::numeric_limits<int>::min() / 2);
    for (auto y = 0; y < analysisData.nBlkY; y++) {
        const auto yBound = yBounds[y];
        const auto* sourceRow = sourceRecords + static_cast<size_t>(y) * blockCountX * recordSize;
        auto* outputRow = outputRecords + static_cast<size_t>(y) * blockCountX * recordSize;
        for (auto x = 0; x < analysisData.nBlkX; x++) {
            const auto* sourceRecord = sourceRow + static_cast<size_t>(x) * recordSize;
            auto* outputRecord = outputRow + static_cast<size_t>(x) * recordSize;
            auto components = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(sourceRecord));
            if constexpr (Geometry == FastGeometryMode::doubleScale) {
                const auto overflow = _mm_or_si128(_mm_cmpgt_epi32(components, maxInput), _mm_cmpgt_epi32(minInput, components));
                if (_mm_movemask_epi8(overflow))
                    throw std::runtime_error("vector scaling overflows integer storage");
                components = _mm_slli_epi32(components, 1);
            } else {
                const auto oddPositive = _mm_and_si128(_mm_cmpgt_epi32(components, zero),
                                                       _mm_cmpeq_epi32(_mm_and_si128(components, one), one));
                components = _mm_add_epi32(_mm_srai_epi32(components, 1), oddPositive);
            }
            components = clampFastComponents(components, xBounds[x], yBound);
            int64_t sad{};
            std::memcpy(&sad, sourceRecord + offsetof(MVToolsVector, sad), sizeof(sad));
            if constexpr (Sad == FastSadMode::quadruple)
                sad = scaleQuadrupleSad(sad);
            else if constexpr (Sad == FastSadMode::quarter)
                sad = scaleQuarterSad(sad);
            const auto output = _mm_unpacklo_epi64(components, _mm_cvtsi64_si128(sad));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(outputRecord), output);
        }
    }
}
#endif

void ScaleGrid::transformFast(const char* const sourceRecords, const size_t vectorCount, const bool valid,
                              std::vector<char>& scaledBlob) const {
    auto* outputRecords = motionVectorGridPrepareBlob(vectorCount, valid, scaledBlob);
#if SCALEGRID_HAS_SSE2
    if (fastSadMode != FastSadMode::generic) {
        if (fastGeometryMode == FastGeometryMode::doubleScale) {
            switch (fastSadMode) {
            case FastSadMode::identity:
                transformFastRecordsSSE2<FastGeometryMode::doubleScale, FastSadMode::identity>(
                    sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds);
                break;
            case FastSadMode::quadruple:
                transformFastRecordsSSE2<FastGeometryMode::doubleScale, FastSadMode::quadruple>(
                    sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds);
                break;
            case FastSadMode::quarter:
                transformFastRecordsSSE2<FastGeometryMode::doubleScale, FastSadMode::quarter>(
                    sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds);
                break;
            case FastSadMode::generic:
                break;
            }
        } else {
            switch (fastSadMode) {
            case FastSadMode::identity:
                transformFastRecordsSSE2<FastGeometryMode::halfScale, FastSadMode::identity>(
                    sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds);
                break;
            case FastSadMode::quadruple:
                transformFastRecordsSSE2<FastGeometryMode::halfScale, FastSadMode::quadruple>(
                    sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds);
                break;
            case FastSadMode::quarter:
                transformFastRecordsSSE2<FastGeometryMode::halfScale, FastSadMode::quarter>(
                    sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds);
                break;
            case FastSadMode::generic:
                break;
            }
        }
        return;
    }
#endif
    if (fastGeometryMode == FastGeometryMode::doubleScale) {
        switch (fastSadMode) {
        case FastSadMode::identity:
            transformFastRecords<FastGeometryMode::doubleScale, FastSadMode::identity>(
                sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds, sadScale);
            break;
        case FastSadMode::quadruple:
            transformFastRecords<FastGeometryMode::doubleScale, FastSadMode::quadruple>(
                sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds, sadScale);
            break;
        case FastSadMode::quarter:
            transformFastRecords<FastGeometryMode::doubleScale, FastSadMode::quarter>(
                sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds, sadScale);
            break;
        case FastSadMode::generic:
            transformFastRecords<FastGeometryMode::doubleScale, FastSadMode::generic>(
                sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds, sadScale);
            break;
        }
    } else {
        switch (fastSadMode) {
        case FastSadMode::identity:
            transformFastRecords<FastGeometryMode::halfScale, FastSadMode::identity>(
                sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds, sadScale);
            break;
        case FastSadMode::quadruple:
            transformFastRecords<FastGeometryMode::halfScale, FastSadMode::quadruple>(
                sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds, sadScale);
            break;
        case FastSadMode::quarter:
            transformFastRecords<FastGeometryMode::halfScale, FastSadMode::quarter>(
                sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds, sadScale);
            break;
        case FastSadMode::generic:
            transformFastRecords<FastGeometryMode::halfScale, FastSadMode::generic>(
                sourceRecords, outputRecords, scaledAnalysisData, xBounds, yBounds, sadScale);
            break;
        }
    }
}

const VSFrame* VS_CC ScaleGrid::getFrame(int n, int activationReason, void* instanceData,
                                         [[maybe_unused]] void** frameData,
                                         VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const ScaleGrid*>(instanceData) };
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto src = vsapi->getFrameFilter(n, d->node, frameCtx);
        try {
            const auto srcProps = vsapi->getFramePropertiesRO(src);
            MVAnalysisData analysisData{};
            if (!readMVAnalysisData(srcProps, analysisData, vsapi) ||
                std::memcmp(&analysisData, &d->sourceAnalysisData, sizeof(analysisData)) != 0)
                throw "per-frame MVTools metadata does not match the first frame";
            int err{};
            const auto* vectorBlob = vsapi->mapGetData(srcProps, MVToolsVectorsKey, 0, &err);
            const auto vectorBlobSize = err ? 0 : vsapi->mapGetDataSize(srcProps, MVToolsVectorsKey, 0, nullptr);
            const char* sourceRecords{};
            size_t vectorCount{};
            if (err || !readMotionVectorBlobRecords(vectorBlob, vectorBlobSize, analysisData, sourceRecords, vectorCount))
                throw "failed to read MVTools vector blob";
            const auto valid = readMotionVectorBlobValidity(vectorBlob, vectorBlobSize);
            const auto includeSadStats = hasPublicSadStats(srcProps, vsapi);
            const auto includeMotionStats = hasPublicMotionStats(srcProps, vsapi);
            static thread_local std::vector<char> scaledBlob;
            scaledBlob.clear();
            MotionVectorFrameStats stats{};
            if (d->fastGeometryMode != FastGeometryMode::none && !includeSadStats && !includeMotionStats) {
                d->transformFast(sourceRecords, vectorCount, valid, scaledBlob);
            } else {
                std::vector<MVToolsVector> vectors;
                vectors.resize(vectorCount);
                std::memcpy(vectors.data(), sourceRecords, vectorCount * sizeof(MVToolsVector));
                const auto stepX = d->scaledAnalysisData.nBlkSizeX - d->scaledAnalysisData.nOverlapX;
                const auto stepY = d->scaledAnalysisData.nBlkSizeY - d->scaledAnalysisData.nOverlapY;
                for (size_t i = 0; i < vectors.size(); i++) {
                    auto& vector = vectors[i];
                    const auto blockXIndex = static_cast<int>(i % static_cast<size_t>(d->scaledAnalysisData.nBlkX));
                    const auto blockYIndex = static_cast<int>(i / static_cast<size_t>(d->scaledAnalysisData.nBlkX));
                    const auto blockX = computeBlockCoordinate(blockXIndex, stepX, d->scaledAnalysisData.nHPadding, "horizontal");
                    const auto blockY = computeBlockCoordinate(blockYIndex, stepY, d->scaledAnalysisData.nVPadding, "vertical");
                    vector.x = scaleVectorComponent(vector.x, d->scaleX, "horizontal");
                    vector.y = scaleVectorComponent(vector.y, d->scaleY, "vertical");
                    vector.sad = scaleSadValue(vector.sad, d->sadScale);
                    vector.x = motionVectorGridClampComponent(vector.x, d->scaledAnalysisData.nPel, blockX,
                                                              d->scaledAnalysisData.nBlkSizeX, d->scaledAnalysisData.nWidth,
                                                              d->scaledAnalysisData.nHPadding);
                    vector.y = motionVectorGridClampComponent(vector.y, d->scaledAnalysisData.nPel, blockY,
                                                              d->scaledAnalysisData.nBlkSizeY, d->scaledAnalysisData.nHeight,
                                                              d->scaledAnalysisData.nVPadding);
                }
                // Derived statistics always describe the conceptual scaled grid, even when metadata output is disabled.
                motionVectorGridPackBlob(vectors, valid, d->scaledAnalysisData, scaledBlob, &stats,
                                         includeSadStats, includeMotionStats);
            }
            VSFrame* dst{};
            if (d->scaleClip) {
                std::array<const VSFrame*, 4> planeSrc{};
                std::array<int, 4> planes{};
                for (auto plane = 0; plane < d->outputVi.format.numPlanes; plane++) {
                    planeSrc[plane] = d->blankCarrier;
                    planes[plane] = plane;
                }
                dst = vsapi->newVideoFrame2(&d->outputVi.format, d->outputVi.width, d->outputVi.height,
                                            planeSrc.data(), planes.data(), src, core);
            } else {
                dst = vsapi->copyFrame(src, core);
            }
            if (!dst)
                throw "failed to create scaled carrier frame";
            auto dstProps = vsapi->getFramePropertiesRW(dst);
            deletePublicStats(dstProps, vsapi);
            const auto& outputAnalysisData = d->scaleMeta ? d->scaledAnalysisData : d->sourceAnalysisData;
            motionVectorGridSetProperties(dstProps, outputAnalysisData, scaledBlob.data(), static_cast<int>(scaledBlob.size()),
                                          stats, includeSadStats, includeMotionStats, vsapi);
            vsapi->freeFrame(src);
            return dst;
        } catch (const std::exception& error) {
            vsapi->freeFrame(src);
            vsapi->setFilterError(("ScaleGrid: "s + error.what()).c_str(), frameCtx);
            return nullptr;
        } catch (const char* error) {
            vsapi->freeFrame(src);
            vsapi->setFilterError(("ScaleGrid: "s + error).c_str(), frameCtx);
            return nullptr;
        }
    }
    return nullptr;
}

void VS_CC ScaleGrid::free(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<ScaleGrid*>(instanceData) };
    vsapi->freeNode(d->node);
    if (d->blankCarrier)
        vsapi->freeFrame(d->blankCarrier);
    delete d;
}

void VS_CC ScaleGrid::create(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto data{ std::make_unique<ScaleGrid>() };
    const VSFrame* firstFrame{};
    try {
        data->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        const auto hasScaleX = vsapi->mapNumElements(in, "scale_x") > 0;
        const auto hasScaleY = vsapi->mapNumElements(in, "scale_y") > 0;
        const auto hasSadScale = vsapi->mapNumElements(in, "sad_scale") > 0;
        if (!hasScaleX && !hasScaleY && !hasSadScale)
            throw "at least one of scale_x, scale_y, or sad_scale must be supplied";
        int err{};
        if (hasScaleX) {
            data->scaleX = vsapi->mapGetFloat(in, "scale_x", 0, &err);
            if (err)
                throw "scale_x must be a float";
        }
        if (hasScaleY) {
            data->scaleY = vsapi->mapGetFloat(in, "scale_y", 0, &err);
            if (err)
                throw "scale_y must be a float";
        }
        if (hasScaleX && !hasScaleY)
            data->scaleY = data->scaleX;
        else if (!hasScaleX && hasScaleY)
            data->scaleX = data->scaleY;
        if (!std::isfinite(data->scaleX) || data->scaleX <= 0.0)
            throw "scale_x must be finite and greater than 0";
        if (!std::isfinite(data->scaleY) || data->scaleY <= 0.0)
            throw "scale_y must be finite and greater than 0";
        data->sadScale = hasSadScale ? vsapi->mapGetFloat(in, "sad_scale", 0, &err) : data->scaleX * data->scaleY;
        if (hasSadScale && err)
            throw "sad_scale must be a float";
        if (!std::isfinite(data->sadScale) || data->sadScale <= 0.0)
            throw "sad_scale must be finite and greater than 0";
        data->scaleMeta = !!vsapi->mapGetInt(in, "scale_meta", 0, &err);
        if (err)
            data->scaleMeta = true;
        data->scaleClip = !!vsapi->mapGetInt(in, "scale_clip", 0, &err);
        if (err)
            data->scaleClip = true;
        data->strict = !!vsapi->mapGetInt(in, "strict", 0, &err);
        if (err)
            data->strict = true;
        if (data->scaleX == 1.0 && data->scaleY == 1.0 && data->sadScale == 1.0) {
            vsapi->mapSetNode(out, "clip", data->node, maReplace);
            vsapi->freeNode(data->node);
            data->node = nullptr;
            return;
        }
        data->sourceVi = *vsapi->getVideoInfo(data->node);
        data->outputVi = data->sourceVi;
        if (!vsh::isConstantVideoFormat(&data->sourceVi))
            throw "clip must have constant dimensions and format";
        std::array<char, 1024> errorMsg{};
        firstFrame = vsapi->getFrame(0, data->node, errorMsg.data(), static_cast<int>(errorMsg.size()));
        if (!firstFrame)
            throw std::runtime_error(std::string("failed to retrieve first clip frame: ") + errorMsg.data());
        const auto firstProps = vsapi->getFramePropertiesRO(firstFrame);
        if (!readMVAnalysisData(firstProps, data->sourceAnalysisData, vsapi))
            throw "clip is missing MVTools vector metadata";
        validateInputAnalysisData(data->sourceAnalysisData);
        if (data->sourceVi.width != data->sourceAnalysisData.nWidth || data->sourceVi.height != data->sourceAnalysisData.nHeight)
            throw "vector carrier dimensions do not match MVTools vector metadata";
        int blobErr{};
        const auto* firstBlob = vsapi->mapGetData(firstProps, MVToolsVectorsKey, 0, &blobErr);
        const auto firstBlobSize = blobErr ? 0 : vsapi->mapGetDataSize(firstProps, MVToolsVectorsKey, 0, nullptr);
        const char* firstRecords{};
        size_t firstVectorCount{};
        if (blobErr || !readMotionVectorBlobRecords(firstBlob, firstBlobSize, data->sourceAnalysisData, firstRecords, firstVectorCount))
            throw "failed to read first MVTools vector blob";
        data->scaledAnalysisData = createScaledAnalysisData(data->sourceAnalysisData, data->scaleX, data->scaleY);
        validateScaledGrid(data->scaledAnalysisData, data->strict);
        data->configureFastPath();
        if (data->scaleClip) {
            data->outputVi.width = data->scaledAnalysisData.nWidth;
            data->outputVi.height = data->scaledAnalysisData.nHeight;
            const auto xRatio = 1 << data->outputVi.format.subSamplingW;
            const auto yRatio = 1 << data->outputVi.format.subSamplingH;
            if (data->outputVi.format.numPlanes > 1 && (data->outputVi.width % xRatio || data->outputVi.height % yRatio))
                throw "scaled carrier dimensions are incompatible with the clip's chroma subsampling";
            data->blankCarrier = vsapi->newVideoFrame(&data->outputVi.format, data->outputVi.width,
                                                      data->outputVi.height, nullptr, core);
            if (!data->blankCarrier)
                throw "failed to create scaled carrier template";
            ScaleGrid::zeroFrame(data->blankCarrier, vsapi);
        }
        vsapi->freeFrame(firstFrame);
        firstFrame = nullptr;
        VSFilterDependency deps[]{ { data->node, rpStrictSpatial } };
        vsapi->createVideoFilter(out, "ScaleGrid", &data->outputVi, ScaleGrid::getFrame, ScaleGrid::free,
                                 fmParallel, deps, 1, data.get(), core);
        data.release();
    } catch (const std::exception& error) {
        vsapi->mapSetError(out, ("ScaleGrid: "s + error.what()).c_str());
        vsapi->freeFrame(firstFrame);
        if (data->blankCarrier)
            vsapi->freeFrame(data->blankCarrier);
        vsapi->freeNode(data->node);
    } catch (const char* error) {
        vsapi->mapSetError(out, ("ScaleGrid: "s + error).c_str());
        vsapi->freeFrame(firstFrame);
        if (data->blankCarrier)
            vsapi->freeFrame(data->blankCarrier);
        vsapi->freeNode(data->node);
    }
}

void ScaleGrid::registerFunction(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->registerFunction("ScaleGrid",
                             "clip:vnode;"
                             "scale_x:float:opt;"
                             "scale_y:float:opt;"
                             "sad_scale:float:opt;"
                             "scale_meta:int:opt;"
                             "scale_clip:int:opt;"
                             "strict:int:opt;",
                             "clip:vnode;",
                             ScaleGrid::create, nullptr, plugin);
}

}

void registerScaleGrid(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    ScaleGrid::registerFunction(plugin, vspapi);
}
