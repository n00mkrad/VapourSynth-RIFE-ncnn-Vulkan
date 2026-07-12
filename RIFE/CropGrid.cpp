// Copyright (c) 2021-2022 HolyWu
// SPDX-License-Identifier: MIT

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "CropGrid.h"
#include "MotionVectorGrid.h"
#include "VSHelper4.h"

using namespace std::literals;

namespace {

constexpr auto MVToolsAnalysisDataKey = "MVTools_MVAnalysisData";
constexpr auto MVToolsVectorsKey = "MVTools_vectors";
constexpr auto MVToolsSuperHeightKey = "Super_height";
constexpr auto MVToolsSuperHPadKey = "Super_hpad";
constexpr auto MVToolsSuperVPadKey = "Super_vpad";
constexpr auto MVToolsSuperPelKey = "Super_pel";
constexpr auto MVToolsSuperModeYUVKey = "Super_modeyuv";
constexpr auto MVToolsSuperLevelsKey = "Super_levels";
constexpr auto RIFEMVAvgSadKey = "RMV_AvgSad";
constexpr auto RIFEMVMaxSadKey = "RMV_MaxSad";
constexpr auto RIFEMVMinSadKey = "RMV_MinSad";
constexpr auto RIFEMVAvgAbsDxKey = "RMV_AvgAbsDx";
constexpr auto RIFEMVAvgAbsDyKey = "RMV_AvgAbsDy";
constexpr auto RIFEMVAvgAbsMotionKey = "RMV_AvgAbsMotion";
constexpr auto RIFEMVPanAmountKey = "RMV_PanAmount";

class CropGrid final {
public:
    static void registerFunction(VSPlugin* plugin, const VSPLUGINAPI* vspapi);

private:
    enum class Mode : uint8_t {
        Vector,
        Super,
    };

    struct SuperData final {
        int height;
        int hpad;
        int vpad;
        int pel;
        int modeYUV;
        int levels;
    };

    VSNode* node{};
    VSVideoInfo vi{};
    Mode mode{ Mode::Vector };
    SuperData super{};
    int left{};
    int right{};
    int top{};
    int bottom{};
    int cropLeftPx{};
    int cropRightPx{};
    int cropTopPx{};
    int cropBottomPx{};
    int sourceWidth{};
    int sourceHeight{};
    int outputSourceWidth{};
    int outputSourceHeight{};
    int stepX{};
    int stepY{};

    static void VS_CC create(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi);
    static const VSFrame* VS_CC getFrame(int n, int activationReason, void* instanceData,
                                         void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi);
    static void VS_CC free(void* instanceData, VSCore* core, const VSAPI* vsapi);

    static bool readMVAnalysisData(const VSMap* props, MVAnalysisData& analysisData, const VSAPI* vsapi) noexcept;
    static bool readMVToolsSuperData(const VSMap* props, SuperData& super, const VSAPI* vsapi) noexcept;
    static int readMotionVectorBlobValidity(const char* vectorBlob, int vectorBlobSize) noexcept;
    static bool hasPublicSadStats(const VSMap* props, const VSAPI* vsapi) noexcept;
    static bool hasPublicMotionStats(const VSMap* props, const VSAPI* vsapi) noexcept;
    static void deleteMotionVectorPublicFrameStats(VSMap* props, const VSAPI* vsapi);
    static bool unpackMotionVectorBlob(const char* vectorBlob, int vectorBlobSize,
                                       const MVAnalysisData& analysisData, std::vector<MVToolsVector>& vectors);

    static int cropGridPlaneHeightLuma(int srcHeight, int level, int yRatioUV, int vpad) noexcept;
    static int cropGridPlaneWidthLuma(int srcWidth, int level, int xRatioUV, int hpad) noexcept;
    static ptrdiff_t cropGridPlaneSuperOffset(bool chroma, int srcHeight, int level, int pel,
                                              int vpad, ptrdiff_t planePitch, int yRatioUV) noexcept;
    static int cropGridSuperHeight(int srcHeight, int levels, int pel, int vpad,
                                   int superWidth, int yRatioUV) noexcept;
    static int cropGridSuperWidth(int srcWidth, int hpad, int xRatioUV) noexcept;
    static void validateCropSubsamplingAlignment(const VSVideoInfo& vi, int cropLeftPx, int cropRightPx,
                                                 int cropTopPx, int cropBottomPx);
    static void copyCroppedVideoFramePixels(const VSFrame* src, VSFrame* dst, int cropLeftPx, int cropTopPx,
                                            const VSAPI* vsapi);
    static void zeroVideoFrame(VSFrame* frame, const VSAPI* vsapi);
    static void padCroppedSuperPlane(uint8_t* plane, ptrdiff_t stride, int width, int height,
                                     int hpad, int vpad, int bytesPerSample);
    static bool cropMotionVectorGrid(const std::vector<MVToolsVector>& srcVectors, const MVAnalysisData& srcAnalysisData,
                                     int left, int right, int top, int bottom,
                                     std::vector<MVToolsVector>& dstVectors, MVAnalysisData& dstAnalysisData);
    static void setMVToolsSuperProperties(VSMap* props, const SuperData& super, const VSAPI* vsapi);
    static void copyCroppedSuperFrame(const VSFrame* src, VSFrame* dst, const CropGrid& data, const VSAPI* vsapi);

    static void validateAnalysisData(const MVAnalysisData& analysisData);
    static void configureData(CropGrid& data, const MVAnalysisData& analysisData);
    static void configureStepData(CropGrid& data, const MVAnalysisData& analysisData);
};

bool CropGrid::readMVAnalysisData(const VSMap* props, MVAnalysisData& analysisData, const VSAPI* vsapi) noexcept {
    int err{};
    const auto* data = vsapi->mapGetData(props, MVToolsAnalysisDataKey, 0, &err);
    if (err || vsapi->mapGetDataSize(props, MVToolsAnalysisDataKey, 0, nullptr) != sizeof(MVAnalysisData))
        return false;

    std::memcpy(&analysisData, data, sizeof(analysisData));
    return true;
}

bool CropGrid::readMVToolsSuperData(const VSMap* props, SuperData& super, const VSAPI* vsapi) noexcept {
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

int CropGrid::readMotionVectorBlobValidity(const char* vectorBlob, const int vectorBlobSize) noexcept {
    if (!vectorBlob || vectorBlobSize < static_cast<int>(sizeof(MVArraySizeType) * 2))
        return 0;

    MVArraySizeType valid{};
    std::memcpy(&valid, vectorBlob + sizeof(MVArraySizeType), sizeof(valid));
    return valid != 0;
}

bool CropGrid::hasPublicSadStats(const VSMap* props, const VSAPI* vsapi) noexcept {
    return vsapi->mapGetType(props, RIFEMVAvgSadKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVMaxSadKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVMinSadKey) != ptUnset;
}

bool CropGrid::hasPublicMotionStats(const VSMap* props, const VSAPI* vsapi) noexcept {
    return vsapi->mapGetType(props, RIFEMVAvgAbsDxKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVAvgAbsDyKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVAvgAbsMotionKey) != ptUnset &&
           vsapi->mapGetType(props, RIFEMVPanAmountKey) != ptUnset;
}

void CropGrid::deleteMotionVectorPublicFrameStats(VSMap* props, const VSAPI* vsapi) {
    vsapi->mapDeleteKey(props, RIFEMVAvgSadKey);
    vsapi->mapDeleteKey(props, RIFEMVMaxSadKey);
    vsapi->mapDeleteKey(props, RIFEMVMinSadKey);
    vsapi->mapDeleteKey(props, RIFEMVAvgAbsDxKey);
    vsapi->mapDeleteKey(props, RIFEMVAvgAbsDyKey);
    vsapi->mapDeleteKey(props, RIFEMVAvgAbsMotionKey);
    vsapi->mapDeleteKey(props, RIFEMVPanAmountKey);
}

int CropGrid::cropGridPlaneHeightLuma(const int srcHeight, const int level, const int yRatioUV, const int vpad) noexcept {
    auto height = srcHeight;
    for (auto i = 1; i <= level; i++)
        height = vpad >= yRatioUV ? ((height / yRatioUV + 1) / 2) * yRatioUV : ((height / yRatioUV) / 2) * yRatioUV;
    return height;
}

int CropGrid::cropGridPlaneWidthLuma(const int srcWidth, const int level, const int xRatioUV, const int hpad) noexcept {
    auto width = srcWidth;
    for (auto i = 1; i <= level; i++)
        width = hpad >= xRatioUV ? ((width / xRatioUV + 1) / 2) * xRatioUV : ((width / xRatioUV) / 2) * xRatioUV;
    return width;
}

ptrdiff_t CropGrid::cropGridPlaneSuperOffset(const bool chroma, const int srcHeight, const int level, const int pel,
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

int CropGrid::cropGridSuperHeight(const int srcHeight, const int levels, const int pel,
                                  const int vpad, const int superWidth, const int yRatioUV) noexcept {
    auto height = static_cast<int>(cropGridPlaneSuperOffset(false, srcHeight, levels, pel, vpad, superWidth, yRatioUV) / superWidth);
    if (yRatioUV == 2 && (height & 1))
        height++;
    return height;
}

int CropGrid::cropGridSuperWidth(const int srcWidth, const int hpad, const int xRatioUV) noexcept {
    auto width = srcWidth + hpad * 2;
    if (xRatioUV == 2 && (width & 1))
        width++;
    return width;
}

void CropGrid::validateCropSubsamplingAlignment(const VSVideoInfo& vi, const int cropLeftPx, const int cropRightPx,
                                                const int cropTopPx, const int cropBottomPx) {
    const auto xRatioUV = 1 << vi.format.subSamplingW;
    const auto yRatioUV = 1 << vi.format.subSamplingH;
    if (vi.format.numPlanes > 1 &&
        (cropLeftPx % xRatioUV || cropRightPx % xRatioUV || cropTopPx % yRatioUV || cropBottomPx % yRatioUV))
        throw "crop values produce pixel crops that are not compatible with the clip's chroma subsampling";
}

void CropGrid::copyCroppedVideoFramePixels(const VSFrame* src, VSFrame* dst, const int cropLeftPx, const int cropTopPx,
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

void CropGrid::zeroVideoFrame(VSFrame* frame, const VSAPI* vsapi) {
    const auto* format = vsapi->getVideoFrameFormat(frame);
    for (auto plane = 0; plane < format->numPlanes; plane++) {
        auto* dstp = vsapi->getWritePtr(frame, plane);
        const auto stride = vsapi->getStride(frame, plane);
        const auto height = vsapi->getFrameHeight(frame, plane);
        for (auto y = 0; y < height; y++)
            std::memset(dstp + static_cast<size_t>(y) * stride, 0, stride);
    }
}

void CropGrid::padCroppedSuperPlane(uint8_t* plane, const ptrdiff_t stride, const int width, const int height,
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

bool CropGrid::cropMotionVectorGrid(const std::vector<MVToolsVector>& srcVectors, const MVAnalysisData& srcAnalysisData,
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
            vector.x = motionVectorGridClampComponent(vector.x, dstAnalysisData.nPel, blockX, dstAnalysisData.nBlkSizeX,
                                                      dstAnalysisData.nWidth, dstAnalysisData.nHPadding);
            vector.y = motionVectorGridClampComponent(vector.y, dstAnalysisData.nPel, blockY, dstAnalysisData.nBlkSizeY,
                                                      dstAnalysisData.nHeight, dstAnalysisData.nVPadding);
        }
    }

    return true;
}

void CropGrid::setMVToolsSuperProperties(VSMap* props, const SuperData& super, const VSAPI* vsapi) {
    vsapi->mapSetInt(props, MVToolsSuperHeightKey, super.height, maReplace);
    vsapi->mapSetInt(props, MVToolsSuperHPadKey, super.hpad, maReplace);
    vsapi->mapSetInt(props, MVToolsSuperVPadKey, super.vpad, maReplace);
    vsapi->mapSetInt(props, MVToolsSuperPelKey, super.pel, maReplace);
    vsapi->mapSetInt(props, MVToolsSuperModeYUVKey, super.modeYUV, maReplace);
    vsapi->mapSetInt(props, MVToolsSuperLevelsKey, super.levels, maReplace);
}

void CropGrid::copyCroppedSuperFrame(const VSFrame* src, VSFrame* dst, const CropGrid& d, const VSAPI* vsapi) {
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

const VSFrame* VS_CC CropGrid::getFrame(int n, int activationReason, void* instanceData,
                                        [[maybe_unused]] void** frameData,
                                        VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const CropGrid*>(instanceData) };

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto src = vsapi->getFrameFilter(n, d->node, frameCtx);

        try {
            if (d->mode == Mode::Super) {
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
            motionVectorGridPackBlob(croppedVectors, readMotionVectorBlobValidity(vectorBlob, vectorBlobSize), croppedAnalysisData,
                                     croppedBlob, &stats, includeSadStats, includeMotionStats);

            auto dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
            copyCroppedVideoFramePixels(src, dst, d->cropLeftPx, d->cropTopPx, vsapi);
            auto dstProps = vsapi->getFramePropertiesRW(dst);
            deleteMotionVectorPublicFrameStats(dstProps, vsapi);
            motionVectorGridSetProperties(dstProps, croppedAnalysisData, croppedBlob.data(), static_cast<int>(croppedBlob.size()),
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

void VS_CC CropGrid::free(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<CropGrid*>(instanceData) };
    vsapi->freeNode(d->node);
    delete d;
}

void CropGrid::validateAnalysisData(const MVAnalysisData& analysisData) {
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

void CropGrid::configureData(CropGrid& data, const MVAnalysisData& analysisData) {
    validateAnalysisData(analysisData);
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

void CropGrid::configureStepData(CropGrid& data, const MVAnalysisData& analysisData) {
    validateAnalysisData(analysisData);
    data.stepX = analysisData.nBlkSizeX - analysisData.nOverlapX;
    data.stepY = analysisData.nBlkSizeY - analysisData.nOverlapY;
    if (data.stepX <= 0 || data.stepY <= 0)
        throw "invalid MVTools block step metadata";

    data.cropLeftPx = data.left * data.stepX;
    data.cropRightPx = data.right * data.stepX;
    data.cropTopPx = data.top * data.stepY;
    data.cropBottomPx = data.bottom * data.stepY;
}

bool CropGrid::unpackMotionVectorBlob(const char* vectorBlob, const int vectorBlobSize,
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

void VS_CC CropGrid::create(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto data{ std::make_unique<CropGrid>() };
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
        data->mode = err ? Mode::Vector : Mode::Super;

        std::array<char, 1024> errorMsg{};
        firstFrame = vsapi->getFrame(0, data->node, errorMsg.data(), static_cast<int>(errorMsg.size()));
        if (!firstFrame)
            throw std::runtime_error(std::string("failed to retrieve first clip frame: ") + errorMsg.data());

        if (data->mode == Mode::Vector) {
            MVAnalysisData analysisData{};
            if (!readMVAnalysisData(vsapi->getFramePropertiesRO(firstFrame), analysisData, vsapi))
                throw "clip is not a single-level MVTools vector clip; pass vectors= when cropping an mv.Super clip";
            configureData(*data, analysisData);
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
            configureStepData(*data, analysisData);
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
        vsapi->createVideoFilter(out, "CropGrid", &data->vi, CropGrid::getFrame, CropGrid::free,
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

void CropGrid::registerFunction(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->registerFunction("CropGrid",
                             "clip:vnode;"
                             "left:int:opt;"
                             "right:int:opt;"
                             "top:int:opt;"
                             "bottom:int:opt;"
                             "vectors:vnode:opt;",
                             "clip:vnode;",
                             CropGrid::create, nullptr, plugin);
}

}

void registerCropGrid(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    CropGrid::registerFunction(plugin, vspapi);
}
