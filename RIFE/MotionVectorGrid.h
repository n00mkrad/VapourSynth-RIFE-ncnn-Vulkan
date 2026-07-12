// Copyright (c) 2021-2022 HolyWu
// SPDX-License-Identifier: MIT

#ifndef MOTION_VECTOR_GRID_H
#define MOTION_VECTOR_GRID_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "VapourSynth4.h"

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
    int64_t averageSad8x8;
    int64_t maxSad8x8;
    int64_t minSad8x8;
    double averageAbsDx;
    double averageAbsDy;
    double averageAbsMotion;
    double panAmount;
};

int motionVectorGridClampComponent(int value, int pel, int blockCoord,
                                   int blockSize, int size, int padding) noexcept;

char* motionVectorGridPrepareBlob(size_t vectorCount, bool valid, std::vector<char>& blob);

void motionVectorGridPackBlob(const std::vector<MVToolsVector>& vectors, bool valid,
                              const MVAnalysisData& analysisData, std::vector<char>& blob,
                              MotionVectorFrameStats* stats, bool includeSadStats,
                              bool includeMotionStats);

void motionVectorGridSetProperties(VSMap* props, const MVAnalysisData& analysisData,
                                   const char* vectorBlob, int vectorBlobSize,
                                   const MotionVectorFrameStats& stats, bool includeSadStats,
                                   bool includeMotionStats, const VSAPI* vsapi);

#endif
