/****************************************************************************/
/* Copyright (c) 2009, Javor Kalojanov
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/
/****************************************************************************/

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LAZYTWOLEVELGRIDBUILDER_H_INCLUDED_0558B4C5_1D58_4EE4_A846_870E8F039B46
#define LAZYTWOLEVELGRIDBUILDER_H_INCLUDED_0558B4C5_1D58_4EE4_A846_870E8F039B46

#include "../../CUDAConfig.h"
#include "../../CUDAStdAfx.h"

#include "../Structure/TwoLevelGrid.h"

#include "GridBuilder.h"
#include "FastBuildKernels.h"
#include "TwoLevelBuildKernels.h"
#include "SimpleGridTraverser.h"

#include "../../Intersector/SimpleIntersector.hpp"
#include "../../Intersector/HybridIntersector.hpp"

#include "../../Utils/Scan.h"
#include "../../Utils/Sort.h"


template< 
    class tPrimitiveStorage,
    class tRayGenerator,
    class tControlStructure,
    template <class> class tIntersector >
GLOBAL void tracePilotRays(
    tPrimitiveStorage       aStorage,
    cudaPitchedPtr          aGpuTopLevelCells,
    tControlStructure       aGridParameters,
    tRayGenerator           aRayGenerator,
    const uint              aNumRays,
    char*                   aGlobalMemoryPtr)
{
    typedef tIntersector<tPrimitiveStorage>                     t_Intersector;
    typedef SimpleGridPilotRayTraverser<
        tControlStructure, tPrimitiveStorage, t_Intersector>    t_Traverser;
    typedef typename t_Traverser::TraversalState                t_State;

    extern SHARED uint sharedMem[];

    vec3f* rayOrg =
        (vec3f*)(sharedMem + t_Intersector::SHAREDMEMSIZE);
    vec3f* rayDir = rayOrg + RENDERTHREADSX * RENDERTHREADSY;


    for(uint myRayIndex = globalThreadId1D(20)*20;
        ANY(myRayIndex < aNumRays);
        myRayIndex += numThreads() * 400)
    {

        //////////////////////////////////////////////////////////////////////////
        //Initialization
        t_Traverser     traverser;
        t_State         state;

        float rayT   = aRayGenerator(rayOrg[threadId1D()], rayDir[threadId1D()],
            myRayIndex, aNumRays);

        uint  bestHit;
        //////////////////////////////////////////////////////////////////////////

        traverser.traverse(rayOrg, rayDir, rayT, bestHit, state,
            aGridParameters, aStorage, sharedMem, aGpuTopLevelCells,
            aGlobalMemoryPtr);

        //*(aGlobalMemoryPtr + (uint)state.cellId.x +
        //    (uint)state.cellId.y * aGridParameters.res[0] +
        //    (uint)state.cellId.z * aGridParameters.res[0] *
        //    aGridParameters.res[1]) = 1;
    }
}

template<
    class tPrimitiveStorage,
    class tPilotPrimaryRayGenerator,
    class tPilotRayControlStructure,
    template <class> class tPilotRayIntersector >

class LazyTwoLevelGridBuilder
{

    uint mNumPrimitives;
    cudaEvent_t mStart, mDataUpload;
    cudaEvent_t mTopLevel, mPilotRays, mLeafCellCount, mLeafRefsCount,
        mLeafRefsWrite, mSortLeafPairs, mSecondLevel, mDataTransfer, mEnd;

    cudaPitchedPtr gpuTopLevelCells;
public:
    HOST void setNumPrimitives(const uint aNum)
    {
        mNumPrimitives = aNum;
    }

    HOST void init(
        TwoLevelGrid&                                   oGrid,
        tPrimitiveStorage&                              oFaceSoup,
        const FWObject::t_FaceIterator&                 aBegin,
        const FWObject::t_FaceIterator&                 aEnd,
        const FWObject&                                 aData)
    {
        //////////////////////////////////////////////////////////////////////////
        //initialize grid parameters
        cudaEventCreate(&mStart);
        cudaEventCreate(&mDataUpload);
        cudaEventRecord(mStart, 0);
        //////////////////////////////////////////////////////////////////////////

        mNumPrimitives = 0u;
        for(FWObject::t_FaceIterator it = aBegin; it != aEnd; ++mNumPrimitives, ++it)
        {
            oGrid.bounds.extend(aData.getVertex(it->vert1));
            oGrid.bounds.extend(aData.getVertex(it->vert2));
            oGrid.bounds.extend(aData.getVertex(it->vert3));
        }

        //cudastd::log::out << "Number of primitives: " << mNumPrimitives << "\n";

        oFaceSoup.upload(aBegin, aEnd, aData);

        vec3f diagonal = oGrid.bounds.max - oGrid.bounds.min;

        const float volume = diagonal.x * diagonal.y * diagonal.z;
        const float lambda = 5.f;
        const float magicConstant =
            powf(lambda * static_cast<float>(mNumPrimitives) / volume, 0.3333333f);

        diagonal *= magicConstant;

        oGrid.resX = static_cast<int>(diagonal.x);
        oGrid.resY = static_cast<int>(diagonal.y);
        oGrid.resZ = static_cast<int>(diagonal.z);

        oGrid.resX = cudastd::max( 2, oGrid.resX / 6 );
        oGrid.resY = cudastd::max( 2, oGrid.resY / 6 );
        oGrid.resZ = cudastd::max( 2, oGrid.resZ / 6 );

        gpuTopLevelCells = oGrid.allocateDeviceCells();
        oGrid.setDeviceCellsToZero(gpuTopLevelCells);

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mDataUpload, 0);
        cudaEventSynchronize(mDataUpload);
        //////////////////////////////////////////////////////////////////////////

    }//void init(...)

    HOST uint buildTopLevel(        
        TwoLevelGrid&                                   oGrid,
        tPrimitiveStorage&                              oFaceSoup,
        uint*&                                          oTmpPairs,
        cudaStream_t                                    aStream = 0)
    {
        vec3f cellSize = (oGrid.bounds.max - oGrid.bounds.min) /
            oGrid.getResolution();

        vec3f cellSizeRCP = vec3f::rep(1.f) / cellSize;

        dim3 blockTotalSize(NUMBUILDTHREADS);
        dim3 gridTotalSize (NUMBUILDBLOCKS);

        uint* hlpReduceDevice;
        CUDA_SAFE_CALL( cudaMalloc((void**)&hlpReduceDevice, (gridTotalSize.x + 1) * sizeof(uint)));

        //set first to 0 and leave it out
        CUDA_SAFE_CALL( cudaMemset(hlpReduceDevice, 0, sizeof(uint)) );
        ++hlpReduceDevice;

        countRefs< tPrimitiveStorage >
            <<< gridTotalSize, blockTotalSize,
            blockTotalSize.x * sizeof(uint), aStream >>>(
            oFaceSoup,
            mNumPrimitives,
            oGrid.getResolution(), 
            oGrid.bounds.min,
            cellSize,
            cellSizeRCP,
            hlpReduceDevice);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");


        InclusiveScan scan;
        scan(hlpReduceDevice, gridTotalSize.x, aStream);
        --hlpReduceDevice; //make exclusive scan out of inclusive

        uint* totalSizePageLocked;
        CUDA_SAFE_CALL( cudaMallocHost((void**)&totalSizePageLocked, sizeof(uint)) );
        CUDA_SAFE_CALL( cudaMemcpyAsync(totalSizePageLocked,
            (hlpReduceDevice + gridTotalSize.x),
            sizeof(uint), cudaMemcpyDeviceToHost, aStream) );
        CUDA_SAFE_CALL( cudaStreamSynchronize(aStream) );
        const uint indexArraySize = totalSizePageLocked[0];
        CUDA_SAFE_CALL( cudaFreeHost(totalSizePageLocked) );

        // allocate temporary index array;
        CUDA_SAFE_CALL( cudaMalloc((void**)&oTmpPairs,
            indexArraySize * sizeof(uint2)));

        dim3 blockUnsortedGrid(NUMBUILDTHREADS);
        dim3 gridUnsortedGrid (NUMBUILDBLOCKS);

        buildUnsortedGrid< tPrimitiveStorage >
            <<< gridUnsortedGrid, blockUnsortedGrid, sizeof(uint), aStream >>>(
            oFaceSoup,
            oTmpPairs,
            mNumPrimitives,
            hlpReduceDevice,
            oGrid.getResolution(),
            oGrid.bounds.min,
            cellSize,
            cellSizeRCP);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        CUDA_SAFE_CALL( cudaFree( hlpReduceDevice ));

        // allocate help array for sorting
        uint* hlpPairs;
        CUDA_SAFE_CALL( cudaMalloc((void**)&hlpPairs, indexArraySize * sizeof(uint2)));


        const uint numCellsPlus1 = oGrid.resX * oGrid.resY * oGrid.resZ;
        uint numBits = 32u;
        for(; numCellsPlus1 >> numBits == 0u ; numBits -= 8u){}
        numBits = cudastd::min(32u, numBits + 8u);

        Sort radixSort;
        radixSort((uint2*)oTmpPairs, (uint2*)hlpPairs, indexArraySize, numBits, aStream);

        CUDA_SAFE_CALL( cudaFree( hlpPairs ) );

        CUDA_SAFE_CALL( cudaMalloc((void**)&oFaceSoup.indices, indexArraySize * sizeof(uint)));

        dim3 blockPrepRng(NUMBUILDTHREADS);
        dim3 gridPrepRng (NUMBUILDBLOCKS);

        prepareCellRanges< tPrimitiveStorage >
            <<< gridPrepRng, blockPrepRng,
            (2 + blockPrepRng.x) * sizeof(uint), aStream>>>(
            oFaceSoup,
            (uint2*)oTmpPairs,
            indexArraySize,
            gpuTopLevelCells,
            static_cast<uint>(oGrid.resX),
            static_cast<uint>(oGrid.resY),
            static_cast<uint>(oGrid.resZ)
            );

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        return indexArraySize;

    }//void buildTopLevel(...)

    HOST void build(
        TwoLevelGrid&                                   oGrid,
        tPrimitiveStorage&                              oFaceSoup)
    {
        //////////////////////////////////////////////////////////////////////////
        cudaEventCreate(&mTopLevel);
        cudaEventCreate(&mPilotRays);
        cudaEventCreate(&mLeafCellCount);
        cudaEventCreate(&mLeafRefsCount);
        cudaEventCreate(&mLeafRefsWrite);
        cudaEventCreate(&mSortLeafPairs);
        cudaEventCreate(&mSecondLevel);
        cudaEventCreate(&mEnd);
        //////////////////////////////////////////////////////////////////////////

        uint* tmpIndices;

        uint numTopLvlRefs = buildTopLevel(oGrid, oFaceSoup, tmpIndices);

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //uint2 lastCellRangeHost;
        //uint2* lastCellRangeDevice = ((uint2*)
        //    ((char*)gpuTopLevelCells.ptr + 
        //    + (oGrid.resY - 1) * gpuTopLevelCells.pitch
        //    + (oGrid.resZ - 1) * gpuTopLevelCells.pitch * gpuTopLevelCells.ysize)
        //    + (oGrid.resX - 1));
        //CUDA_SAFE_CALL(cudaMemcpy(&lastCellRangeHost, lastCellRangeDevice, sizeof(uint2), cudaMemcpyDeviceToHost) );
        //cudastd::log::out << "last top level cell range: " << lastCellRangeHost.x << " " << lastCellRangeHost.y << "\n";
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mTopLevel, 0);
        //////////////////////////////////////////////////////////////////////////

        char* isLeafFlagsInv;
        CUDA_SAFE_CALL( cudaMalloc((void**)&isLeafFlagsInv, oGrid.resX * oGrid.resY * oGrid.resZ * sizeof(char)));
        CUDA_SAFE_CALL( cudaMemset(isLeafFlagsInv, 0, oGrid.resX * oGrid.resY * oGrid.resZ * sizeof(char)) );

        const uint sharedMemoryTrace =
            RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //rayOrg
            RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //rayDir
            tPilotRayIntersector<tPrimitiveStorage>::SHAREDMEMSIZE * 4u +                 //Intersector
            0u;

        const uint numRays = gRESX * gRESY;
        //CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcNumRays", &numRays, sizeof(uint)) );

        dim3 threadBlockTrace( RENDERTHREADSX, RENDERTHREADSY );
        dim3 blockGridTrace  ( RENDERBLOCKSX, RENDERBLOCKSY );

        tPilotPrimaryRayGenerator primaryRayGenerator;

        tracePilotRays <tPrimitiveStorage,
                        tPilotPrimaryRayGenerator,
                        tPilotRayControlStructure,
                        tPilotRayIntersector >
            <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace >>>
            (oFaceSoup,
            gpuTopLevelCells,
            oGrid.getParameters(),
            primaryRayGenerator,
            numRays,
            isLeafFlagsInv);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //char* isLeafFlagInvHost;
        //CUDA_SAFE_CALL( cudaMallocHost((void**)&isLeafFlagInvHost, oGrid.resX * oGrid.resY * oGrid.resZ * sizeof(char)));
        //CUDA_SAFE_CALL( cudaMemcpy(isLeafFlagInvHost, isLeafFlagsInv, oGrid.resX * oGrid.resY * oGrid.resZ * sizeof(char), cudaMemcpyDeviceToHost ));
        //uint cellCount = 0;
        //for(int i = 0; i < oGrid.resX * oGrid.resY * oGrid.resZ; ++i)
        //{
        //    if (isLeafFlagInvHost[i] != 0)
        //    {
        //        ++cellCount;
        //        cudastd::log::out << " " << i;
        //    }
        //}
        //cudastd::log::out << "\n";
        //cudastd::log::out << "Number of non-leaf cells : " << cellCount << "\n";
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mPilotRays, 0);
        //////////////////////////////////////////////////////////////////////////

        uint* leafCellCounts;
        CUDA_SAFE_CALL( cudaMalloc((void**)&leafCellCounts, (oGrid.resX * oGrid.resY * oGrid.resZ + 1) * sizeof(uint)));

        //set last to 0 and leave it out
        CUDA_SAFE_CALL( cudaMemset(leafCellCounts + oGrid.resX * oGrid.resY * oGrid.resZ, 0, sizeof(uint)) );
        //++leafCellCounts;

        dim3 blockCellCount(oGrid.resX);
        dim3 gridCellCount(oGrid.resY, oGrid.resZ);

        countLeafLevelCells< true > <<< gridCellCount, blockCellCount >>>(
            oGrid.getCellSize(),
            gpuTopLevelCells,
            leafCellCounts,
            isLeafFlagsInv
            );

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        CUDA_SAFE_CALL( cudaFree (isLeafFlagsInv) );

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //uint* leafCellCountsHost;
        //CUDA_SAFE_CALL( cudaMallocHost((void**)&leafCellCountsHost, (oGrid.resX * oGrid.resY * oGrid.resZ + 1) * sizeof(uint)) );
        //--leafCellCounts;
        //CUDA_SAFE_CALL( cudaMemcpy(leafCellCountsHost, leafCellCounts, (oGrid.resX * oGrid.resY * oGrid.resZ + 1) * sizeof(uint), cudaMemcpyDeviceToHost ));
        //++leafCellCounts;
        //cudastd::log::out << "leaf cells counts: ";
        //for(int it = 0; it < (oGrid.resX * oGrid.resY * oGrid.resZ + 1); ++it)
        //{
        //    cudastd::log::out << leafCellCountsHost[it] << " ";
        //}
        //cudastd::log::out << "\n";
        //////////////////////////////////////////////////////////////////////////

        ExclusiveScan escan;
        escan(leafCellCounts, oGrid.resX * oGrid.resY * oGrid.resZ + 1);

        //--leafCellCounts;//make exclusive scan out of inclusive

        prepareTopLevelCellRanges<<< gridCellCount, blockCellCount >>>(
            leafCellCounts,
            gpuTopLevelCells
            );

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //CUDA_SAFE_CALL( cudaMemcpy(leafCellCountsHost, leafCellCounts, (oGrid.resX * oGrid.resY * oGrid.resZ + 1) * sizeof(uint), cudaMemcpyDeviceToHost ));
        //cudastd::log::out << "scanned leaf cells counts: ";
        //for(int it = 0; it < (oGrid.resX * oGrid.resY * oGrid.resZ + 1); ++it)
        //{
        //    cudastd::log::out << leafCellCountsHost[it] << " ";
        //}
        //cudastd::log::out << "\n";
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //cudaPitchedPtr cpuCells = oGrid.allocateHostCells();
        //oGrid.copyCellsDeviceToHost(cpuCells, gpuTopLevelCells);
        //cudastd::log::out << "Top level cell res:\n";
        //for(uint cellId = 0; cellId < (uint)oGrid.resX * oGrid.resY * oGrid.resZ; ++cellId)
        //{
        //    if (oGrid.cpuCells[cellId][0] > 21u ||
        //        oGrid.cpuCells[cellId][1] > 21u ||
        //        oGrid.cpuCells[cellId][2] > 21u)
        //    {
        //        cudastd::log::out << 
        //            oGrid.cpuCells[cellId][0] << " " << 
        //            oGrid.cpuCells[cellId][1] << " " <<
        //            oGrid.cpuCells[cellId][2] << " " <<
        //            oGrid.cpuCells[cellId].getLeafRangeBegin() << "\n";
        //    }        
        //}
        //////////////////////////////////////////////////////////////////////////

        uint numLeafCells;
        CUDA_SAFE_CALL(cudaMemcpy(&numLeafCells, (leafCellCounts + oGrid.resX * oGrid.resY * oGrid.resZ), sizeof(uint), cudaMemcpyDeviceToHost) );

        CUDA_SAFE_CALL( cudaFree(leafCellCounts) );

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafCellCount, 0);
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //uint* debugInfo;
        //CUDA_SAFE_CALL( cudaMalloc((void**)&debugInfo, (81) * sizeof(uint)) );
        //CUDA_SAFE_CALL( cudaMemset(debugInfo, 0, sizeof(uint)) );
        //////////////////////////////////////////////////////////////////////////

        dim3 blockRefCount = NUMBUILDTHREADS_TLG;
        dim3 gridRefCount = NUMBUILDBLOCKS_TLG;

        uint* leafRefsCounts;
        CUDA_SAFE_CALL( cudaMalloc((void**)&leafRefsCounts, (gridRefCount.x + 1) * sizeof(uint)) );

        //set first to 0 and leave it out
        CUDA_SAFE_CALL( cudaMemset(leafRefsCounts + gridRefCount.x, 0, sizeof(uint)) );
        ++leafRefsCounts;

        countLeafLevelRefs< tPrimitiveStorage > 
            <<< gridRefCount, blockRefCount,  blockRefCount.x * sizeof(uint) * 2 >>>(
            oFaceSoup,
            numTopLvlRefs,
            (uint2*)tmpIndices,
            gpuTopLevelCells,
            //oGrid.getResolution(),
            static_cast<uint>(oGrid.resX),
            static_cast<uint>(oGrid.resY),
            static_cast<uint>(oGrid.resZ),
            oGrid.bounds.min,
            oGrid.getCellSize(),
            leafRefsCounts
            //////////////////////////////////////////////////////////////////////////
            //DEBUG
            //, debugInfo
            //////////////////////////////////////////////////////////////////////////
            );

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //uint* debugInfoHost;
        //CUDA_SAFE_CALL( cudaMallocHost((void**)&debugInfoHost, (81) * sizeof(uint)) );
        //CUDA_SAFE_CALL( cudaMemcpy(debugInfoHost, debugInfo, (81) * sizeof(uint), cudaMemcpyDeviceToHost ));
        //for(uint i = 0; debugInfoHost[0] != 0u && i < 4; ++i)
        //{
        //    cudastd::log::out << "buggy insertion : "   << i <<" :\n";
        //    cudastd::log::out << "indexPair.x        ; "   << debugInfoHost           [i*20 +  1] <<"\n";
        //    cudastd::log::out << "topLvlCellX        ; "   << debugInfoHost           [i*20 +  2] <<"\n";
        //    cudastd::log::out << "topLvlCellY        ; "   << debugInfoHost           [i*20 +  3] <<"\n";
        //    cudastd::log::out << "topLvlCellZ        ; "   << debugInfoHost           [i*20 +  4] <<"\n";
        //    cudastd::log::out << "topLvlCellRes.x    ; "   << ((float*)debugInfoHost) [i*20 +  5] <<"\n";
        //    cudastd::log::out << "topLvlCellRes.y    ; "   << ((float*)debugInfoHost) [i*20 +  6] <<"\n";
        //    cudastd::log::out << "topLvlCellRes.z    ; "   << ((float*)debugInfoHost) [i*20 +  7] <<"\n";
        //    cudastd::log::out << "minCellIdX         ; "   << ((int*)debugInfoHost)   [i*20 +  8] <<"\n";
        //    cudastd::log::out << "minCellIdY         ; "   << ((int*)debugInfoHost)   [i*20 +  9] <<"\n";
        //    cudastd::log::out << "minCellIdZ         ; "   << ((int*)debugInfoHost)   [i*20 + 10] <<"\n";
        //    cudastd::log::out << "maxCellIdP1X       ; "   << ((int*)debugInfoHost)   [i*20 + 11] <<"\n";
        //    cudastd::log::out << "maxCellIdP1Y       ; "   << ((int*)debugInfoHost)   [i*20 + 12] <<"\n";
        //    cudastd::log::out << "maxCellIdP1Z       ; "   << ((int*)debugInfoHost)   [i*20 + 13] <<"\n";
        //    cudastd::log::out << "minCellIdf.x       ; "   << ((float*)debugInfoHost) [i*20 + 14] <<"\n";
        //    cudastd::log::out << "minCellIdf.y       ; "   << ((float*)debugInfoHost) [i*20 + 15] <<"\n";
        //    cudastd::log::out << "minCellIdf.z       ; "   << ((float*)debugInfoHost) [i*20 + 16] <<"\n";
        //    cudastd::log::out << "maxCellIdPlus1f.x  ; "   << ((float*)debugInfoHost) [i*20 + 17] <<"\n";
        //    cudastd::log::out << "maxCellIdPlus1f.y  ; "   << ((float*)debugInfoHost) [i*20 + 18] <<"\n";
        //    cudastd::log::out << "maxCellIdPlus1f.z  ; "   << ((float*)debugInfoHost) [i*20 + 19] <<"\n";
        //    cudastd::log::out << "numCells           ; "   << debugInfoHost           [i*20 + 20] <<"\n";
        //                          
        //}
        //CUDA_SAFE_CALL(cudaFree(debugInfo));
        //CUDA_SAFE_CALL(cudaFreeHost(debugInfoHost));
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //uint* leafRefsCountsHost;
        //CUDA_SAFE_CALL( cudaMallocHost((void**)&leafRefsCountsHost, (NUMBUILDBLOCKS_TLG + 1) * sizeof(uint)) );
        //--leafRefsCounts;
        //CUDA_SAFE_CALL( cudaMemcpy(leafRefsCountsHost, leafRefsCounts, (NUMBUILDBLOCKS_TLG + 1) * sizeof(uint), cudaMemcpyDeviceToHost ));
        //++leafRefsCounts;
        //cudastd::log::out << "leaf reference counts: ";
        //for(int it = 0; it < NUMBUILDBLOCKS_TLG + 1; ++it)
        //{
        //    cudastd::log::out << leafRefsCountsHost[it] << " ";
        //}
        //cudastd::log::out << "\n";
        //////////////////////////////////////////////////////////////////////////

        InclusiveScan iscan;
        iscan(leafRefsCounts, gridRefCount.x);
        --leafRefsCounts; //make exclusive scan out of inclusive

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafRefsCount, 0);
        //////////////////////////////////////////////////////////////////////////

        uint numRefs;
        CUDA_SAFE_CALL( cudaMemcpy(&numRefs, (leafRefsCounts + gridRefCount.x), sizeof(uint), cudaMemcpyDeviceToHost) );


        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //CUDA_SAFE_CALL( cudaMemcpy(leafRefsCountsHost, leafRefsCounts, (NUMBUILDBLOCKS_TLG + 1) * sizeof(uint), cudaMemcpyDeviceToHost ) );
        //cudastd::log::out << "scanned leaf reference counts: ";
        //for(int it = 0; it < NUMBUILDBLOCKS_TLG + 1; ++it)
        //{
        //    cudastd::log::out << leafRefsCountsHost[it] << " ";
        //}
        //cudastd::log::out << "\n";
        //////////////////////////////////////////////////////////////////////////

        uint* leafLevelPairs;
        CUDA_SAFE_CALL(cudaMalloc((void**)&leafLevelPairs, numRefs * sizeof(uint2)));

        dim3 blockRefWrite = NUMBUILDTHREADS_TLG;
        dim3 gridRefWrite = NUMBUILDBLOCKS_TLG;

        writeLeafLevelRefs< tPrimitiveStorage >
            <<< gridRefWrite, blockRefWrite,  blockRefWrite.x * sizeof(uint) + sizeof(uint) >>>(
            oFaceSoup,
            numTopLvlRefs,
            (uint2*)tmpIndices,
            gpuTopLevelCells,
            numLeafCells,
            leafRefsCounts,
            //oGrid.getResolution(),
            static_cast<uint>(oGrid.resX),
            static_cast<uint>(oGrid.resY),
            static_cast<uint>(oGrid.resZ),
            oGrid.bounds.min,
            oGrid.getCellSize(),
            leafLevelPairs
            );

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        CUDA_SAFE_CALL( cudaFree( tmpIndices ) );
        CUDA_SAFE_CALL( cudaFree( leafRefsCounts) );

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mLeafRefsWrite, 0);
        //////////////////////////////////////////////////////////////////////////

        uint* hlpPairs;
        CUDA_SAFE_CALL( cudaMalloc((void**)&hlpPairs, numRefs * sizeof(uint2)));

        uint numBits = 32u;
        for(; numLeafCells >> numBits == 0u ; numBits -= 8u){}
        numBits = cudastd::min(32u, numBits + 8u);

        Sort radixSort;
        radixSort((uint2*)leafLevelPairs, (uint2*)hlpPairs, numRefs, numBits);

        CUDA_SAFE_CALL( cudaFree( hlpPairs ) );

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //uint2* hostPairs;
        //CUDA_SAFE_CALL( cudaMallocHost((void**)&hostPairs, numRefs * sizeof(uint2)) );
        //CUDA_SAFE_CALL( cudaMemcpy(hostPairs, leafLevelPairs, numRefs * sizeof(uint2), cudaMemcpyDeviceToHost) );
        //cudastd::log::out << "Sorted leaf pairs:\n";
        //uint numRealPairs = 0u;
        //for(uint it = 0; it < numRefs; ++it)
        //{
        //    if (hostPairs[it].x < numLeafCells)
        //    {
        //        ++numRealPairs;
        //    }
        //    //cudastd::log::out << "( " << hostPairs[it].x << " | " << hostPairs[it].y  << " ) ";
        //}
        //cudastd::log::out <<"Number of actual references: "<< numRealPairs <<"\n";
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mSortLeafPairs, 0);
        //////////////////////////////////////////////////////////////////////////

        CUDA_SAFE_CALL( cudaFree(oFaceSoup.indices) );
        CUDA_SAFE_CALL( cudaMalloc((void**)&oFaceSoup.indices, numRefs * sizeof(uint)));
        oGrid.allocateDeviceLeaves(numLeafCells);
        oGrid.setDeviceLeavesToZero();

        dim3 blockPrepRng(NUMBUILDTHREADS_TLG);
        dim3 gridPrepRng (NUMBUILDBLOCKS_TLG);

        prepareLeafCellRanges< tPrimitiveStorage >
            <<< gridPrepRng, blockPrepRng,
            (2 + blockPrepRng.x) * sizeof(uint) >>>(
            oFaceSoup,
            (uint2*)leafLevelPairs,
            numRefs,
            (uint2*)oGrid.gpuLeaves
            );

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        CUDA_SAFE_CALL( cudaFree(leafLevelPairs) );

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //oGrid.allocateHostLeaves(oGrid.leavesCount);
        //oGrid.copyLeavesDeviceToHost();
        //cudastd::log::out << "Grid leaves:\n";
        //for(uint it = 0; it < oGrid.leavesCount; ++it)
        //{
        //    cudastd::log::out << oGrid.cpuLeaves[it].x << " " << oGrid.cpuLeaves[it].y << " | ";
        //}
        //cudastd::log::out << "\n";
        //////////////////////////////////////////////////////////////////////////


        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mSecondLevel, 0);
        //////////////////////////////////////////////////////////////////////////

        oGrid.copyCellsDeviceToDeviceArray(gpuTopLevelCells);
        oGrid.bindDeviceDataToTexture();

        //oGrid.freeCellMemoryHost();
        oGrid.freeCellMemoryDevice();
        //oGrid.freeLeafMemoryHost();

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mEnd, 0);
        //////////////////////////////////////////////////////////////////////////

        cudastd::log::out << "Top  level cells:     " << oGrid.resX * oGrid.resY * oGrid.resZ << "\n";
        cudastd::log::out << "Top  level refs:      " << numTopLvlRefs << "\n";
        cudastd::log::out << "Leaf level cells:     " << numLeafCells << "\n";
        cudastd::log::out << "Leaf level refs:      " << numRefs << "\n";
        outputStats();
        cleanup();

    }

    HOST void rebuild(
        TwoLevelGrid&                                   oGrid,
        tPrimitiveStorage&                              oFaceSoup,
        cudaStream_t&                                   aStream)
    {
        //Not implemented
    }

    HOST void cleanup()
    {
        cudaEventDestroy(mStart);
        cudaEventDestroy(mDataUpload);
        cudaEventDestroy(mTopLevel);
        cudaEventDestroy(mPilotRays);
        cudaEventDestroy(mLeafCellCount);
        cudaEventDestroy(mLeafRefsCount);
        cudaEventDestroy(mLeafRefsWrite);
        cudaEventDestroy(mSortLeafPairs);
        cudaEventDestroy(mSecondLevel);
        cudaEventDestroy(mEnd);
    }

    HOST void outputStats()
    {
        //////////////////////////////////////////////////////////////////////////
        float elapsedTime;
        cudastd::log::floatPrecision(4);       
        cudaEventElapsedTime(&elapsedTime, mStart, mDataUpload);
        cudastd::log::out << "Data upload:      " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mDataUpload, mTopLevel);
        cudastd::log::out << "Top Level:        " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mTopLevel, mSecondLevel);
        cudastd::log::out << "Leaf Level:       " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mTopLevel, mPilotRays);
        cudastd::log::out << "Pilot rays:       " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mPilotRays, mLeafCellCount);
        cudastd::log::out << "Leaf Cells Count: " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mLeafCellCount, mLeafRefsCount);
        cudastd::log::out << "Leaf Refs Count:  " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mLeafRefsCount, mLeafRefsWrite);
        cudastd::log::out << "Leaf Refs Write:  " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mLeafRefsWrite, mSortLeafPairs);
        cudastd::log::out << "Leaf Refs Sort:   " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime,mSortLeafPairs, mSecondLevel);
        cudastd::log::out << "Prep. Leaf cells: " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mSecondLevel, mEnd);
        cudastd::log::out << "Data upload:      " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mDataUpload, mEnd);
        cudastd::log::out << "Total:            " << elapsedTime << "ms\n";
        //////////////////////////////////////////////////////////////////////////
    }


};


#undef PilotRaysIntersector_t

#endif // LAZYTWOLEVELGRIDBUILDER_H_INCLUDED_0558B4C5_1D58_4EE4_A846_870E8F039B46
