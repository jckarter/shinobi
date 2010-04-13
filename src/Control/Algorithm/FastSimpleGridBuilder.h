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

#ifndef FASTSIMPLEGRIDBUILDER_H_INCLUDED_C1F24FB0_AB29_48E3_B29A_1B91640B0E9E
#define FASTSIMPLEGRIDBUILDER_H_INCLUDED_C1F24FB0_AB29_48E3_B29A_1B91640B0E9E

#include "../../CUDAConfig.h"
#include "../../CUDAStdAfx.h"

#include "../Structure/SimpleGrid.h"

#include "FastBuildKernels.h"
#include "GridBuilder.h"
#include "SAHGridBuilder.h"

#include "../../Utils/Scan.h"
#include "../../Utils/Sort.h"

template<class tPrimitiveStorage>
class FastSimpleGridBuilder
{

    uint mNumPrimitives;
    cudaEvent_t mStart, mDataUpload;
    cudaEvent_t mSizeEstimation, mReduceCellIds, mWriteUnsortedCells, mSort, mEnd;
    cudaPitchedPtr cpuCells;
    cudaPitchedPtr gpuCells;

public:
    HOST void setNumPrimitives(const uint aNum)
    {
        mNumPrimitives = aNum;
    }

    HOST void init(
        SimpleGrid&                                     oGrid,
        tPrimitiveStorage&                            oFaceSoup,
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

#ifdef SAHGRID
        SAHGridBuilder<tPrimitiveStorage>::computeResolution(mNumPrimitives, oGrid, oFaceSoup);
#else
        GridBuilder<tPrimitiveStorage>::computeResolution(mNumPrimitives, oGrid);
#endif
        gpuCells = oGrid.allocateDeviceCells();
        oGrid.setDeviceCellsToZero(gpuCells);

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mDataUpload, 0);
        cudaEventSynchronize(mDataUpload);
        //////////////////////////////////////////////////////////////////////////

    }


    HOST void build(
        SimpleGrid&                                     oGrid,
        tPrimitiveStorage&                            oFaceSoup)
    {

        //////////////////////////////////////////////////////////////////////////
        cudaEventCreate(&mSizeEstimation);
        cudaEventCreate(&mWriteUnsortedCells);
        cudaEventCreate(&mSort);
        cudaEventCreate(&mReduceCellIds);
        cudaEventCreate(&mEnd);
        //////////////////////////////////////////////////////////////////////////

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

        countRefs< tPrimitiveStorage > <<< gridTotalSize, blockTotalSize,
            blockTotalSize.x * (sizeof(uint) + sizeof(vec3f)) >>>(
            oFaceSoup,
            mNumPrimitives,
            oGrid.getResolution(), 
            oGrid.bounds.min,
            cellSize,
            cellSizeRCP,
            hlpReduceDevice);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        InclusiveScan scan;
        scan(hlpReduceDevice, gridTotalSize.x);
        --hlpReduceDevice; //make exclusive scan out of inclusive

        uint totalSize;
        CUDA_SAFE_CALL( cudaMemcpy(&totalSize, (hlpReduceDevice + gridTotalSize.x), sizeof(uint), cudaMemcpyDeviceToHost) );
        const uint indexArraySize = totalSize;

        // allocate temporary index array;
        uint* tmpIndices;
        CUDA_SAFE_CALL( cudaMalloc((void**)&tmpIndices, indexArraySize * sizeof(uint2)));

        cudaEventRecord(mSizeEstimation, 0);
        cudaEventSynchronize(mSizeEstimation);

        dim3 blockUnsortedGrid(NUMBUILDTHREADS);
        dim3 gridUnsortedGrid (NUMBUILDBLOCKS);

        buildUnsortedGrid< tPrimitiveStorage > 
            <<< gridUnsortedGrid, blockUnsortedGrid,
            sizeof(uint) + sizeof(vec3f) * blockUnsortedGrid.x >>>(
            oFaceSoup,
            tmpIndices,
            mNumPrimitives,
            hlpReduceDevice,
            oGrid.getResolution(),
            oGrid.bounds.min,
            cellSize,
            cellSizeRCP);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        cudaEventRecord(mWriteUnsortedCells, 0);
        cudaEventSynchronize(mWriteUnsortedCells);

        CUDA_SAFE_CALL( cudaFree( hlpReduceDevice ));

        // allocate help array for sorting
        uint* hlpIndices;
        CUDA_SAFE_CALL( cudaMalloc((void**)&hlpIndices, indexArraySize * sizeof(uint2)));


        const uint numCellsPlus1 = oGrid.resX * oGrid.resY * oGrid.resZ;
        uint numBits = 32u;
        for(; numCellsPlus1 >> numBits == 0u ; numBits -= 8u){}
        numBits = cudastd::max(8u, cudastd::min(32u, numBits + 8u));

        //cudastd::log::out << "Number: " << numCellsPlus1 <<" bits: " << numBits << "\n";

        Sort radixSort;
        radixSort((uint2*)tmpIndices, (uint2*)hlpIndices, indexArraySize, numBits);
        
        //RadixSort((KeyValuePair*)tmpIndices, (KeyValuePair*)hlpIndices, indexArraySize, numBits);

        CUDA_SAFE_CALL( cudaFree( hlpIndices ) );

        cudaEventRecord(mSort, 0);
        cudaEventSynchronize(mSort);

        ////////////////////////////////////////////////////////////////////////////
        ////DEBUG
        //uint* tmpIndicesHost;
        //CUDA_SAFE_CALL( cudaMallocHost((void**)&tmpIndicesHost, indexArraySize * sizeof(uint2)));

        //CUDA_SAFE_CALL( cudaMemcpy( tmpIndicesHost, tmpIndices, indexArraySize * sizeof(uint2),  cudaMemcpyDeviceToHost));

        //uint2* cellsOnHost;
        //CUDA_SAFE_CALL( cudaMallocHost((void**)&cellsOnHost, (uint)oGrid.resX * oGrid.resY * oGrid.resZ * sizeof(uint2)));
        //for(uint k = 0; k < (uint)oGrid.resX * oGrid.resY * oGrid.resZ; ++k)
        //{
        //    cellsOnHost[k] = make_uint2(0u, 0u);
        //}

        //for(uint bla = 0; bla < indexArraySize; bla += 1)
        //{
        //    if (tmpIndicesHost[2 * bla] > (uint)oGrid.resX * oGrid.resY * oGrid.resZ)
        //    {
        //        cudastd::log::out << "( " << tmpIndicesHost[2 * bla] << "," << 
        //            tmpIndicesHost[2 * bla + 1]<< " ) ";
        //        cudastd::log::out << "\n";
        //    }

        //    if (bla < indexArraySize - 1  && tmpIndicesHost[2 * bla] != tmpIndicesHost[2 * bla + 2])
        //    {
        //        //cudastd::log::out << "( " << tmpIndicesHost[2 * bla] << "," << 
        //        //    tmpIndicesHost[2 * bla + 1]<< " ) ";
        //        //cudastd::log::out << "( " << tmpIndicesHost[2 * bla + 2] << "," << 
        //        //    tmpIndicesHost[2 * bla + 3]<< " ) ";
        //        //cudastd::log::out << "\n";

        //        cellsOnHost[tmpIndicesHost[2 * bla]].y = bla + 1;
        //        //cellsOnHost[tmpIndicesHost[2 * bla + 2]].x = bla + 1;
        //    }

        //    if (bla > 0 && tmpIndicesHost[2 * bla - 2] != tmpIndicesHost[2 * bla])
        //    {
        //        //cudastd::log::out << "( " << tmpIndicesHost[2 * bla - 2] << "," << 
        //        //    tmpIndicesHost[2 * bla - 1]<< " ) ";
        //        //cudastd::log::out << "( " << tmpIndicesHost[2 * bla] << "," << 
        //        //    tmpIndicesHost[2 * bla + 1]<< " ) ";
        //        //cudastd::log::out << "\n";

        //        //cellsOnHost[tmpIndicesHost[2 * bla - 2]].y = bla;
        //        cellsOnHost[tmpIndicesHost[2 * bla]].x = bla;
        //    }
        //}
        ////////////////////////////////////////////////////////////////////////////

        CUDA_SAFE_CALL( cudaMalloc((void**)&oFaceSoup.indices, indexArraySize * sizeof(uint)));


        dim3 blockPrepRng(NUMBUILDTHREADS);
        dim3 gridPrepRng (NUMBUILDBLOCKS);

        prepareCellRanges< tPrimitiveStorage > <<< gridPrepRng, blockPrepRng, (2 + blockPrepRng.x) * sizeof(uint)>>>(
            oFaceSoup,
            (uint2*)tmpIndices,
            indexArraySize,
            gpuCells,
            static_cast<uint>(oGrid.resX),
            static_cast<uint>(oGrid.resY),
            static_cast<uint>(oGrid.resZ)
            );

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        ////////////////////////////////////////////////////////////////////////////
        ////DEBUG
        //dim3 blockChkCells(oGrid.resX);
        //dim3 gridChkCells(oGrid.resY, oGrid.resZ);

        //cudastd::log::out << oGrid.resX << " " << oGrid.resY << " " << oGrid.resZ << "\n";

        //checkGridCells<<< gridPrepRng, blockChkCells >>>(oFaceSoup, gpuCells, oGrid.getResolution());

        //CUT_CHECK_ERROR("Kernel Execution failed.\n");

        //oGrid.copyCellsDeviceToHost(cpuCells, gpuCells);

        //for(uint k = 0; k < (uint)oGrid.resX * oGrid.resY * oGrid.resZ; ++k)
        //{
        //    if (cellsOnHost[k].x != ((uint2*)cpuCells.ptr)[k].x ||
        //        cellsOnHost[k].y != ((uint2*)cpuCells.ptr)[k].y)
        //    {
        //        cudastd::log::out << "index : " << k << "\n";
        //        cudastd::log::out << "h( " << cellsOnHost[k].x << "," << cellsOnHost[k].y << " ) ";
        //        cudastd::log::out << "d( " << ((uint2*)cpuCells.ptr)[k].x << "," << 
        //            ((uint2*)cpuCells.ptr)[k].y << " ) ";
        //        cudastd::log::out << "\n";
        //    }
        //    
        //}
        ////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mReduceCellIds, 0);
        cudaEventSynchronize(mReduceCellIds);
        //////////////////////////////////////////////////////////////////////////

        CUDA_SAFE_CALL( cudaFree( tmpIndices ) );

        //////////////////////////////////////////////////////////////////////////
        //if the grid cells are too large for device-device copy    
        //cpuCells = oGrid.allocateHostCells();
        //oGrid.copyCellsDeviceToHost(cpuCells, gpuCells);
        //oGrid.freeCellMemoryDevice();
        //oGrid.bindHostDataToTexture(cpuCells);
        //oGrid.freeCellMemoryHost();
        //////////////////////////////////
        //else
        //allocate CUDA array for cell texture
        oGrid.bindDeviceDataToTexture(gpuCells);
        //free device memory
        oGrid.freeCellMemoryDevice();
        //endif
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mEnd, 0);
        cudaEventSynchronize(mEnd);
        //////////////////////////////////////////////////////////////////////////

        //cudastd::log::out << "Number of instances:" << indexArraySize << "\n";
        //outputStats();

        cleanup();
    }

    HOST void rebuild(
        SimpleGrid&         oGrid,
        tPrimitiveStorage&  oFaceSoup,
        cudaStream_t&       aStream
        )
    {

        if (oFaceSoup.indices != NULL)
        {
            CUDA_SAFE_CALL( cudaFree( oFaceSoup.indices ) );
        }

#ifndef SAHGRID
        GridBuilder<tPrimitiveStorage>::computeResolution(mNumPrimitives, oGrid);
#endif
        gpuCells = oGrid.allocateDeviceCells();
        oGrid.setDeviceCellsToZero(gpuCells);

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
            blockTotalSize.x * (sizeof(uint) + sizeof(vec3f)), aStream >>>(
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
        uint* tmpIndices;
        CUDA_SAFE_CALL( cudaMalloc((void**)&tmpIndices,
            indexArraySize * sizeof(uint2)));

        dim3 blockUnsortedGrid(NUMBUILDTHREADS);
        dim3 gridUnsortedGrid (NUMBUILDBLOCKS);

        buildUnsortedGrid< tPrimitiveStorage >
            <<< gridUnsortedGrid, blockUnsortedGrid, sizeof(uint) + sizeof(vec3f) * blockUnsortedGrid.x, aStream >>>(
            oFaceSoup,
            tmpIndices,
            mNumPrimitives,
            hlpReduceDevice,
            oGrid.getResolution(),
            oGrid.bounds.min,
            cellSize,
            cellSizeRCP);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        CUDA_SAFE_CALL( cudaFree( hlpReduceDevice ));

        // allocate help array for sorting
        uint* hlpIndices;
        CUDA_SAFE_CALL( cudaMalloc((void**)&hlpIndices, indexArraySize * sizeof(uint2)));


        const uint numCellsPlus1 = oGrid.resX * oGrid.resY * oGrid.resZ;
        uint numBits = 32u;
        for(; numCellsPlus1 >> numBits == 0u ; numBits -= 8u){}
        numBits = cudastd::min(32u, numBits + 8u);

        Sort radixSort;
        radixSort((uint2*)tmpIndices, (uint2*)hlpIndices, indexArraySize, numBits, aStream);

        CUDA_SAFE_CALL( cudaFree( hlpIndices ) );

        CUDA_SAFE_CALL( cudaMalloc((void**)&oFaceSoup.indices, indexArraySize * sizeof(uint)));

        dim3 blockPrepRng(NUMBUILDTHREADS);
        dim3 gridPrepRng (NUMBUILDBLOCKS);

        prepareCellRanges< tPrimitiveStorage >
            <<< gridPrepRng, blockPrepRng,
            (2 + blockPrepRng.x) * sizeof(uint), aStream>>>(
            oFaceSoup,
            (uint2*)tmpIndices,
            indexArraySize,
            gpuCells,
            static_cast<uint>(oGrid.resX),
            static_cast<uint>(oGrid.resY),
            static_cast<uint>(oGrid.resZ)
            );

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        CUDA_SAFE_CALL( cudaFree( tmpIndices ) );

        oGrid.reBindDeviceDataToTexture(gpuCells, aStream);

        //free device memory
        oGrid.freeCellMemoryDevice();
    }

    HOST void cleanup()
    {
        cudaEventDestroy(mStart);
        cudaEventDestroy(mDataUpload);
        cudaEventDestroy(mSizeEstimation);
        cudaEventDestroy(mWriteUnsortedCells);
        cudaEventDestroy(mSort);
        cudaEventDestroy(mReduceCellIds);
        cudaEventDestroy(mEnd);
    }

    HOST void outputStats()
    {
        //////////////////////////////////////////////////////////////////////////
        float elapsedTime;
        cudastd::log::floatPrecision(4);       
        cudaEventElapsedTime(&elapsedTime, mStart, mDataUpload);
        cudastd::log::out << "Data upload:     " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mDataUpload, mSizeEstimation);
        cudastd::log::out << "Size estimation: " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mSizeEstimation, mWriteUnsortedCells);
        cudastd::log::out << "Write cells:     " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mWriteUnsortedCells, mSort);
        cudastd::log::out << "Sort:            " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mSort, mReduceCellIds);
        cudastd::log::out << "Reduce:          " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mReduceCellIds, mEnd);
        cudastd::log::out << "Bind to texture: " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mDataUpload, mEnd);
        cudastd::log::out << "Total:           " << elapsedTime << "ms\n";
        //////////////////////////////////////////////////////////////////////////
    }
};

#endif // FASTSIMPLEGRIDBUILDER_H_INCLUDED_C1F24FB0_AB29_48E3_B29A_1B91640B0E9E
