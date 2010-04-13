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

#ifndef SIMPLEGRIDBUILDER_H_INCLUDED_201B9104_21C0_45B9_B392_9B2B711A734F
#define SIMPLEGRIDBUILDER_H_INCLUDED_201B9104_21C0_45B9_B392_9B2B711A734F

#include "../../CUDAStdAfx.h"

#include "../Structure/SimpleGrid.h"

#include "BuildKernels.h"
#include "GridBuilder.h"

template<class tPrimitiveStorage>
class SimpleGridBuilder
{
    uint mNumPrimitives;
    cudaEvent_t mStart, mDataUpload;
    cudaEvent_t mSizeEstimation, mPrefixSum, mFillGridEvent, mEnd;
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

        //cudastd::log::out << "Number of primitives:" << mNumPrimitives << "\n";
        oFaceSoup.upload(aBegin, aEnd, aData);

        GridBuilder<tPrimitiveStorage>::computeResolution(mNumPrimitives, oGrid);

        cpuCells = oGrid.allocateHostCells();
        gpuCells = oGrid.allocateDeviceCells();
        oGrid.setDeviceCellsToZero(gpuCells);
        
        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mDataUpload, 0);
        cudaEventSynchronize(mDataUpload);
        //////////////////////////////////////////////////////////////////////////

    }

    HOST void build(
        SimpleGrid&                                     oGrid,
        tPrimitiveStorage&                              oFaceSoup)
    {

        //////////////////////////////////////////////////////////////////////////
        cudaEventCreate(&mSizeEstimation);
        cudaEventCreate(&mPrefixSum);
        cudaEventCreate(&mFillGridEvent);
        cudaEventCreate(&mEnd);
        cudaEventRecord(mSizeEstimation, 0);
        //////////////////////////////////////////////////////////////////////////
        
        vec3f cellSize = (oGrid.bounds.max - oGrid.bounds.min) /
            oGrid.getResolution();

        vec3f cellSizeRCP = vec3f::rep(1.f) / cellSize;
        // call Kernel to compute cell Triangle count
        dim3 blockEstimateCellSize(128);
        dim3 gridEstimateCellSize (32);

        estimateCellSize< tPrimitiveStorage > <<< gridEstimateCellSize, blockEstimateCellSize>>>(
                oFaceSoup,
                mNumPrimitives,
                gpuCells,
                oGrid.getResolution(), 
                oGrid.bounds.min,
                cellSize,
                cellSizeRCP);
        
        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        cudaEventRecord(mSizeEstimation, 0);
        cudaEventSynchronize(mSizeEstimation);

        //copy data from device
        oGrid.copyCellsDeviceToHost(cpuCells, gpuCells);

        // Do prefix sum on CPU.
        uint sum = 0;
        for (int z = 0; z < oGrid.resZ; ++z)
        {
            for (int y = 0; y < oGrid.resY; ++y)
            {
                for (int x = 0; x < oGrid.resX; ++x)
                {
                    uint2* cell = (uint2*)((char*)cpuCells.ptr
                        + y * cpuCells.pitch
                        + z * cpuCells.pitch * cpuCells.ysize) + x;
                    
                    cell->x = sum;
                    sum += cell->y;
                    cell->y = cell->x;
                }//end for x
            }//end for y
        }//end for z

        const uint indexArraySize = sum;
        // getInstancesArraySize(primitiveCount, oFaceSoup, oGrid);

        // allocate Index Array;
        CUDA_SAFE_CALL( cudaMalloc((void**)&oFaceSoup.indices ,
            indexArraySize * sizeof(uint)));

        // copy data to device again
        oGrid.copyCellsHostToDevice(gpuCells,cpuCells);

        cudaEventRecord(mPrefixSum, 0);
        cudaEventSynchronize(mPrefixSum);

        dim3 blockFillGrid(128);
        dim3 gridFillGrid (32);

        fillGrid< tPrimitiveStorage > <<< gridFillGrid, blockFillGrid >>>(
            oFaceSoup,
            mNumPrimitives,
            gpuCells,
            oGrid.getResolution(),
            oGrid.bounds.min,
            cellSize,
            cellSizeRCP);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        cudaEventRecord(mFillGridEvent, 0);
        cudaEventSynchronize(mFillGridEvent);

        //allocate CUDA array for cell texture
        oGrid.bindDeviceDataToTexture(gpuCells);

        //free device memory
        oGrid.freeCellMemoryDevice();
        //free host memory
        oGrid.freeCellMemoryHost();

        //////////////////////////////////////////////////////////////////////////
        cudaEventRecord(mEnd, 0);
        cudaEventSynchronize(mEnd);
        //////////////////////////////////////////////////////////////////////////
        
        //cudastd::log::out << "Number of instances:" << indexArraySize << "\n";
        //outputStats();

        cleanup();
        
    }

    //dummy
    HOST void rebuild(
        SimpleGrid&         oGrid,
        tPrimitiveStorage&  oFaceSoup,
        cudaStream_t&       aStream)
    {

        if (oFaceSoup.indices != NULL)
        {
            CUDA_SAFE_CALL( cudaFree( oFaceSoup.indices ) );
        }

#ifndef SAHGRID
        GridBuilder<tPrimitiveStorage>::computeResolution(mNumPrimitives, oGrid);
#endif
        cpuCells = oGrid.allocateHostCells();
        gpuCells = oGrid.allocateDeviceCells();
        oGrid.setDeviceCellsToZero(gpuCells);

        vec3f cellSize = (oGrid.bounds.max - oGrid.bounds.min) /
            oGrid.getResolution();

        vec3f cellSizeRCP = vec3f::rep(1.f) / cellSize;
        // call Kernel to compute cell Triangle count
        dim3 blockEstimateCellSize(128);
        dim3 gridEstimateCellSize (32);

        estimateCellSize< tPrimitiveStorage > <<< gridEstimateCellSize, blockEstimateCellSize>>>(
            oFaceSoup,
            mNumPrimitives,
            gpuCells,
            oGrid.getResolution(), 
            oGrid.bounds.min,
            cellSize,
            cellSizeRCP);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        //copy data from device
        oGrid.copyCellsDeviceToHost(cpuCells, gpuCells);

        // Do prefix sum on CPU.
        uint sum = 0;
        for (int z = 0; z < oGrid.resZ; ++z)
        {
            for (int y = 0; y < oGrid.resY; ++y)
            {
                for (int x = 0; x < oGrid.resX; ++x)
                {
                    uint2* cell = (uint2*)((char*)cpuCells.ptr
                        + y * cpuCells.pitch
                        + z * cpuCells.pitch * cpuCells.ysize) + x;

                    cell->x = sum;
                    sum += cell->y;
                    cell->y = cell->x;
                }//end for x
            }//end for y
        }//end for z

        const uint indexArraySize = sum;
        // getInstancesArraySize(primitiveCount, oFaceSoup, oGrid);

        // allocate Index Array;
        CUDA_SAFE_CALL( cudaMalloc((void**)&oFaceSoup.indices ,
            indexArraySize * sizeof(uint)));

        // copy data to device again
        oGrid.copyCellsHostToDevice(gpuCells,cpuCells);

        dim3 blockFillGrid(128);
        dim3 gridFillGrid (32);

        fillGrid< tPrimitiveStorage > <<< gridFillGrid, blockFillGrid >>>(
            oFaceSoup,
            mNumPrimitives,
            gpuCells,
            oGrid.getResolution(),
            oGrid.bounds.min,
            cellSize,
            cellSizeRCP);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        //allocate CUDA array for cell texture
        oGrid.reBindDeviceDataToTexture(gpuCells, aStream);

        //free device memory
        oGrid.freeCellMemoryDevice();
        //free host memory
        oGrid.freeCellMemoryHost();

    }

    HOST void cleanup()
    {
        cudaEventDestroy(mStart);
        cudaEventDestroy(mDataUpload);
        cudaEventDestroy(mSizeEstimation);
        cudaEventDestroy(mPrefixSum);
        cudaEventDestroy(mFillGridEvent);
        cudaEventDestroy(mEnd);
    }

    HOST void outputStats()
    {
        //////////////////////////////////////////////////////////////////////////
        float elapsedTime;
        cudastd::log::floatPrecision(4);       
        cudaEventElapsedTime(&elapsedTime, mStart, mDataUpload);
        cudastd::log::out << "Data upload:       " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mDataUpload, mSizeEstimation);
        cudastd::log::out << "Space estimation:  " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mSizeEstimation, mPrefixSum);
        cudastd::log::out << "Prefix sum:        " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mPrefixSum, mFillGridEvent);
        cudastd::log::out << "Construction time: " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mFillGridEvent, mEnd);
        cudastd::log::out << "Bind to texture:   " << elapsedTime << "ms\n";
        cudaEventElapsedTime(&elapsedTime, mDataUpload, mEnd);
        cudastd::log::out << "Total:             " << elapsedTime << "ms\n";
        /////////////////////////////////////////////////////////////////////////
    }
};

#endif // SIMPLEGRIDBUILDER_H_INCLUDED_201B9104_21C0_45B9_B392_9B2B711A734F
