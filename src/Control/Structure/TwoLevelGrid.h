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

#ifndef TWOLEVELGRID_H_INCLUDED_3386E0EE_E8A3_469B_BA24_12AB64EB2939
#define TWOLEVELGRID_H_INCLUDED_3386E0EE_E8A3_469B_BA24_12AB64EB2939

#include "../../CUDAStdAfx.h"
#include "TwoLevelGrid.hpp"
#include "../../Core/Algebra.hpp"
#include "../../Primitive/BBox.hpp"

texture< uint2, 3, cudaReadModeElementType > texTopLevelCells;
texture< uint2, 1, cudaReadModeElementType > texLeafCells;

class TwoLevelGrid
{
public:


    typedef uint2                           t_Leaf;
    typedef TwoLevelGridCell                t_Cell;
    
    int resX, resY, resZ;
    BBox bounds;
    //NOTE: cells must point to page-locked memory!
    t_Cell* cpuCells;
    t_Cell* gpuCells;
    t_Leaf* cpuLeaves;
    t_Leaf* gpuLeaves;

    //bookkeeping for deallocation
    char* gpuTopLevelCellsPitchPtr;
    char* cpuTopLevelCellsPitchPtr;

    uint leavesCount, instancesCount;
    cudaArray* topLevelCellsArray;

    //////////////////////////////////////////////////////////////////////////
    //construction related
    //////////////////////////////////////////////////////////////////////////
    t_Cell& getCell(uint aId)
    {
        return cpuCells[aId];
    }

    vec3f getResolution() const
    {
        vec3f retval;
        retval.x = static_cast<float>(resX);
        retval.y = static_cast<float>(resY);
        retval.z = static_cast<float>(resZ);
        return retval;
    }

    vec3f getCellSize() const
    {
        return (bounds.max - bounds.min) / getResolution();
    }

    vec3f getCellSizeRCP() const
    {
        return getResolution()/ (bounds.max - bounds.min);
    }

    //////////////////////////////////////////////////////////////////////////
    //data transfer related
    //////////////////////////////////////////////////////////////////////////
    GridParameters getParameters() const
    {
        GridParameters retval;
        retval.bounds = bounds;
        retval.res[0] = resX;
        retval.res[1] = resY;
        retval.res[2] = resZ;
        retval.cellSize = getCellSize();
        retval.cellSizeRCP = getCellSizeRCP();


        return retval;
    }

    HOST void upload()
    {

        //allocate array on device
        cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();
        cudaExtent res = make_cudaExtent(resX, resY, resZ);
        CUDA_SAFE_CALL( cudaMalloc3DArray(&topLevelCellsArray, &chanelFormatDesc, res) );


        //copy data to device
        cudaPitchedPtr pitchedPtr;
        //NOTE: cells must point to page-locked memory!
        pitchedPtr.ptr = (uint2* )cpuCells;
        pitchedPtr.pitch = resX * sizeof(t_Cell);
        pitchedPtr.xsize = resX;
        pitchedPtr.ysize = resY;

        cudaMemcpy3DParms cpyParams = { 0 };
        cpyParams.srcPtr    = pitchedPtr;
        cpyParams.dstArray  = topLevelCellsArray;
        cpyParams.extent    = res;
        cpyParams.kind      = cudaMemcpyHostToDevice;
        CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParams) );

        //bind texture to device array
        CUDA_SAFE_CALL( cudaBindTextureToArray(texTopLevelCells, topLevelCellsArray, chanelFormatDesc) );

        //free host memory
        CUDA_SAFE_CALL( cudaFreeHost(pitchedPtr.ptr) );

        const size_t leavesSize = leavesCount * sizeof(t_Leaf);

        CUDA_SAFE_CALL( cudaMalloc( (void**)&gpuLeaves, leavesSize) );


        CUDA_SAFE_CALL( 
            cudaMemcpy(gpuLeaves, cpuLeaves, leavesSize, cudaMemcpyHostToDevice));

        cudaChannelFormatDesc chanelFormatDescLeaves =
            cudaCreateChannelDesc<uint2>();

        CUDA_SAFE_CALL( cudaBindTexture(NULL, texLeafCells, (void*) gpuLeaves,
            chanelFormatDescLeaves, leavesSize) );

    }

    void copyCellsDeviceToHost(
        const cudaPitchedPtr& aHostCells,
        const cudaPitchedPtr& aDeviceCells)
    {
        cudaMemcpy3DParms cpyParamsDownloadPtr = { 0 };
        cpyParamsDownloadPtr.srcPtr  = aDeviceCells;
        cpyParamsDownloadPtr.dstPtr  = aHostCells;
        cpyParamsDownloadPtr.extent  = make_cudaExtent(resX * sizeof(t_Cell), resY, resZ);
        cpyParamsDownloadPtr.kind    = cudaMemcpyDeviceToHost;

        CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsDownloadPtr) );
    }

    void copyCellsHostToDevice(
        const cudaPitchedPtr& aDeviceCells,
        const cudaPitchedPtr& aHostCells)
    {
        cudaMemcpy3DParms cpyParamsUploadPtr = { 0 };
        cpyParamsUploadPtr.srcPtr  = aHostCells;
        cpyParamsUploadPtr.dstPtr  = aDeviceCells;
        cpyParamsUploadPtr.extent  = make_cudaExtent(resX * sizeof(t_Cell), resY, resZ);
        cpyParamsUploadPtr.kind    = cudaMemcpyHostToDevice;

        CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsUploadPtr) );
    }

    void copyCellsHostToDeviceArray(
        const cudaPitchedPtr& aHostCells)
    {
        cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();
        cudaExtent res = make_cudaExtent(resX, resY, resZ);
        CUDA_SAFE_CALL( cudaMalloc3DArray(&topLevelCellsArray, &chanelFormatDesc, res) );

        cudaMemcpy3DParms cpyParamsUploadPtr = { 0 };
        cpyParamsUploadPtr.srcPtr   = aHostCells;
        cpyParamsUploadPtr.dstArray = topLevelCellsArray;
        cpyParamsUploadPtr.extent   = make_cudaExtent(resX, resY, resZ);
        cpyParamsUploadPtr.kind     = cudaMemcpyHostToDevice;

        CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsUploadPtr) );
    }

    void copyCellsDeviceToDeviceArray(
        const cudaPitchedPtr& aDeviceCells)
    {
        cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();
        cudaExtent res = make_cudaExtent(resX, resY, resZ);
        CUDA_SAFE_CALL( cudaMalloc3DArray(&topLevelCellsArray, &chanelFormatDesc, res) );

        cudaMemcpy3DParms cpyParamsUploadPtr = { 0 };
        cpyParamsUploadPtr.srcPtr   = aDeviceCells;
        cpyParamsUploadPtr.dstArray = topLevelCellsArray;
        cpyParamsUploadPtr.extent   = make_cudaExtent(resX, resY, resZ);
        cpyParamsUploadPtr.kind     = cudaMemcpyDeviceToDevice;

        CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsUploadPtr) );
    }

    void copyLeavesHostToDevice()
    {
        const uint leavesSize = leavesCount * sizeof(t_Leaf);

        CUDA_SAFE_CALL( 
            cudaMemcpy(gpuLeaves, cpuLeaves, leavesSize, cudaMemcpyHostToDevice));
    }

    void copyLeavesDeviceToHost()
    {
        const uint leavesSize = leavesCount * sizeof(t_Leaf);

        CUDA_SAFE_CALL( 
            cudaMemcpy(cpuLeaves, gpuLeaves, leavesSize, cudaMemcpyDeviceToHost));
    }



    void bindDeviceDataToTexture()
    {
        //cells
        cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();

        CUDA_SAFE_CALL( cudaBindTextureToArray(texTopLevelCells, topLevelCellsArray, chanelFormatDesc) );

        //leaves
        cudaChannelFormatDesc chanelFormatDescLeaves =
            cudaCreateChannelDesc<uint2>();

        const uint leavesSize = leavesCount * sizeof(t_Leaf);

        CUDA_SAFE_CALL( cudaBindTexture(NULL, texLeafCells, (void*) gpuLeaves,
            chanelFormatDescLeaves, leavesSize) );
    }

    //////////////////////////////////////////////////////////////////////////
    //memory allocation
    //////////////////////////////////////////////////////////////////////////
    cudaPitchedPtr allocateHostCells()
    {
        checkResolution();

        CUDA_SAFE_CALL( cudaMallocHost((void**)&cpuCells,
            resX * resY * resZ * sizeof(t_Cell)));

        cudaPitchedPtr pitchedPtrCPUCells = 
            make_cudaPitchedPtr(cpuCells, resX * sizeof(t_Cell), resX, resY);

        cpuTopLevelCellsPitchPtr = (char*)pitchedPtrCPUCells.ptr;

        return pitchedPtrCPUCells;
    }

    cudaPitchedPtr allocateDeviceCells()
    {
        checkResolution();

        cudaPitchedPtr pitchedPtrGPUCells =
            make_cudaPitchedPtr(gpuCells, resX * sizeof(t_Cell), resX, resY);

        cudaExtent cellDataExtent = 
            make_cudaExtent(resX * sizeof(t_Cell), resY, resZ);

        CUDA_SAFE_CALL( cudaMalloc3D(&pitchedPtrGPUCells, cellDataExtent) );

        gpuTopLevelCellsPitchPtr = (char*)pitchedPtrGPUCells.ptr;

        return pitchedPtrGPUCells;
    }

    t_Leaf* allocateHostLeaves(const uint aNumLeaves)
    {
        leavesCount = aNumLeaves;
        
        CUDA_SAFE_CALL(cudaMallocHost((void**)&cpuLeaves, leavesCount * sizeof(uint2)));
        
        return cpuLeaves;
    }

    t_Leaf* allocateDeviceLeaves(const uint aNumLeaves)
    {
        leavesCount = aNumLeaves;

        CUDA_SAFE_CALL(cudaMalloc((void**)&gpuLeaves, leavesCount * sizeof(uint2)));

        return cpuLeaves;
    }

    void setDeviceLeavesToZero()
    {
        CUDA_SAFE_CALL( cudaMemset(gpuLeaves, 0, leavesCount * sizeof(uint2) ) );
    }

    void setDeviceCellsToZero(const cudaPitchedPtr& aDeviceCells)
    {
        CUDA_SAFE_CALL( cudaMemset(aDeviceCells.ptr, 0 ,
            aDeviceCells.pitch * resY * resZ ) );

        //does not work!
        //cudaExtent cellDataExtent = 
        //    make_cudaExtent(aDeviceCells.pitch, resY, resZ);
        //CUDA_SAFE_CALL( cudaMemset3D(aDeviceCells, 0, memExtent) );
    }

    //////////////////////////////////////////////////////////////////////////
    //memory deallocation
    //////////////////////////////////////////////////////////////////////////
    void freeCellMemoryDevice()
    {
        CUDA_SAFE_CALL( cudaFree(gpuTopLevelCellsPitchPtr) );
    }

    void freeCellMemoryHost()
    {
        CUDA_SAFE_CALL( cudaFreeHost(cpuTopLevelCellsPitchPtr) );
    }


    void freeLeafMemoryDevice()
    {
        CUDA_SAFE_CALL( cudaFree(gpuLeaves) );
    }

    void freeLeafMemoryHost()
    {
        CUDA_SAFE_CALL( cudaFreeHost(cpuLeaves) );
    }

    HOST void cleanup()
    {
        freeLeafMemoryDevice();
        CUDA_SAFE_CALL( cudaUnbindTexture(texLeafCells) );
        CUDA_SAFE_CALL( cudaFreeArray(topLevelCellsArray) );
        CUDA_SAFE_CALL( cudaUnbindTexture(texTopLevelCells) );

    }
    //////////////////////////////////////////////////////////////////////////
    //debug related
    //////////////////////////////////////////////////////////////////////////
    void checkResolution()
    {
        if (resX <= 0 || resY <= 0 || resZ <= 0)
        {
            cudastd::log::out << "Invalid grid resolution!" 
                << " Setting grid resolution to 32 x 32 x 32\n";
            resX = resY = resZ = 32;
        }
    }
};

#endif // TWOLEVELGRID_H_INCLUDED_3386E0EE_E8A3_469B_BA24_12AB64EB2939
