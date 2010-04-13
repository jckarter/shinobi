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

#ifndef SIMPLEGRID_H_INCLUDED_8D64DE12_C558_478D_ACC6_8E8E776002F0
#define SIMPLEGRID_H_INCLUDED_8D64DE12_C558_478D_ACC6_8E8E776002F0

#include "../../CUDAStdAfx.h"
#include "Grid.hpp"
#include "../../Core/Algebra.hpp"
#include "../../Primitive/BBox.hpp"

//////////////////////////////////////////////////////////////////////////
//textures
texture< uint2, 3, cudaReadModeElementType >        texGridCells;
//////////////////////////////////////////////////////////////////////////

struct SimpleGrid
{
    typedef uint2 Cell;

    int resX, resY, resZ;
    BBox bounds;
    Cell* cpuCells;
    Cell* gpuCells;
    //bookkeeping for deallocation
    char* gpuCellsPitchPtr;
    char* cpuCellsPitchPtr;
    cudaArray* cellArray;

     SimpleGrid()
         :resX(0), resY(0), resZ(0), bounds(BBox::empty()),
         cpuCells(NULL), gpuCells(NULL)
     {}

    //////////////////////////////////////////////////////////////////////////
    //traversal related
    //////////////////////////////////////////////////////////////////////////
    uint getCellId(uint aIdX, uint aIdY, uint aIdZ) const
    {
        return aIdX + aIdY * resX + aIdZ * resX * resY;
    }

    uint getCellId(const vec3i& aId) const
    {
        return aId.x + aId.y * resX + aId.z * resX * resY;
    }

    const Cell& getCell(uint aId) const
    {
        return cpuCells[aId];
    }

    const Cell& getCell(const vec3i& aId) const
    {
        return cpuCells[getCellId(aId)];
    }

    //////////////////////////////////////////////////////////////////////////
    //construction related
    //////////////////////////////////////////////////////////////////////////
    Cell& getCell(uint aId)
    {
        return cpuCells[aId];
    }

    const vec3f getResolution() const
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
        return getResolution() / (bounds.max - bounds.min);
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

    void copyCellsDeviceToHost(
        const cudaPitchedPtr& aHostCells,
        const cudaPitchedPtr& aDeviceCells)
    {
        cudaMemcpy3DParms cpyParamsDownloadPtr = { 0 };
        cpyParamsDownloadPtr.srcPtr  = aDeviceCells;
        cpyParamsDownloadPtr.dstPtr  = aHostCells;
        cpyParamsDownloadPtr.extent  = make_cudaExtent(resX * sizeof(Cell), resY, resZ);
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
        cpyParamsUploadPtr.extent  = make_cudaExtent(resX * sizeof(Cell), resY, resZ);
        cpyParamsUploadPtr.kind    = cudaMemcpyHostToDevice;

        CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParamsUploadPtr) );
    }

    void bindDeviceDataToTexture(const cudaPitchedPtr& aDeviceCells)
    {
        cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();
        cudaExtent res = make_cudaExtent(resX, resY, resZ);
        CUDA_SAFE_CALL( cudaMalloc3DArray(&cellArray, &chanelFormatDesc, res) );

        cudaMemcpy3DParms cpyParams = { 0 };
        cpyParams.srcPtr    = aDeviceCells;
        cpyParams.dstArray  = cellArray;
        cpyParams.extent    = res;
        cpyParams.kind      = cudaMemcpyDeviceToDevice;


        CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParams) );

        CUDA_SAFE_CALL( cudaBindTextureToArray(texGridCells, cellArray, chanelFormatDesc) );
    }

    void reBindDeviceDataToTexture(const cudaPitchedPtr& aDeviceCells, cudaStream_t& aStream)
    {
        CUDA_SAFE_CALL( cudaFreeArray(cellArray) );
        CUDA_SAFE_CALL( cudaUnbindTexture(texGridCells) );

        cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();
        cudaExtent res = make_cudaExtent(resX, resY, resZ);
        CUDA_SAFE_CALL( cudaMalloc3DArray(&cellArray, &chanelFormatDesc, res) );

        cudaMemcpy3DParms cpyParams = { 0 };
        cpyParams.srcPtr    = aDeviceCells;
        cpyParams.dstArray  = cellArray;
        cpyParams.extent    = res;
        cpyParams.kind      = cudaMemcpyDeviceToDevice;


        CUDA_SAFE_CALL( cudaMemcpy3DAsync(&cpyParams, aStream) );

        CUDA_SAFE_CALL( cudaBindTextureToArray(texGridCells, cellArray, chanelFormatDesc) );
    }

    void bindHostDataToTexture(const cudaPitchedPtr& aHostCells)
    {
        cudaChannelFormatDesc chanelFormatDesc = cudaCreateChannelDesc<uint2>();
        cudaExtent res = make_cudaExtent(resX, resY, resZ);
        CUDA_SAFE_CALL( cudaMalloc3DArray(&cellArray, &chanelFormatDesc, res) );

        cudaMemcpy3DParms cpyParams = { 0 };
        cpyParams.srcPtr    = aHostCells;
        cpyParams.dstArray  = cellArray;
        cpyParams.extent    = res;
        cpyParams.kind      = cudaMemcpyHostToDevice;

        CUDA_SAFE_CALL( cudaMemcpy3D(&cpyParams) );

        CUDA_SAFE_CALL( cudaBindTextureToArray(texGridCells, cellArray, chanelFormatDesc) );
    }

    //////////////////////////////////////////////////////////////////////////
    //memory allocation
    //////////////////////////////////////////////////////////////////////////
    cudaPitchedPtr allocateHostCells()
    {
        checkResolution();

        CUDA_SAFE_CALL( cudaMallocHost((void**)&cpuCells,
            resX * resY * resZ * sizeof(Cell)));

        cudaPitchedPtr pitchedPtrCPUCells = 
            make_cudaPitchedPtr(cpuCells, resX * sizeof(Cell), resX, resY);
        
        cpuCellsPitchPtr = (char*)pitchedPtrCPUCells.ptr;
        
        return pitchedPtrCPUCells;
    }

    cudaPitchedPtr allocateDeviceCells()
    {
        checkResolution();

        cudaPitchedPtr pitchedPtrGPUCells =
            make_cudaPitchedPtr(gpuCells, resX * sizeof(Cell), resX, resY);

        cudaExtent cellDataExtent = 
            make_cudaExtent(resX * sizeof(Cell), resY, resZ);

        CUDA_SAFE_CALL( cudaMalloc3D(&pitchedPtrGPUCells, cellDataExtent) );

        gpuCellsPitchPtr = (char*)pitchedPtrGPUCells.ptr;

        return pitchedPtrGPUCells;
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
        CUDA_SAFE_CALL( cudaFree(gpuCellsPitchPtr) );
    }

    void freeCellMemoryHost()
    {
        CUDA_SAFE_CALL( cudaFreeHost(cpuCellsPitchPtr) );
    }

    void cleanup()
    {
        CUDA_SAFE_CALL( cudaFreeArray(cellArray) );
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

#endif // SIMPLEGRID_H_INCLUDED_8D64DE12_C558_478D_ACC6_8E8E776002F0
