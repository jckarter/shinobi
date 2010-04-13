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

#ifndef SAHGRIDBUILDER_H_INCLUDED_3CF102C8_1F3D_49B6_BD8B_C3F6B71F0982
#define SAHGRIDBUILDER_H_INCLUDED_3CF102C8_1F3D_49B6_BD8B_C3F6B71F0982

#include "../../CUDAStdAfx.h"
#include "../../CUDAConfig.h"
#include "../Structure/SimpleGrid.h"
#include "../Structure/FaceSoup.h"
#include "../../Loaders/FWObject.hpp"
#include "FastBuildKernels.h"

template<class tPrimitiveStorage>
class SAHGridBuilder
{
public:
    static HOST uint getInstancesArraySize(
        uint                                            aNumPrimitives,
        tPrimitiveStorage&                            oFaceSoup,
        const SimpleGrid&                               oGrid )
    {
        //////////////////////////////////////////////////////////////////////////
        //timing
        //cudaEvent_t startEvent, deviceEvent, hostEvent;
        //cudaEventCreate(&startEvent);
        //cudaEventCreate(&deviceEvent);
        //cudaEventCreate(&hostEvent);
        //cudaEventRecord(startEvent, 0);
        //////////////////////////////////////////////////////////////////////////

        vec3f cellSize = (oGrid.bounds.max - oGrid.bounds.min) /
            oGrid.getResolution();

        vec3f cellSizeRCP = vec3f::rep(1.f) / cellSize;

        dim3 block(NUMBUILDTHREADS);
        dim3 grid (NUMBUILDBLOCKS);

        uint* hlpReduceDevice;
        CUDA_SAFE_CALL( cudaMalloc((void**)&hlpReduceDevice, grid.x * sizeof(uint)));

        countRefs < tPrimitiveStorage > <<< grid, block, block.x * sizeof(uint)>>>(
            oFaceSoup,
            aNumPrimitives,
            oGrid.getResolution(), 
            oGrid.bounds.min,
            cellSize,
            cellSizeRCP,
            hlpReduceDevice);

        //////////////////////////////////////////////////////////////////////////
        //timing
        //cudaEventRecord(deviceEvent, 0);
        //cudaEventSynchronize(deviceEvent);
        //////////////////////////////////////////////////////////////////////////

        //download intermediate reduction result
        uint* hlpReduceHost;
        CUDA_SAFE_CALL( cudaMallocHost((void**)&hlpReduceHost, grid.x * sizeof(uint)));
        CUDA_SAFE_CALL( cudaMemcpy( hlpReduceHost, hlpReduceDevice, grid.x * sizeof(uint),  cudaMemcpyDeviceToHost));

        uint totalSize = 0u;
        for (uint i = 0; i < block.x; ++i)
        {
            totalSize += hlpReduceHost[i]; 
        }

        CUDA_SAFE_CALL( cudaFree( hlpReduceDevice ));
        CUDA_SAFE_CALL( cudaFreeHost( hlpReduceHost ));

        //////////////////////////////////////////////////////////////////////////
        //timing
        //cudaEventRecord(hostEvent, 0);
        //cudaEventSynchronize(hostEvent);
        //cudastd::log::floatPrecision(4);
        //float deviceTime, hostTime;
        //cudaEventElapsedTime(&deviceTime, startEvent, deviceEvent);
        //cudaEventElapsedTime(&hostTime, deviceEvent, hostEvent);
        //cudastd::log::out << "Size estimation\n" 
        //    << "device: " << deviceTime
        //    << " host: "  << hostTime
        //    << " total: " << deviceTime + hostTime
        //    << " (ms)\n";
        //cudaEventDestroy(startEvent);
        //cudaEventDestroy(deviceEvent);
        //cudaEventDestroy(hostEvent);
        //////////////////////////////////////////////////////////////////////////

        return totalSize;
    }

    static HOST void computeResolution(
        uint                                            aNumPrimitives,
        SimpleGrid&                                     oGrid,
        tPrimitiveStorage&                            oFaceSoup)
    {
        vec3f diagonal = oGrid.bounds.max - oGrid.bounds.min;

        const float volume = diagonal.x * diagonal.y * diagonal.z;
        const float lambda = 5.f;
        const float magicConstant =
            powf(lambda * static_cast<float>(aNumPrimitives) / volume, 0.3333333f);

        diagonal *= magicConstant;

        int minResX = static_cast<int>(0.75f * diagonal.x);
        int minResY = static_cast<int>(0.75f * diagonal.y);
        int minResZ = static_cast<int>(0.75f * diagonal.z);
        int maxResX = static_cast<int>(1.25f * diagonal.x);
        int maxResY = static_cast<int>(1.25f * diagonal.y);
        int maxResZ = static_cast<int>(1.25f * diagonal.z);

        float bestCost = FLT_MAX;
        int bestResolutionX, bestResolutionY, bestResolutionZ;
        const vec3f boundsSize = (oGrid.bounds.max - oGrid.bounds.min);
        const float sceneBoundsArea =  2.f * ( 
            boundsSize.x * boundsSize.x + 
            boundsSize.y * boundsSize.y +
            boundsSize.z * boundsSize.z);
        int resolutionId = 0;
        const int numResolutions = (maxResX - minResX + 1) *
            (maxResY - minResY + 1) * (maxResZ - minResZ + 1);
        for (int resZ = minResZ; resZ <= maxResZ; ++resZ)
        {
            for (int resY = minResY; resY <= maxResY; ++resY)
            {
                for (int resX = minResX; resX <= maxResX; ++resX)
                {

                    oGrid.resX = resX;
                    oGrid.resY = resY;
                    oGrid.resZ = resZ;

                    const vec3f cellSize = (oGrid.bounds.max - oGrid.bounds.min) /
                        oGrid.getResolution();

                    const vec3f cellSizeRCP = vec3f::rep(1.f) / cellSize;
                    const float cellSurfaceArea = 2.f * ( 
                        cellSize.x * cellSize.x + 
                        cellSize.y * cellSize.y +
                        cellSize.z * cellSize.z);

                    const float resolutionCost = 
                        0.85f * static_cast<float>(getInstancesArraySize(aNumPrimitives, oFaceSoup, oGrid)) * cellSurfaceArea / sceneBoundsArea + //intersection
                        0.15f * (oGrid.getResolution().x + oGrid.getResolution().y + oGrid.getResolution().z); //traversal

                    if (resolutionCost < bestCost)
                    {
                        bestCost = resolutionCost;
                        bestResolutionX = resX;
                        bestResolutionY = resY;
                        bestResolutionZ = resZ;
                    }
                    cudastd::log::out << "Tested resolution : "
                        << resolutionId++
                        <<" of "
                        << numResolutions
                        << "\r";
                }
            }
        }

        oGrid.resX = bestResolutionX;
        oGrid.resY = bestResolutionY;
        oGrid.resZ = bestResolutionZ;


        cudastd::log::out << "\nDefault resolution : "
            << static_cast<int>(diagonal.x) << " "
            << static_cast<int>(diagonal.y) << " "
            << static_cast<int>(diagonal.z) << "\n";

        cudastd::log::out << "Best resolution    : "
            << bestResolutionX << " "
            << bestResolutionY << " "
            << bestResolutionZ << "\n";
    }
};


#endif // SAHGRIDBUILDER_H_INCLUDED_3CF102C8_1F3D_49B6_BD8B_C3F6B71F0982
