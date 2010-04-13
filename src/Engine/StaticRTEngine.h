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

#ifndef STATICRTENGINE_H_INCLUDED_B01BFA00_C5C9_4E2C_9C5F_FB61A3D38833
#define STATICRTENGINE_H_INCLUDED_B01BFA00_C5C9_4E2C_9C5F_FB61A3D38833

#include "../CUDAConfig.h"
#include "../Shinobi.h"

class StaticRTEngine
{
public:

    static void renderImage(
        t_PrimitiveStorage0&    aTriangles,
        t_MaterialStorage&      aMaterials,
        FrameBuffer&            aFrameBuffer,
        FrameBuffer&            aFinalImage)
    {
        for(int imageId = 0; imageId < NUMIMAGES; ++imageId)
        {

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

           t_Integrator integrator;

           integrator.integrate(aTriangles, aMaterials, aFrameBuffer,
               aFinalImage, imageId);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            cudastd::log::out << "Image " << imageId;
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            cudastd::log::out << " rendering time: " << elapsedTime << "ms\r";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    static void renderFrame(
        t_PrimitiveStorage0&    aTriangles,
        t_MaterialStorage&      aMaterials,
        FrameBuffer&            aFrameBuffer,
        FrameBuffer&            aFinalImage,
        const int               aImageId,
        float&                  oRenderTime)
    {

        t_Integrator integrator;

        cudaEvent_t start, traceEvent, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&traceEvent);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        
        integrator.tracePrimary(aTriangles, aMaterials, aFrameBuffer,
            aFinalImage, aImageId);

        cudaEventRecord(traceEvent, 0);
        cudaEventSynchronize(traceEvent);

        integrator.shade(aTriangles, aMaterials, aFrameBuffer,
            aFinalImage, aImageId);

        integrator.cleanup();

        //integrator.integrate(aTriangles, aMaterials, aFrameBuffer, aFinalImage,
        //    aImageId);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&oRenderTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(traceEvent);
        cudaEventDestroy(stop);


    }

    static void renderImageSingleKernel(
        t_PrimitiveStorage0&    aTriangles,
        t_MaterialStorage&      aMaterials,
        FrameBuffer&            aFrameBuffer,
        FrameBuffer&            aFinalImage)
    {
        const uint numRays = gRESX * gRESY;

        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcNumRays", &numRays, sizeof(uint)) );

        const uint sharedMemorySize =
            RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //rayOrg
            RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //rayDir
            t_DeviceIntegrator::t_Intersector::SHAREDMEMSIZE * 4u +   //Intersector
            RENDERTHREADSY * 2 * sizeof(uint) +                 //Persistent threads    
            0u;

        const uint globalMemorySize = sizeof(uint); //Persistent threads global pool
        int* globalMemoryPtr;
        CUDA_SAFE_CALL( cudaMalloc((void**)&globalMemoryPtr, globalMemorySize));

        dim3 threadBlock( RENDERTHREADSX, RENDERTHREADSY );
        dim3 blockGrid  ( RENDERBLOCKSX, RENDERBLOCKSY );

        for(int imageId = 0; imageId < NUMIMAGES; ++imageId)
        {

            CUDA_SAFE_CALL( cudaMemset( globalMemoryPtr, 0, sizeof(uint)) );
            CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcImageId", &imageId, sizeof(int)) );

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            render<<< blockGrid, threadBlock, sharedMemorySize >>>(
                aTriangles,
                aMaterials,
                aFrameBuffer.deviceData,
                aFinalImage.deviceData,
                globalMemoryPtr);

            CUT_CHECK_ERROR("Kernel Execution failed.\n");

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            CUDA_SAFE_CALL( cudaFree(globalMemoryPtr) );

            cudastd::log::out << "Image " << imageId;
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            cudastd::log::out << " rendering time: " << elapsedTime << "ms\r";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    static void renderFrameSingleKernel(
        t_PrimitiveStorage0&    aTriangles,
        t_MaterialStorage&      aMaterials,
        FrameBuffer&            aFrameBuffer,
        FrameBuffer&            aFinalImage,
        const int               aImageId,
        float&                  oRenderTime)
    {
        const uint numRays = gRESX * gRESY;

        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcNumRays", &numRays, sizeof(uint)) );

        const uint sharedMemorySize =
            RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //rayOrg
            RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //rayDir
            t_DeviceIntegrator::t_Intersector::SHAREDMEMSIZE * 4u +   //Intersector
            RENDERTHREADSY * 2 * sizeof(uint) +                 //Persistent threads    
            0u;

        const uint globalMemorySize = sizeof(uint); //Persistent threads global pool
        int* globalMemoryPtr;
        CUDA_SAFE_CALL( cudaMalloc((void**)&globalMemoryPtr, globalMemorySize));
        CUDA_SAFE_CALL( cudaMemset( globalMemoryPtr, 0, sizeof(uint)) );

        dim3 threadBlock( RENDERTHREADSX, RENDERTHREADSY );
        dim3 blockGrid  ( RENDERBLOCKSX, RENDERBLOCKSY );

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcImageId", &aImageId, sizeof(int)) );

        renderIncremental<<< blockGrid, threadBlock, sharedMemorySize >>>(
            aTriangles,
            aMaterials,
            aFrameBuffer.deviceData,
            aFinalImage.deviceData,
            globalMemoryPtr);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");


        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        CUDA_SAFE_CALL( cudaFree(globalMemoryPtr));

        cudaEventElapsedTime(&oRenderTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    static void build(
        t_Grid&                         aGrid,
        t_PrimitiveStorage0&            aGeometry,
        const FWObject::t_FaceIterator& aBegin,
        const FWObject::t_FaceIterator& aEnd,
        const FWObject&                 aData)

    {

        t_Builder0               builder;
        builder.init(aGrid, aGeometry, aBegin, aEnd, aData);
        builder.build(aGrid, aGeometry);
    }
};


#endif // STATICRTENGINE_H_INCLUDED_B01BFA00_C5C9_4E2C_9C5F_FB61A3D38833
