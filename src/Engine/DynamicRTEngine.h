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

#ifndef DYNAMICRTENGINE_H_INCLUDED_0BB02AA0_F7C5_403D_A8F1_AC423AE33BAF
#define DYNAMICRTENGINE_H_INCLUDED_0BB02AA0_F7C5_403D_A8F1_AC423AE33BAF

#include "../CUDAConfig.h"
#include "../Shinobi.h"

class DynamicRTEngine
{
public:
    static uint                         sFrameId;
    static cudaStream_t                 sRenderEngineSreams[2];
    static Storage_t<0>                 sStorage0;
    static Storage_t<1>                 sStorage1;
    static t_Builder0                   sBuilder0;
    static t_Builder1                   sBuilder1;

    static t_Integrator0                sIntegrator0;
    static t_Integrator1                sIntegrator1;



    static void init(
        t_Grid&                         oGrid,
        const FWObject::t_FaceIterator& aBegin,
        const FWObject::t_FaceIterator& aEnd,
        const FWObject&                 aData)
    {
        sFrameId = 0u;
        cudaStreamCreate(&(sRenderEngineSreams[0]));
        cudaStreamCreate(&(sRenderEngineSreams[1])); 

        sBuilder0.init(oGrid, sStorage0, aBegin, aEnd, aData);
        sBuilder0.build(oGrid, sStorage0);

#if defined TWOLEVELGRID && __CUDA_ARCH__ < 120
        //prevents crash but still incorrect
        sBuilder1.init(oGrid, sStorage1, aBegin, aEnd, aData);
        sBuilder1.build(oGrid, sStorage1);
#endif // TWOLEVELGRID

        t_GridParameters gridParams = oGrid.getParameters();
        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcGridParameters", &gridParams,
            sizeof(t_GridParameters)) );
    }

    static void renderFrame(
        t_Grid&                 oGrid,
        const FWObject&         aKeyFrame1,
        const FWObject&         aKeyFrame2,
        const float             aCoeff,
        t_MaterialStorage&      aMaterials,
        FrameBuffer&            aFrameBuffer,
        FrameBuffer&            aFinalImage,
        const int               aImageId,
        float&                  oRenderTime,
        float&                  oBuildTime)
    {
        cudaEvent_t begin, build, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&build);
        cudaEventCreate(&end);
        cudaEventRecord(begin, 0);

        sFrameId = (sFrameId == 0u) ? 1u : 0u;
        const uint uploadStreamId = sFrameId;

        //////////////////////////////////////////////////////////////////////////
        //render and upload data for next frame (use separate stream)
        if (uploadStreamId == 0u)
        {
            sStorage0.uploadVertices(aKeyFrame1, aKeyFrame2, aCoeff,
                sRenderEngineSreams[0], oGrid.bounds.min, oGrid.bounds.max);

            for(int imageId = 0; imageId < NUMIMAGES; ++imageId)
            {
                sIntegrator1.integrate(sStorage1, aMaterials, aFrameBuffer,
                    aFinalImage, imageId, sRenderEngineSreams[1]);
            }
        }
        else
        {
            sStorage1.uploadVertices(aKeyFrame1, aKeyFrame2, aCoeff,
                sRenderEngineSreams[1], oGrid.bounds.min, oGrid.bounds.max);

            for(int imageId = 0; imageId < NUMIMAGES; ++imageId)
            {
                sIntegrator0.integrate(sStorage0, aMaterials, aFrameBuffer,
                    aFinalImage, imageId, sRenderEngineSreams[0]);
            }

        }
        //////////////////////////////////////////////////////////////////////////

        cudaEventRecord(build, 0);
        cudaEventSynchronize(build);

        //////////////////////////////////////////////////////////////////////////
        //build acceleration structure for next frame
        if (uploadStreamId == 0u)
        {
            sBuilder0.setNumPrimitives((uint)aKeyFrame2.getNumFaces());
            sBuilder0.rebuild(oGrid, sStorage0, sRenderEngineSreams[0]);

            t_GridParameters gridParams = oGrid.getParameters();
            CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcGridParameters", &gridParams,
                sizeof(t_GridParameters)) );

            oGrid.bounds = BBox::empty();
        }
        else
        {
            sBuilder1.setNumPrimitives((uint)aKeyFrame2.getNumFaces());
            sBuilder1.rebuild(oGrid, sStorage1, sRenderEngineSreams[1]);

            t_GridParameters gridParams = oGrid.getParameters();
            CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcGridParameters", &gridParams,
                sizeof(t_GridParameters)) );

            oGrid.bounds = BBox::empty();
        }
        //////////////////////////////////////////////////////////////////////////

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);

        cudaEventElapsedTime(&oRenderTime, begin, build);
        cudaEventElapsedTime(&oBuildTime, build, end);
        cudaEventDestroy(begin);
        cudaEventDestroy(build);
        cudaEventDestroy(end);
    }

    static void renderFrameSingleKernel(
        t_Grid&                 oGrid,
        const FWObject&         aKeyFrame1,
        const FWObject&         aKeyFrame2,
        const float             aCoeff,
        t_MaterialStorage&      aMaterials,
        FrameBuffer&            aFrameBuffer,
        FrameBuffer&            aFinalImage,
        const int               aImageId,
        float&                  oRenderTime,
        float&                  oBuildTime)
    {
        cudaEvent_t begin, build, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&build);
        cudaEventCreate(&end);
        cudaEventRecord(begin, 0);

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

        const uint numRays = gRESX * gRESY;

        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcNumRays", &numRays, sizeof(uint)) );

        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcImageId", &aImageId, sizeof(int)) );

        const uint renderStreamId = sFrameId;
        sFrameId = (sFrameId == 0u) ? 1u : 0u;
        const uint uploadStreamId = sFrameId;

        //////////////////////////////////////////////////////////////////////////
        //render and upload data for next frame (use separate stream)
        if (uploadStreamId == 0u)
        {
            sStorage0.uploadVertices(aKeyFrame1, aKeyFrame2, aCoeff,
                sRenderEngineSreams[0], oGrid.bounds.min, oGrid.bounds.max);

            for(int imageId = 0; imageId < NUMIMAGES; ++imageId)
            {
                render<<< blockGrid, threadBlock,
                    sharedMemorySize, sRenderEngineSreams[renderStreamId] >>>(
                    sStorage1,
                    aMaterials,
                    aFrameBuffer.deviceData,
                    aFinalImage.deviceData,
                    globalMemoryPtr);

                CUT_CHECK_ERROR("Kernel Execution failed.\n");

                CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcImageId", &imageId, sizeof(int)) );
            }
        }
        else
        {
            sStorage1.uploadVertices(aKeyFrame1, aKeyFrame2, aCoeff,
                sRenderEngineSreams[1], oGrid.bounds.min, oGrid.bounds.max);

            for(int imageId = 0; imageId < NUMIMAGES; ++imageId)
            {
                render<<< blockGrid, threadBlock,
                    sharedMemorySize, sRenderEngineSreams[renderStreamId] >>>(
                    sStorage0,
                    aMaterials,
                    aFrameBuffer.deviceData,
                    aFinalImage.deviceData,
                    globalMemoryPtr);

                CUT_CHECK_ERROR("Kernel Execution failed.\n");

                CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcImageId", &imageId, sizeof(int)) );
            }

        }
        //////////////////////////////////////////////////////////////////////////

        cudaEventRecord(build, 0);
        cudaEventSynchronize(build);

        //////////////////////////////////////////////////////////////////////////
        //build acceleration structure for next frame
        if (uploadStreamId == 0u)
        {
            sBuilder0.setNumPrimitives((uint)aKeyFrame2.getNumFaces());
            sBuilder0.rebuild(oGrid, sStorage0, sRenderEngineSreams[0]);

            t_GridParameters gridParams = oGrid.getParameters();
            CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcGridParameters", &gridParams,
                sizeof(t_GridParameters)) );

            oGrid.bounds = BBox::empty();
        }
        else
        {
            sBuilder1.setNumPrimitives((uint)aKeyFrame2.getNumFaces());
            sBuilder1.rebuild(oGrid, sStorage1, sRenderEngineSreams[1]);

            t_GridParameters gridParams = oGrid.getParameters();
            CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcGridParameters", &gridParams,
                sizeof(t_GridParameters)) );

            oGrid.bounds = BBox::empty();
        }
        //////////////////////////////////////////////////////////////////////////


        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);

        CUDA_SAFE_CALL( cudaFree(globalMemoryPtr) );

        cudaEventElapsedTime(&oRenderTime, begin, build);
        cudaEventElapsedTime(&oBuildTime, build, end);
        cudaEventDestroy(begin);
        cudaEventDestroy(build);
        cudaEventDestroy(end);
    }

};


#endif // DYNAMICRTENGINE_H_INCLUDED_0BB02AA0_F7C5_403D_A8F1_AC423AE33BAF
