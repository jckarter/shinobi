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

#ifndef DIRECTILLUMINATIONINTEGRATOR_H_INCLUDED_1CB35551_BD2F_4854_9CD4_552CA5AA80A0
#define DIRECTILLUMINATIONINTEGRATOR_H_INCLUDED_1CB35551_BD2F_4854_9CD4_552CA5AA80A0

#include "../CUDAConfig.h"
#include "../CUDAStdAfx.h"

#include "../Engine/RayTracingTypes.h"
#include "../Engine/RayTracingConstants.h"

#include "../Engine/RayBuffers.h"
#include "../Engine/RayTracingKernels.h"

#include "SimpleIntegrator.h"


#define  NUMOCCLUSIONSAMPLES 4

//assumes that block size is multiple of number of samples per area light
template< 
    class tPrimitiveStorage,
    class tMaterialStorage,
    class tInputBuffer,
    class tOcclusionBuffer>
GLOBAL void computeDirectIllumination(
                        tPrimitiveStorage       aStorage,
                        tMaterialStorage        aMaterials,
                        tInputBuffer            aInputBuffer,
                        tOcclusionBuffer        aOcclusionBuffer,
                        vec3f*                  oFrameBuffer,
                        vec3f*                  oFinalImage,
                        int*                    aGlobalMemoryPtr)
{
    extern SHARED uint sharedMem[];
    vec3f* sharedVec = (vec3f*)sharedMem;

    vec3f rayOrg;
    vec3f rayDir;

    for(uint myRayIndex = globalThreadId1D(); 
        myRayIndex < dcNumRays + blockSize() - 1;
        myRayIndex += numThreads())
    {

        const uint myPixelIndex = min(dcNumPixels - 1, myRayIndex / dcSamples);

        SYNCTHREADS;

        //////////////////////////////////////////////////////////////////////////
        //load occlusion information in shared memory
        if (myRayIndex < dcNumRays)
        {
            sharedVec[threadId1D()] = aOcclusionBuffer.loadLigtVec(myRayIndex);
        }

        //////////////////////////////////////////////////////////////////////////
        //Initialization
        float rayT;
        uint  bestHit;

        aInputBuffer.load(rayOrg, rayDir, rayT, bestHit,
            myPixelIndex, dcNumPixels);
        //////////////////////////////////////////////////////////////////////////


        SYNCTHREADS;

        if (rayT < FLT_MAX)
        {
            if (myRayIndex < dcNumRays && sharedVec[threadId1D()].x < FLT_MAX)
            {
                bestHit = aStorage.indices[bestHit];
                vec3f normal = aStorage(bestHit).vertices[0];
                vec3f edge1 = aStorage(bestHit).vertices[1];
                vec3f edge2 = aStorage(bestHit).vertices[2];

                edge1 = edge1 - normal;
                edge2 = edge2 - normal;

                normal = ~(edge1 % edge2);

                float4 tmp = aMaterials.getDiffuseReflectance(
                    aStorage.getMaterialId(bestHit));

                vec3f diffReflectance;
                diffReflectance.x = tmp.x;
                diffReflectance.y = tmp.y;
                diffReflectance.z = tmp.z;

                float attenuation = 1.f /
                    sharedVec[threadId1D()].dot(sharedVec[threadId1D()]);

                float cosLightNormal = dcLightSource.normal.dot(
                    ~sharedVec[threadId1D()]);

                float cosNormalEyeDir = normal.dot(~sharedVec[threadId1D()]);

                //overwrite light vec with  the incident radiance
                sharedVec[threadId1D()] = 
                    dcLightSource.intensity *
                    dcLightSource.getArea() *
                    diffReflectance *
                    max(0.f, -cosLightNormal) *
                    max(0.f, cosNormalEyeDir) *
                    attenuation;
            }
            else if (sharedVec[threadId1D()].x == FLT_MAX)
            {
                sharedVec[threadId1D()] = vec3f::rep(0.f);
            }//endif point receives direct illumination
        }
        else
        {
            sharedVec[threadId1D()].x = BACKGROUND_R;
            sharedVec[threadId1D()].y = BACKGROUND_G;
            sharedVec[threadId1D()].z = BACKGROUND_B;
        }//endif hit point exists

        //one thread per pixel instead of per occlusion sample
        if (threadId1D() % dcSamples == 0u)
        {
            vec3f& oRadiance =
                oFrameBuffer[myPixelIndex];
            oRadiance = vec3f::rep(0.f);

            for(uint i = 0; i < dcSamples; ++i)
            {
                oRadiance =  oRadiance + sharedVec[threadId1D() + i] 
                / (float)dcSamples;
            }

            float newSampleWeight = 1.f / (float)(dcImageId + 1);
            float oldSamplesWeight = 1.f - newSampleWeight;

            oFinalImage[myPixelIndex] =
                oFinalImage[myPixelIndex] * oldSamplesWeight +
                oFrameBuffer[myPixelIndex] * newSampleWeight;
        }


    }
}

template<
    class tPrimitiveStorage,
    class tPrimaryRayGenerator,
        template <class> class tPrimaryIntersector,
        template <class> class tAlternativeIntersector >

class DirectIlluminationIntegrator
{

    int* mGlobalMemoryPtr;

public:
    typedef tPrimaryRayGenerator            t_PrimaryRayGenerator;
    typedef SimpleRayBuffer                 t_RayBuffer;
    typedef tPrimaryIntersector<tPrimitiveStorage> t_PrimaryIntersector;
    typedef tAlternativeIntersector<tPrimitiveStorage> t_AlternativeIntersector;
    typedef Traverser_t<t_GridParameters, tPrimitiveStorage, t_PrimaryIntersector> 
        t_Traverser;
    typedef ShadowTraverser_t<t_GridParameters, tPrimitiveStorage, t_PrimaryIntersector> 
        t_ShadowTraverser;


    t_RayBuffer rayBuffer;

    DirectIlluminationIntegrator():rayBuffer(t_RayBuffer(NULL))
    {}

    HOST void tracePrimary(
        tPrimitiveStorage&      aStorage,
        t_MaterialStorage&      aMaterials,
        FrameBuffer&            aFrameBuffer,
        FrameBuffer&            aFinalImage,
        const int               aImageId,
        const cudaStream_t&     aStream = 0
        )
    {
        const uint sharedMemoryTrace =
            //RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //rayOrg
            //RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //rayDir
            t_Traverser::SHAREDMEMSIZE * 4u +                   //Traverser
            t_PrimaryIntersector::SHAREDMEMSIZE * 4u +                 //Intersector
            RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * 2 * sizeof(uint) + //Persistent threads    
            0u;

        const uint numRays = gRESX * gRESY;
        const uint globalMemorySize = sizeof(uint) +    //Persistent threads
            numRays * sizeof(vec3f) +                   //rayOrg
            numRays * sizeof(vec3f) +                   //rayDir
            numRays * sizeof(float) +                   //rayT
            numRays * sizeof(uint) +                    //primitive Id
            gRESX * gRESY * NUMOCCLUSIONSAMPLES * sizeof(vec3f) + //light vector
            0u;

        CUDA_SAFE_CALL( cudaMalloc((void**)&mGlobalMemoryPtr, globalMemorySize));
        CUDA_SAFE_CALL( cudaMemset( mGlobalMemoryPtr, 0, sizeof(uint)) );

        dim3 threadBlockTrace( RENDERTHREADSX, RENDERTHREADSY );
        dim3 blockGridTrace  ( RENDERBLOCKSX, RENDERBLOCKSY );


        rayBuffer = t_RayBuffer(mGlobalMemoryPtr + 1);
        t_PrimaryRayGenerator primaryRayGenerator;

        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcImageId", &aImageId, sizeof(int)) );

        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcNumRays", &numRays, sizeof(uint)) );


        trace<tPrimitiveStorage,
            t_PrimaryRayGenerator,
            t_GridParameters,
            tPrimaryIntersector,
            Traverser_t,
            t_RayBuffer>
            <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace, aStream >>>(
            aStorage,
            primaryRayGenerator,
            rayBuffer,
            mGlobalMemoryPtr);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");
    }


    HOST void shade(
        tPrimitiveStorage&      aStorage,
        t_MaterialStorage&      aMaterials,
        FrameBuffer&            aFrameBuffer,
        FrameBuffer&            aFinalImage,
        const int               aImageId,
        const cudaStream_t&     aStream = 0)
    {
        typedef AreaLightShadowRayGenerator<t_RayBuffer>
            t_ShadowRayGenerator;
        typedef OcclusionRayBuffer  t_OcclusionBuffer;

        const uint sharedMemoryTrace =
            //RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //rayOrg
            //RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //rayDir
            t_ShadowTraverser::SHAREDMEMSIZE * 4u +                   //Traverser
            t_AlternativeIntersector::SHAREDMEMSIZE * 4u +                 //Intersector
            RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * 2 * sizeof(uint) + //Persistent threads    
            0u;

        const uint numShadowSamples = NUMOCCLUSIONSAMPLES;
        const uint numRays = gRESX * gRESY * NUMOCCLUSIONSAMPLES;
        const uint numPixels = gRESX * gRESY;

        CUDA_SAFE_CALL( cudaMemset( mGlobalMemoryPtr, 0, sizeof(uint)) );

        dim3 threadBlockTrace( RENDERTHREADSX, RENDERTHREADSY );
        dim3 blockGridTrace  ( RENDERBLOCKSX, RENDERBLOCKSY );


        rayBuffer = t_RayBuffer(mGlobalMemoryPtr + 1);
        t_ShadowRayGenerator shadowRayGenerator(rayBuffer);
        t_OcclusionBuffer occlusionBuffer(mGlobalMemoryPtr + 
            1 +                             //Persistent threads
            numPixels * 3 +                 //rayOrg
            numPixels * 3 +                 //rayDir
            numPixels +                     //rayT
            numPixels +                     //primitive Id
            0u);

        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcImageId", &aImageId, sizeof(int)) );

        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcNumRays", &numRays, sizeof(uint)) );

        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcSamples", &numShadowSamples, sizeof(uint)) );

        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcNumPixels", &numPixels, sizeof(uint)) );

        //trace shadow rays
        trace<tPrimitiveStorage,
            t_ShadowRayGenerator,
            t_GridParameters,
            tAlternativeIntersector,
            ShadowTraverser_t,
            t_OcclusionBuffer>
            <<< blockGridTrace, threadBlockTrace, sharedMemoryTrace, aStream >>>(
            aStorage,
            shadowRayGenerator,
            occlusionBuffer,
            mGlobalMemoryPtr);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        const uint sharedMemoryShade =
            RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //light vector   
            0u;

        dim3 threadBlockShade( RENDERTHREADSX, RENDERTHREADSY );
        dim3 blockGridShade  ( RENDERBLOCKSX, RENDERBLOCKSY );

        computeDirectIllumination<tPrimitiveStorage,
            t_MaterialStorage,
            t_RayBuffer, t_OcclusionBuffer>
            <<< blockGridShade, threadBlockShade, sharedMemoryShade, aStream >>>(
            aStorage,
            aMaterials,
            rayBuffer,
            occlusionBuffer,
            aFrameBuffer.deviceData,
            aFinalImage.deviceData,
            mGlobalMemoryPtr);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

    }

    HOST void cleanup()
    {
        CUDA_SAFE_CALL( cudaFree(mGlobalMemoryPtr));
    }

    HOST void integrate(
        tPrimitiveStorage&      aStorage,
        t_MaterialStorage&      aMaterials,
        FrameBuffer&            aFrameBuffer,
        FrameBuffer&            aFinalImage,
        const int               aImageId,
        const cudaStream_t&     aStream = 0
        )
    {

        tracePrimary(aStorage, aMaterials, aFrameBuffer,
            aFinalImage, aImageId, aStream);

        shade(aStorage, aMaterials, aFrameBuffer,
            aFinalImage, aImageId, aStream);

        cleanup();
    }
};

#endif // DIRECTILLUMINATIONINTEGRATOR_H_INCLUDED_1CB35551_BD2F_4854_9CD4_552CA5AA80A0
