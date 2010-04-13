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

#ifndef SIMPLEINTEGRATOR_H_INCLUDED_BF62D38E_F9C5_45CB_A68E_12706DA07E07
#define SIMPLEINTEGRATOR_H_INCLUDED_BF62D38E_F9C5_45CB_A68E_12706DA07E07

#include "../CUDAConfig.h"
#include "../CUDAStdAfx.h"

#include "../Engine/RayTracingTypes.h"
#include "../Engine/RayTracingConstants.h"

#include "../Engine/RayBuffers.h"
#include "../Engine/RayTracingKernels.h"

template< 
    class tPrimitiveStorage,
    class tMaterialStorage,
    class tInputBuffer >
GLOBAL void simpleShade(
        tPrimitiveStorage       aStorage,
        tMaterialStorage        aMaterials,
        tInputBuffer            aInputBuffer,
        const uint              aNumRays,
        vec3f*                  oFrameBuffer,
        vec3f*                  oFinalImage,
        int*                    aGlobalMemoryPtr)
{
    extern SHARED uint sharedMem[];

    vec3f* rayOrg = (vec3f*)(sharedMem);
    vec3f* rayDir = rayOrg + RENDERTHREADSX * RENDERTHREADSY;

    for(uint myRayIndex = globalThreadId1D(); myRayIndex < aNumRays;
        myRayIndex += numThreads())
    {
        //////////////////////////////////////////////////////////////////////////
        //Initialization
        float rayT;
        uint  bestHit;

        aInputBuffer.load(rayOrg[threadId1D()], rayDir[threadId1D()],
            rayT, bestHit, myRayIndex, aNumRays);
        //////////////////////////////////////////////////////////////////////////


        vec3f& oRadiance = oFrameBuffer[myRayIndex];
        if (rayT < FLT_MAX)
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

            oRadiance =  diffReflectance * fabsf(rayDir[threadId1D()].dot(normal));
        }
        else
        {
            oRadiance.x = BACKGROUND_R;
            oRadiance.y = BACKGROUND_G;
            oRadiance.z = BACKGROUND_B;
        }

        float newSampleWeight = 1.f / (float)(dcImageId + 1);
        float oldSamplesWeight = 1.f - newSampleWeight;

        oFinalImage[myRayIndex] =
            oFinalImage[myRayIndex] * oldSamplesWeight +
            oFrameBuffer[myRayIndex] * newSampleWeight;

    }
}


    
template<
    class tPrimitiveStorage,
    class tPrimaryRayGenerator,
    template <class> class tPrimaryIntersector,
    template <class> class tAlternativeIntersector>

class SimpleIntegrator
{
    int* mGlobalMemoryPtr;

public:
    typedef tPrimaryRayGenerator            t_PrimaryRayGenerator;
    typedef SimpleRayBuffer                 t_RayBuffer;
    typedef tPrimaryIntersector<tPrimitiveStorage> t_Intersector;
    typedef Traverser_t<t_GridParameters, tPrimitiveStorage, t_Intersector> 
        t_Traverser;

    t_RayBuffer rayBuffer;

    SimpleIntegrator():rayBuffer(t_RayBuffer(NULL))
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
            t_Intersector::SHAREDMEMSIZE * 4u +                 //Intersector
            RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * 2 * sizeof(uint) + //Persistent threads    
            0u;

        const uint numRays = gRESX * gRESY;
        const uint globalMemorySize = sizeof(uint) +    //Persistent threads
            numRays * sizeof(vec3f) +                   //rayOrg
            numRays * sizeof(vec3f) +                   //rayDir
            numRays * sizeof(float) +                   //rayT
            numRays * sizeof(uint) +                    //primitive Id
            //gRESX * gRESY * NUMOCCLUSIONSAMPLES * sizeof(vec3f) + //light vector
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
        const cudaStream_t&     aStream = 0
        )
    {
        const uint sharedMemoryShade =
            RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //rayOrg
            RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +   //rayDir
            RENDERTHREADSY * 2 * sizeof(uint) +                 //Persistent threads    
            0u;

        dim3 threadBlockShade( RENDERTHREADSX, RENDERTHREADSY );
        dim3 blockGridShade  ( RENDERBLOCKSX, RENDERBLOCKSY );


        simpleShade<tPrimitiveStorage,
            t_MaterialStorage,
            t_RayBuffer>
            <<< blockGridShade, threadBlockShade, sharedMemoryShade, aStream >>>(
            aStorage,
            aMaterials,
            rayBuffer,
            gRESX*gRESY,
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

#endif // SIMPLEINTEGRATOR_H_INCLUDED_BF62D38E_F9C5_45CB_A68E_12706DA07E07
