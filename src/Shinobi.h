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

#ifndef SHINOBI_H_INCLUDED_7B6264F6_C50C_4293_A1EC_5CDA5D62F93A
#define SHINOBI_H_INCLUDED_7B6264F6_C50C_4293_A1EC_5CDA5D62F93A

#include "CUDAStdAfx.h"

#include "CUDAConfig.h"

#include "Engine/RayTracingTypes.h"
#include "Engine/RayTracingConstants.h"
#include "Engine/RayGenerators.h"

#include "Integrator/SimpleIntegrator.hpp"
#include "Integrator/BasicPathTracer.hpp"
#include "Integrator/SimplePathTracer.hpp"
#include "Integrator/IndirectIntegrator.hpp"

#include "Integrator/SimpleIntegrator.h"
#include "Integrator/DirectIlluminationIntegrator.h"

#include "Utils/RandomNumberGenerators.hpp"

#ifdef ANIMATION
# define RayGenerator_t                 RegularPrimaryRayGenerator
#else
# define RayGenerator_t                 RandomPrimaryRayGenerator
#endif

typedef RayGenerator_t<true>            t_PrimaryRayGenerator;
typedef RayGenerator_t<true>            t_PilotRayGenerator;

#if __CUDA_ARCH__ >= 120
#   define Intersector_t                SimpleIntersector
#   define PrimaryIntersector_t         SimpleIntersector
#   define AlternativeIntersector_t     HybridIntersector
#   define PilotRayIntersector_t        PilotRayIntersector
#else
#   define Intersector_t                SimpleIntersector
#   define PrimaryIntersector_t         SimpleIntersector
#   define AlternativeIntersector_t     SimpleIntersector
#   define PilotRayIntersector_t        SimpleIntersector

#endif


#if defined TWOLEVELGRID && defined LAZYBUILD && (__CUDA_ARCH__ >= 120) 

typedef LazyTwoLevelGridBuilder< Storage_t<0>, t_PilotRayGenerator,
    t_GridParameters, PilotRayIntersector_t >         t_Builder0;
typedef LazyTwoLevelGridBuilder< Storage_t<1>, t_PilotRayGenerator,
    t_GridParameters, PilotRayIntersector_t >         t_Builder1;

#endif //TWOLEVELGRID && LAZYBUILD && __CUDA_ARCH__ >= 120



#define Integrator_t                    DirectIlluminationIntegrator

typedef Integrator_t<
        t_PrimitiveStorage0,
        t_PrimaryRayGenerator,
        PrimaryIntersector_t,
        AlternativeIntersector_t>       t_Integrator;

typedef Integrator_t<
        t_PrimitiveStorage0,
        t_PrimaryRayGenerator,
        PrimaryIntersector_t,
        AlternativeIntersector_t>       t_Integrator0;

typedef Integrator_t<
        t_PrimitiveStorage1,
        t_PrimaryRayGenerator,
        PrimaryIntersector_t,
        AlternativeIntersector_t>       t_Integrator1;


#ifdef ANIMATION
#   define DeviceIntegrator_t           DeviceSimpleIntegrator
#else
#   define DeviceIntegrator_t           SimplePathTracer
#endif

typedef DeviceIntegrator_t<
        AreaLightSource,
        t_GridParameters,
        t_PrimitiveStorage0,
        t_MaterialStorage,
        SimpleTraverser_t,
        Intersector_t >                 t_DeviceIntegrator;

////////////////////////////////////////////////////////////////////////////////
#pragma region Ray Tracing Kernel
////////////////////////////////////////////////////////////////////////////////

DEVICE void initCameraRay(vec3f& aRayOrg, vec3f& aRayDir, const float aRayId)
{
    aRayOrg = dcCamera.getPosition();

    float screenX, screenY;
    dcRegularPixelSampler.getPixelCoords(aRayId, (float)dcImageId,
        screenX, screenY);
    aRayDir = dcCamera.getDirection(screenX, screenY);

}

DEVICE void initCameraRayRandom(vec3f& aRayOrg, vec3f& aRayDir, const float aRayId)
{
    typedef KISSRandomNumberGenerator       t_RNG;


    aRayOrg = dcCamera.getPosition();

    t_RNG genRand(  1236789u + aRayId * 977 + aRayId,
        369u + aRayId * 35537 + globalThreadId1D(719),
        351288629u + aRayId + globalThreadId1D(45751),
        416191069u );

    float screenX = genRand();
    float screenY = genRand();
    dcRandomPixelSampler.getPixelCoords(aRayId, screenX, screenY);
    aRayDir = dcCamera.getDirection(screenX, screenY);

}

template< class tPrimitiveStorage >
GLOBAL void render(
                   tPrimitiveStorage aStorage,
                   t_MaterialStorage aMaterialStorage,
                   vec3f* oFrameBuffer,
                   vec3f* oFinalImageBuffer,
                   int* aGlobalMemoryPtr)
{
    extern SHARED uint sharedMem[];

    vec3f* rayOrg =
        (vec3f*)(sharedMem + t_DeviceIntegrator::t_Intersector::SHAREDMEMSIZE);
    vec3f* rayDir = rayOrg + RENDERTHREADSX * RENDERTHREADSY;

#if __CUDA_ARCH__ >= 110
    volatile uint*  nextRayArray = (uint*)(rayDir + RENDERTHREADSX * RENDERTHREADSY);
    volatile uint*  rayCountArray = nextRayArray + RENDERTHREADSY;

    if (threadId1DInWarp() == 0u)
    {
        rayCountArray[warpId()] = 0u;
    }

    volatile uint& localPoolNextRay = nextRayArray[warpId()];
    volatile uint& localPoolRayCount = rayCountArray[warpId()];

    while (true)
    {
        if (localPoolRayCount==0 && threadId1DInWarp() == 0)
        {
            localPoolNextRay = atomicAdd(&aGlobalMemoryPtr[0], BATCHSIZE);
            localPoolRayCount = BATCHSIZE;
        }
        // get rays from local pool
        uint myRayIndex = localPoolNextRay + threadId1DInWarp();
        if (ALL(myRayIndex >= dcNumRays))
        {
            return;
        }

        if (myRayIndex >= dcNumRays) //keep whole warp active
        {
            myRayIndex = dcNumRays;
        }

        if (threadId1DInWarp() == 0)
        {
            localPoolNextRay += WARPSIZE;
            localPoolRayCount -= WARPSIZE;
        }
#else
    for(uint myRayIndex = globalThreadId1D(); myRayIndex < dcNumRays;
        myRayIndex += numThreads())
    {
#endif
        initCameraRay(rayOrg[threadId1D()], rayDir[threadId1D()],
            (float)myRayIndex);

        DeviceIntegrator_t<
            AreaLightSource,
            t_GridParameters,
            tPrimitiveStorage,
            t_MaterialStorage,
            SimpleTraverser_t,
            Intersector_t >  ()(
            rayOrg,
            rayDir,
            dcGridParameters,
            aStorage,
            aMaterialStorage,
            dcLightSource,
            oFrameBuffer[myRayIndex],
            sharedMem);

        oFinalImageBuffer[myRayIndex] = oFinalImageBuffer[myRayIndex] +
            oFrameBuffer[myRayIndex] / (float) NUMIMAGES;

    }
}

template< class tPrimitiveStorage >
GLOBAL void renderIncremental(
                              tPrimitiveStorage aStorage,
                              t_MaterialStorage aMaterialStorage,
                              vec3f* oFrameBuffer,
                              vec3f* oFinalImageBuffer,
                              int* aGlobalMemoryPtr)
{
    extern SHARED uint sharedMem[];

    vec3f* rayOrg =
        (vec3f*)(sharedMem + t_DeviceIntegrator::t_Intersector::SHAREDMEMSIZE);
    vec3f* rayDir = rayOrg + RENDERTHREADSX * RENDERTHREADSY;

#if __CUDA_ARCH__ >= 110
    volatile uint*  nextRayArray = (uint*)(rayDir + RENDERTHREADSX * RENDERTHREADSY);
    volatile uint*  rayCountArray = nextRayArray + RENDERTHREADSY;

    if (threadId1DInWarp() == 0u)
    {
        rayCountArray[warpId()] = 0u;
    }

    volatile uint& localPoolNextRay = nextRayArray[warpId()];
    volatile uint& localPoolRayCount = rayCountArray[warpId()];

    while (true)
    {
        if (localPoolRayCount==0 && threadId1DInWarp() == 0)
        {
            localPoolNextRay = atomicAdd(&aGlobalMemoryPtr[0], BATCHSIZE);
            localPoolRayCount = BATCHSIZE;
        }
        // get rays from local pool
        uint myRayIndex = localPoolNextRay + threadId1DInWarp();
        if (ALL(myRayIndex >= dcNumRays))
        {
            return;
        }

        if (myRayIndex >= dcNumRays) //keep whole warp active
        {
            myRayIndex = dcNumRays;
        }

        if (threadId1DInWarp() == 0)
        {
            localPoolNextRay += WARPSIZE;
            localPoolRayCount -= WARPSIZE;
        }
#else
    for(uint myRayIndex = globalThreadId1D(); myRayIndex < dcNumRays;
        myRayIndex += numThreads())
    {
#endif
        initCameraRayRandom(rayOrg[threadId1D()], rayDir[threadId1D()],
            (float)myRayIndex);

        int seed = (int)(myRayIndex  + dcImageId * 103333);

        DeviceIntegrator_t<
            AreaLightSource,
            t_GridParameters,
            tPrimitiveStorage,
            t_MaterialStorage,
            SimpleTraverser_t,
            Intersector_t >  ()(
            rayOrg,
            rayDir,
            dcGridParameters,
            aStorage,
            aMaterialStorage,
            dcLightSource,
            oFrameBuffer[myRayIndex],
            sharedMem,
            seed);

        float newSampleWeight = 1.f / (float)(dcImageId + 1);
        float oldSamplesWeight = 1.f - newSampleWeight;

        oFinalImageBuffer[myRayIndex] =
            oFinalImageBuffer[myRayIndex] * oldSamplesWeight +
            oFrameBuffer[myRayIndex] * newSampleWeight;
    }
}
////////////////////////////////////////////////////////////////////////////////
#pragma endregion // Ray Tracing Kernel
////////////////////////////////////////////////////////////////////////////////

#endif // SHINOBI_H_INCLUDED_7B6264F6_C50C_4293_A1EC_5CDA5D62F93A
