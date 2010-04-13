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

#ifndef RAYTRACINGKERNELS_H_INCLUDED_34DF52E3_9D6B_4FAE_BF5E_BD8745B9CDB8
#define RAYTRACINGKERNELS_H_INCLUDED_34DF52E3_9D6B_4FAE_BF5E_BD8745B9CDB8

#include "../CUDAConfig.h"
#include "../CUDAStdAfx.h"

#include "RayTracingTypes.h"
#include "RayTracingConstants.h"

template< 
    class tPrimitiveStorage,
    class tRayGenerator,
    class tControlStructure,
    template <class> class tIntersector,
    template <class, class, class> class tTraverser,
    class tOutputBuffer >
GLOBAL void trace(
                   tPrimitiveStorage        aStorage,
                   tRayGenerator            aRayGenerator,
                   tOutputBuffer            oOutputBuffer,
                   int*                     aGlobalMemoryPtr)
{
    typedef tIntersector<tPrimitiveStorage>                     t_Intersector;
    typedef tTraverser<
        tControlStructure, tPrimitiveStorage, t_Intersector>    t_Traverser;

    extern SHARED uint sharedMem[];

    //vec3f* rayOrg =
    //    (vec3f*)(sharedMem + t_Intersector::SHAREDMEMSIZE);
    //vec3f* rayDir = rayOrg + RENDERTHREADSX * RENDERTHREADSY;

#if __CUDA_ARCH__ >= 110
    volatile uint*  nextRayArray = sharedMem;
    volatile uint*  rayCountArray = nextRayArray + RENDERTHREADSY;

    if (threadId1DInWarp32() == 0u)
    {
        rayCountArray[warpId32()] = 0u;
    }

    volatile uint& localPoolNextRay = nextRayArray[warpId32()];
    volatile uint& localPoolRayCount = rayCountArray[warpId32()];

    while (true)
    {
        if (localPoolRayCount==0 && threadId1DInWarp32() == 0)
        {
            localPoolNextRay = atomicAdd(&aGlobalMemoryPtr[0], BATCHSIZE);
            localPoolRayCount = BATCHSIZE;
        }
        // get rays from local pool
        uint myRayIndex = localPoolNextRay + threadId1DInWarp32();
        if (ALL(myRayIndex >= dcNumRays))
        {
            return;
        }

        if (myRayIndex >= dcNumRays) //keep whole warp active
        {
            myRayIndex = dcNumRays;
        }

        if (threadId1DInWarp32() == 0)
        {
            localPoolNextRay += WARPSIZE;
            localPoolRayCount -= WARPSIZE;
        }
#else
    for(uint myRayIndex = globalThreadId1D(); myRayIndex < dcNumRays;
        myRayIndex += numThreads())
    {
#endif

        //////////////////////////////////////////////////////////////////////////
        //Initialization
        t_Traverser     traverser;

        uint* sharedMemNew = sharedMem + RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * 2;
        vec3f rayOrg;
        vec3f& rayDir = (((vec3f*)sharedMemNew)[threadId1D32()]);

        float rayT   = aRayGenerator(rayOrg, rayDir, myRayIndex, dcNumRays);
        rayDir.x = 1.f / rayDir.x;
        rayDir.y = 1.f / rayDir.y;
        rayDir.z = 1.f / rayDir.z;

        uint  bestHit;
        //////////////////////////////////////////////////////////////////////////

#ifdef GATHERSTATISTICS
        vec3f dummy;
        traverser.traverse(rayOrg, rayT, bestHit, dcGridParameters, aStorage,
            sharedMemNew, dummy);

#else
        traverser.traverse(rayOrg, rayT, bestHit, dcGridParameters, aStorage, 
            sharedMemNew);
#endif

        rayDir.x = 1.f / rayDir.x;
        rayDir.y = 1.f / rayDir.y;
        rayDir.z = 1.f / rayDir.z;

        oOutputBuffer.store(rayOrg, rayDir, rayT, bestHit, myRayIndex, dcNumRays);


    }
}

#endif // RAYTRACINGKERNELS_H_INCLUDED_34DF52E3_9D6B_4FAE_BF5E_BD8745B9CDB8
