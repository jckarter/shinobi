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

#ifndef HYBRIDINTERSECTOR_HPP_INCLUDED_057C3BEE_0D14_4FDE_88A8_16F4CCAF091F
#define HYBRIDINTERSECTOR_HPP_INCLUDED_057C3BEE_0D14_4FDE_88A8_16F4CCAF091F

#include "../CUDAConfig.h"

#include "../Core/Algebra.hpp"

#define BUFFERSIZE 12

template<class tStorageStructure>
class PilotRayIntersector
{
public:
    //////////////////////////////////////////////////////////////////////////
    //amount of memory required for buffering
    //////////////////////////////////////////////////////////////////////////
    static const uint   SHAREDMEMSIZE   = 
        //RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE * 6 + //ray
        RENDERTHREADSX * RENDERTHREADSY +                           //intersectionT
        RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE +   //threadIdBuffer
        RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE +   //primIdBuffer
        RENDERTHREADSX * RENDERTHREADSY / WARPSIZE   +              //threadCount
        0u;
    static const uint   GLOBALMEMSIZE   =
        0u;
    //////////////////////////////////////////////////////////////////////////
    
    DEVICE void operator() (
        vec3f*                                          aRayOrg,
        vec3f*                                          aRayDir,
        float&                                          oRayT,
        uint2&                                          aIdRange,
        const tStorageStructure&                        aScene,
        uint*                                           aSharedMem,
        uint&                                           oBestHit) const
    {
        for(;;)//while(ANY(aIdRange.y > aIdRange.x))
        {
            //NOTE: Device capability 1.2 or higher required
            if (ANY(aIdRange.y == aIdRange.x) &&
                ANY(aIdRange.y - aIdRange.x >= HALFWARPSIZE))
            {
                //////////////////////////////////////////////////////////////////////////
                //Warp is incoherent and exists thread with large workload
                //////////////////////////////////////////////////////////////////////////
                
                //////////////////////////////////////////////////////////////////////////
                //Perform compaction to determine count and indices
                //of threads with large workload
                //Currently memory used to store rays is spilled in registers
                //////////////////////////////////////////////////////////////////////////
                uint* sharedMem = (uint*)(aRayOrg + warpId() * WARPSIZE);
                uint backup1 = sharedMem[threadId1DInWarp()];
                
                sharedMem[threadId1DInWarp()] = 0u;
                sharedMem += WARPSIZE;
                uint backup2 = sharedMem[threadId1DInWarp()];
                sharedMem[threadId1DInWarp()] =
                    (aIdRange.y - aIdRange.x >= HALFWARPSIZE) ? 1u : 0u;

                sharedMem[threadId1DInWarp()] += sharedMem[threadId1DInWarp() -  1];
                sharedMem[threadId1DInWarp()] += sharedMem[threadId1DInWarp() -  2];
                sharedMem[threadId1DInWarp()] += sharedMem[threadId1DInWarp() -  4];
                sharedMem[threadId1DInWarp()] += sharedMem[threadId1DInWarp() -  8];
                sharedMem[threadId1DInWarp()] += sharedMem[threadId1DInWarp() - 16];

                uint* threadCount         = aSharedMem +
                    RENDERTHREADSX * RENDERTHREADSY +                           //intersectionT
                    RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE +   //threadIdBuffer
                    RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE +   //primIdBuffer
                    0u;

                threadCount[warpId()] = sharedMem[WARPSIZE - 1];

                sharedMem[threadId1DInWarp()] = sharedMem[threadId1DInWarp() -  1];

                //////////////////////////////////////////////////////////////////////////
                //End of compaction, now insert ray & triangle indices in the buffer
                //////////////////////////////////////////////////////////////////////////

                uint* threadIds = aSharedMem +
                    RENDERTHREADSX * RENDERTHREADSY +                       //intersectionT
                    warpId() * BUFFERSIZE;

                uint* primIds = (aSharedMem +
                    RENDERTHREADSX * RENDERTHREADSY +                       //intersectionT
                    RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE) + //threadIdBuffer
                    warpId() * BUFFERSIZE;
                
                //Determine count and indices of threads with large workload
                if (aIdRange.y - aIdRange.x >= HALFWARPSIZE
                    && sharedMem[threadId1DInWarp()] < BUFFERSIZE)
                {
                    threadIds[sharedMem[threadId1DInWarp()]] = threadId1D();
                    primIds[sharedMem[threadId1DInWarp()]] = aIdRange.x;
                    //intersection with these follows
                    aIdRange.x += HALFWARPSIZE; 
                }

                //Spill back ray data
                sharedMem[threadId1DInWarp()] = backup2;
                sharedMem -= WARPSIZE;
                sharedMem[threadId1DInWarp()] = backup1;

                //////////////////////////////////////////////////////////////////////////
                //Compute Intersection (parallelize horizontal)
                //////////////////////////////////////////////////////////////////////////
                for (int numIterations = min(BUFFERSIZE, threadCount[warpId()]); 
                    numIterations >= 1;
                    numIterations -= 2)
                {
                    float* distances = (float*)(aSharedMem);

                    const uint primitiveId = 
                        (threadId1DInWarp() < HALFWARPSIZE) ?
                        primIds[numIterations - 1] + threadId1DInWarp() :
                        primIds[max(numIterations - 2, 0)] +
                            threadId1DInWarp() - HALFWARPSIZE;
                    //Moeller/Trumbore
                    vec3f org   = aScene[primitiveId].vertices[0];
                    vec3f edge1 = aScene[primitiveId].vertices[1];
                    vec3f edge2 = aScene[primitiveId].vertices[2];

                    edge1 = edge1 - org;
                    edge2 = edge2 - org;

                    const uint threadId =  (threadId1DInWarp() < HALFWARPSIZE) ?
                        threadIds[numIterations - 1] :
                    threadIds[max(numIterations - 2, 0)];

                    vec3f pvec      = aRayDir[threadId] % edge2;
                    float detRCP    = 1.f / edge1.dot(pvec);

                    vec3f tvec  = aRayOrg[threadId] - org;
                    float alpha = detRCP * tvec.dot(pvec);

                    tvec        = tvec % edge1;
                    float beta  = detRCP * tvec.dot(aRayDir[threadId]);

                    distances[threadId1D()] = detRCP * edge2.dot(tvec);

                    if(alpha < 0.f         ||
                        beta < 0.f         ||
                        alpha + beta > 1.f ||
                        distances[threadId1D()] <= 0.001f)
                    {
                        distances[threadId1D()] = FLT_MAX;
                    }

                    //////////////////////////////////////////////////////////////////////////
                    //Reduce to find best hit-point (currently sequential)
                    //////////////////////////////////////////////////////////////////////////
                    uint bestPrim = 0u;

                    distances +=
                        warpId() * WARPSIZE + 
                        ((threadId1DInWarp() < HALFWARPSIZE) ? 0 : HALFWARPSIZE);
                        //HALFWARPSIZE * isInSecondHalfWarp();

                    for (uint it = 1; it < HALFWARPSIZE; ++it)
                    {
                        bestPrim = 
                            (distances[it] < distances[bestPrim]) ? it : bestPrim;
                    }

                    //Prevents "warp serialize"
                    if (threadId1DInWarp() == 0)
                    {
                        distances[0] = distances[bestPrim];
                        primIds[numIterations - 1] += bestPrim;
                    }

                    //Prevents "warp serialize"
                    if (threadId1DInWarp() == HALFWARPSIZE && numIterations > 1)
                    {
                        distances[0] = distances[bestPrim];
                        primIds[numIterations - 2] += bestPrim;
                    }


                    //write out first hit
                    if (threadId1D() == threadIds[numIterations - 1])
                    {
                        distances -=
                            (threadId1DInWarp() < HALFWARPSIZE) ? 0 : HALFWARPSIZE;
                        if (oRayT > distances[0])
                        {
                            oRayT = distances[0];
                            oBestHit = primIds[numIterations - 1];
                        }
                    }
                    //write out second hit
                    if (threadId1D() == threadIds[max(numIterations - 2, 0)])
                    {
                        distances +=
                            (threadId1DInWarp() < HALFWARPSIZE) ? HALFWARPSIZE : 0;
                        if (oRayT > distances[0])
                        {
                            oRayT = distances[0];
                            oBestHit = primIds[max(numIterations - 2, 0)];
                        }
                    }

                }//end for intersection calculations

            }//end if (exists thread with large workload)
            else 
            {
                //////////////////////////////////////////////////////////////////////////
                //Warp is coherent or there are no threads with large workload
                //////////////////////////////////////////////////////////////////////////
                intersect(
                    aRayOrg,
                    aRayDir,
                    oRayT,
                    aIdRange,
                    aScene,
                    aSharedMem,
                    oBestHit);
            }//end if (exists thread with some workload)

            if (ALL( aIdRange.y == aIdRange.x ))
            {
                return;
            }
        }
    }
private:
    DEVICE void intersect (
        vec3f*                                          aRayOrg,
        vec3f*                                          aRayDir,
        float&                                          oRayT,
        uint2&                                          aIdRange,
        const tStorageStructure&                        aScene,
        uint*                                           aDummy,
        uint&                                           oBestHit) const
    {
        if (aIdRange.y > aIdRange.x)
        {
            const uint primitiveId = aIdRange.x++;

            vec3f org   = aScene[primitiveId].vertices[0];
            vec3f edge1 = aScene[primitiveId].vertices[1];
            vec3f edge2 = aScene[primitiveId].vertices[2];

            edge1 = edge1 - org;
            edge2 = edge2 - org;

            vec3f pvec      = aRayDir[threadId1D()] % edge2;
            float detRCP    = 1.f / edge1.dot(pvec);

            //if(fabsf(detRCP) <= EPS_RCP) terminate

            vec3f tvec  = aRayOrg[threadId1D()] - org;
            float alpha = detRCP * tvec.dot(pvec);

            //if(alpha < 0.f || alpha > 1.f) terminate

            tvec        = tvec % edge1;
            float beta  = detRCP * tvec.dot(aRayDir[threadId1D()]);

            //if(beta < 0.f || beta + alpha > 1.f) terminate

            float dist  = detRCP * edge2.dot(tvec);

            if (alpha >= 0.f        &&
                beta >= 0.f         &&
                alpha + beta <= 1.f &&
                dist > 0.001f       &&
                dist < oRayT)
            {
                oRayT   = dist;
                oBestHit = primitiveId;
            }
        }
    }

    DEVICE void intersectCached (
        vec3f*                                          aRayOrg,
        vec3f*                                          aRayDir,
        float&                                          oRayT,
        uint2&                                          aIdRange,
        const tStorageStructure&                        aScene,
        uint*                                           aSharedMem,
        uint&                                           oBestHit) const
    {
        //////////////////////////////////////////////////////////////////////////
        //Load 3 triangles in shared memory
        //////////////////////////////////////////////////////////////////////////
        uint* primIds = (aSharedMem +
            RENDERTHREADSX * RENDERTHREADSY +                       //intersectionT
            RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE) + //threadIdBuffer
            warpId() * BUFFERSIZE;

        uint* threadCount         = aSharedMem +
            RENDERTHREADSX * RENDERTHREADSY +                           //intersectionT
            RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE +   //threadIdBuffer
            RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE +   //primIdBuffer
            warpId();

        if (threadId1DInWarp() == 0u)
        {
            *threadCount = 0u;
        }

        //uint* sharedMem = (uint*)(aRayOrg + warpId() * WARPSIZE);
        //uint backup1 = sharedMem[threadId1DInWarp()];

        //sharedMem[threadId1DInWarp()] = 0u;
        //sharedMem += WARPSIZE;
        //uint backup2 = sharedMem[threadId1DInWarp()];
        //sharedMem[threadId1DInWarp()] =
        //    (aIdRange.y > aIdRange.x) ? 1u : 0u;

        //sharedMem[threadId1DInWarp()] += sharedMem[threadId1DInWarp() -  1];
        //sharedMem[threadId1DInWarp()] += sharedMem[threadId1DInWarp() -  2];
        //sharedMem[threadId1DInWarp()] += sharedMem[threadId1DInWarp() -  4];
        //sharedMem[threadId1DInWarp()] += sharedMem[threadId1DInWarp() -  8];
        //sharedMem[threadId1DInWarp()] += sharedMem[threadId1DInWarp() - 16];

        Triangle* cachedPrimitive = (Triangle*)(aSharedMem + warpId() * WARPSIZE);

        const uint primitiveId = aIdRange.x;
        if (aIdRange.y > aIdRange.x)
        {
            uint slotId = atomicAdd(threadCount, 1u);
            if (slotId < 3)
            {
                primIds[slotId] = primitiveId;
                cachedPrimitive[slotId] = aScene[primitiveId];
            }
        }

        //Spill back ray data
        //sharedMem[threadId1DInWarp()] = backup2;
        //sharedMem -= WARPSIZE;
        //sharedMem[threadId1DInWarp()] = backup1;
        //////////////////////////////////////////////////////////////////////////
        //Intersect
        //////////////////////////////////////////////////////////////////////////

        if (aIdRange.y > aIdRange.x)
        {
            ++aIdRange.x;

            vec3f org, edge1, edge2;
            if (primitiveId == primIds[0])
            {
                org     = cachedPrimitive[0].vertices[0];
                edge1   = cachedPrimitive[0].vertices[1];
                edge2   = cachedPrimitive[0].vertices[2];

            }
            else if (primitiveId == primIds[1])
            {
                org     = cachedPrimitive[1].vertices[0];
                edge1   = cachedPrimitive[1].vertices[1];
                edge2   = cachedPrimitive[1].vertices[2];
            }
            else if (primitiveId == primIds[2])
            {
                org     = cachedPrimitive[2].vertices[0];
                edge1   = cachedPrimitive[2].vertices[1];
                edge2   = cachedPrimitive[2].vertices[2];
            }
            else
            {
                org     = aScene[primitiveId].vertices[0];
                edge1   = aScene[primitiveId].vertices[1];
                edge2   = aScene[primitiveId].vertices[2];
            }

            edge1 = edge1 - org;
            edge2 = edge2 - org;

            vec3f pvec      = aRayDir[threadId1D()] % edge2;
            float detRCP    = 1.f / edge1.dot(pvec);

            //if(fabsf(detRCP) <= EPS_RCP) terminate

            vec3f tvec  = aRayOrg[threadId1D()] - org;
            float alpha = detRCP * tvec.dot(pvec);

            //if(alpha < 0.f || alpha > 1.f) terminate

            tvec        = tvec % edge1;
            float beta  = detRCP * tvec.dot(aRayDir[threadId1D()]);

            //if(beta < 0.f || beta + alpha > 1.f) terminate

            float dist  = detRCP * edge2.dot(tvec);

            if (alpha >= 0.f        &&
                beta >= 0.f         &&
                alpha + beta <= 1.f &&
                dist > 0.001f       &&
                dist < oRayT)
            {
                oRayT   = dist;
                oBestHit = primitiveId;
            }
        }
    }
};

//Shevtsov
//uint triId = aScene.indices[primitiveId];
//float4 data1 = aScene.getAccelDataChunck(3 * triId);   // nu,  nv,  np, pu
//float4 data2 = aScene.getAccelDataChunck(3 * triId + 1); // pv,  e0u, e0v, e1u
//float4 data3 = aScene.getAccelDataChunck(3 * triId + 2); // e1v, dimW, dimU, dimV

//int dimW = floatAsInt(data3.y);
//int dimU = floatAsInt(data3.z);
//int dimV = floatAsInt(data3.w);

//const uint threadId =  (threadId1DInWarp() < HALFWARPSIZE) ?
//    threadIds[numIterations - 1] :
//    threadIds[max(numIterations - 2, 0)];

//float det  = (aRayDir[threadId][dimU] * data1.x) +
//    (aRayDir[threadId][dimV] * data1.y) +
//    aRayDir[threadId][dimW];

//float dett = data1.z - ((aRayOrg[threadId][dimU] * data1.x) +
//    (aRayOrg[threadId][dimV] * data1.y) +
//    aRayOrg[threadId][dimW]);

//float Du   = aRayDir[threadId][dimU] * dett -
//    (data1.w - aRayOrg[threadId][dimU]) * det;
//float Dv   = aRayDir[threadId][dimV] * dett -
//    (data2.x - aRayOrg[threadId][dimV]) * det;
//float detu = (data3.x * Du) - (data2.w * Dv);
//float detv = (data2.y * Dv) - (data2.z * Du);

//uint mask = 
//    (floatAsInt(det - detu - detv) ^ floatAsInt(detu)) |
//    (floatAsInt(detu) ^ floatAsInt(detv));

//distances[threadId1D()] = fastDivide(dett, det);
//if (mask >> 31 || distances[threadId1D()] <= 0.001f)
//{
//    distances[threadId1D()] = FLT_MAX;
//}

template<class tStorageStructure>
class HybridIntersector
{
public:
    //////////////////////////////////////////////////////////////////////////
    //amount of memory required for buffering
    //////////////////////////////////////////////////////////////////////////
    static const uint   SHAREDMEMSIZE   = 
        RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE * 3 + //origins
        RENDERTHREADSX * RENDERTHREADSY +                           //intersectionT
        RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE +   //threadIdBuffer
        RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE +   //primIdBuffer
        RENDERTHREADSX * RENDERTHREADSY / WARPSIZE   +              //threadCount
        0u;
    static const uint   GLOBALMEMSIZE   =
        0u;
    //////////////////////////////////////////////////////////////////////////

    enum{
        ORIGINS_OFFSET      =   RENDERTHREADSX * RENDERTHREADSY * 3,
        
        RAY_DISTANCE_OFFSET =   RENDERTHREADSX * RENDERTHREADSY * 3 + //directions
                                RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE * 3, //origins
        
        THREAD_IDS_OFFSET   =   RENDERTHREADSX * RENDERTHREADSY * 3 +                       //directions
                                RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE * 3 + //origins
                                RENDERTHREADSX * RENDERTHREADSY,                       //intersectionT
        
        PRIMITIVE_IDS_OFFSET=   RENDERTHREADSX * RENDERTHREADSY * 3 +                       //directions
                                RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE * 3 + //origins
                                RENDERTHREADSX * RENDERTHREADSY +                       //intersectionT
                                RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE, //threadIdBuffer
        
        THREAD_COUNT_OFFSET =   RENDERTHREADSX * RENDERTHREADSY * 3 +                       //directions
                                RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE * 3 + //origins
                                RENDERTHREADSX * RENDERTHREADSY +                           //intersectionT
                                RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE +   //threadIdBuffer
                                RENDERTHREADSX * RENDERTHREADSY / WARPSIZE * BUFFERSIZE,   //primIdBuffer
    };

    DEVICE void operator() (
        const vec3f&                                    aRayOrg,
        const vec3f&                                    aRayDirRCP,
        float&                                          oRayT,
        uint2&                                          aIdRange,
        const tStorageStructure&                        aScene,
        uint*                                           aSharedMem,
        uint&                                           oBestHit) const
    {
 
        if (ANY(aIdRange.y - aIdRange.x >= HALFWARPSIZE) && 
            ANY(aIdRange.y - aIdRange.x < HALFWARPSIZE))
        {
            int* threadCount   = (int*)aSharedMem + THREAD_COUNT_OFFSET + warpId32();

            if (threadId1DInWarp32() == 0)
            {
                *threadCount = 0;
            }

            vec3f* origins = (vec3f*)(aSharedMem + ORIGINS_OFFSET) + 
                warpId32() * BUFFERSIZE;

            uint* threadIds = aSharedMem + THREAD_IDS_OFFSET + 
                warpId32() * BUFFERSIZE;

            uint* primIds = aSharedMem + PRIMITIVE_IDS_OFFSET +
                warpId32() * BUFFERSIZE;

            //Determine count and indices of threads with large workload
            if (aIdRange.y - aIdRange.x >= HALFWARPSIZE)
            {
                int slotId = atomicAdd(threadCount, 1);
                if (slotId < BUFFERSIZE)
                {
                    origins[slotId] = aRayOrg;
                    threadIds[slotId] = threadId1DInWarp32();
                    primIds[slotId] = aIdRange.x;
                    //intersection with these follows
                    aIdRange.x += HALFWARPSIZE; 
                }
            }

            //////////////////////////////////////////////////////////////////////////
            //Compute Intersection (parallelize horizontal)
            //////////////////////////////////////////////////////////////////////////
            *threadCount = min(BUFFERSIZE, *threadCount);
            for (int& numIterations = *threadCount;
                numIterations >= 1;
                numIterations -= 2)
            {                    
                const uint bufferPos = (threadId1DInWarp32() < HALFWARPSIZE) ?
                    numIterations - 1 : max(numIterations - 2, 0);

                const uint primitiveId = primIds[bufferPos] +
                    (threadId1DInWarp32() & (HALFWARPSIZE - 1u));

                const uint& threadId = threadIds[bufferPos];

                //Moeller/Trumbore
                vec3f org   = aScene[primitiveId].vertices[0];
                vec3f edge1 = aScene[primitiveId].vertices[1];
                vec3f edge2 = aScene[primitiveId].vertices[2];

                edge1 = edge1 - org;
                edge2 = edge2 - org;

                const vec3f rayDir =
                    vec3f::fastDivide(vec3f::rep(1.f),
                    ((vec3f*)aSharedMem + warpId() * WARPSIZE)[threadId]);

                vec3f pvec      = rayDir % edge2;
                float detRCP    = 1.f / edge1.dot(pvec);

                vec3f tvec  = origins[bufferPos] - org;
                float alpha = detRCP * tvec.dot(pvec);

                tvec        = tvec % edge1;
                float beta  = detRCP * tvec.dot(rayDir);

                float* distances = (float*)(aSharedMem + 
                    RAY_DISTANCE_OFFSET + warpId32() * WARPSIZE);

                distances[threadId1DInWarp32()] = detRCP * edge2.dot(tvec);

                if(alpha < 0.f         ||
                    beta < 0.f         ||
                    alpha + beta > 1.f ||
                    distances[threadId1DInWarp32()] <= EPS)
                {
                    distances[threadId1DInWarp32()] = FLT_MAX;
                }

                //////////////////////////////////////////////////////////////////////////
                //Reduce to find best hit-point (currently sequential)
                //////////////////////////////////////////////////////////////////////////
                uint bestPrim = 0u;

                distances +=
                    (threadId1DInWarp32() >> LOG2HALFWARPSIZE) * HALFWARPSIZE;

                for (uint it = 1; it < HALFWARPSIZE; ++it)
                {
                    bestPrim = 
                        (distances[it] < distances[bestPrim]) ? it : bestPrim;
                }

                //Prevents "warp serialize"
                if (threadId1DInWarp32() == 0 || threadId1DInWarp32() == HALFWARPSIZE)
                {
                    distances[0] = distances[bestPrim];
                    distances[1] = intAsFloat(primIds[bufferPos] + bestPrim);
                }

                distances -=
                    (threadId1DInWarp32() >> LOG2HALFWARPSIZE) * HALFWARPSIZE;

                //Prevents "warp serialize"
                if (threadId1DInWarp32() == 0)
                {
                    distances[2] = distances[HALFWARPSIZE];
                    distances[3] = distances[HALFWARPSIZE + 1];
                }

                //write out first hit
                if (threadId1DInWarp32() == threadIds[numIterations - 1])
                {
                    if (oRayT >= distances[0])
                    {
                        oRayT = distances[0];
                        oBestHit = floatAsInt(distances[1]);
                    }
                }

                //write out second hit
                if (threadId1DInWarp32() == threadIds[max(numIterations - 2, 0)])
                {
                    if (oRayT >= distances[2])
                    {
                        oRayT = distances[2];
                        oBestHit = floatAsInt(distances[3]);;
                    }
                }

            }//end for intersection calculations
        }


        //////////////////////////////////////////////////////////////////////////
        //Warp is coherent or there are no threads with large workload
        //////////////////////////////////////////////////////////////////////////
        while (aIdRange.y > aIdRange.x)
        {
            const uint primitiveId = aIdRange.x++;

            vec3f org   = aScene[primitiveId].vertices[0];
            vec3f edge1 = aScene[primitiveId].vertices[1];
            vec3f edge2 = aScene[primitiveId].vertices[2];

            const vec3f rayDir = vec3f::fastDivide(vec3f::rep(1.f), aRayDirRCP);

            edge1 = edge1 - org;
            edge2 = edge2 - org;

            vec3f pvec      = rayDir % edge2;
            float detRCP    = 1.f / edge1.dot(pvec);

            //if(fabsf(detRCP) <= EPS_RCP) continue;

            vec3f tvec  = aRayOrg - org;
            float alpha = detRCP * tvec.dot(pvec);

            //if(alpha < 0.f || alpha > 1.f) continue;

            tvec        = tvec % edge1;
            float beta  = detRCP * tvec.dot(rayDir);

            //if(beta < 0.f || beta + alpha > 1.f) continue;

            float dist  = detRCP * edge2.dot(tvec);

            if (alpha >= 0.f        &&
                beta >= 0.f         &&
                alpha + beta <= 1.f &&
                dist > EPS          &&
                dist < oRayT)
            {
                oRayT  = dist;
                oBestHit = primitiveId;
            }
        }
    }
};


#undef BUFFERSIZE

#endif // HYBRIDINTERSECTOR_HPP_INCLUDED_057C3BEE_0D14_4FDE_88A8_16F4CCAF091F
