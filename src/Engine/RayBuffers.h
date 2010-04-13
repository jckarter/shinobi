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

#ifndef RAYBUFFERS_H_INCLUDED_62D567EF_F944_4636_8467_CF45C3A63EF8
#define RAYBUFFERS_H_INCLUDED_62D567EF_F944_4636_8467_CF45C3A63EF8

#include "../CUDAConfig.h"
#include "../CUDAStdAfx.h"

#include "RayTracingTypes.h"
#include "RayTracingConstants.h"



class SimpleRayBuffer
{
    static const int ELEMENTSIZE = 8;
    void* mMemoryPtr;
public:
    SimpleRayBuffer(void* aMemPtr): mMemoryPtr(aMemPtr)
    {}

    HOST DEVICE void* getData()
    {
        return mMemoryPtr;
    }

    HOST DEVICE void store(const vec3f& aRayOrg, const vec3f& aRayDir, const float aRayT,
        const uint aBestHit, const uint aRayId, const uint aNumRays)
    {
        vec3f hitPoint = aRayOrg + (aRayT - EPS) * aRayDir;

        float* rayOut = ((float*)mMemoryPtr) + aRayId;
        *rayOut = hitPoint.x;
        rayOut += aNumRays;
        *rayOut = hitPoint.y;
        rayOut += aNumRays;
        *rayOut = hitPoint.z;

        rayOut += aNumRays;
        *rayOut = aRayDir.x;
        rayOut += aNumRays;
        *rayOut = aRayDir.y;
        rayOut += aNumRays;
        *rayOut = aRayDir.z;

        rayOut += aNumRays;
        *rayOut = aRayT;

        rayOut += aNumRays;
        *((uint*)rayOut) = aBestHit;

    }

    HOST DEVICE void load(vec3f& oRayOrg, vec3f& oRayDir, float& oRayT,
        uint& oBestHit, const uint aRayId, const uint aNumRays)
    {
        float* rayOut = ((float*)mMemoryPtr) + aRayId;
        oRayOrg.x = *rayOut;
        rayOut += aNumRays;
        oRayOrg.y = *rayOut;
        rayOut += aNumRays;
        oRayOrg.z = *rayOut;

        rayOut += aNumRays;
        oRayDir.x = *rayOut;
        rayOut += aNumRays;
        oRayDir.y = *rayOut;
        rayOut += aNumRays;
        oRayDir.z = *rayOut;

        rayOut += aNumRays;
        oRayT = *rayOut;

        rayOut += aNumRays;
        oBestHit = *((uint*)rayOut);

    }

    HOST DEVICE vec3f loadOrigin(const uint aRayId, const uint aNumRays)
    {
        vec3f retval;

        float* ptr = (float*)mMemoryPtr + aRayId;
        retval.x = *ptr;
        ptr += aNumRays;
        retval.y = *ptr;
        ptr += aNumRays;
        retval.z = *ptr;

        return retval;
    }

    HOST DEVICE vec3f loadDirection(const uint aRayId, const uint aNumRays)
    {
        vec3f retval;

        float* ptr = (float*)mMemoryPtr + aRayId + aNumRays * 3;
        retval.x = *ptr;
        ptr += aNumRays;
        retval.y = *ptr;
        ptr += aNumRays;
        retval.z = *ptr;

        return retval;
    }

    HOST DEVICE float loadDistance(const uint aRayId, const uint aNumRays)
    {
        return *((float*)mMemoryPtr + aRayId + aNumRays * 6);
    }

    HOST DEVICE uint loadBestHit(const uint aRayId, const uint aNumRays)
    {
        return *((uint*)mMemoryPtr + aRayId + aNumRays * 7);
    }
};


class OcclusionRayBuffer
{
    void* mMemoryPtr;
public:
    OcclusionRayBuffer(void* aMemPtr): mMemoryPtr(aMemPtr)
    {}

    HOST DEVICE void* getData()
    {
        return mMemoryPtr;
    }

    HOST DEVICE void store(const vec3f& aRayOrg, const vec3f& aRayDir, const float aRayT,
        const uint aBestHit, const uint aRayId, const uint aNumRays)
    {
        //if not occluded, store distance to light source
        vec3f* rayOut = ((vec3f*)mMemoryPtr) + aRayId;
        if (aRayT >= 0.9999f)
        {
            *rayOut = aRayDir;
        }
        else
        {
            *rayOut = vec3f::rep(FLT_MAX);
        }
    }

    HOST DEVICE vec3f loadLigtVec(const uint aSampleId)
    {
        return *((vec3f*)mMemoryPtr + aSampleId);
    }    
    
};

#endif // RAYBUFFERS_H_INCLUDED_62D567EF_F944_4636_8467_CF45C3A63EF8
