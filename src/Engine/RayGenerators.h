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

#ifndef RAYGENERATORS_H_INCLUDED_3709907C_4651_4741_BCAD_2D20C55C60D4
#define RAYGENERATORS_H_INCLUDED_3709907C_4651_4741_BCAD_2D20C55C60D4

#include "../CUDAConfig.h"
#include "../CUDAStdAfx.h"

#include "RayTracingConstants.h"

template< bool taSafe >
class RegularPrimaryRayGenerator
{
public:
    DEVICE float operator()(vec3f& oRayOrg, vec3f& oRayDir, const uint aRayId,
        const uint aNumRays)
    {
        oRayOrg = dcCamera.getPosition();

        float screenX, screenY;
        if (taSafe)//compile-time decision
        {
            dcRegularPixelSampler.getPixelCoords(
                (float)(min(aRayId, aNumRays - 1u)), (float)dcImageId, screenX,
                screenY);
        }
        else
        {
            dcRegularPixelSampler.getPixelCoords(
                (float)aRayId, (float)dcImageId, screenX, screenY);
        }
        oRayDir = dcCamera.getDirection(screenX, screenY);

        return FLT_MAX;
    }
};

template < bool taSafe >
class RandomPrimaryRayGenerator
{
public:
    DEVICE float operator()(vec3f& oRayOrg, vec3f& oRayDir, const uint aRayId,
        const uint aNumRays)
    {
        typedef KISSRandomNumberGenerator       t_RNG;

        t_RNG genRand(
            1236789u + aRayId * 977u + aRayId,
            369u + aRayId * 35537u + globalThreadId1D(719u),
            351288629u + aRayId + globalThreadId1D(45751u),        
            416191069u );

        float screenX = genRand();
        float screenY = genRand();

        if (taSafe)//compile-time decision
        {
            dcRandomPixelSampler.getPixelCoords(
                (float)(min(aRayId, aNumRays - 1u)), screenX, screenY);
        }
        else
        {
            dcRandomPixelSampler.getPixelCoords((float)aRayId, screenX, screenY);
        }

        oRayDir = dcCamera.getDirection(screenX, screenY);
        oRayOrg = dcCamera.getPosition();

        return FLT_MAX;
    }
};

template<class tRayBuffer>
class AreaLightShadowRayGenerator
{
    tRayBuffer mBuffer;
public:
    AreaLightShadowRayGenerator(const tRayBuffer& aBuffer):mBuffer(aBuffer)
    {}

    DEVICE float operator()(vec3f& oRayOrg, vec3f& oRayDir, const uint aRayId,
        const uint aNumRays)
    {
        uint myPixelIndex = aRayId / dcSamples;
        uint numPixels = dcNumPixels;

        float rayT = mBuffer.loadDistance(myPixelIndex, numPixels);

        typedef KISSRandomNumberGenerator       t_RNG;

        t_RNG genRand(  3643u + aRayId * 4154207u + aRayId,
            1761919u + aRayId * 2746753u + globalThreadId1D(8116093u),
            331801u + aRayId + globalThreadId1D(91438197u),
            10499029u );

        float r1 = genRand();
        float r2 = genRand();

        oRayDir = dcLightSource.getPoint(r1, r2);

        oRayOrg = mBuffer.loadOrigin(myPixelIndex, numPixels);
        //+ (rayT - EPS) * mBuffer.loadDirection(myPixelIndex, numPixels);

        oRayDir = oRayDir - oRayOrg;

        if (rayT >= FLT_MAX)
        {
            return 0.5f;
        }
        else
        {
            return FLT_MAX;
        }
    }
};

#endif // RAYGENERATORS_H_INCLUDED_3709907C_4651_4741_BCAD_2D20C55C60D4
