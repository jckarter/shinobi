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

#ifndef SIMPLEINTERSECTOR_HPP_INCLUDED_6BF1AF36_926B_4392_8DF1_2E09F5338666
#define SIMPLEINTERSECTOR_HPP_INCLUDED_6BF1AF36_926B_4392_8DF1_2E09F5338666

#include "../CUDAConfig.h"

#include "../Core/Algebra.hpp"

template<class tStorageStructure>
class SimpleIntersector
{
    template<class tStorage, bool taRayDirIsRCP>
    class MollerTrumboreIntersectionTest
    {
    public:
        DEVICE void operator()(
            const vec3f&                                    aRayOrg,
            const vec3f&                                    aRayDir,
            float&                                          oRayT,
            const uint2&                                    aIdRange,
            const tStorageStructure&                        aScene,
            uint&                                           oBestHit) const
        {
            for (uint it = aIdRange.x; it != aIdRange.y; ++ it)
            {
                vec3f org   = aScene[it].vertices[0];
                vec3f edge1 = aScene[it].vertices[1];
                vec3f edge2 = aScene[it].vertices[2];

                if (taRayDirIsRCP)//compile time decision
                {
                    const vec3f rayDir = vec3f::fastDivide(vec3f::rep(1.f), aRayDir);

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
                        oBestHit = it;
                    }
                }
                else
                {
                    edge1 = edge1 - org;
                    edge2 = edge2 - org;

                    vec3f pvec      = aRayDir % edge2;
                    float detRCP    = 1.f / edge1.dot(pvec);

                    //if(fabsf(detRCP) <= EPS_RCP) continue;

                    vec3f tvec  = aRayOrg - org;
                    float alpha = detRCP * tvec.dot(pvec);

                    //if(alpha < 0.f || alpha > 1.f) continue;

                    tvec        = tvec % edge1;
                    float beta  = detRCP * tvec.dot(aRayDir);

                    //if(beta < 0.f || beta + alpha > 1.f) continue;

                    float dist  = detRCP * edge2.dot(tvec);

                    if (alpha >= 0.f        &&
                        beta >= 0.f         &&
                        alpha + beta <= 1.f &&
                        dist > EPS          &&
                        dist < oRayT)
                    {
                        oRayT  = dist;
                        oBestHit = it;
                    }
                }
            }
        }
    };

    template<class tStorage, bool taRayDirIsRCP>
    class ShevtsovIntersectionTest
    {
    public:
        DEVICE void operator()(
            const vec3f&                                    aRayOrg,
            const vec3f&                                    aRayDir,
            float&                                          oRayT,
            const uint2&                                    aIdRange,
            const tStorageStructure&                        aScene,
            uint&                                           oBestHit) const
        {
            for (uint it = aIdRange.x; it != aIdRange.y; ++ it)
            {
                uint triId = aScene.indices[it];
                float4 data1 = aScene.getAccelDataChunck(3 * triId);   // nu,  nv,  np, pu
                float4 data2 = aScene.getAccelDataChunck(3 * triId + 1); // pv,  e0u, e0v, e1u
                float4 data3 = aScene.getAccelDataChunck(3 * triId + 2); // e1v, dimW, dimU, dimV

                if (taRayDirIsRCP)//compile time decision
                {
                    const vec3f rayDir = vec3f::fastDivide(vec3f::rep(1.f), aRayDir);

                    int dimW = floatAsInt(data3.y);
                    int dimU = floatAsInt(data3.z);
                    int dimV = floatAsInt(data3.w);

                    float det  = (rayDir[dimU] * data1.x) +
                        (rayDir[dimV] * data1.y) +
                        rayDir[dimW];

                    float dett = data1.z - ((aRayOrg[dimU] * data1.x) +
                        (aRayOrg[dimV] * data1.y) +
                        aRayOrg[dimW]);

                    float Du   = rayDir[dimU] * dett -
                        (data1.w - aRayOrg[dimU]) * det;
                    float Dv   = rayDir[dimV] * dett -
                        (data2.x - aRayOrg[dimV]) * det;
                    float detu = (data3.x * Du) - (data2.w * Dv);
                    float detv = (data2.y * Dv) - (data2.z * Du);

                    uint mask = 
                        (floatAsInt(det - detu - detv) ^ floatAsInt(detu)) |
                        (floatAsInt(detu) ^ floatAsInt(detv));

                    if (mask >> 31) // No hit!
                        continue;


                    float dist = fastDivide(dett, det);
                    if (dist > 0.f && dist < oRayT)
                    {
                        oRayT  = dist;
                        oBestHit = it;
                    }
                }
                else
                {
                    int dimW = floatAsInt(data3.y);
                    int dimU = floatAsInt(data3.z);
                    int dimV = floatAsInt(data3.w);

                    float det  = (aRayDir[dimU] * data1.x) +
                        (aRayDir[dimV] * data1.y) +
                        aRayDir[dimW];

                    float dett = data1.z - ((aRayOrg[dimU] * data1.x) +
                        (aRayOrg[dimV] * data1.y) +
                        aRayOrg[dimW]);

                    float Du   = aRayDir[dimU] * dett -
                        (data1.w - aRayOrg[dimU]) * det;
                    float Dv   = aRayDir[dimV] * dett -
                        (data2.x - aRayOrg[dimV]) * det;
                    float detu = (data3.x * Du) - (data2.w * Dv);
                    float detv = (data2.y * Dv) - (data2.z * Du);

                    uint mask = 
                        (floatAsInt(det - detu - detv) ^ floatAsInt(detu)) |
                        (floatAsInt(detu) ^ floatAsInt(detv));

                    if (mask >> 31) // No hit!
                        continue;


                    float dist = fastDivide(dett, det);
                    if (dist > 0.f && dist < oRayT)
                    {
                        oRayT  = dist;
                        oBestHit = it;
                    }
                }

            }

            for (uint it = aIdRange.x; it != aIdRange.y; ++ it)
            {
                vec3f org   = aScene[it].vertices[0];
                vec3f edge1 = aScene[it].vertices[1];
                vec3f edge2 = aScene[it].vertices[2];

                if (taRayDirIsRCP)//compile time decision
                {
                    const vec3f rayDir = vec3f::fastDivide(vec3f::rep(1.f), aRayDir);

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
                        oBestHit = it;
                    }
                }
                else
                {
                    edge1 = edge1 - org;
                    edge2 = edge2 - org;

                    vec3f pvec      = aRayDir % edge2;
                    float detRCP    = 1.f / edge1.dot(pvec);

                    //if(fabsf(detRCP) <= EPS_RCP) continue;

                    vec3f tvec  = aRayOrg - org;
                    float alpha = detRCP * tvec.dot(pvec);

                    //if(alpha < 0.f || alpha > 1.f) continue;

                    tvec        = tvec % edge1;
                    float beta  = detRCP * tvec.dot(aRayDir);

                    //if(beta < 0.f || beta + alpha > 1.f) continue;

                    float dist  = detRCP * edge2.dot(tvec);

                    if (alpha >= 0.f        &&
                        beta >= 0.f         &&
                        alpha + beta <= 1.f &&
                        dist > EPS          &&
                        dist < oRayT)
                    {
                        oRayT  = dist;
                        oBestHit = it;
                    }
                }
            }
        }
    };


    MollerTrumboreIntersectionTest<tStorageStructure, true> mMTIntersectorRCP;
    MollerTrumboreIntersectionTest<tStorageStructure, false> mMTIntersector;
    ShevtsovIntersectionTest<tStorageStructure, true> mSIntersectorRCP;
    ShevtsovIntersectionTest<tStorageStructure, false> mSIntersector;

public:
    //////////////////////////////////////////////////////////////////////////
    //amount of shared memory required for buffering
    //////////////////////////////////////////////////////////////////////////
    static const uint    SHAREDMEMSIZE = 0u; //dummy
    static const uint    GLOBALMEMSIZE = 0u; //dummy
    //////////////////////////////////////////////////////////////////////////

    DEVICE void operator() (
        const vec3f*                                    aRayOrg,
        const vec3f*                                    aRayDir,
        float&                                          oRayT,
        const uint2&                                    aIdRange,
        const tStorageStructure&                        aScene,
        uint*                                           aSharedMem, //dummy
        uint&                                           oBestHit) const
    {
        return mMTIntersector(aRayOrg[threadId1D()], aRayDir[threadId1D()], 
            oRayT, aIdRange, aScene, oBestHit);

    }

    DEVICE void operator() (
        const vec3f&                                    aRayOrg,
        const vec3f&                                    aRayDirRCP,
        float&                                          oRayT,
        const uint2&                                    aIdRange,
        const tStorageStructure&                        aScene,
        uint*                                           aSharedMem, //dummy
        uint&                                           oBestHit) const
    {
        return mMTIntersectorRCP(aRayOrg, aRayDirRCP, oRayT, aIdRange, aScene,
            oBestHit);
    }
};

#endif // SIMPLEINTERSECTOR_HPP_INCLUDED_6BF1AF36_926B_4392_8DF1_2E09F5338666
