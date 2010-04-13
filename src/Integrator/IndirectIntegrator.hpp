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

#ifndef INDIRECTINTEGRATOR_HPP_INCLUDED_9F65EADC_A3DD_4790_A44E_A2A9F5E65C2E
#define INDIRECTINTEGRATOR_HPP_INCLUDED_9F65EADC_A3DD_4790_A44E_A2A9F5E65C2E


#include "../CUDAConfig.h"
#include "../CUDAStdAfx.h"

#include "../Core/Algebra.hpp"
#include "../Primitive/Ray.hpp"

#include "../Utils/RandomNumberGenerators.hpp"
#include "../Utils/HemisphereSamplers.hpp"

template<
    class tLightSource,
    class tControlStructure,
    class tStorageStructure,
    class tMaterialStorageStructure,
        template <class, class, class> class tTraverser,
        template <class> class tIntersector >

class IndirectIntegrator
{
public:
    typedef tIntersector<tStorageStructure>                     t_Intersector;
    typedef tTraverser<
        tControlStructure, tStorageStructure, t_Intersector>    t_Traverser;
    typedef typename t_Traverser::TraversalState                t_State;
    typedef KISSRandomNumberGenerator                           t_RNG;

    DEVICE void operator() (
        vec3f*                                          aRayOrg,
        vec3f*                                          aRayDir,
        const tControlStructure&                        aGridParameters,
        const tStorageStructure&                        aScene,
        const tMaterialStorageStructure&                aMaterialStorage,
        const tLightSource&                             aLightSource,
        vec3f&                                          oRadiance,
        uint*                                           aSharedMem,
        int                                             aSeed = 0)
    {
        //////////////////////////////////////////////////////////////////////////
        //Traversal initialization
        t_Traverser     traverser;
        t_State         state;

        //////////////////////////////////////////////////////////////////////////
        //Integration Initialization
        CosineHemisphereSampler genDir;
        t_RNG genRand(  1236789u + globalThreadId1D(),
            369u + globalThreadId1D(aSeed),
            351288629u + globalThreadId1D(aSeed),
            416191069u );
        //t_RNG genRand(globalThreadId1D() + RESX * RESY * (*aImageId));

#ifndef GATHERSTATISTICS
        oRadiance = vec3f::rep(1.f);
#else
        oRadiance.x = 1.f; //number of rays traced
        oRadiance.y = 0.f; //number of rays active after path termination
        oRadiance.z = 0.f; //number of intersection tests
#endif

        //////////////////////////////////////////////////////////////////////////
        //Integration loop
        float rayT   = FLT_MAX;
        uint  bestHit;
        uint bounceCount = 0u;

        //NOTE: Device capability 1.2 or higher required
        while (ANY(rayT > 0.f))//Termination criteria
        {
#ifndef GATHERSTATISTICS
            traverser.traverse(aRayOrg, aRayDir, rayT, bestHit, state,
                aGridParameters, aScene, aSharedMem);
#else
            traverser.traverse(aRayOrg, aRayDir, rayT, bestHit, state,
                aGridParameters, aScene, aSharedMem, oRadiance);

            oRadiance.y += 1.f;
#endif
            if ((rayT < FLT_MAX) && (rayT > 0.f))
            {
                bestHit = aScene.indices[bestHit];
                vec3f normal = aScene(bestHit).vertices[0];
                //register reuse: state.tMax should be edge1
                state.tMax = aScene(bestHit).vertices[1];
                //register reuse: state.cellId should be edge2
                state.cellId = aScene(bestHit).vertices[2];

                state.tMax = state.tMax - normal;
                state.cellId = state.cellId - normal;

                normal = ~(state.tMax % state.cellId);
                state.tMax = ~state.tMax;
                state.cellId = normal % state.tMax;

                if (normal.dot(aRayDir[threadId1D()]) > 0.f)
                {
                    normal = -normal;
                }

                float4 diffReflectance = aMaterialStorage.getDiffuseReflectance(
                    aScene.getMaterialId(bestHit));

                if( bounceCount < 1 )
                {

                    ++bounceCount;
#ifndef GATHERSTATISTICS

                    oRadiance.x = oRadiance.x * diffReflectance.x;
                    oRadiance.y = oRadiance.y * diffReflectance.y;
                    oRadiance.z = oRadiance.z * diffReflectance.z;
#else
                    oRadiance.x += 1.f;
#endif
                    //generate new ray
                    aRayOrg[threadId1D()] = aRayOrg[threadId1D()]
                    + rayT * aRayDir[threadId1D()] + normal * EPS;

                    vec3f randDir = genDir(genRand(), genRand());
                    aRayDir[threadId1D()] = state.tMax * randDir.x +
                        state.cellId * randDir.y + normal * randDir.z;

                    rayT = FLT_MAX;
                }
                else
                {
                    //terminate path and save last hit-point
                    aRayDir[threadId1D()] = aRayOrg[threadId1D()]
                    + (rayT - 0.0001f) * aRayDir[threadId1D()];
#ifndef GATHERSTATISTICS
                    oRadiance.x = oRadiance.x * diffReflectance.x;
                    oRadiance.y = oRadiance.y * diffReflectance.y;
                    oRadiance.z = oRadiance.z * diffReflectance.z;
#endif
                    //save normal for later
                    aRayOrg[threadId1D()] = normal;

                    rayT = -1.f;
                }
            }
            else if (rayT > 0.f)
            {
#ifndef GATHERSTATISTICS
                //terminate path
                oRadiance.x = oRadiance.x * BACKGROUND_R;
                oRadiance.y = oRadiance.y * BACKGROUND_G;
                oRadiance.z = oRadiance.z * BACKGROUND_B;
#endif
                rayT = -2.f;
            }//end if (rayT < FLT_MAX) && (rayT > 0.f)
        }
        //end integration loop
        //////////////////////////////////////////////////////////////////////////
        //trace ray to light source

        //register reuse state.cellId should be vec3f surfaceNormal
        state.cellId = aRayOrg[threadId1D()];

        aRayOrg[threadId1D()] = aLightSource.getPoint(genRand(), genRand());
        aRayDir[threadId1D()] = aRayDir[threadId1D()] - aRayOrg[threadId1D()];

        if (rayT > -2.f)
        {
            rayT = FLT_MAX;
#ifndef GATHERSTATISTICS
            oRadiance = oRadiance * fabsf(state.cellId.dot(~aRayDir[threadId1D()]));
#endif
        }

#ifndef GATHERSTATISTICS
        traverser.traverseShadowRay(aRayOrg, aRayDir, rayT, bestHit, state,
            aGridParameters, aScene, aSharedMem);

        if (rayT >= 0.9999f)
        {
            oRadiance = oRadiance * aLightSource.intensity;
            oRadiance = oRadiance * max(0.f, aLightSource.normal.dot(~aRayDir[threadId1D()]));
        }
        else if (rayT > -2.f)
        {
            //occluded
            oRadiance = vec3f::rep(0.f);
        }
#else
        traverser.traverseShadowRay(aRayOrg, aRayDir, rayT, bestHit, state,
            aGridParameters, aScene, aSharedMem, oRadiance);

        oRadiance.x += 1.f;
        oRadiance.y += 1.f;
#endif
    }
};

#endif // INDIRECTINTEGRATOR_HPP_INCLUDED_9F65EADC_A3DD_4790_A44E_A2A9F5E65C2E
