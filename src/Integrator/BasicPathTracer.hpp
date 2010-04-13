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

#ifndef BASICPATHTRACER_HPP_INCLUDED_1DD691B6_5FD3_4CBC_A887_29B0941D3CAB
#define BASICPATHTRACER_HPP_INCLUDED_1DD691B6_5FD3_4CBC_A887_29B0941D3CAB

//////////////////////////////////////////////////////////////////////////
//Very basic implementation of path tracing.
//Path are terminated based on Russian Roulette.
//All objects have hard coded diffuse reflectance and no specular reflectance.
//Paths that leave the scene are assumed to have hit a light source of
//hard-coded intensity (assume environment map with constant intensity).
//No direct illumination is computed apart from the above-mentioned.
//////////////////////////////////////////////////////////////////////////

#include "../CUDAConfig.h"

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

class BasicPathTracer
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
        t_RNG genRand(  12872141u + aSeed * 426997u + aSeed,
            2611909u + aSeed * 14910827u + globalThreadId1D(1887143u),
            1010567u + aSeed * 2757577u + globalThreadId1D(45751u),
            416191069u);
        //t_RNG genRand(globalThreadId1D() + gRESX * gRESY * (*aImageId));

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
                float rnum = genRand();
                if( rnum < (0.5f * 0.222f + 0.5f * 0.7067f + 0.5f * 0.071f))
                {
                    vec3f normal = aScene[bestHit].vertices[0];
                    //register reuse: state.tMax should be edge1
                    state.tMax = aScene[bestHit].vertices[1];
                    //register reuse: state.cellId should be edge2
                    state.cellId = aScene[bestHit].vertices[2];

                    state.tMax = state.tMax - normal;
                    state.cellId = state.cellId - normal;

                    normal = ~(state.tMax % state.cellId);
                    state.tMax = ~state.tMax;
                    state.cellId = normal % state.tMax;

                    if (normal.dot(aRayDir[threadId1D()]) > 0.f)
                    {
                        normal = -normal;
                    }
#ifndef GATHERSTATISTICS
                    oRadiance = oRadiance *
                        //diffuse reflectance
                        0.5f /
                        //albedo
                        (0.5f * 0.222f + 0.5f * 0.7067f + 0.5f * 0.0713f);
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
#ifndef GATHERSTATISTICS
                    //terminate path
                    oRadiance = vec3f::rep(0.f);
#endif
                    rayT = -1.f;
                }
            }
            else if (rayT > 0.f)
            {
#ifndef GATHERSTATISTICS
                //light source
                oRadiance.x = oRadiance.x * BACKGROUND_R;
                oRadiance.y = oRadiance.y * BACKGROUND_G;
                oRadiance.z = oRadiance.z * BACKGROUND_B;
#endif
                rayT = -1.f;
            }//end if (rayT < FLT_MAX) && (rayT > 0.f)
        }
        //end integration loop
        //////////////////////////////////////////////////////////////////////////
    }
};

#endif // BASICPATHTRACER_HPP_INCLUDED_1DD691B6_5FD3_4CBC_A887_29B0941D3CAB
