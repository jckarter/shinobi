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

#ifndef SIMPLEPATHTRACER_HPP_INCLUDED_EBB40964_45F4_46FD_9069_81DC664AF8C2
#define SIMPLEPATHTRACER_HPP_INCLUDED_EBB40964_45F4_46FD_9069_81DC664AF8C2

//////////////////////////////////////////////////////////////////////////
//Simple implementation of path tracing.
//Paths are terminated based on Russian Roulette.
//All objects have diffuse reflectance and no specular reflectance.
//At the end of each path is computed the direct illumination from the light
//source(s).
//////////////////////////////////////////////////////////////////////////

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

class SimplePathTracer
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
            2611909u + aSeed * 14910827u + globalThreadId1D(1887143u) + aSeed,
            1010567u + aSeed * 2757577u + globalThreadId1D(45751u) + aSeed,
            416191069u);
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

                //vec3f::getLocalCoordinates(normal, state.tMax, state.cellId);

                if (normal.dot(aRayDir[threadId1D()]) > 0.f)
                {
                    normal = -normal;
                }

                float4 diffReflectance = aMaterialStorage.getDiffuseReflectance(
                    aScene.getMaterialId(bestHit));
                //float albedo = (diffReflectance.x * 0.222f +
                //    diffReflectance.y * 0.7067f +
                //    diffReflectance.z * 0.0713f);
                float albedo = 0.5f;
                float rnum = genRand();

                if( rnum < albedo )
                {
#ifndef GATHERSTATISTICS
                    oRadiance = oRadiance / albedo;
                    //pi is to account for cosine weighted hemisphere sampling
                    oRadiance.x = oRadiance.x * diffReflectance.x * M_PI;
                    oRadiance.y = oRadiance.y * diffReflectance.y * M_PI;
                    oRadiance.z = oRadiance.z * diffReflectance.z * M_PI;
#else
                    oRadiance.x += 1.f;
#endif
                    //generate new ray
                    aRayOrg[threadId1D()] = aRayOrg[threadId1D()]
                    + rayT * aRayDir[threadId1D()] + normal *  0.001f;

                    vec3f randDir = genDir(genRand(), genRand());
                    aRayDir[threadId1D()] = state.tMax * randDir.x +
                        state.cellId * randDir.y + normal * randDir.z;

                    rayT = FLT_MAX;
                }
                else
                {
                    //terminate path and save last hit-point
                    aRayDir[threadId1D()] = aRayOrg[threadId1D()]
                    + (rayT - 0.001f) * aRayDir[threadId1D()];
#ifndef GATHERSTATISTICS
                    oRadiance.x = oRadiance.x * diffReflectance.x / (1 - albedo);
                    oRadiance.y = oRadiance.y * diffReflectance.y / (1 - albedo);
                    oRadiance.z = oRadiance.z * diffReflectance.z / (1 - albedo);
#endif
                    //save cos between normal and incoming direction
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

        //register reuse state.cellId.x should be vec3f surfaceNormal
        state.cellId = aRayOrg[threadId1D()];

        aRayOrg[threadId1D()] = aLightSource.getPoint(genRand(), genRand());
        aRayDir[threadId1D()] = aRayDir[threadId1D()] - aRayOrg[threadId1D()];

        if (rayT > -2.f)
        {
            rayT = FLT_MAX;
#ifndef GATHERSTATISTICS
            oRadiance = oRadiance * max(0.f, -state.cellId.dot(~aRayDir[threadId1D()]));
#endif
        }

#ifndef GATHERSTATISTICS
        traverser.traverseShadowRay(aRayOrg, aRayDir, rayT, bestHit, state,
            aGridParameters, aScene, aSharedMem);

        if (rayT >= 0.9999f)
        {
            float attenuation = aRayDir[threadId1D()].dot(aRayDir[threadId1D()]);
            oRadiance = oRadiance *
                aLightSource.intensity *
                dcLightSource.getArea() *
                max(0.f, aLightSource.normal.dot(~aRayDir[threadId1D()])) /
                attenuation;
            //oRadiance = oRadiance / aRayDir[threadId1D()].len() / aRayDir[threadId1D()].len() * max(0.f, aLightSource.normal.dot(aRayDir[threadId1D()]));


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

#endif // SIMPLEPATHTRACER_HPP_INCLUDED_EBB40964_45F4_46FD_9069_81DC664AF8C2
