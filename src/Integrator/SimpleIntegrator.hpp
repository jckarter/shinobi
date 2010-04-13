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

#ifndef SIMPLEINTEGRATOR_HPP_INCLUDED_0A45B3B2_99BD_4245_9F3B_A4401133444E
#define SIMPLEINTEGRATOR_HPP_INCLUDED_0A45B3B2_99BD_4245_9F3B_A4401133444E

#include "../CUDAConfig.h"

#include "../Core/Algebra.hpp"
#include "../Primitive/Ray.hpp"

template<
    class tLightSource,
    class tControlStructure,
    class tStorageStructure,
    class tMaterialStorageStructure,
        template <class, class, class> class tTraverser,
        template <class> class tIntersector >

class DeviceSimpleIntegrator
{
public:
    typedef tIntersector<tStorageStructure>                     t_Intersector;
    typedef tTraverser<
        tControlStructure, tStorageStructure, t_Intersector>    t_Traverser;
    typedef typename t_Traverser::TraversalState                t_State;
    
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
        float rayT   = FLT_MAX;
        uint  bestHit;

        //////////////////////////////////////////////////////////////////////////
        //Traversal initialization
        t_Traverser     traverser;
        t_State         state;

#ifndef GATHERSTATISTICS
        traverser.traverse(aRayOrg, aRayDir, rayT, bestHit, state,
            aGridParameters, aScene, aSharedMem);

        if (rayT < FLT_MAX)
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
            float4 diffReflectance = aMaterialStorage.getDiffuseReflectance(aScene.getMaterialId(bestHit));
            state.cellId.x = diffReflectance.x;
            state.cellId.y = diffReflectance.y;
            state.cellId.z = diffReflectance.z;

            oRadiance =  state.cellId * fabsf(aRayDir[threadId1D()].dot(normal));
        }
        else
        {
            oRadiance.x = BACKGROUND_R;
            oRadiance.y = BACKGROUND_G;
            oRadiance.z = BACKGROUND_B;
        }
#else
        oRadiance.x = 1.f; //number of rays traced
        oRadiance.y = 1.f; //number of active rays
        oRadiance.z = 0.f; //number of intersection tests

        traverser.traverse(aRayOrg, aRayDir, rayT, bestHit, state,
            aGridParameters, aScene, aSharedMem, oRadiance);
#endif

    }
};



#endif // SIMPLEINTEGRATOR_HPP_INCLUDED_0A45B3B2_99BD_4245_9F3B_A4401133444E
