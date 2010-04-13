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

#ifndef TWOLEVELGRIDBUILDER_H_INCLUDED_58363B73_9669_41EE_895B_34C38BC43731
#define TWOLEVELGRIDBUILDER_H_INCLUDED_58363B73_9669_41EE_895B_34C38BC43731

#include "../../CUDAStdAfx.h"

#include "../Structure/TwoLevelGrid.h"

#include "TwoLevelGridBuilder.hpp"
#include "GridBuilder.h"

template<class tPrimitiveStorage>
class TwoLevelGridBuilder
{

    uint mNumPrimitives;
public:
    HOST void setNumPrimitives(const uint aNum)
    {
        mNumPrimitives = aNum;
    }

    void init(
        TwoLevelGrid&                                   oGrid,
        tPrimitiveStorage&                              oFaceSoup,
        const FWObject::t_FaceIterator&                 aBegin,
        const FWObject::t_FaceIterator&                 aEnd,
        const FWObject&                                 aData)
    {
        //initialize grid parameters
        mNumPrimitives = 0u;

        for(FWObject::t_FaceIterator bboxIt = aBegin; 
            bboxIt != aEnd; ++mNumPrimitives, ++bboxIt)
        {
            oGrid.bounds.extend(aData.getVertex(bboxIt->vert1));
            oGrid.bounds.extend(aData.getVertex(bboxIt->vert2));
            oGrid.bounds.extend(aData.getVertex(bboxIt->vert3));
        }

        //cudastd::log::out << "Number of primitives:" << mNumPrimitives << "\n";

        vec3f diagonal = oGrid.bounds.max - oGrid.bounds.min;
        const float volume = diagonal.x * diagonal.y * diagonal.z;
        const float lambda = 5.f;
        const float magicConstant =
            powf(lambda * static_cast<float>(mNumPrimitives) / volume, 0.3333333f);

        diagonal *= magicConstant;

        oGrid.resX = static_cast<int>(diagonal.x * 0.25f);
        oGrid.resY = static_cast<int>(diagonal.y * 0.25f);
        oGrid.resZ = static_cast<int>(diagonal.z * 0.25f);

        vec3f cellSize = (oGrid.bounds.max - oGrid.bounds.min) /
            oGrid.getResolution();

        vec3f cellSizeRCP = vec3f::rep(1.f) / cellSize;

        CUDA_SAFE_CALL(
            cudaMallocHost((void**)&oGrid.cpuCells,
            oGrid.resX * oGrid.resY * oGrid.resZ * sizeof(TwoLevelGrid::t_Cell)));

        uint* faceSoupData;
        uint numReferences;

        TwoLevelGridBuilderHost::hostBuild(
            oGrid.cpuCells,
            oGrid.cpuLeaves,
            oGrid.leavesCount,
            faceSoupData,
            numReferences,
            oGrid.resX,
            oGrid.resY,
            oGrid.resZ,
            oGrid.getResolution(),
            oGrid.bounds.min,
            oGrid.bounds.max,
            aBegin,
            aEnd,
            aData);

        //upload cells to device
        oGrid.upload();

        // allocate and copy index array on the device
        CUDA_SAFE_CALL( cudaMalloc((void**)&oFaceSoup.indices , numReferences * sizeof(uint)));
        CUDA_SAFE_CALL( cudaMemcpy(oFaceSoup.indices, faceSoupData, numReferences * sizeof(uint), cudaMemcpyHostToDevice) );

        oFaceSoup.upload(aBegin, aEnd, aData);

    }//void init(...)

    //dummy
    HOST void build(
        TwoLevelGrid&                                   oGrid,
        tPrimitiveStorage&                                       oFaceSoup)
    {}

    //dummy
    HOST void rebuild(
        TwoLevelGrid&       oGrid,
        tPrimitiveStorage&  oFaceSoup,
        cudaStream_t&       aStream)
    {}

};

#endif // TWOLEVELGRIDBUILDER_H_INCLUDED_58363B73_9669_41EE_895B_34C38BC43731
