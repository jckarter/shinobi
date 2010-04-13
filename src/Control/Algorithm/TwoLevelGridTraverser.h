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

#ifndef TWOLEVELGRIDTRAVERSER_H_INCLUDED_CE63E087_B88D_4675_932B_0A7BAD291CCF
#define TWOLEVELGRIDTRAVERSER_H_INCLUDED_CE63E087_B88D_4675_932B_0A7BAD291CCF

#include "../../CUDAStdAfx.h"
#include "../../Core/Algebra.hpp"
#include "../Structure/TwoLevelGrid.h"

template <
    class tControlStructure,
    class tStorageStructure,
    class tIntersector>
class SimpleTwoLevelGridTraverser
{
public:
    struct TraversalState
    {
        vec3f tMax;
        vec3f cellId;
        //////////////////////////////////////////////////////////////////////////
        //omitted for memory/computation trade-off
        //vec3f tDelta;
        //int stepInt[3], justOut[3];
        //
        //stepInt[0] = (aRay.dir.x > 0.f) ? 1 : -1;
        //stepInt[1] = (aRay.dir.y > 0.f) ? 1 : -1;
        //stepInt[2] = (aRay.dir.z > 0.f) ? 1 : -1;
        //justOut[0] = (aRay.dir.x > 0.f) ? aGridParams.res[0] : -1;
        //justOut[1] = (aRay.dir.y > 0.f) ? aGridParams.res[1] : -1;
        //justOut[2] = (aRay.dir.z > 0.f) ? aGridParams.res[2] : -1;
        //tDelta[0]  =  aGridParams.getCellSize()[0] / fabsf(aRay.dir[0]);
        //tDelta[1]  =  aGridParams.getCellSize()[1] / fabsf(aRay.dir[1]);
        //tDelta[2]  =  aGridParams.getCellSize()[2] / fabsf(aRay.dir[2]);
        //////////////////////////////////////////////////////////////////////////

    };

    DEVICE void traverse(
        vec3f*                                          aRayOrg,
        vec3f*                                          aRayDir,
        float&                                          oRayT,
        uint&                                           oBestHit,
        TraversalState&                                 aState,
        const tControlStructure&                        aGridParameters,
        const tStorageStructure&                        aScene,
        uint*                                           aSharedMem
#ifdef GATHERSTATISTICS
        , vec3f&                                         oRadiance
#endif
        )
    {
        bool traversalFlag = (oRayT > 0.f) &&
            traverseInit(aRayOrg[threadId1D()], aRayDir[threadId1D()], aGridParameters, aState);

        //////////////////////////////////////////////////////////////////////////
        //Traversal loop
        while (ANY(traversalFlag))
        {
            TwoLevelGrid::t_Cell cell;

            if (traversalFlag /*&& !cell.notEmpty()*/)
            {
                //NOTE: Do not normalize coordinates!
                cell = 
                    TwoLevelGrid::t_Cell(
                    tex3D(texTopLevelCells, aState.cellId.x,
                    aState.cellId.y, aState.cellId.z));

                //traversalFlag &=
                //    traverseMacroCell(aRayDir[threadId1D()], aGridParameters, aState);
            }


            bool subFlag = traversalFlag && cell.notEmpty();
            TraversalState subState;

            if (subFlag)
            {
                const vec3f minBound = aState.cellId * aGridParameters.getCellSize()
                    + aGridParameters.bounds.min;

                //ray/box intersection with the cell
                vec3f tmp;
                tmp.x = (aRayDir[threadId1D()].x > 0.f) ? minBound.x : minBound.x + aGridParameters.getCellSize().x;
                tmp.y = (aRayDir[threadId1D()].y > 0.f) ? minBound.y : minBound.y + aGridParameters.getCellSize().y;
                tmp.z = (aRayDir[threadId1D()].z > 0.f) ? minBound.z : minBound.z + aGridParameters.getCellSize().z;

                const vec3f tMin = (tmp - aRayOrg[threadId1D()]) / aRayDir[threadId1D()];

                const float dist = fmaxf(0.f, fmaxf(fmaxf(tMin.x, tMin.y), tMin.z));

                const vec3f entryPoint = aRayOrg[threadId1D()] + 
                    dist * aRayDir[threadId1D()];

                vec3f cellRes;
                cellRes.x = (float)cell[0];
                cellRes.y = (float)cell[1];
                cellRes.z = (float)cell[2];

                const vec3f subCellSizeRCP = aGridParameters.getCellSizeRCP() * cellRes; 

                subState.cellId.x = floorf((entryPoint - minBound).x * subCellSizeRCP.x);
                subState.cellId.y = floorf((entryPoint - minBound).y * subCellSizeRCP.y);
                subState.cellId.z = floorf((entryPoint - minBound).z * subCellSizeRCP.z);

                subState.cellId = vec3f::min(cellRes - vec3f::rep(1.f), 
                    vec3f::max(subState.cellId, vec3f::rep(0.f)));

                subState.tMax.x = ((subState.cellId.x + 
                    ((aRayDir[threadId1D()].x > 0.f) ? 1 : 0 ))
                    / subCellSizeRCP.x + minBound.x
                    - aRayOrg[threadId1D()].x) / aRayDir[threadId1D()].x;
                subState.tMax.y = ((subState.cellId.y + 
                    ((aRayDir[threadId1D()].y > 0.f) ? 1 : 0 ))
                    / subCellSizeRCP.y + minBound.y
                    - aRayOrg[threadId1D()].y) / aRayDir[threadId1D()].y;
                subState.tMax.z = ((subState.cellId.z +
                    ((aRayDir[threadId1D()].z > 0.f) ? 1 : 0 ))
                    / subCellSizeRCP.z + minBound.z
                    - aRayOrg[threadId1D()].z) / aRayDir[threadId1D()].z;
            }

            while (ANY(subFlag))
            {
                uint2 cellRange = make_uint2(0u, 0u);

                if (subFlag)
                {
                    //NOTE: Do not normalize coordinates!
                    cellRange = tex1Dfetch(texLeafCells, cell.getLeafRangeBegin()
                        + (uint)(subState.cellId.x + (float)cell[0] * subState.cellId.y
                        + (float)cell[0] * (float)cell[1] * subState.cellId.z));
                }

#ifdef GATHERSTATISTICS
                oRadiance.z += (float)(cellRange.y - cellRange.x);
#endif
                tIntersector    intersect;
                intersect(aRayOrg, aRayDir, oRayT, cellRange, aScene, aSharedMem, oBestHit);

                subFlag &=
                    oRayT > subState.tMax.x
                    ||  oRayT > subState.tMax.y
                    ||  oRayT > subState.tMax.z;

                subFlag &=
                    traverseCell(aRayDir[threadId1D()], aGridParameters.getCellSize(), cell, subState);

            }

            traversalFlag &=
                oRayT > aState.tMax.x
                ||  oRayT > aState.tMax.y
                ||  oRayT > aState.tMax.z;

            traversalFlag &=
                traverseMacroCell(aRayDir[threadId1D()], aGridParameters, aState);

        }
        //end of traversal loop
        //////////////////////////////////////////////////////////////////////////
    }

    DEVICE void traverseShadowRay(
        vec3f*                                          aRayOrg,
        vec3f*                                          aRayDir,
        float&                                          oRayT,
        uint&                                           oBestHit,
        TraversalState&                                 aState,
        const tControlStructure&                        aGridParameters,
        const tStorageStructure&                        aScene,
        uint*                                           aSharedMem
#ifdef GATHERSTATISTICS
        , vec3f&                                         oRadiance
#endif
        )
    {
        bool traversalFlag =
            traverseInit(aRayOrg[threadId1D()], aRayDir[threadId1D()], aGridParameters, aState);

        //////////////////////////////////////////////////////////////////////////
        //Traversal loop
        while (ANY(traversalFlag))
        {
            TwoLevelGrid::t_Cell cell;

            if (traversalFlag /*&& !cell.notEmpty()*/)
            {
                //NOTE: Do not normalize coordinates!
                cell = 
                    TwoLevelGrid::t_Cell(
                    tex3D(texTopLevelCells, aState.cellId.x,
                    aState.cellId.y, aState.cellId.z));
                // 
                //                 traversalFlag &=
                //                     traverseMacroCell(aRayDir[threadId1D()], aGridParameters, aState);
            }

            bool subFlag = traversalFlag && cell.notEmpty();
            TraversalState subState;

            if (subFlag)
            {
                const vec3f minBound = aState.cellId * aGridParameters.getCellSize()
                    + aGridParameters.bounds.min;

                //ray/box intersection with the cell
                vec3f tmp;
                tmp.x = (aRayDir[threadId1D()].x > 0.f) ? minBound.x : minBound.x + aGridParameters.getCellSize().x;
                tmp.y = (aRayDir[threadId1D()].y > 0.f) ? minBound.y : minBound.y + aGridParameters.getCellSize().y;
                tmp.z = (aRayDir[threadId1D()].z > 0.f) ? minBound.z : minBound.z + aGridParameters.getCellSize().z;

                const vec3f tMin = (tmp - aRayOrg[threadId1D()]) / aRayDir[threadId1D()];

                const float dist = fmaxf(0.f, fmaxf(fmaxf(tMin.x, tMin.y), tMin.z));

                const vec3f entryPoint = aRayOrg[threadId1D()] + 
                    dist * aRayDir[threadId1D()];

                vec3f cellRes;
                cellRes.x = (float)cell[0];
                cellRes.y = (float)cell[1];
                cellRes.z = (float)cell[2];

                const vec3f subCellSizeRCP = aGridParameters.getCellSizeRCP() * cellRes; 

                subState.cellId.x = floorf((entryPoint - minBound).x * subCellSizeRCP.x);
                subState.cellId.y = floorf((entryPoint - minBound).y * subCellSizeRCP.y);
                subState.cellId.z = floorf((entryPoint - minBound).z * subCellSizeRCP.z);

                subState.cellId = vec3f::min(cellRes - vec3f::rep(1.f), 
                    vec3f::max(subState.cellId, vec3f::rep(0.f)));

                subState.tMax.x = ((subState.cellId.x + 
                    ((aRayDir[threadId1D()].x > 0.f) ? 1 : 0 ))
                    / subCellSizeRCP.x + minBound.x
                    - aRayOrg[threadId1D()].x) / aRayDir[threadId1D()].x;
                subState.tMax.y = ((subState.cellId.y + 
                    ((aRayDir[threadId1D()].y > 0.f) ? 1 : 0 ))
                    / subCellSizeRCP.y + minBound.y
                    - aRayOrg[threadId1D()].y) / aRayDir[threadId1D()].y;
                subState.tMax.z = ((subState.cellId.z +
                    ((aRayDir[threadId1D()].z > 0.f) ? 1 : 0 ))
                    / subCellSizeRCP.z + minBound.z
                    - aRayOrg[threadId1D()].z) / aRayDir[threadId1D()].z;
            }

            while (ANY(subFlag))
            {
                uint2 cellRange = make_uint2(0u, 0u);

                if (subFlag)
                {
                    //NOTE: Do not normalize coordinates!
                    cellRange = tex1Dfetch(texLeafCells, cell.getLeafRangeBegin()
                        + (uint)(subState.cellId.x + cell[0] * subState.cellId.y
                        + cell[0] * cell[1] * subState.cellId.z));
                }

#ifdef GATHERSTATISTICS
                oRadiance.z += (float)(cellRange.y - cellRange.x);
#endif


                tIntersector   intersect;
                intersect(aRayOrg, aRayDir, oRayT, cellRange, aScene, aSharedMem, oBestHit);

                subFlag &= (oRayT >= 0.9999f);
                subFlag &=
                    traverseCell(aRayDir[threadId1D()], aGridParameters.getCellSize(), cell, subState); 
            }

            traversalFlag &= (oRayT >= 0.9999f);
            traversalFlag &=
                traverseMacroCell(aRayDir[threadId1D()], aGridParameters, aState);

        }
        //end of traversal loop
        //////////////////////////////////////////////////////////////////////////
    }

    DEVICE bool traverseInit(
        const vec3f&                aRayOrg,
        const vec3f&                aRayDir,
        const tControlStructure&    aGridParams,
        TraversalState&             oState) const
    {
        //////////////////////////////////////////////////////////////////////////
        //ray/box intersection test
        float tEntry, tExit;
        aGridParams.bounds.clip(aRayOrg, aRayDir, tEntry, tExit);

        if (tExit <= tEntry || tExit < 0.f)
        {
            return false;
        }
        //end ray/box intersection test
        //////////////////////////////////////////////////////////////////////////

        const vec3f entryPt = (tEntry >= 0.f) ?
            aRayOrg + (tEntry + EPS) * aRayDir : aRayOrg;

        oState.cellId = 
            (entryPt - aGridParams.bounds.min) * aGridParams.getCellSizeRCP();

        oState.cellId.x = floorf(oState.cellId.x);
        oState.cellId.y = floorf(oState.cellId.y);
        oState.cellId.z = floorf(oState.cellId.z);

        vec3f tmp;
        tmp.x = (aRayDir.x > 0.f) ? 1.f : 0.f;
        tmp.y = (aRayDir.y > 0.f) ? 1.f : 0.f;
        tmp.z = (aRayDir.z > 0.f) ? 1.f : 0.f;

        oState.tMax = ((oState.cellId + tmp) * aGridParams.getCellSize()
            + aGridParams.bounds.min - aRayOrg) / aRayDir;

        //         tmp.x = (aRay.dir.x > 0.f) ? 1.f : -1.f;
        //         tmp.y = (aRay.dir.y > 0.f) ? 1.f : -1.f;
        //         tmp.z = (aRay.dir.z > 0.f) ? 1.f : -1.f;
        //         oState.tDelta = aGridParams.getCellSize() / aRay.dir * tmp;

        return (oState.cellId.x != ((aRayDir.x > 0.f) ? aGridParams.res[0] : -1)) 
            && (oState.cellId.y != ((aRayDir.y > 0.f) ? aGridParams.res[1] : -1))
            && (oState.cellId.z != ((aRayDir.z > 0.f) ? aGridParams.res[2] : -1));
    }

    DEVICE bool traverseMacroCell(
        const vec3f&                aRayDir,
        const tControlStructure&    aGridParams,
        TraversalState&             oState) const
    {
#define MIN_DIMENSION(aX, aY, aZ)	                           \
    (aX < aY) ? ((aX < aZ) ? 0 : 2)	: ((aY < aZ) ? 1 : 2)

        const int tMinDimension =
            MIN_DIMENSION(oState.tMax[0], oState.tMax[1], oState.tMax[2]);

#undef  MIN_DIMENSION

        oState.cellId[tMinDimension] += (aRayDir[tMinDimension] > 0.f) ? 1.f : -1.f;
        oState.tMax[tMinDimension] +=
            fastDivide(aGridParams.bounds.max[tMinDimension] -
            aGridParams.bounds.min[tMinDimension],
            aGridParams.res[tMinDimension] * fabsf(aRayDir[tMinDimension]) );

        //if (aRayDir[tMinDimension] > 0.f)
        //{
        //    return (fabsf(oState.cellId[tMinDimension]  - 
        //        (float)aGridParams.res[tMinDimension]) < 0.1f) ? false : true;
        //}
        //else
        //{
        //    return (fabsf(oState.cellId[tMinDimension] + 1.f) < 0.1f) ? false : true;
        //}
        return
            !(fabsf(oState.cellId[tMinDimension]  - (float)aGridParams.res[tMinDimension]) < 0.1f 
            || fabsf(oState.cellId[tMinDimension] + 1.f) < 0.1f);
    }

    DEVICE bool traverseCell(
        const vec3f&                aRayDir,
        const vec3f&                aCellDiagonal,
        const TwoLevelGrid::t_Cell& aCellParams,
        TraversalState&             oState) const
    {
#define MIN_DIMENSION(aX, aY, aZ)	                           \
    (aX < aY) ? ((aX < aZ) ? 0 : 2)	: ((aY < aZ) ? 1 : 2)

        const int tMinDimension =
            MIN_DIMENSION(oState.tMax[0], oState.tMax[1], oState.tMax[2]);

#undef  MIN_DIMENSION

        oState.cellId[tMinDimension] += (aRayDir[tMinDimension] > 0.f) ? 1.f : -1.f;
        oState.tMax[tMinDimension] +=
            fastDivide(aCellDiagonal[tMinDimension],
            aCellParams[tMinDimension] * fabsf(aRayDir[tMinDimension]) );

        //if (aRayDir[tMinDimension] > 0.f)
        //{
        //    return (fabsf(oState.cellId[tMinDimension]  - 
        //        (float)aCellParams[tMinDimension]) < 0.1f) ? false : true;
        //}
        //else
        //{
        //    return (fabsf(oState.cellId[tMinDimension] + 1.f) < 0.1f) ? false : true;
        //}
        return
            !(fabsf(oState.cellId[tMinDimension]  - (float)aCellParams[tMinDimension]) < 0.1f 
            || fabsf(oState.cellId[tMinDimension] + 1.f) < 0.1f);
    }

};


template <
    class tControlStructure,
    class tStorageStructure,
    class tIntersector>
class TwoLevelGridTraverser
{
public:
    static const uint SHAREDMEMSIZE = 
        RENDERTHREADSX * RENDERTHREADSY * 3; //3 floats per thread
    struct TraversalState
    {
        vec3f tMax;
        vec3f cellId;
        float tEntry;
    };

    struct SmallTraversalState
    {
        vec3f tMax;
        vec3f cellId;
    };

    DEVICE void traverse(
        const vec3f&                                    aRayOrg,
        float&                                          oRayT,
        uint&                                           oBestHit,
        const tControlStructure&                        aGridParameters,
        const tStorageStructure&                        aScene,
        uint*                                           aSharedMem
#ifdef GATHERSTATISTICS
        , vec3f&                                         oRadiance
#endif
        )
    {
        vec3f& rayDirRCP = (((vec3f*)aSharedMem)[threadId1D32()]);

        flag4 flag;
        flag.setFlag1(oRayT > 0.f);

        TraversalState  aState;

        const bool topLvlFlag = 
            traverseInit(aRayOrg, rayDirRCP, aGridParameters, aState);
        if (!topLvlFlag)
            flag.setFlag1To0();

        //////////////////////////////////////////////////////////////////////////
        //Traversal loop
        while (ANY(flag.getFlag1()))
        {
            TwoLevelGrid::t_Cell cell;

            if (flag.getFlag1())
            {
                cell = 
                    TwoLevelGrid::t_Cell(
                    tex3D(texTopLevelCells, aState.cellId.x,
                    aState.cellId.y, aState.cellId.z));
            }


            const bool secondLvlFlag = flag.getFlag1() && cell.notEmpty();
            flag.setFlag2(secondLvlFlag);

            SmallTraversalState subState;
            vec3f subCellSize;

            if (flag.getFlag2())
            {

                const vec3f entryPoint = aRayOrg + 
                    vec3f::fastDivide(vec3f::rep(aState.tEntry), rayDirRCP);

                vec3f cellRes;
                cellRes.x = (float)cell[0];
                cellRes.y = (float)cell[1];
                cellRes.z = (float)cell[2];

                const vec3f subCellSizeRCP = aGridParameters.getCellSizeRCP() * cellRes; 
                const vec3f minBound = aState.cellId * aGridParameters.getCellSize()
                    + aGridParameters.bounds.min;

                subState.cellId.x = floorf((entryPoint - minBound).x * subCellSizeRCP.x);
                subState.cellId.y = floorf((entryPoint - minBound).y * subCellSizeRCP.y);
                subState.cellId.z = floorf((entryPoint - minBound).z * subCellSizeRCP.z);

                subCellSize.x = fastDivide(1.f, subCellSizeRCP.x);
                subCellSize.y = fastDivide(1.f, subCellSizeRCP.y);
                subCellSize.z = fastDivide(1.f, subCellSizeRCP.z);

                subState.cellId = vec3f::min(cellRes - vec3f::rep(1.f), 
                    vec3f::max(subState.cellId, vec3f::rep(0.f)));

                subState.tMax.x = ((subState.cellId.x + 
                    ((rayDirRCP.x > 0.f) ? 1 : 0 ))
                    * subCellSize.x + minBound.x
                    - aRayOrg.x) * rayDirRCP.x;
                subState.tMax.y = ((subState.cellId.y + 
                    ((rayDirRCP.y > 0.f) ? 1 : 0 ))
                    * subCellSize.y + minBound.y
                    - aRayOrg.y) * rayDirRCP.y;
                subState.tMax.z = ((subState.cellId.z +
                    ((rayDirRCP.z > 0.f) ? 1 : 0 ))
                    * subCellSize.z + minBound.z
                    - aRayOrg.z) * rayDirRCP.z;
            }

            while (ANY(flag.getFlag2()))
            {
                uint2 cellRange = make_uint2(0u, 0u);

                if (flag.getFlag2())
                {
                    //NOTE: Do not normalize coordinates!
                    cellRange = tex1Dfetch(texLeafCells, cell.getLeafRangeBegin()
                        + (uint)(subState.cellId.x + (float)cell[0] * subState.cellId.y
                        + (float)cell[0] * (float)cell[1] * subState.cellId.z));
                }

#ifdef GATHERSTATISTICS
                oRadiance.z += (float)(cellRange.y - cellRange.x);
#endif
                tIntersector    intersect;
                intersect(aRayOrg, rayDirRCP, oRayT, cellRange, aScene, aSharedMem, oBestHit);

                const bool keepRayActive =
                    oRayT > subState.tMax.x
                    ||  oRayT > subState.tMax.y
                    ||  oRayT > subState.tMax.z;

                if (!keepRayActive)
                    flag.setFlag2To0();

                if (flag.getFlag2())
                {
                    const bool secondLvlFlag =
                        traverseCell(rayDirRCP, subCellSize, cell, subState);

                    flag.setFlag2(secondLvlFlag);
                }

            }

            const bool keepRayActive =
                oRayT > aState.tMax.x
                ||  oRayT > aState.tMax.y
                ||  oRayT > aState.tMax.z;

            if (!keepRayActive)
                flag.setFlag1To0();

            if (flag.getFlag1())
            {
                const bool topLvlFlag =
                    traverseMacroCell(rayDirRCP, aGridParameters, aState);
                flag.setFlag1(topLvlFlag);
            }

        }
        //end of traversal loop
        //////////////////////////////////////////////////////////////////////////
    }

    DEVICE bool traverseInit(
        const vec3f&                aRayOrg,
        const vec3f&                aRayDirRCP,
        const tControlStructure&    aGridParams,
        TraversalState&             oState) const
    {
        //////////////////////////////////////////////////////////////////////////
        //ray/box intersection test
        float tExit;
        aGridParams.bounds.fastClip(aRayOrg, aRayDirRCP, oState.tEntry, tExit);

        if (tExit <= oState.tEntry || tExit < 0.f)
        {
            return false;
        }
        //end ray/box intersection test
        //////////////////////////////////////////////////////////////////////////

        oState.tEntry = fmaxf(0.f, oState.tEntry);
        const vec3f entryPt = aRayOrg +
            vec3f::fastDivide(vec3f::rep(oState.tEntry + EPS) , aRayDirRCP);

        oState.cellId = 
            (entryPt - aGridParams.bounds.min) * aGridParams.getCellSizeRCP();

        oState.cellId.x = floorf(oState.cellId.x);
        oState.cellId.y = floorf(oState.cellId.y);
        oState.cellId.z = floorf(oState.cellId.z);

        vec3f tmp;
        tmp.x = (aRayDirRCP.x > 0.f) ? 1.f : 0.f;
        tmp.y = (aRayDirRCP.y > 0.f) ? 1.f : 0.f;
        tmp.z = (aRayDirRCP.z > 0.f) ? 1.f : 0.f;

        oState.tMax = ((oState.cellId + tmp) * aGridParams.getCellSize()
            + aGridParams.bounds.min - aRayOrg) * aRayDirRCP;

        return (oState.cellId.x != ((aRayDirRCP.x > 0.f) ? aGridParams.res[0] : -1)) 
            && (oState.cellId.y != ((aRayDirRCP.y > 0.f) ? aGridParams.res[1] : -1))
            && (oState.cellId.z != ((aRayDirRCP.z > 0.f) ? aGridParams.res[2] : -1));
    }

    DEVICE bool traverseMacroCell(
        const vec3f&                aRayDirRCP,
        const tControlStructure&    aGridParams,
        TraversalState&             oState) const
    {
#define MIN_DIMENSION(aX, aY, aZ)	                           \
    (aX < aY) ? ((aX < aZ) ? 0 : 2)	: ((aY < aZ) ? 1 : 2)

        const int tMinDimension =
            MIN_DIMENSION(oState.tMax[0], oState.tMax[1], oState.tMax[2]);

#undef  MIN_DIMENSION

        oState.tEntry = oState.tMax[tMinDimension];
        oState.cellId[tMinDimension] += (aRayDirRCP[tMinDimension] > 0.f) ? 1.f : -1.f;
        oState.tMax[tMinDimension] += aGridParams.cellSize[tMinDimension] *
            fabsf(aRayDirRCP[tMinDimension]);

        return
            !(fabsf(oState.cellId[tMinDimension]  - (float)aGridParams.res[tMinDimension]) < 0.1f 
            || fabsf(oState.cellId[tMinDimension] + 1.f) < 0.1f);
    }

    DEVICE bool traverseCell(
        const vec3f&                aRayDirRCP,
        const vec3f&                aSubCellSize,
        const TwoLevelGrid::t_Cell& aCellParams,
        SmallTraversalState&        oState) const
    {
#define MIN_DIMENSION(aX, aY, aZ)	                           \
    (aX < aY) ? ((aX < aZ) ? 0 : 2)	: ((aY < aZ) ? 1 : 2)

        const int tMinDimension =
            MIN_DIMENSION(oState.tMax[0], oState.tMax[1], oState.tMax[2]);

#undef  MIN_DIMENSION

        oState.cellId[tMinDimension] += (aRayDirRCP[tMinDimension] > 0.f) ? 1.f : -1.f;
        oState.tMax[tMinDimension] += aSubCellSize[tMinDimension] * fabsf(aRayDirRCP[tMinDimension]);

        return
            !(fabsf(oState.cellId[tMinDimension]  - (float)aCellParams[tMinDimension]) < 0.1f 
            || fabsf(oState.cellId[tMinDimension] + 1.f) < 0.1f);
    }

};



template <
    class tControlStructure,
    class tStorageStructure,
    class tIntersector>
class TwoLevelGridShadowTraverser:
    public TwoLevelGridTraverser<tControlStructure, tStorageStructure, tIntersector>
{
    using typename TwoLevelGridTraverser<
        tControlStructure, tStorageStructure, tIntersector>::TraversalState;

    using typename TwoLevelGridTraverser<
        tControlStructure, tStorageStructure, tIntersector>::SmallTraversalState;

public:
    TwoLevelGridShadowTraverser():
      TwoLevelGridTraverser<tControlStructure, tStorageStructure,
          tIntersector>()
    {}

    DEVICE void traverse(
        const vec3f&                                    aRayOrg,
        float&                                          oRayT,
        uint&                                           oBestHit,
        const tControlStructure&                        aGridParameters,
        const tStorageStructure&                        aScene,
        uint*                                           aSharedMem
#ifdef GATHERSTATISTICS
        , vec3f&                                         oRadiance
#endif
        )
    {
        vec3f& rayDirRCP = (((vec3f*)aSharedMem)[threadId1D32()]);

        flag4 flag;
        flag.setFlag1(true);

        if (oRayT < 0.9999f)
            flag.setFlag1To0();

        TraversalState aState;

        traverseInit(aRayOrg, rayDirRCP, aGridParameters, aState);

        //////////////////////////////////////////////////////////////////////////
        //Traversal loop
        while (ANY(flag.getFlag1()))
        {
            TwoLevelGrid::t_Cell cell;

            if (flag.getFlag1())
            {
                cell = 
                    TwoLevelGrid::t_Cell(
                    tex3D(texTopLevelCells, aState.cellId.x,
                    aState.cellId.y, aState.cellId.z));
            }


            const bool secondLvlFlag = flag.getFlag1() && cell.notEmpty();
            flag.setFlag2(secondLvlFlag);

            SmallTraversalState subState;
            vec3f subCellSize;

            if (flag.getFlag2())
            {

                const vec3f entryPoint = aRayOrg + 
                    vec3f::fastDivide(vec3f::rep(aState.tEntry), rayDirRCP);

                vec3f cellRes;
                cellRes.x = (float)cell[0];
                cellRes.y = (float)cell[1];
                cellRes.z = (float)cell[2];

                const vec3f subCellSizeRCP = aGridParameters.getCellSizeRCP() * cellRes; 
                const vec3f minBound = aState.cellId * aGridParameters.getCellSize()
                    + aGridParameters.bounds.min;

                subState.cellId.x = floorf((entryPoint - minBound).x * subCellSizeRCP.x);
                subState.cellId.y = floorf((entryPoint - minBound).y * subCellSizeRCP.y);
                subState.cellId.z = floorf((entryPoint - minBound).z * subCellSizeRCP.z);

                subCellSize.x = fastDivide(1.f, subCellSizeRCP.x);
                subCellSize.y = fastDivide(1.f, subCellSizeRCP.y);
                subCellSize.z = fastDivide(1.f, subCellSizeRCP.z);

                subState.cellId = vec3f::min(cellRes - vec3f::rep(1.f), 
                    vec3f::max(subState.cellId, vec3f::rep(0.f)));

                subState.tMax.x = ((subState.cellId.x + 
                    ((rayDirRCP.x > 0.f) ? 1 : 0 ))
                    * subCellSize.x + minBound.x
                    - aRayOrg.x) * rayDirRCP.x;
                subState.tMax.y = ((subState.cellId.y + 
                    ((rayDirRCP.y > 0.f) ? 1 : 0 ))
                    * subCellSize.y + minBound.y
                    - aRayOrg.y) * rayDirRCP.y;
                subState.tMax.z = ((subState.cellId.z +
                    ((rayDirRCP.z > 0.f) ? 1 : 0 ))
                    * subCellSize.z + minBound.z
                    - aRayOrg.z) * rayDirRCP.z;
            }

            while (ANY(flag.getFlag2()))
            {
                uint2 cellRange = make_uint2(0u, 0u);

                if (flag.getFlag2())
                {
                    //NOTE: Do not normalize coordinates!
                    cellRange = tex1Dfetch(texLeafCells, cell.getLeafRangeBegin()
                        + (uint)(subState.cellId.x + (float)cell[0] * subState.cellId.y
                        + (float)cell[0] * (float)cell[1] * subState.cellId.z));
                }

#ifdef GATHERSTATISTICS
                oRadiance.z += (float)(cellRange.y - cellRange.x);
#endif
                tIntersector    intersect;
                intersect(aRayOrg, rayDirRCP, oRayT, cellRange, aScene, aSharedMem, oBestHit);


                if (oRayT < 0.9999f)
                    flag.setFlag2To0();

                if (flag.getFlag2())
                {
                    const bool secondLvlFlag =
                        traverseCell(rayDirRCP, subCellSize, cell, subState);

                    flag.setFlag2(secondLvlFlag);
                }

            }

            if (oRayT < 0.9999f)
                flag.setFlag1To0();

            if (flag.getFlag1())
            {
                const bool topLvlFlag =
                    traverseMacroCell(rayDirRCP, aGridParameters, aState);
                flag.setFlag1(topLvlFlag);
            }

        }
        //end of traversal loop
        //////////////////////////////////////////////////////////////////////////
    }

};

#endif // TWOLEVELGRIDTRAVERSER_H_INCLUDED_CE63E087_B88D_4675_932B_0A7BAD291CCF
