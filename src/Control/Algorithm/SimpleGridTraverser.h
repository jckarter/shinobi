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

#ifndef SIMPLEGRIDTRAVERSER_H_INCLUDED_6685BE83_1558_414F_B922_65B9A1918840
#define SIMPLEGRIDTRAVERSER_H_INCLUDED_6685BE83_1558_414F_B922_65B9A1918840

#include "../../CUDAStdAfx.h"
#include "../../Core/Algebra.hpp"
#include "../Structure/SimpleGrid.h"


template <
    class tControlStructure,
    class tStorageStructure,
    class tIntersector>
class SimpleGridTraverser
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
            uint2 cellRange = make_uint2(0u, 0u);

            if (traversalFlag)
            {
                //NOTE: Do not normalize coordinates!
                cellRange = tex3D(texGridCells, aState.cellId.x,
                    aState.cellId.y, aState.cellId.z);
            }

#ifdef GATHERSTATISTICS
            oRadiance.z += (float)(cellRange.y - cellRange.x);
#endif

            tIntersector   intersect;
            intersect(aRayOrg, aRayDir, oRayT, cellRange, aScene, aSharedMem, oBestHit);

            traversalFlag &=
                oRayT > aState.tMax.x
                ||  oRayT > aState.tMax.y
                ||  oRayT > aState.tMax.z;

            traversalFlag &=
                traverseCell(aRayDir[threadId1D()], aGridParameters, aState);

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
            uint2 cellRange = make_uint2(0u, 0u);

            if (traversalFlag)
            {
                //NOTE: Do not normalize coordinates!
                cellRange = tex3D(texGridCells, aState.cellId.x,
                    aState.cellId.y, aState.cellId.z);
            }

#ifdef GATHERSTATISTICS
            oRadiance.z += (float)(cellRange.y - cellRange.x);
#endif
            tIntersector   intersect;
            intersect(aRayOrg, aRayDir, oRayT, cellRange, aScene, aSharedMem, oBestHit);

            traversalFlag &= (oRayT >= 0.9999f);
            traversalFlag &=
                traverseCell(aRayDir[threadId1D()], aGridParameters, aState);

        }
        //end of traversal loop
        //////////////////////////////////////////////////////////////////////////
    }

    DEVICE bool traverseInit(
        const vec3f&            aRayOrg,
        const vec3f&            aRayDir,
        const tControlStructure&  aGridParams,
        TraversalState&         oState) const
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

    DEVICE bool traverseCell(
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

};

template <
    class tControlStructure,
    class tStorageStructure,
    class tIntersector>
class UniformGridTraverser
{
public:
    static const uint SHAREDMEMSIZE = 
        RENDERTHREADSX * RENDERTHREADSY * 3; //3 floats per thread

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
        vec3f& rayDirRCP = (((vec3f*)aSharedMem)[threadId1D()]);

        TraversalState aState;
        bool traversalFlag = (oRayT > 0.f) &&
            traverseInit(aRayOrg, rayDirRCP, aGridParameters, aState);

        //////////////////////////////////////////////////////////////////////////
        //Traversal loop
        while (ANY(traversalFlag))
        {
            uint2 cellRange = make_uint2(0u, 0u);

            if (traversalFlag)
            {
                //NOTE: Do not normalize coordinates!
                cellRange = tex3D(texGridCells, aState.cellId.x,
                    aState.cellId.y, aState.cellId.z);
            }

#ifdef GATHERSTATISTICS
            oRadiance.z += (float)(cellRange.y - cellRange.x);
#endif

            tIntersector   intersect;
            intersect(aRayOrg, rayDirRCP, oRayT, cellRange, aScene, aSharedMem, oBestHit);

            traversalFlag &=
                oRayT > aState.tMax.x
                ||  oRayT > aState.tMax.y
                ||  oRayT > aState.tMax.z;

            traversalFlag &=
                traverseCell(rayDirRCP, aGridParameters, aState);

        }
        //end of traversal loop
        //////////////////////////////////////////////////////////////////////////
    }

    DEVICE bool traverseInit(
        const vec3f&            aRayOrg,
        const vec3f&                aRayDirRCP,
        const tControlStructure&  aGridParams,
        TraversalState&         oState) const
    {
        //////////////////////////////////////////////////////////////////////////
        //ray/box intersection test
        float tEntry, tExit;
        aGridParams.bounds.fastClip(aRayOrg, aRayDirRCP, tEntry, tExit);

        if (tExit <= tEntry || tExit < 0.f)
        {
            return false;
        }
        //end ray/box intersection test
        //////////////////////////////////////////////////////////////////////////

        const vec3f entryPt = (tEntry >= 0.f) ?
            aRayOrg + vec3f::rep(tEntry + EPS) / aRayDirRCP : aRayOrg;

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

        //         tmp.x = (aRay.dir.x > 0.f) ? 1.f : -1.f;
        //         tmp.y = (aRay.dir.y > 0.f) ? 1.f : -1.f;
        //         tmp.z = (aRay.dir.z > 0.f) ? 1.f : -1.f;
        //         oState.tDelta = aGridParams.getCellSize() / aRay.dir * tmp;

        return (oState.cellId.x != ((aRayDirRCP.x > 0.f) ? aGridParams.res[0] : -1)) 
            && (oState.cellId.y != ((aRayDirRCP.y > 0.f) ? aGridParams.res[1] : -1))
            && (oState.cellId.z != ((aRayDirRCP.z > 0.f) ? aGridParams.res[2] : -1));
    }

    DEVICE bool traverseCell(
        const vec3f&                aRayDirRCP,
        const tControlStructure&    aGridParams,
        TraversalState&             oState) const
    {
#define MIN_DIMENSION(aX, aY, aZ)	                           \
    (aX < aY) ? ((aX < aZ) ? 0 : 2)	: ((aY < aZ) ? 1 : 2)

        const int tMinDimension =
            MIN_DIMENSION(oState.tMax[0], oState.tMax[1], oState.tMax[2]);

#undef  MIN_DIMENSION

        oState.cellId[tMinDimension] += (aRayDirRCP[tMinDimension] > 0.f) ? 1.f : -1.f;
        oState.tMax[tMinDimension] +=aGridParams.cellSize[tMinDimension] * 
            fabsf(aRayDirRCP[tMinDimension]);

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

};

template <
    class tControlStructure,
    class tStorageStructure,
    class tIntersector>
class UniformGridShadowTraverser : 
    public UniformGridTraverser<tControlStructure, tStorageStructure, tIntersector>
{
    using typename 
        UniformGridTraverser<
            tControlStructure,
            tStorageStructure,
            tIntersector>::TraversalState;
public:
    UniformGridShadowTraverser():
      UniformGridTraverser<tControlStructure, tStorageStructure, tIntersector>()
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

        vec3f& rayDirRCP = (((vec3f*)aSharedMem)[threadId1D()]);

        TraversalState aState;
        bool traversalFlag = oRayT >= 0.9999f &&
            traverseInit(aRayOrg, rayDirRCP, aGridParameters, aState);

        //////////////////////////////////////////////////////////////////////////
        //Traversal loop
        while (ANY(traversalFlag))
        {
            uint2 cellRange = make_uint2(0u, 0u);

            if (traversalFlag)
            {
                //NOTE: Do not normalize coordinates!
                cellRange = tex3D(texGridCells, aState.cellId.x,
                    aState.cellId.y, aState.cellId.z);
            }

#ifdef GATHERSTATISTICS
            oRadiance.z += (float)(cellRange.y - cellRange.x);
#endif

            tIntersector   intersect;
            intersect(aRayOrg, rayDirRCP, oRayT, cellRange, aScene, 
                aSharedMem, oBestHit);

            traversalFlag &= (oRayT >= 0.9999f);
            traversalFlag &=
                traverseCell(rayDirRCP, aGridParameters, aState);
        }
        //end of traversal loop
        //////////////////////////////////////////////////////////////////////////
    }
};


template <
    class tControlStructure,
    class tStorageStructure,
    class tIntersector>
class SimpleGridPilotRayTraverser : 
    public SimpleGridTraverser<tControlStructure, tStorageStructure, tIntersector>
{
    using typename 
        SimpleGridTraverser<
        tControlStructure,
        tStorageStructure,
        tIntersector>::TraversalState;
public:
    SimpleGridPilotRayTraverser():
      SimpleGridTraverser<tControlStructure, tStorageStructure, tIntersector>()
      {}

DEVICE void traverse(
      vec3f*                                        aRayOrg,
      vec3f*                                        aRayDir,
      float&                                        oRayT,
      uint&                                         oBestHit,
      TraversalState&                               aState,
      const tControlStructure&                      aGridParameters,          
      const tStorageStructure&                      aScene,
      uint*                                         aSharedMem,

      cudaPitchedPtr                                aGpuTopLevelCells,
      char*                                         aGlobalMemoryPtr
      )
  {
      bool traversalFlag = (oRayT > 0.f) &&
          traverseInit(aRayOrg[threadId1D()], aRayDir[threadId1D()],
          aGridParameters, aState);

      //vec3f mostExpensiveCellId = aState.cellId;
      //uint maxCellCost = 0u;
      //////////////////////////////////////////////////////////////////////////
      //Traversal loop
      while (ANY(traversalFlag))
      {
          uint2 cellRange = make_uint2(0u, 0u);

          if (traversalFlag)
          {
              //NOTE: Do not normalize coordinates!
              cellRange = *((uint2*)((char*)aGpuTopLevelCells.ptr
                  + (uint)aState.cellId.y * aGpuTopLevelCells.pitch
                  + (uint)aState.cellId.z * aGpuTopLevelCells.pitch * 
                  aGpuTopLevelCells.ysize) + (uint)aState.cellId.x);
          }

          if (0 < cellRange.y - cellRange.x)
          {
              //maxCellCost = cellRange.y - cellRange.x;
              //mostExpensiveCellId = aState.cellId;

              aGlobalMemoryPtr[(uint)(aState.cellId.x +
                  aState.cellId.y * aGridParameters.res[0] +
                  aState.cellId.z * aGridParameters.res[0] *
                  aGridParameters.res[1])] = 1;
          }

          tIntersector   intersect;
          intersect(aRayOrg, aRayDir, oRayT, cellRange, aScene, aSharedMem, oBestHit);

          traversalFlag &=
              oRayT > aState.tMax.x
              ||  oRayT > aState.tMax.y
              ||  oRayT > aState.tMax.z;

          traversalFlag &=
              traverseCell(aRayDir[threadId1D()], aGridParameters, aState);

      }
      //end of traversal loop
      //////////////////////////////////////////////////////////////////////////

      //aState.cellId = mostExpensiveCellId;
  }

};
#endif // SIMPLEGRIDTRAVERSER_H_INCLUDED_6685BE83_1558_414F_B922_65B9A1918840
