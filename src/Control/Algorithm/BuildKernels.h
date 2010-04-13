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

#ifndef GRIDOPERATOR_H_INCLUDED_1A94BF40_DABC_476F_8611_A4AF24798068
#define GRIDOPERATOR_H_INCLUDED_1A94BF40_DABC_476F_8611_A4AF24798068

#include "../../CUDAStdAfx.h"
#include "../Structure/FaceSoup.h"


template < bool taWriteTriangleIndex, bool taPreciseInsertion, class tPrimitiveStorage >
class GridOperator
{
    //////////////////////////////////////////////////////////////////////////
    //axis tests
    //////////////////////////////////////////////////////////////////////////

    DEVICE bool axisTest(
        const float a,
        const float b,
        const float fa,
        const float fb,
        const float v0a,
        const float v0b,
        const float v1a,
        const float v1b,
        const float aCellSizeHALFa,
        const float aCellSizeHALFb)
    {
        const float p0 = a * v0a + b * v0b;
        const float p1 = a * v1a + b * v1b;

        const float minP = fminf(p0, p1);
        const float maxP = fmaxf(p0, p1);

        const float rad = fa * aCellSizeHALFa + fb * aCellSizeHALFb;

        return ! (minP > rad + EPS || maxP + EPS < -rad);

    }

#define AXISTEST_X01(e, fe, v0, v1, v2, s)                                     \
    axisTest(e.z, -e.y, fe.z, fe.y, v0.y, v0.z, v2.y, v2.z, s.y, s.z)

#define AXISTEST_X2(e, fe, v0, v1, v2, s)                                      \
    axisTest(e.z, -e.y, fe.z, fe.y, v0.y, v0.z, v1.y, v1.z, s.y, s.z)

#define AXISTEST_Y02(e, fe, v0, v1, v2, s)                                     \
    axisTest(-e.z, e.x, fe.z, fe.x, v0.x, v0.z, v2.x, v2.z, s.x, s.z)

#define AXISTEST_Y1(e, fe, v0, v1, v2, s)                                      \
    axisTest(-e.z, e.x, fe.z, fe.x, v0.x, v0.z, v1.x, v1.z, s.x, s.z)

#define AXISTEST_Z12(e, fe, v0, v1, v2, s)                                     \
    axisTest(e.y, -e.x, fe.y, fe.x, v1.x, v1.y, v2.x, v2.y, s.x, s.y)

#define AXISTEST_Z0(e, fe, v0, v1, v2, s)                                      \
    axisTest(e.y, -e.x, fe.y, fe.x, v0.x, v0.y, v1.x, v1.y, s.x, s.y)

    //////////////////////////////////////////////////////////////////////////

public:
    DEVICE void operator()(
        tPrimitiveStorage aFaceSoup,
        const uint aNumTriangles,
        cudaPitchedPtr aGridCellsPtr,
        const vec3f aGridRes,
        const vec3f aBoundsMin,
        const vec3f aCellSize,
        const vec3f aCellSizeRCP
        )
    {
        for(int triangleId = globalThreadId1D(); triangleId < aNumTriangles; triangleId += numThreads())
        {
            const Triangle tri = aFaceSoup(triangleId);
            BBox triangleBounds = BBox::empty();
            triangleBounds.extend(tri.vertices[0]);
            triangleBounds.extend(tri.vertices[1]);
            triangleBounds.extend(tri.vertices[2]);
            const vec3f minCellIdf =
                vec3f::max(vec3f::rep(0.f), (triangleBounds.min - aBoundsMin) * aCellSizeRCP + vec3f::rep(-EPS));
            const vec3f maxCellIdf =
                vec3f::min(aGridRes - vec3f::rep(1.f), (triangleBounds.max - aBoundsMin) * aCellSizeRCP + vec3f::rep(EPS));

            const uint minCellIdX =  (uint)(minCellIdf.x);
            const uint minCellIdY =  (uint)(minCellIdf.y);
            const uint minCellIdZ =  (uint)(minCellIdf.z);

            const uint maxCellIdX =  (uint)(maxCellIdf.x);
            const uint maxCellIdY =  (uint)(maxCellIdf.y);
            const uint maxCellIdZ =  (uint)(maxCellIdf.z);

            if (    taPreciseInsertion
                &&  maxCellIdX - minCellIdX > 0
                &&  maxCellIdY - minCellIdY > 0
                &&  maxCellIdZ - minCellIdZ > 0
                )
            {
                const vec3f normal =
                    ~((tri.vertices[1] - tri.vertices[0]) %
                    (tri.vertices[2] - tri.vertices[0]));

                const vec3f gridCellSizeHALF = aCellSize * 0.5f;
                vec3f minCellCenter;
                minCellCenter.x = (float)(minCellIdX);
                minCellCenter.y = (float)(minCellIdY);
                minCellCenter.z = (float)(minCellIdZ);
                minCellCenter = minCellCenter * aCellSize;
                minCellCenter = minCellCenter + aBoundsMin + gridCellSizeHALF;

                float cellCenterZ = minCellCenter.z - aCellSize.z;

                for (uint z = minCellIdZ; z <= maxCellIdZ; ++z)
                {
                    cellCenterZ += aCellSize.z;
                    float cellCenterY = minCellCenter.y - aCellSize.y;

                    for (uint y = minCellIdY; y <= maxCellIdY; ++y)
                    {
                        cellCenterY += aCellSize.y;
                        float cellCenterX = minCellCenter.x - aCellSize.x;

                        for (uint x = minCellIdX; x <= maxCellIdX; ++x)
                        {
                            cellCenterX += aCellSize.x;
                            vec3f cellCenter;
                            cellCenter.x = cellCenterX;
                            cellCenter.y = cellCenterY;
                            cellCenter.z = cellCenterZ;
                            const vec3f distToPlane = normal *
                                (cellCenter - tri.vertices[0]).dot(normal);

                            uint2* cell = (uint2*)((char*)aGridCellsPtr.ptr
                                + y * aGridCellsPtr.pitch
                                + z * aGridCellsPtr.pitch * aGridCellsPtr.ysize) + x;

                            //////////////////////////////////////////////////////////////////////////
                            //coordinate transform origin -> cellCenter
                            const vec3f v0 = tri.vertices[0] - cellCenter;
                            const vec3f v1 = tri.vertices[1] - cellCenter;
                            const vec3f v2 = tri.vertices[2] - cellCenter;
                            const vec3f e0 = v1 - v0;
                            const vec3f e1 = v2 - v1;
                            const vec3f e2 = v0 - v2;
                            
                            bool passedAllTests = true;
                            //9 tests for separating axis
                            vec3f fe;
                            fe.x = fabsf(e0.x);
                            fe.y = fabsf(e0.y);
                            fe.z = fabsf(e0.z);

                            passedAllTests = passedAllTests && AXISTEST_X01(e0, fe, v0, v1, v2, gridCellSizeHALF);
                            passedAllTests = passedAllTests && AXISTEST_Y02(e0, fe, v0, v1, v2, gridCellSizeHALF);
                            passedAllTests = passedAllTests && AXISTEST_Z12(e0, fe, v0, v1, v2, gridCellSizeHALF);

                            fe.x = fabsf(e1.x);
                            fe.y = fabsf(e1.y);
                            fe.z = fabsf(e1.z);

                            passedAllTests = passedAllTests && AXISTEST_X01(e1, fe, v0, v1, v2, gridCellSizeHALF);
                            passedAllTests = passedAllTests && AXISTEST_Y02(e1, fe, v0, v1, v2, gridCellSizeHALF);
                            passedAllTests = passedAllTests && AXISTEST_Z0(e1, fe, v0, v1, v2, gridCellSizeHALF);

                            fe.x = fabsf(e2.x);
                            fe.y = fabsf(e2.y);
                            fe.z = fabsf(e2.z);

                            passedAllTests = passedAllTests && AXISTEST_X2(e2, fe, v0, v1, v2, gridCellSizeHALF);
                            passedAllTests = passedAllTests && AXISTEST_Y1(e2, fe, v0, v1, v2, gridCellSizeHALF);
                            passedAllTests = passedAllTests && AXISTEST_Z12(e2, fe, v0, v1, v2, gridCellSizeHALF);

                            //////////////////////////////////////////////////////////////////////////
                            if (passedAllTests &&
                                fabsf(distToPlane.x) <= gridCellSizeHALF.x + EPS &&
                                fabsf(distToPlane.y) <= gridCellSizeHALF.y + EPS &&
                                fabsf(distToPlane.z) <= gridCellSizeHALF.z + EPS )
                            {
                                if (taWriteTriangleIndex) //compile time decision
                                {
                                    const uint index = atomicInc(&cell->y, 0x7fffffff);

                                    aFaceSoup.indices[index] = triangleId;
                                }
                                else
                                {
                                    atomicInc(&cell->y, 0x7fffffff);
                                }
                            }
                        }//end for z
                    }//end for y
                }//end for x
            }
            else
            {
                for (uint z = minCellIdZ; z <= maxCellIdZ; ++z)
                {
                    for (uint y = minCellIdY; y <= maxCellIdY; ++y)
                    {
                        for (uint x = minCellIdX; x <= maxCellIdX; ++x)
                        {
                            uint2* cell = (uint2*)((char*)aGridCellsPtr.ptr
                                + y * aGridCellsPtr.pitch
                                + z * aGridCellsPtr.pitch * aGridCellsPtr.ysize) + x;

                            if (taWriteTriangleIndex) //compile time decision
                            {
                                const uint index = atomicInc(&cell->y, 0x7fffffff);

                                aFaceSoup.indices[index] = triangleId;
                            }
                            else
                            {
                                atomicInc(&cell->y, 0x7fffffff);
                            }
                        }//end for z
                    }//end for y
                }//end for x
            }//end if (maxCellIdX - minCellIdX > 0...
        }//end for(int triangleId = globalThreadId1D()...
    }

#undef AXISTEST_X01
#undef AXISTEST_X2
#undef AXISTEST_Y02
#undef AXISTEST_Y1
#undef AXISTEST_Z12
#undef AXISTEST_Z0

};

template<class tPrimitiveStorage>
GLOBAL void estimateCellSize(tPrimitiveStorage aFaceSoup,
                             const uint aNumTriangles,
                             cudaPitchedPtr aGridCellsPtr,
                             const vec3f aGridRes,
                             const vec3f aBoundsMin,
                             const vec3f aCellSize,
                             const vec3f aCellSizeRCP)
{
    GridOperator< false, true, tPrimitiveStorage > ()(aFaceSoup, aNumTriangles, aGridCellsPtr, aGridRes,
        aBoundsMin, aCellSize, aCellSizeRCP);

}

template<class tPrimitiveStorage>
GLOBAL void fillGrid(tPrimitiveStorage aFaceSoup,
                     const uint aNumTrianlges,
                     cudaPitchedPtr aGridCellsPtr,
                     const vec3f aGridRes,
                     const vec3f aBoundsMin,
                     const vec3f aCellSize,
                     const vec3f aCellSizeRCP)
{
    GridOperator< true, true, tPrimitiveStorage> () (aFaceSoup, aNumTrianlges, aGridCellsPtr, aGridRes,
        aBoundsMin, aCellSize, aCellSizeRCP);

}


#endif // GRIDOPERATOR_H_INCLUDED_1A94BF40_DABC_476F_8611_A4AF24798068
