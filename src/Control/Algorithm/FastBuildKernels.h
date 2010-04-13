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

#ifndef FASTBUILDKERNELS_H_INCLUDED_AE81568F_99A7_4D2C_8C3E_D2180D26D33F
#define FASTBUILDKERNELS_H_INCLUDED_AE81568F_99A7_4D2C_8C3E_D2180D26D33F

template<class tPrimitiveStorage>
GLOBAL void countRefs(
                      tPrimitiveStorage aFaceSoup,
                      const uint        aNumTriangles,
                      const vec3f       aGridRes,
                      const vec3f       aBoundsMin,
                      const vec3f       aCellSize,
                      const vec3f       aCellSizeRCP,
                      uint*             oRefCounts
                      )
{
    extern SHARED uint shMem[];
    shMem[threadId1D()] = 0u;

    for(int triangleId = globalThreadId1D(); triangleId < aNumTriangles; triangleId += numThreads())
    {
        const Triangle tri = aFaceSoup(triangleId);
        BBox triangleBounds = BBox::empty();
        triangleBounds.extend(tri.vertices[0]);
        triangleBounds.extend(tri.vertices[1]);
        triangleBounds.extend(tri.vertices[2]);
        
        vec3f& minCellIdf = ((vec3f*)(shMem + blockSize()))[threadId1D()];
        minCellIdf =
            vec3f::max(vec3f::rep(0.f), (triangleBounds.min - aBoundsMin) * aCellSizeRCP + vec3f::rep(-EPS));
        const vec3f maxCellIdf =
            vec3f::min(aGridRes - vec3f::rep(1.f), (triangleBounds.max - aBoundsMin) * aCellSizeRCP + vec3f::rep(EPS));

        const uint minCellIdX =  (uint)(minCellIdf.x);
        const uint minCellIdY =  (uint)(minCellIdf.y);
        const uint minCellIdZ =  (uint)(minCellIdf.z);

        const uint maxCellIdX =  (uint)(maxCellIdf.x);
        const uint maxCellIdY =  (uint)(maxCellIdf.y);
        const uint maxCellIdZ =  (uint)(maxCellIdf.z);

        shMem[threadId1D()] += (maxCellIdX - minCellIdX + 1)
            * (maxCellIdY - minCellIdY + 1)
            * (maxCellIdZ - minCellIdZ + 1);
    }

    SYNCTHREADS;

    //reduction
    if (NUMBUILDTHREADS >= 256) { if (threadId1D() < 128) { shMem[threadId1D()] += shMem[threadId1D() + 128]; } SYNCTHREADS;   }
    if (NUMBUILDTHREADS >= 128) { if (threadId1D() <  64) { shMem[threadId1D()] += shMem[threadId1D() +  64]; } SYNCTHREADS;   }
    if (NUMBUILDTHREADS >=  64) { if (threadId1D() <  32) { shMem[threadId1D()] += shMem[threadId1D() +  32]; } EMUSYNCTHREADS;}
    if (NUMBUILDTHREADS >=  32) { if (threadId1D() <  16) { shMem[threadId1D()] += shMem[threadId1D() +  16]; } EMUSYNCTHREADS;}
    if (NUMBUILDTHREADS >=  16) { if (threadId1D() <   8) { shMem[threadId1D()] += shMem[threadId1D() +   8]; } EMUSYNCTHREADS;}
    if (NUMBUILDTHREADS >=   8) { if (threadId1D() <   4) { shMem[threadId1D()] += shMem[threadId1D() +   4]; } EMUSYNCTHREADS;}
    if (NUMBUILDTHREADS >=   4) { if (threadId1D() <   2) { shMem[threadId1D()] += shMem[threadId1D() +   2]; } EMUSYNCTHREADS;}
    if (NUMBUILDTHREADS >=   2) { if (threadId1D() <   1) { shMem[threadId1D()] += shMem[threadId1D() +   1]; } EMUSYNCTHREADS;}

    // write out block sum 
    if (threadId1D() == 0) oRefCounts[blockId1D()] = shMem[0];
}

template<class tPrimitiveStorage>
GLOBAL void buildUnsortedGrid(
                              tPrimitiveStorage aFaceSoup,
                              uint*             oIndices,
                              const uint        aNumTriangles,
                              uint*             aStartId,
                              const vec3f       aGridRes,
                              const vec3f       aBoundsMin,
                              const vec3f       aCellSize,
                              const vec3f       aCellSizeRCP
                              )
{
    extern SHARED uint shMem[];

    if (threadId1D() == 0)
    {
        shMem[0] = aStartId[blockId1D()];
    }
    SYNCTHREADS;

    for(int triangleId = globalThreadId1D(); triangleId < aNumTriangles; triangleId += numThreads())
    {
        const Triangle tri = aFaceSoup(triangleId);
        BBox triangleBounds = BBox::empty();
        triangleBounds.extend(tri.vertices[0]);
        triangleBounds.extend(tri.vertices[1]);
        triangleBounds.extend(tri.vertices[2]);
        const vec3f minCellIdf =
            vec3f::max(vec3f::rep(0.f), 
            (triangleBounds.min - aBoundsMin) * aCellSizeRCP + vec3f::rep(-EPS));
        const vec3f maxCellIdf =
            vec3f::min(aGridRes - vec3f::rep(1.f),
            (triangleBounds.max - aBoundsMin) * aCellSizeRCP + vec3f::rep(EPS));

        const uint minCellIdX =  (uint)(minCellIdf.x);
        const uint minCellIdY =  (uint)(minCellIdf.y);
        const uint minCellIdZ =  (uint)(minCellIdf.z);

        const uint maxCellIdX =  (uint)(maxCellIdf.x);
        const uint maxCellIdY =  (uint)(maxCellIdf.y);
        const uint maxCellIdZ =  (uint)(maxCellIdf.z);

        const uint numCells =
            (maxCellIdX - minCellIdX + 1u) *
            (maxCellIdY - minCellIdY + 1u) *
            (maxCellIdZ - minCellIdZ + 1u);

        uint nextSlot  = atomicAdd(&shMem[0], numCells);

        if (    maxCellIdX - minCellIdX > 0
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
            minCellCenter =  minCellCenter * aCellSize;
            minCellCenter = minCellCenter + aBoundsMin + gridCellSizeHALF;

            vec3f& cellCenter = ((vec3f*)(shMem + 1))[threadId1D()];
            cellCenter.z = minCellCenter.z - aCellSize.z;

            for (uint z = minCellIdZ; z <= maxCellIdZ; ++z)
            {
                cellCenter.z += aCellSize.z;
                cellCenter.y = minCellCenter.y - aCellSize.y;

                for (uint y = minCellIdY; y <= maxCellIdY; ++y)
                {
                    cellCenter.y += aCellSize.y;
                    cellCenter.x = minCellCenter.x - aCellSize.x;

                    for (uint x = minCellIdX; x <= maxCellIdX; ++x, ++nextSlot)
                    {
                        cellCenter.x += aCellSize.x;

                        const vec3f distToPlane = normal *
                            (cellCenter - tri.vertices[0]).dot(normal);

                        if (fabsf(distToPlane.x) <= gridCellSizeHALF.x + EPS &&
                            fabsf(distToPlane.y) <= gridCellSizeHALF.y + EPS &&
                            fabsf(distToPlane.z) <= gridCellSizeHALF.z + EPS )
                        {
                            oIndices[2 * nextSlot] = x +
                                y * (uint)aGridRes.x +
                                z * (uint)(aGridRes.x * aGridRes.y);

                            oIndices[2 * nextSlot + 1] =
                                triangleId;
                        }
                        else
                        {
                            oIndices[2 * nextSlot] = 
                                (uint)(aGridRes.x * aGridRes.y * aGridRes.z);

                            oIndices[2 * nextSlot + 1] = 
                                triangleId;
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
                    for (uint x = minCellIdX; x <= maxCellIdX; ++x, ++nextSlot)
                    {
                        oIndices[2 * nextSlot] = x +
                            y * (uint)aGridRes.x +
                            z * (uint)(aGridRes.x * aGridRes.y);
                        oIndices[2 * nextSlot + 1] = triangleId;
                    }//end for z
                }//end for y
            }//end for x
        }//end if (maxCellIdX - minCellIdX > 0...

    }
}


template<class tPrimitiveStorage>
GLOBAL void prepareCellRanges(
                              tPrimitiveStorage oFaceSoup,
                              uint2*            aSortedPairs,
                              const uint        aNumTrianlgeInstances,
                              cudaPitchedPtr    aGridCellsPtr,
                              const uint        aGridResX,
                              const uint        aGridResY,
                              const uint        aGridResZ
                              )
{
    extern SHARED uint shMem[];

    //padding
    if (threadId1D() == 0)
    {
        shMem[0] = 0u;
        shMem[blockSize()] = 0u;
    }

    uint *padShMem = shMem + 1;
    padShMem[threadId1D()] = 0u;

    SYNCTHREADS;

    for(int instanceId = globalThreadId1D();
        instanceId < aNumTrianlgeInstances + blockSize() - 1;
        instanceId += numThreads())
    {
        //load blockSize() + 2 input elements in shared memory

        SYNCTHREADS;

        if (threadId1D() == 0 && instanceId > 0u)
        {
            //padding left
            shMem[0] = aSortedPairs[instanceId - 1].x;
        }
        if (threadId1D() == 0 && instanceId + blockSize() < aNumTrianlgeInstances)
        {
            //padding right
            padShMem[blockSize()] = aSortedPairs[instanceId + blockSize()].x;
        }
        if (instanceId < aNumTrianlgeInstances)
        {
            padShMem[threadId1D()] = aSortedPairs[instanceId].x;
        }

        SYNCTHREADS;

        //Check if the two neighboring cell indices are different
        //which means that at this point there is an end of and a begin of a range

        //compare left neighbor
        if (instanceId > 0 && instanceId < aNumTrianlgeInstances && padShMem[threadId1D()] != shMem[threadId1D()])
        {
            //begin of range
            uint cellIdX =  padShMem[threadId1D()] % aGridResX;
            uint cellIdY = (padShMem[threadId1D()] % (aGridResX * aGridResY)) / aGridResX;
            uint cellIdZ =  padShMem[threadId1D()] / (aGridResX * aGridResY);

            uint2* cell = (uint2*)((char*)aGridCellsPtr.ptr
                + cellIdY * aGridCellsPtr.pitch
                + cellIdZ * aGridCellsPtr.pitch * aGridCellsPtr.ysize) + cellIdX;

            cell->x = instanceId;
        }

        //compare right neighbor
        if (instanceId < aNumTrianlgeInstances && padShMem[threadId1D()] != padShMem[threadId1D() + 1])
        {
            //end of range
            uint cellIdX =  padShMem[threadId1D()] % aGridResX;
            uint cellIdY = (padShMem[threadId1D()] % (aGridResX * aGridResY)) / aGridResX;
            uint cellIdZ =  padShMem[threadId1D()] / (aGridResX * aGridResY);

            uint2* cell = (uint2*)((char*)aGridCellsPtr.ptr
                + cellIdY * aGridCellsPtr.pitch
                + cellIdZ * aGridCellsPtr.pitch * aGridCellsPtr.ysize) + cellIdX;

            cell->y = instanceId + 1;
        }

    }//end for(uint startId = blockId1D() * blockSize()...

    SYNCTHREADS;

    //compact triangle indices from aInIndices to oFaceSoup.indices
    for(int instanceId = globalThreadId1D();
        instanceId < aNumTrianlgeInstances;
        instanceId += numThreads())
    {
        oFaceSoup.indices[instanceId] = aSortedPairs[instanceId].y;
    }

}


template<class tPrimitiveStorage>
GLOBAL void checkGridCells(tPrimitiveStorage    triangles,
                           cudaPitchedPtr       aGridCellsPtr,
                           const vec3f          aGridRes)
{

    uint2* cell = (uint2*)((char*)aGridCellsPtr.ptr
        + blockIdx.x * aGridCellsPtr.pitch
        + blockIdx.y * aGridCellsPtr.pitch * aGridCellsPtr.ysize) + threadIdx.x;

    uint2 cellRange = *cell;

    for(uint it = cellRange.x; it != cellRange.y; ++ it)
    {
        Triangle tri = triangles[it];
    }

    cellRange.x = aGridCellsPtr.pitch / sizeof(uint2);
}


#endif // FASTBUILDKERNELS_H_INCLUDED_AE81568F_99A7_4D2C_8C3E_D2180D26D33F
