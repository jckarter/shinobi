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

#ifndef TWOLEVELBUILDKERNELS_H_INCLUDED_D8A2A2E8_80A2_4054_9FBF_B6866B5C45BC
#define TWOLEVELBUILDKERNELS_H_INCLUDED_D8A2A2E8_80A2_4054_9FBF_B6866B5C45BC

#include "../../CUDAConfig.h"
#include "../../CUDAStdAfx.h"

//Computes number of second-level-cells of a gird
//A cell=(x,y,z) is mapped to a thread like this:
//  x = threadIdx.x
//  y = blockIdx.x
//  z = blcokIdx.y
//The input cells are modified for rendering purposes
template<bool taExternLeafFlag>
GLOBAL void countLeafLevelCells(
                                const vec3f     aCellSize,
                                cudaPitchedPtr  oTopLevelCells,
                                uint*           oCellCounts,
                                char*           aLeafFlagInv = NULL
                                )
{
    const float cellVolume      = aCellSize.x * aCellSize.y * aCellSize.z;
    const float lambda          = 1.2f;

    uint cellId = globalThreadId1D();
    const uint2 cellRange = *((uint2*)((char*)oTopLevelCells.ptr + 
        + blockIdx.x * oTopLevelCells.pitch
        + blockIdx.y * oTopLevelCells.pitch * oTopLevelCells.ysize) + threadIdx.x);

    const uint numCellRefs = cellRange.y - cellRange.x;
    const float magicConstant  = 
        powf(lambda * static_cast<float>(numCellRefs) / cellVolume, 0.3333333f);

    //const vec3f res = vec3f::min(vec3f::rep(255.f), 
    //    vec3f::max(vec3f::rep(1.f), aCellSize * magicConstant));
    const vec3f res = aCellSize * magicConstant;

    bool isLeaf;
    if (taExternLeafFlag)
    {
        isLeaf = aLeafFlagInv[threadIdx.x + blockIdx.x * blockSize() +
            blockIdx.y * blockSize() * gridDim.x] == 0
            || numCellRefs <= 16u;
    }
    else
    {
        isLeaf = numCellRefs <= 16u;
    }

    const uint resX = (isLeaf) ? 1u : static_cast<uint>(res.x);
    const uint resY = (isLeaf) ? 1u : static_cast<uint>(res.y);
    const uint resZ = (isLeaf) ? 1u : static_cast<uint>(res.z);

    //number of cells
    oCellCounts[cellId] = resX * resY * resZ;

    //prepare top level cell for rendering
    TwoLevelGridCell cell;
    cell.clear();

    cell.setNotEmpty();
    if (numCellRefs == 0)
    {
        cell.setEmpty();
    }

    cell.setX(resX);
    cell.setY(resY);
    cell.setZ(resZ);
    cell.setLeafRangeBegin(cellId);

    *((TwoLevelGridCell*)((char*)oTopLevelCells.ptr + 
        + blockIdx.x * oTopLevelCells.pitch
        + blockIdx.y * oTopLevelCells.pitch * oTopLevelCells.ysize)
        + threadIdx.x) = cell;

}

//Writes where the array of sub-cells starts for each top level cell
//A cell=(x,y,z) is mapped to a thread like this:
//  x = threadIdx.x
//  y = blockIdx.x
//  z = blcokIdx.y
//This completes the preparation of the top level cells for rendering
GLOBAL void prepareTopLevelCellRanges(             
                                      uint*             aCellCounts,
                                      cudaPitchedPtr    oTopLevelCells
                                      )
{
    uint cellId = globalThreadId1D();

    const uint cellCount = aCellCounts[cellId];

    TwoLevelGridCell* cell = ((TwoLevelGridCell*)((char*)oTopLevelCells.ptr + 
        + blockIdx.x * oTopLevelCells.pitch
        + blockIdx.y * oTopLevelCells.pitch * oTopLevelCells.ysize)
        + threadIdx.x);

    cell->setLeafRangeBegin(cellCount);

}

template<class tPrimitiveStorage>
GLOBAL void countLeafLevelRefs(
                               tPrimitiveStorage aFaceSoup,
                               const uint        aNumTopLevelRefs,
                               const uint2*      aTopLevelSortedPairs,
                               cudaPitchedPtr    aTopLevelCells,
                               //const vec3f       aGridRes,
                               const uint        aGridResX,
                               const uint        aGridResY,
                               const uint        aGridResZ,
                               const vec3f       aBoundsMin,
                               const vec3f       aCellSize,
                               uint*             oRefCounts
                               //////////////////////////////////////////////////////////////////////////
                               //DEBUG
                               //, uint*             debugInfo
                               //////////////////////////////////////////////////////////////////////////
                               )
{
    extern SHARED uint shMem[];
    shMem[threadId1D()] = 0u;

    //uint* numTopLevelCells = shMem + blockSize() + threadId1D();
    uint numTopLevelCells = aGridResX * aGridResY * aGridResZ;

    for(uint refId = globalThreadId1D(); refId < aNumTopLevelRefs; refId += numThreads())
    {
        const uint2 indexPair = aTopLevelSortedPairs[refId];

        if (indexPair.x >= /***/numTopLevelCells)
        {
            break;
        }

        const Triangle tri = aFaceSoup(indexPair.y);

        //////////////////////////////////////////////////////////////////////////
        //correct, but serious precision issues
        //float tmp;
        //tmp = static_cast<float>( indexPair.x ) / aGridRes.x;
        //const int topLvlCellX = static_cast<uint>( (tmp - truncf(tmp)) * aGridRes.x );
        //tmp = static_cast<float>( indexPair.x ) / (aGridRes.x * aGridRes.y);
        //const int topLvlCellY = static_cast<uint>( (tmp - truncf(tmp)) * aGridRes.y );
        //const int topLvlCellZ = static_cast<uint>( truncf(tmp) );
        //////////////////////////////////////////////////////////////////////////

        const uint topLvlCellX = indexPair.x % aGridResX;
        const uint topLvlCellY = (indexPair.x %(aGridResX * aGridResY)) / aGridResX;
        const uint topLvlCellZ = indexPair.x / (aGridResX * aGridResY);

        BBox triangleBounds = BBox::empty();
        triangleBounds.extend(tri.vertices[0]);
        triangleBounds.extend(tri.vertices[1]);
        triangleBounds.extend(tri.vertices[2]);

        const TwoLevelGridCell topLvlCell = *((TwoLevelGridCell*)
            ((char*)aTopLevelCells.ptr + 
            + topLvlCellY * aTopLevelCells.pitch
            + topLvlCellZ * aTopLevelCells.pitch * aTopLevelCells.ysize)
            + topLvlCellX);

        vec3f topLvlCellRes;
        topLvlCellRes.x = static_cast<float>(topLvlCell[0]);
        topLvlCellRes.y = static_cast<float>(topLvlCell[1]);
        topLvlCellRes.z = static_cast<float>(topLvlCell[2]);

        vec3f topLvlCellOrigin;
        topLvlCellOrigin.x = static_cast<float>(topLvlCellX) * aCellSize.x + aBoundsMin.x;
        topLvlCellOrigin.y = static_cast<float>(topLvlCellY) * aCellSize.y + aBoundsMin.y;
        topLvlCellOrigin.z = static_cast<float>(topLvlCellZ) * aCellSize.z + aBoundsMin.z;

        const vec3f subCellSizeRCP = topLvlCellRes / aCellSize;

        //triangleBounds.tighten(topLvlCellOrigin, topLvlCellOrigin + aCellSize);

        vec3f& minCellIdf = ((vec3f*)(shMem + blockSize()))[threadId1D()];
        minCellIdf =
        //const vec3f minCellIdf =
            (triangleBounds.min - topLvlCellOrigin) * subCellSizeRCP + vec3f::rep(-0.001f);
        const vec3f maxCellIdPlus1f =
            (triangleBounds.max - topLvlCellOrigin) * subCellSizeRCP + vec3f::rep(1.001f);

        const int minCellIdX =  max(0, (int)(minCellIdf.x));
        const int minCellIdY =  max(0, (int)(minCellIdf.y));
        const int minCellIdZ =  max(0, (int)(minCellIdf.z));

        const int maxCellIdP1X =  min((int)topLvlCellRes.x, (int)(maxCellIdPlus1f.x));
        const int maxCellIdP1Y =  min((int)topLvlCellRes.y, (int)(maxCellIdPlus1f.y));
        const int maxCellIdP1Z =  min((int)topLvlCellRes.z, (int)(maxCellIdPlus1f.z));

        const int numCells =
            (maxCellIdP1X - minCellIdX) *
            (maxCellIdP1Y - minCellIdY) *
            (maxCellIdP1Z - minCellIdZ);

        //////////////////////////////////////////////////////////////////////////
        //DEBUG
        //if (numCells > 0 && numCells < 4096)
        //{
        //    shMem[threadId1D()] += numCells;
        //}
        //else
        //{
        //    uint infoId = atomicAdd(&debugInfo[0], 20);
        //    if(infoId < 20 * 4)
        //    {
        //        debugInfo           [infoId +  1]   = indexPair.x        ;
        //        debugInfo           [infoId +  2]   = topLvlCellX        ;
        //        debugInfo           [infoId +  3]   = topLvlCellY        ;
        //        debugInfo           [infoId +  4]   = topLvlCellZ        ;
        //        ((float*)debugInfo) [infoId +  5]   = topLvlCellRes.x    ;
        //        ((float*)debugInfo) [infoId +  6]   = topLvlCellRes.y    ;
        //        ((float*)debugInfo) [infoId +  7]   = topLvlCellRes.z    ;
        //        ((int*)debugInfo)   [infoId +  8]   = minCellIdX         ;
        //        ((int*)debugInfo)   [infoId +  9]   = minCellIdY         ;
        //        ((int*)debugInfo)   [infoId + 10]   = minCellIdZ         ;
        //        ((int*)debugInfo)   [infoId + 11]   = maxCellIdP1X       ;
        //        ((int*)debugInfo)   [infoId + 12]   = maxCellIdP1Y       ;
        //        ((int*)debugInfo)   [infoId + 13]   = maxCellIdP1Z       ;
        //        ((float*)debugInfo) [infoId + 14]   = minCellIdf.x       ;
        //        ((float*)debugInfo) [infoId + 15]   = minCellIdf.y       ;
        //        ((float*)debugInfo) [infoId + 16]   = minCellIdf.z       ;
        //        ((float*)debugInfo) [infoId + 17]   = maxCellIdPlus1f.x  ;
        //        ((float*)debugInfo) [infoId + 18]   = maxCellIdPlus1f.y  ;
        //        ((float*)debugInfo) [infoId + 19]   = maxCellIdPlus1f.z  ;
        //        debugInfo           [infoId + 20]   = numCells           ;
        //    }
        //}
        //////////////////////////////////////////////////////////////////////////

        shMem[threadId1D()] += numCells;
    }

    SYNCTHREADS;

    //reduction
    if (NUMBUILDTHREADS_TLG >= 512) { if (threadId1D() < 256) { shMem[threadId1D()] += shMem[threadId1D() + 256]; } SYNCTHREADS;   }
    if (NUMBUILDTHREADS_TLG >= 256) { if (threadId1D() < 128) { shMem[threadId1D()] += shMem[threadId1D() + 128]; } SYNCTHREADS;   }
    if (NUMBUILDTHREADS_TLG >= 128) { if (threadId1D() <  64) { shMem[threadId1D()] += shMem[threadId1D() +  64]; } SYNCTHREADS;   }
    if (NUMBUILDTHREADS_TLG >=  64) { if (threadId1D() <  32) { shMem[threadId1D()] += shMem[threadId1D() +  32]; } EMUSYNCTHREADS;}
    if (NUMBUILDTHREADS_TLG >=  32) { if (threadId1D() <  16) { shMem[threadId1D()] += shMem[threadId1D() +  16]; } EMUSYNCTHREADS;}
    if (NUMBUILDTHREADS_TLG >=  16) { if (threadId1D() <   8) { shMem[threadId1D()] += shMem[threadId1D() +   8]; } EMUSYNCTHREADS;}
    if (NUMBUILDTHREADS_TLG >=   8) { if (threadId1D() <   4) { shMem[threadId1D()] += shMem[threadId1D() +   4]; } EMUSYNCTHREADS;}
    if (NUMBUILDTHREADS_TLG >=   4) { if (threadId1D() <   2) { shMem[threadId1D()] += shMem[threadId1D() +   2]; } EMUSYNCTHREADS;}
    if (NUMBUILDTHREADS_TLG >=   2) { if (threadId1D() <   1) { shMem[threadId1D()] += shMem[threadId1D() +   1]; } EMUSYNCTHREADS;}

    // write out block sum 
    if (threadId1D() == 0) oRefCounts[blockId1D()] = shMem[0];
}

template<class tPrimitiveStorage>
GLOBAL void writeLeafLevelRefs(
                               tPrimitiveStorage aFaceSoup,
                               const uint        aNumTopLevelRefs,
                               const uint2*      aTopLevelSortedPairs,
                               cudaPitchedPtr    aTopLevelCells,
                               const uint        aNumLeafLevelCells,
                               uint*             aStartId,
                               //const vec3f       aGridRes,
                               const uint        aGridResX,
                               const uint        aGridResY,
                               const uint        aGridResZ,
                               const vec3f       aBoundsMin,
                               const vec3f       aCellSize,
                               uint*             oPairs
                               )
{
    extern SHARED uint shMem[];

    if (threadId1D() == 0)
    {
        shMem[0] = aStartId[blockId1D()];
    }
    SYNCTHREADS;

    uint numTopLevelCells = aGridResX * aGridResY * aGridResZ;

    for(uint refId = globalThreadId1D(); refId < aNumTopLevelRefs; refId += numThreads())
    {
        const uint2 indexPair = aTopLevelSortedPairs[refId];

        if (indexPair.x >= numTopLevelCells)
        {
            break;
        }

        const Triangle tri = aFaceSoup(indexPair.y);

        //////////////////////////////////////////////////////////////////////////
        //correct, but serious precision issues
        //float tmp;
        //tmp = static_cast<float>( indexPair.x ) / aGridRes.x;
        //const int topLvlCellX = static_cast<uint>( (tmp - truncf(tmp)) * aGridRes.x );
        //tmp = static_cast<float>( indexPair.x ) / (aGridRes.x * aGridRes.y);
        //const int topLvlCellY = static_cast<uint>( (tmp - truncf(tmp)) * aGridRes.y );
        //const int topLvlCellZ = static_cast<uint>( truncf(tmp) );
        //////////////////////////////////////////////////////////////////////////

        const uint topLvlCellX = indexPair.x % aGridResX;
        const uint topLvlCellY = (indexPair.x %(aGridResX * aGridResY)) / aGridResX;
        const uint topLvlCellZ = indexPair.x / (aGridResX * aGridResY);

        BBox triangleBounds = BBox::empty();
        triangleBounds.extend(tri.vertices[0]);
        triangleBounds.extend(tri.vertices[1]);
        triangleBounds.extend(tri.vertices[2]);

        const TwoLevelGridCell topLvlCell = *((TwoLevelGridCell*)
            ((char*)aTopLevelCells.ptr + 
            + topLvlCellY * aTopLevelCells.pitch
            + topLvlCellZ * aTopLevelCells.pitch * aTopLevelCells.ysize)
            + topLvlCellX);

        vec3f topLvlCellRes;
        topLvlCellRes.x = static_cast<float>(topLvlCell[0]);
        topLvlCellRes.y = static_cast<float>(topLvlCell[1]);
        topLvlCellRes.z = static_cast<float>(topLvlCell[2]);

        vec3f topLvlCellOrigin;
        topLvlCellOrigin.x = static_cast<float>(topLvlCellX) * aCellSize.x + aBoundsMin.x;
        topLvlCellOrigin.y = static_cast<float>(topLvlCellY) * aCellSize.y + aBoundsMin.y;
        topLvlCellOrigin.z = static_cast<float>(topLvlCellZ) * aCellSize.z + aBoundsMin.z;

        const vec3f subCellSizeRCP = topLvlCellRes / aCellSize;

        //triangleBounds.tighten(topLvlCellOrigin, topLvlCellOrigin + aCellSize);

        const vec3f minCellIdf =
            (triangleBounds.min - topLvlCellOrigin) * subCellSizeRCP + vec3f::rep(-0.001f);
        const vec3f maxCellIdPlus1f =
            (triangleBounds.max - topLvlCellOrigin) * subCellSizeRCP + vec3f::rep(1.001f);

        const int minCellIdX =  max(0, (int)(minCellIdf.x));
        const int minCellIdY =  max(0, (int)(minCellIdf.y));
        const int minCellIdZ =  max(0, (int)(minCellIdf.z));

        const int maxCellIdP1X =  min((int)topLvlCellRes.x, (int)(maxCellIdPlus1f.x));
        const int maxCellIdP1Y =  min((int)topLvlCellRes.y, (int)(maxCellIdPlus1f.y));
        const int maxCellIdP1Z =  min((int)topLvlCellRes.z, (int)(maxCellIdPlus1f.z));

        const int numCells =
            (maxCellIdP1X - minCellIdX) *
            (maxCellIdP1Y - minCellIdY) *
            (maxCellIdP1Z - minCellIdZ);

        uint nextSlot  = atomicAdd(&shMem[0], numCells);

        if (    maxCellIdP1X - minCellIdX > 1
            &&  maxCellIdP1Y - minCellIdY > 1
            &&  maxCellIdP1Z - minCellIdZ > 1
            )
        {
            const vec3f normal =
                ~((tri.vertices[1] - tri.vertices[0]) %
                (tri.vertices[2] - tri.vertices[0]));

            const vec3f subCellSize = aCellSize / topLvlCellRes;

            vec3f minCellCenter;
            minCellCenter.x = (float)(minCellIdX);
            minCellCenter.y = (float)(minCellIdY);
            minCellCenter.z = (float)(minCellIdZ);
            minCellCenter = minCellCenter * subCellSize;
            minCellCenter = minCellCenter + topLvlCellOrigin + subCellSize * 0.5f;

            vec3f& cellCenter = ((vec3f*)(shMem + 1))[threadId1D()];
            cellCenter.z = minCellCenter.z - subCellSize.z;

            for (uint z = minCellIdZ; z < maxCellIdP1Z; ++z)
            {
                cellCenter.z += subCellSize.z;
                cellCenter.y = minCellCenter.y - subCellSize.y;

                for (uint y = minCellIdY; y < maxCellIdP1Y; ++y)
                {
                    cellCenter.y += (aCellSize / topLvlCellRes).y;
                    cellCenter.x = minCellCenter.x - subCellSize.x;

                    for (uint x = minCellIdX; x < maxCellIdP1X; ++x, ++nextSlot)
                    {
                        cellCenter.x += subCellSize.x;

                        const vec3f distToPlane = normal *
                            (cellCenter - tri.vertices[0]).dot(normal);

                        if (fabsf(distToPlane.x) <= subCellSize.x * 0.5f + EPS &&
                            fabsf(distToPlane.y) <= subCellSize.y * 0.5f + EPS &&
                            fabsf(distToPlane.z) <= subCellSize.z * 0.5f + EPS )
                        {
                            oPairs[2 * nextSlot] = x +
                                y * (uint)topLvlCellRes.x +
                                z * (uint)(topLvlCellRes.x * topLvlCellRes.y) +
                                topLvlCell.getLeafRangeBegin();
                            oPairs[2 * nextSlot + 1] = indexPair.y;
                        }
                        else
                        {
                            oPairs[2 * nextSlot] = aNumLeafLevelCells;
                            oPairs[2 * nextSlot + 1] = indexPair.y;

                        }

                    }//end for z
                }//end for y
            }//end for x
        }
        else
        {
            for (uint z = minCellIdZ; z < maxCellIdP1Z; ++z)
            {
                for (uint y = minCellIdY; y < maxCellIdP1Y; ++y)
                {
                    for (uint x = minCellIdX; x < maxCellIdP1X; ++x, ++nextSlot)
                    {
                        oPairs[2 * nextSlot] = x +
                            y * (uint)topLvlCellRes.x +
                            z * (uint)(topLvlCellRes.x * topLvlCellRes.y) +
                            topLvlCell.getLeafRangeBegin();
                        oPairs[2 * nextSlot + 1] = indexPair.y;
                    }//end for z
                }//end for y
            }//end for x
        }//end if if (maxCellIdP1X - minCellIdX > 1...

    }//end  for(uint refId = globalThreadId1D(); ...

}


template<class tPrimitiveStorage>
GLOBAL void prepareLeafCellRanges(
                                  tPrimitiveStorage oFaceSoup,
                                  uint2*            aSortedPairs,
                                  const uint        aNumPairs,
                                  uint2*            oGridCells
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
        instanceId < aNumPairs + blockSize() - 1;
        instanceId += numThreads())
    {
        //load blockSize() + 2 input elements in shared memory

        SYNCTHREADS;


        if (threadId1D() == 0 && instanceId > 0u)
        {
            //padding left
            shMem[0] = aSortedPairs[instanceId - 1].x;
        }
        if (threadId1D() == 0 && instanceId + blockSize() < aNumPairs)
        {
            //padding right
            padShMem[blockSize()] = aSortedPairs[instanceId + blockSize()].x;
        }
        if (instanceId < aNumPairs)
        {
            padShMem[threadId1D()] = aSortedPairs[instanceId].x;
        }

        SYNCTHREADS;

        //Check if the two neighboring cell indices are different
        //which means that at this point there is an end of and a begin of a range

        //compare left neighbor
        if (instanceId > 0 && instanceId < aNumPairs && padShMem[threadId1D()] != shMem[threadId1D()])
        {
            //begin of range
            oGridCells[padShMem[threadId1D()]].x = instanceId;
        }

        //compare right neighbor
        if (instanceId < aNumPairs && padShMem[threadId1D()] != padShMem[threadId1D() + 1])
        {
            //end of range
            oGridCells[padShMem[threadId1D()]].y = instanceId + 1;
        }

    }//end for(uint startId = blockId1D() * blockSize()...

    SYNCTHREADS;

    //compact triangle indices from aInIndices to oFaceSoup.indices
    for(int instanceId = globalThreadId1D();
        instanceId < aNumPairs;
        instanceId += numThreads())
    {
        oFaceSoup.indices[instanceId] = aSortedPairs[instanceId].y;
    }

}

#endif // TWOLEVELBUILDKERNELS_H_INCLUDED_D8A2A2E8_80A2_4054_9FBF_B6866B5C45BC
