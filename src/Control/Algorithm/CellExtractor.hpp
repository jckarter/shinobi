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

#ifndef CELLEXTRACTOR_HPP_INCLUDED_C72E4FA3_4BC6_4E59_9A21_DAFDD82587C9
#define CELLEXTRACTOR_HPP_INCLUDED_C72E4FA3_4BC6_4E59_9A21_DAFDD82587C9

#include "../../Core/Algebra.hpp"
#include "../../Primitive/Triangle.hpp"
#include "../../Primitive/BBox.hpp"


class CellExtractor
{
public:
    void operator()(
        const Triangle& aTriangle,
        std::vector<uint>& oCellIds,
        const vec3f& aGridRes,
        const vec3f& aBoundsMin,
        const vec3f& aBoundsMax,
        const vec3f& aCellSize,
        const vec3f& aCellSizeRCP) const
    {
        BBox triangleBounds = BBox::empty();
        triangleBounds.extend(aTriangle.vertices[0]);
        triangleBounds.extend(aTriangle.vertices[1]);
        triangleBounds.extend(aTriangle.vertices[2]);

        //determine cells overlapped by the triangles bounding box
        const vec3f minCellIdf = 
            vec3f::max(vec3f::rep(0.f), (triangleBounds.min - aBoundsMin) * aCellSizeRCP + vec3f::rep(-EPS));
        const vec3f maxCellIdf = 
            vec3f::min(aGridRes - vec3f::rep(1.f), (triangleBounds.max - aBoundsMin) * aCellSizeRCP + vec3f::rep(EPS));

        const uint minCellIdX =  static_cast<uint>(minCellIdf.x);
        const uint minCellIdY =  static_cast<uint>(minCellIdf.y);
        const uint minCellIdZ =  static_cast<uint>(minCellIdf.z);

        const uint maxCellIdX =  static_cast<uint>(maxCellIdf.x);
        const uint maxCellIdY =  static_cast<uint>(maxCellIdf.y);
        const uint maxCellIdZ =  static_cast<uint>(maxCellIdf.z);

        if (minCellIdX == maxCellIdX &&
            minCellIdY == maxCellIdY &&
            minCellIdZ == maxCellIdZ)
        {
            oCellIds.push_back(getCellId1D(minCellIdX, minCellIdY, minCellIdZ, aGridRes));
            return;
        }

        //see http://mathworld.wolfram.com/Point-PlaneDistance.html
        //for point-to-plane projection
        const vec3f normal = 
            ~((aTriangle.vertices[1] - aTriangle.vertices[0]) % 
              (aTriangle.vertices[2] - aTriangle.vertices[0]));

        const vec3f gridCellSizeHALF = aCellSize * 0.5f;
        vec3f minCellCenter;
        minCellCenter.x = static_cast<float>(minCellIdX);
        minCellCenter.y = static_cast<float>(minCellIdY);
        minCellCenter.z = static_cast<float>(minCellIdZ);
        minCellCenter *= aCellSize;
        minCellCenter += aBoundsMin + gridCellSizeHALF;

        float cellCenterX = minCellCenter.x - aCellSize.x; 

        for (uint x = minCellIdX; x <= maxCellIdX; ++x)
        {
            cellCenterX += aCellSize.x;    
            float cellCenterY = minCellCenter.y - aCellSize.y;

            for (uint y = minCellIdY; y <= maxCellIdY; ++y)
            {
                cellCenterY += aCellSize.y;
                float cellCenterZ = minCellCenter.z - aCellSize.z;

                for (uint z = minCellIdZ; z <= maxCellIdZ; ++z)
                {
                    cellCenterZ += aCellSize.z;
                    vec3f cellCenter;
                    cellCenter.x = cellCenterX;
                    cellCenter.y = cellCenterY;
                    cellCenter.z = cellCenterZ;

                    //check if the vector from the center of the cell
                    //to the triangle plane reaches outside the cell
                    const vec3f distToPlane = normal * 
                        (cellCenter - aTriangle.vertices[0]).dot(normal);

                    if (fabsf(distToPlane.x) <= gridCellSizeHALF.x + EPS &&
                        fabsf(distToPlane.y) <= gridCellSizeHALF.y + EPS &&
                        fabsf(distToPlane.z) <= gridCellSizeHALF.z + EPS)
                    {
                        oCellIds.push_back(getCellId1D(x,y,z, aGridRes));
                    }
                }
            }
        }

    }

    uint getCellId1D(
        const uint aX,
        const uint aY,
        const uint aZ,
        const vec3f& aGridResolution) const
    {
        return aX +
            aY * static_cast<uint>(aGridResolution.x) +
            aZ * static_cast<uint>(aGridResolution.x) *
                 static_cast<uint>(aGridResolution.y);
    }
};

class FastCellExtractor
{
public:
    void operator()(
        const BBox&     aBBox,
        uint2*&         oCells,
        const vec3f&    aGridRes,
        const vec3f&    aBoundsMin,
        const vec3f&    aBoundsMax,
        const vec3f&    aCellSize,
        const vec3f&    aCellSizeRCP,
        int&            oNumInstances) const
    {


        //determine cells overlapped by the triangles bounding box
        const vec3f minCellIdf = 
            vec3f::max(vec3f::rep(0.f), (aBBox.min - aBoundsMin) * aCellSizeRCP + vec3f::rep(-EPS));
        const vec3f maxCellIdf = 
            vec3f::min(aGridRes - vec3f::rep(1.f), (aBBox.max - aBoundsMin) * aCellSizeRCP + vec3f::rep(EPS));

        const uint minCellIdX =  static_cast<uint>(minCellIdf.x);
        const uint minCellIdY =  static_cast<uint>(minCellIdf.y);
        const uint minCellIdZ =  static_cast<uint>(minCellIdf.z);

        const uint maxCellIdX =  static_cast<uint>(maxCellIdf.x);
        const uint maxCellIdY =  static_cast<uint>(maxCellIdf.y);
        const uint maxCellIdZ =  static_cast<uint>(maxCellIdf.z);

        for (uint x = minCellIdX; x <= maxCellIdX; ++x)
        {
            for (uint y = minCellIdY; y <= maxCellIdY; ++y)
            {
                for (uint z = minCellIdZ; z <= maxCellIdZ; ++z)
                {
                    oCells[getCellId1D(x,y,z, aGridRes)].y += 1;
                    ++oNumInstances;
                }
            }
        }

    }

    uint getCellId1D(
        const uint aX,
        const uint aY,
        const uint aZ,
        const vec3f& aGridResolution) const
    {
        return aX +
            aY * static_cast<uint>(aGridResolution.x) +
            aZ * static_cast<uint>(aGridResolution.x) *
            static_cast<uint>(aGridResolution.y);
    }
};

#endif // CELLEXTRACTOR_HPP_INCLUDED_C72E4FA3_4BC6_4E59_9A21_DAFDD82587C9
