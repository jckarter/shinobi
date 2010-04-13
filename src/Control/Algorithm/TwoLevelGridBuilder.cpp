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

#include "StdAfx.hpp"
#include "TwoLevelGridBuilder.hpp"
#include "CellExtractor.hpp"

void TwoLevelGridBuilderHost::hostBuild(
        TwoLevelGridCell*&                              oGridCells,
        uint2*&                                         oLeaves,
        uint&                                           oNumLeaves,
        uint*&                                          oFaceSoupData,
        uint&                                           oNumReferences,
        int                                             aResX,
        int                                             aResY,
        int                                             aResZ,
        vec3f                                           aRes,
        vec3f                                           aMinBound,
        vec3f                                           aMaxBound,
        const FWObject::t_FaceIterator&                 aBegin,
        const FWObject::t_FaceIterator&                 aEnd,
        const FWObject&                                 aData)
{
    vec3f cellSize = (aMaxBound - aMinBound) /
        aRes;

    vec3f cellSizeRCP = vec3f::rep(1.f) / cellSize;

    //initialize temporary cell storage (top level)
    typedef std::vector<std::vector<uint> > t_FaceStorage;
    typedef std::vector<uint>               t_FaceSubStorage;
    t_FaceStorage gridCells;
    gridCells.resize(aResX * aResY * aResZ);

    for (t_FaceStorage::iterator cellIt = gridCells.begin(); 
        cellIt != gridCells.end(); ++cellIt)
    {
        cellIt->clear();
    }

    //fill top level cells with triangle copies
    std::vector<uint> cellIds;
    uint instancesCount = 0u;
    for(FWObject::t_FaceIterator faceIt = aBegin; 
        faceIt != aEnd; ++faceIt)
    {
        cellIds.clear();
        Triangle currentTriangle;
        currentTriangle.vertices[0] = aData.getVertex(faceIt->vert1);
        currentTriangle.vertices[1] = aData.getVertex(faceIt->vert2);
        currentTriangle.vertices[2] = aData.getVertex(faceIt->vert3);

        //NOTE: If /O2 next call will not terminate
        //Reason is probably that passed value of currentTriangle is not
        //initialized. Maybe due to compiler bug.
        CellExtractor aCellExtractor;

        aCellExtractor(currentTriangle, cellIds, aRes,
            aMinBound, aMaxBound, cellSize, cellSizeRCP);

        for (size_t i = 0u; i < cellIds.size(); ++i)
        {
            ++instancesCount;
            gridCells[cellIds[i]].push_back(static_cast<uint>(faceIt - aBegin));
        }
    }

    //construct bottom-level cells for each top-level one
    const vec3f cellDiagonal = cellSize;
    const float cellVolume  = cellDiagonal.x * cellDiagonal.y * cellDiagonal.z;
    const float cellLambda  = 5.f / cellVolume;
    std::vector<uint> faceSoup;
    std::vector<uint2> leaves;
    for (t_FaceStorage::iterator macroCellIt = gridCells.begin();
        macroCellIt != gridCells.end(); ++macroCellIt)
    {
        uint cellId = static_cast<uint>(macroCellIt - gridCells.begin());
        oGridCells[cellId].clear();
        size_t primitiveCount = macroCellIt->size();
        //do not refine if cell is small
        if (primitiveCount <= MINPRIMITIVESPERLEAF)
        {
            oGridCells[cellId].setNotEmpty();
            if (primitiveCount == 0)
            {
                oGridCells[cellId].setEmpty();
            }
            oGridCells[cellId].setX(1u);
            oGridCells[cellId].setY(1u);
            oGridCells[cellId].setZ(1u);
            oGridCells[cellId].setLeafRangeBegin(static_cast<uint>(leaves.size()));

            uint2 leaf;
            leaf.x = static_cast<uint>(faceSoup.size());
            leaf.y = leaf.x + static_cast<uint>(primitiveCount);
            leaves.push_back(leaf);

            for (uint face = 0; face < primitiveCount; ++face)
            {
                faceSoup.push_back((*macroCellIt)[face]);
            }

            continue;
        }
        //cell resolution
        const float magicConst = 
            powf(cellLambda * static_cast<float>(primitiveCount), 0.3333333f);                
        vec3f resolution = cellDiagonal * magicConst;

        oGridCells[cellId].setNotEmpty();
        oGridCells[cellId].setX(static_cast<uint>(resolution.x));
        oGridCells[cellId].setY(static_cast<uint>(resolution.y));
        oGridCells[cellId].setZ(static_cast<uint>(resolution.z));
        oGridCells[cellId].setLeafRangeBegin(static_cast<uint>(leaves.size()));

        resolution.x = static_cast<float>(oGridCells[cellId][0]);
        resolution.y = static_cast<float>(oGridCells[cellId][1]);
        resolution.z = static_cast<float>(oGridCells[cellId][2]);

        //cell bounds (for primitive insertion)
        vec3f minBound;
        minBound.x = static_cast<float>(( cellId % (aResX * aResY) ) % aResX);
        minBound.y = static_cast<float>(( cellId % (aResX * aResY) ) / aResX);
        minBound.z = static_cast<float>( cellId / (aResX * aResY));
        minBound = minBound * cellSize;
        minBound = minBound + aMinBound;
        vec3f maxBound = minBound + cellDiagonal;

        t_FaceStorage localGrid;
        localGrid.resize(oGridCells[cellId][0] * oGridCells[cellId][1] * oGridCells[cellId][2]);

        for (t_FaceStorage::iterator localFaceIt = localGrid.begin();
            localFaceIt != localGrid.end(); ++localFaceIt)
        {
            localFaceIt->clear();
        }


        //compute occupants of each leaf cell
        std::vector<uint> cellIds;
        uint instancesCountLocal = 0u;
        for(t_FaceSubStorage::const_iterator itInner = macroCellIt->begin();
            itInner != macroCellIt->end(); ++itInner)
        {
            cellIds.clear();
            Triangle currentTriangle;
            currentTriangle.vertices[0] = aData.getVertex((aBegin + *itInner)->vert1);
            currentTriangle.vertices[1] = aData.getVertex((aBegin + *itInner)->vert2);
            currentTriangle.vertices[2] = aData.getVertex((aBegin + *itInner)->vert3);

            CellExtractor aCellExtractor;

            aCellExtractor(currentTriangle, cellIds, resolution,
                minBound, maxBound, cellDiagonal / resolution,
                resolution / cellDiagonal);
            for (size_t i = 0u; i < cellIds.size(); ++i)
            {
                ++instancesCountLocal;
                localGrid[cellIds[i]].push_back(*itInner);
            }
        }

        //compute leaf cells and append primitive instances to the soup
        for (t_FaceStorage::const_iterator it = localGrid.begin();
            it != localGrid.end(); ++it)
        {
            uint2 leaf;
            leaf.x = static_cast<uint>(faceSoup.size());
            leaf.y = leaf.x + static_cast<uint>(it->size());
            leaves.push_back(leaf);

            for (uint face = 0; face < it->size(); ++face)
            {
                faceSoup.push_back((*it)[face]);
            }
        }//end for(t_FaceStorage::const_iterator it ... (leaf computation)
    }//end  for (t_FaceStorage::iterator macroCellIt ... (top level cell computation)

    oLeaves = new uint2[leaves.size()];
    oNumLeaves = static_cast<uint>(leaves.size());
    for(size_t i = 0; i < leaves.size(); ++i)
    {
        oLeaves[i] = leaves[i];
    }

    oFaceSoupData = new uint[faceSoup.size()];
    oNumReferences = static_cast<uint>(faceSoup.size());
    for(size_t i = 0; i < faceSoup.size(); ++i)
    {
        oFaceSoupData[i] = faceSoup[i];
    }

}
