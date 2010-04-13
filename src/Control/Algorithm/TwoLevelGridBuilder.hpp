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

#ifndef TWOLEVELGRIDBUILDER_HPP_INCLUDED_F9EE6DF8_8A49_44F0_AED8_5BDF6FAF945F
#define TWOLEVELGRIDBUILDER_HPP_INCLUDED_F9EE6DF8_8A49_44F0_AED8_5BDF6FAF945F

#include "../../Loaders/FWObject.hpp"
#include "../Structure/TwoLevelGrid.hpp"

#define MINPRIMITIVESPERLEAF 16

class TwoLevelGridBuilderHost
{
public:
    static void hostBuild(
        TwoLevelGridCell*&                              oGridCells,
        uint2*&                                         oLeaves,
        uint&                                           oNumLeaves,
        uint*&                                          oFaceSoupData,
        uint&                                           oNumInstances,
        int                                             aResX,
        int                                             aResY,
        int                                             aResZ,
        vec3f                                           aRes,
        vec3f                                           aMinBound,
        vec3f                                           aMaxBound,
        const FWObject::t_FaceIterator&                 aBegin,
        const FWObject::t_FaceIterator&                 aEnd,
        const FWObject&                                 aData);
};

#endif // TWOLEVELGRIDBUILDER_HPP_INCLUDED_F9EE6DF8_8A49_44F0_AED8_5BDF6FAF945F
