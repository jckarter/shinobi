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

#ifndef GRIDBUILDER_H_INCLUDED_854FAA3A_251A_424F_9B02_31EE7EEA3ACB
#define GRIDBUILDER_H_INCLUDED_854FAA3A_251A_424F_9B02_31EE7EEA3ACB

#include "../../CUDAStdAfx.h"
#include "../../CUDAConfig.h"
#include "../Structure/SimpleGrid.h"
#include "../Structure/FaceSoup.h"
#include "../../Loaders/FWObject.hpp"

template<class tPrimitiveStorage>
class GridBuilder
{
public:
    static HOST void computeResolution(
        uint                    aNumPrimitives,
        SimpleGrid&             oGrid)
    {
        vec3f diagonal = oGrid.bounds.max - oGrid.bounds.min;

        const float volume = diagonal.x * diagonal.y * diagonal.z;
        const float lambda = 5.f;
        const float magicConstant =
            powf(lambda * static_cast<float>(aNumPrimitives) / volume, 0.3333333f);

        diagonal *= magicConstant;

        oGrid.resX = static_cast<int>(diagonal.x);
        oGrid.resY = static_cast<int>(diagonal.y);
        oGrid.resZ = static_cast<int>(diagonal.z);

        //cudastd::log::out << "Default resolution : "
        //    << static_cast<int>(diagonal.x) << " "
        //    << static_cast<int>(diagonal.y) << " "
        //    << static_cast<int>(diagonal.z) << "\n";
    }
};

#endif // GRIDBUILDER_H_INCLUDED_854FAA3A_251A_424F_9B02_31EE7EEA3ACB
