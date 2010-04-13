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

#ifndef GRID_HPP_INCLUDED_AC8C1679_96B6_4982_8885_799D99833064
#define GRID_HPP_INCLUDED_AC8C1679_96B6_4982_8885_799D99833064

#include "../../Core/Algebra.hpp"
#include "../../Primitive/BBox.hpp"

struct GridParameters
{
    int res[3];
    vec3f cellSize;
    vec3f cellSizeRCP;
    BBox bounds;

    DEVICE vec3f getCellSize() const
    {
        return cellSize;

        //vec3f retval;
        //retval.x = (float) res[0];
        //retval.y = (float) res[1];
        //retval.z = (float) res[2];
        //return vec3f::fastDivide((bounds.max - bounds.min), retval);
    }

    DEVICE vec3f getCellSizeRCP() const
    {
        return cellSizeRCP;

        //vec3f retval;
        //retval.x = (float) res[0];
        //retval.y = (float) res[1];
        //retval.z = (float) res[2];
        //return vec3f::fastDivide(retval , (bounds.max - bounds.min));
    }
};

#endif // GRID_HPP_INCLUDED_AC8C1679_96B6_4982_8885_799D99833064
