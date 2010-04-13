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

#ifndef LIGHTSOURCE_HPP_INCLUDED_AE49484E_6EEA_49A3_BBF7_AA1E175B4CB9
#define LIGHTSOURCE_HPP_INCLUDED_AE49484E_6EEA_49A3_BBF7_AA1E175B4CB9

#include "../CUDAStdAfx.h"

#include "../Core/Algebra.hpp"

class AreaLightSource
{
public:
    vec3f position, normal, intensity, edge1, edge2;

    DEVICE HOST vec3f getPoint(float aXCoord, float aYCoord) const
    {
        return position + aXCoord * edge1 + aYCoord * edge2;
    }

    DEVICE HOST float getArea() const
    {
        return (edge1 % edge2).len();
    }


};


#endif // LIGHTSOURCE_HPP_INCLUDED_AE49484E_6EEA_49A3_BBF7_AA1E175B4CB9
