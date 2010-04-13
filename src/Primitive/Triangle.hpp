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

#ifndef TRIANGLE_HPP_INCLUDED_0F226604_0ABB_4534_9A08_C907E74A8CB1
#define TRIANGLE_HPP_INCLUDED_0F226604_0ABB_4534_9A08_C907E74A8CB1

#include "../Core/Algebra.hpp"

struct Triangle
{
    vec3f vertices[3];
};

struct ShevtsovTriAccel
{
    float   nu;
    float   nv;
    float   np;
    float   pu;
    float   pv;
    float   e0u;
    float   e0v;
    float   e1u;
    float   e1v;
    int    dimW;
    int    dimU;
    int    dimV;

    ShevtsovTriAccel()
    {}

    ShevtsovTriAccel(Triangle aTriangle)
    {
        vec3f normal = aTriangle.vertices[0];
        //register reuse: state.tMax should be edge1
        vec3f edge0  = aTriangle.vertices[1];
        //register reuse: state.cellId should be edge2
        vec3f edge1  = aTriangle.vertices[2];

        edge0 -= normal;
        edge1 -= normal;

        normal = (edge0 % edge1);

#define MAX_DIMENSION(aX, aY, aZ)	                           \
    (aX > aY) ? ((aX > aZ) ? 0u : 2u)	: ((aY > aZ) ? 1u : 2u)

        dimW =
            MAX_DIMENSION(fabsf(normal[0]), fabsf(normal[1]), fabsf(normal[2]));

#undef  MAX_DIMENSION

        uint mod3[5] = {0,1,2,0,1};
        dimU = mod3[dimW + 1];
        dimV = mod3[dimW + 2];

        nu = normal[dimU] / normal[dimW];
        nv = normal[dimV] / normal[dimW];

        pu = aTriangle.vertices[0][dimU];
        pv = aTriangle.vertices[0][dimV];

        np = nu * aTriangle.vertices[0][dimU]
            + nv * aTriangle.vertices[0][dimV]
            + aTriangle.vertices[0][dimW];

        float minusOnePowW = (dimW == 1) ? 1.f : 1.f;
        e0u = minusOnePowW * edge0[dimU] / normal[dimW];
        e0v = minusOnePowW * edge0[dimV] / normal[dimW];
        e1u = minusOnePowW * edge1[dimU] / normal[dimW];
        e1v = minusOnePowW * edge1[dimV] / normal[dimW];

    }
};
#endif // TRIANGLE_HPP_INCLUDED_0F226604_0ABB_4534_9A08_C907E74A8CB1
