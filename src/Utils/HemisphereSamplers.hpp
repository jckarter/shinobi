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

#ifndef HEMISPHERESAMPLERS_HPP_INCLUDED_BBEEF8B0_BE58_4BD1_AED5_CFA5411D41C1
#define HEMISPHERESAMPLERS_HPP_INCLUDED_BBEEF8B0_BE58_4BD1_AED5_CFA5411D41C1

#include "../CUDAStdAfx.h"
#include "../Core/Algebra.hpp"

class UniformHemisphereSampler
{
public:
    DEVICE vec3f operator() (
        const float aRandNum1,
        const float aRandNum2) const
    {
        //inverse probability of sample: (2.f * M_PI)
        vec3f retval;

        sinCos(2.f * M_PI * aRandNum1, &retval.x, &retval.y);

        retval.x *= sqrtf(1 - aRandNum2 * aRandNum2);
        retval.y *= sqrtf(1 - aRandNum2 * aRandNum2);
        retval.z  = aRandNum2;

        return retval; 
    }
};

class CosineHemisphereSampler
{
public:
    DEVICE vec3f operator() (
        const float aRandNum1,
        const float aRandNum2) const
    {   
        //inverse probability of sample: M_PI / sqrtf(aRandNum2)
        vec3f retval;
        
        sinCos(2.f * M_PI * aRandNum1, &retval.x, &retval.y);

        retval.x *= sqrtf(1 - aRandNum2);
        retval.y *= sqrtf(1 - aRandNum2);
        retval.z  = sqrtf(aRandNum2);

        return retval;
    }
};

class PowerCosineHemisphereSampler
{
public:
    DEVICE vec3f operator() (
        const float aRandNum1,
        const float aRandNum2,
        const float aPower) const
    {
        //inverse probability of sample: 
        //2 * M_PI / ((aPower + 1) * powf(retval.z, aPower))

        vec3f retval;
        retval.z  = fastPow(aRandNum2, fastDivide(1.f, aPower + 1.f));

        sinCos(2.f * M_PI * aRandNum1, &retval.x, &retval.y);

        retval.x *= sqrtf(1 - retval.z * retval.z);
        retval.y *= sqrtf(1 - retval.z * retval.z);

        return retval;
    }
};

#endif // HEMISPHERESAMPLERS_HPP_INCLUDED_BBEEF8B0_BE58_4BD1_AED5_CFA5411D41C1
