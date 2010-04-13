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

#ifndef PIXELSAMPLER_H_INCLUDED_53ED96B4_937B_49D3_9953_145A20687644
#define PIXELSAMPLER_H_INCLUDED_53ED96B4_937B_49D3_9953_145A20687644


#include "../CUDAStdAfx.h"

template<int taNumSamplesX, int taNumSamplesY>
class RegularPixelSampler
{
public:
    float resX, resY;

    DEVICE void getPixelCoords(float aPixelId, float aSampleId,
        float& oPixelX, float& oPixelY) const
    {
        //compute in-pixel offsets
        float tmp = aSampleId / (float) taNumSamplesX;
        oPixelY = (truncf(tmp) + 0.5f) / (float) taNumSamplesY;
        oPixelX = tmp - truncf(tmp) + 0.5f / (float) taNumSamplesX;

        //compute pixel coordinates (and sum up with the offsets)
        tmp = aPixelId / resX;
        oPixelY += truncf(tmp);
        oPixelX += (tmp - truncf(tmp)) * resX;
    }

};

class RandomPixelSampler
{
public:
    float resX, resY;

    //oPixelX and oPixelY have to be initialized with the random offsets
    DEVICE void getPixelCoords(float aPixelId, float& oPixelX, float& oPixelY) const
    {
        //compute pixel coordinates (and sum up with the offsets)
        float tmp = aPixelId / resX;
        oPixelY = truncf(tmp) + oPixelY;
        oPixelX = (tmp - truncf(tmp)) * resX + oPixelX;
    }

};
#endif // PIXELSAMPLER_H_INCLUDED_53ED96B4_937B_49D3_9953_145A20687644
