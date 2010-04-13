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

#ifndef FRAMEBUFFER_H_INCLUDED_E9F3D1EE_2396_43D1_914E_50CE9BA438AD
#define FRAMEBUFFER_H_INCLUDED_E9F3D1EE_2396_43D1_914E_50CE9BA438AD

#include "../Core/Algebra.hpp"
#include "../CUDAStdAfx.h"

struct FrameBuffer
{
    uint resX, resY;
    vec3f* deviceData;

    DEVICE vec3f& operator[](const uint aId)
    {
        return deviceData[aId];
    }

    DEVICE const vec3f& operator[](const uint aId) const
    {
        return deviceData[aId];
    }

    HOST void init(uint aResX, uint aResY)
    {
        resX = aResX;
        resY = aResY;
        CUDA_SAFE_CALL( 
            cudaMalloc((void**) &deviceData, aResX * aResY * sizeof(vec3f)));
    }

    HOST void download(vec3f* aTarget) const
    {
        CUDA_SAFE_CALL(
            cudaMemcpy(
                aTarget,
                deviceData,
                resX * resY * sizeof(vec3f),
                cudaMemcpyDeviceToHost) );
    }

    HOST void download(vec3f* aTarget, const int aResX, const int aResY) const
    {
        CUDA_SAFE_CALL(
            cudaMemcpy(
            aTarget,
            deviceData,
            aResX * aResY * sizeof(vec3f),
            cudaMemcpyDeviceToHost) );
    }

    HOST void setZero()
    {
        CUDA_SAFE_CALL(
            cudaMemset(
                (void*) deviceData,
                0,
                resX * resY * sizeof(vec3f)));
    }

    HOST void cleanup()
    {
        CUDA_SAFE_CALL( cudaFree(deviceData) );
    }
};

#endif // FRAMEBUFFER_H_INCLUDED_E9F3D1EE_2396_43D1_914E_50CE9BA438AD
