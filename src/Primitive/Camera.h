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

#ifndef CAMERA_H_INCLUDED_C17E3896_6CC1_4748_BFD0_A298686254DF
#define CAMERA_H_INCLUDED_C17E3896_6CC1_4748_BFD0_A298686254DF

#include "../Core/Algebra.hpp"

//A perspective camera implementation
class Camera
{
public:
	vec3f mCenter;
	vec3f mTopLeft;
	vec3f mStepX, mStepY;

    HOST void init(const vec3f &aCenter, const vec3f &aForward, const vec3f &aUp, 
        float aVertOpeningAngInGrad, const uint aResX, const uint aResY)
    {
        mCenter = aCenter;

        float aspect_ratio = static_cast<float>(aResX) / static_cast<float>(aResY);

        vec3f forward_axis = ~aForward;
        vec3f right_axis = ~(forward_axis % aUp);
        vec3f up_axis = ~(forward_axis % right_axis);

        float angleInRad = aVertOpeningAngInGrad * static_cast<float>(M_PI) / 180.f;
        vec3f row_vector = 2.f * right_axis * tanf(angleInRad / 2.f) * aspect_ratio;
        vec3f col_vector = 2.f * up_axis * tanf(angleInRad / 2.f);

        mStepX = row_vector / static_cast<float>(aResX);
        mStepY = col_vector / static_cast<float>(aResY);
        mTopLeft = forward_axis - row_vector / 2.f - col_vector / 2.f;

    }

    DEVICE vec3f getPosition() const
    {
        return mCenter;
    }

	DEVICE vec3f getDirection(float aX, float aY) const
    {
		return mTopLeft + aX * mStepX + aY * mStepY;
	}
};

#endif // CAMERA_H_INCLUDED_C17E3896_6CC1_4748_BFD0_A298686254DF
