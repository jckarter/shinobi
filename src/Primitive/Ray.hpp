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

#ifndef __RAY_H_INCLUDED_CA4668D0_F683_4D3D_B7EB_63AC9E7F7633
#define __RAY_H_INCLUDED_CA4668D0_F683_4D3D_B7EB_63AC9E7F7633
#ifdef _MSC_VER
	#pragma once
#endif

#include "../Core/Algebra.hpp"

//A ray class
struct Ray
{
	vec3f org; //origin
	vec3f dir; //direction
    //float t; //best intersection distance

	Ray() {}
	Ray(const vec3f & aOrg, const vec3f & aDir)
		: org(aOrg), dir(aDir) {}

	vec3f getPoint(float aDistance)
	{
		return org + aDistance * dir;
	}
};




#endif //__RAY_H_INCLUDED_CA4668D0_F683_4D3D_B7EB_63AC9E7F7633
