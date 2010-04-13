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

#ifndef __INCLUDE_GUARD_19715045_A7D7_4BC2_B7A5_674B2D36055C
#define __INCLUDE_GUARD_19715045_A7D7_4BC2_B7A5_674B2D36055C
#ifdef _MSC_VER
	#pragma once
#endif

#include "../CUDAStdAfx.h"
#include "Ray.hpp"

//An axis aligned bounding box
struct BBox
{
	vec3f min, max;


	//Returns the entry and exit distances of the ray with the
	//	bounding box.
	//If the first returned distance > the second, than
	//	the ray does not intersect the bounding box at all
	DEVICE HOST void clip(const vec3f &aRayOrg, const vec3f& aRayDir, float& oEntry, float& oExit) const
	{
		const vec3f t1 = (min - aRayOrg) / aRayDir;
		vec3f tMax = (max - aRayOrg) / aRayDir;

		const vec3f tMin = vec3f::min(t1, tMax);
		tMax = vec3f::max(t1, tMax);
#ifdef __CUDACC__
		oEntry = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
		oExit = fminf(fminf(tMax.x, tMax.y), tMax.z);
#else
        oEntry = std::max(std::max(tMin.x, tMin.y), tMin.z);
        oExit = std::min(std::min(tMax.x, tMax.y), tMax.z);
#endif

	}

    DEVICE HOST void fastClip(const vec3f &aRayOrg, const vec3f& aRayDirRCP, float& oEntry, float& oExit) const
    {
        const vec3f t1 = (min - aRayOrg) * aRayDirRCP;
        vec3f tMax = (max - aRayOrg) * aRayDirRCP;

        const vec3f tMin = vec3f::min(t1, tMax);
        tMax = vec3f::max(t1, tMax);
#ifdef __CUDACC__
        oEntry = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
        oExit = fminf(fminf(tMax.x, tMax.y), tMax.z);
#else
        oEntry = std::max(std::max(tMin.x, tMin.y), tMin.z);
        oExit = std::min(std::min(tMax.x, tMax.y), tMax.z);
#endif

    }

	//Extend the bounding box with a point
	DEVICE HOST void extend(const vec3f &aPoint)
	{
		min = vec3f::min(min, aPoint);
		max = vec3f::max(max, aPoint);
	}

	//Extend the bounding box with another bounding box
	DEVICE HOST void extend(const BBox &aBBox)
	{
		min = vec3f::min(min, aBBox.min);
		max = vec3f::max(max, aBBox.max);
	}

    //Tighten the bounding box around another bounding box
    DEVICE HOST void tighten(const BBox &aBBox)
    {
        min = vec3f::max(min, aBBox.min);
        max = vec3f::min(max, aBBox.max);
    }

    //Tighten the bounding box around two points
    DEVICE HOST void tighten(const vec3f &aMin, const vec3f &aMax)
    {
        min = vec3f::max(min, aMin);
        max = vec3f::min(max, aMax);
    }


	//Returns an "empty" bounding box. Extending such a bounding
	//	box with a point will always create a bbox around the point
	//	and with a bbox - will simply copy the bbox.
	DEVICE HOST static BBox empty()
	{
		BBox ret;
		ret.min.x = FLT_MAX;
        ret.min.y = FLT_MAX;
        ret.min.z = FLT_MAX;
        ret.max.x = -FLT_MAX;
        ret.max.y = -FLT_MAX;
        ret.max.z = -FLT_MAX;
		return ret;
	}

	const vec3f diagonal() const
	{
		return max - min;
	}
};


#endif //__INCLUDE_GUARD_19715045_A7D7_4BC2_B7A5_674B2D36055C
