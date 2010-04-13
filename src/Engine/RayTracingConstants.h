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

#ifndef RAYTRACINGCONSTANTS_H_INCLUDED_95EE5F8D_1F13_42AA_9673_48B69CF00CC1
#define RAYTRACINGCONSTANTS_H_INCLUDED_95EE5F8D_1F13_42AA_9673_48B69CF00CC1

#include "../CUDAStdAfx.h"
#include "RayTracingTypes.h"

////////////////////////////////////////////////////////////////////////////////
//global variables
////////////////////////////////////////////////////////////////////////////////
//structures
DEVICE CONSTANT Camera                      dcCamera;
DEVICE CONSTANT AreaLightSource             dcLightSource;
DEVICE CONSTANT t_RegularPixelSampler       dcRegularPixelSampler;
DEVICE CONSTANT t_RandomPixelSampler        dcRandomPixelSampler;
DEVICE CONSTANT t_GridParameters            dcGridParameters;
DEVICE CONSTANT int                         dcImageId;
DEVICE CONSTANT uint                        dcNumRays;
DEVICE CONSTANT uint                        dcSamples;
DEVICE CONSTANT uint                        dcNumPixels;



////////////////////////////////////////////////////////////////////////////////


#endif // RAYTRACINGCONSTANTS_H_INCLUDED_95EE5F8D_1F13_42AA_9673_48B69CF00CC1
