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

#ifndef RAYTRACINGTYPES_H_INCLUDED_328FC2A8_9B26_45B9_8A44_745302E1FE4B
#define RAYTRACINGTYPES_H_INCLUDED_328FC2A8_9B26_45B9_8A44_745302E1FE4B

#include "../CUDAStdAfx.h"

#include "../CUDAConfig.h"

#include "../Primitive/Triangle.hpp"
#include "../Primitive/Camera.h"
#include "../Primitive/PixelSampler.h"
#include "../Primitive/FrameBuffer.h"
#include "../Primitive/LightSource.hpp"
#include "../Primitive/Material.h"


#include "../Intersector/SimpleIntersector.hpp"
#if __CUDA_ARCH__ >= 120
#   include "Intersector/HybridIntersector.hpp"
#endif

#include "../Control/Structure/SimpleGrid.h"
#include "../Control/Structure/TwoLevelGrid.h"
#include "../Control/Structure/FaceSoup.h"
#include "../Control/Structure/TriangleSoup.h"

#include "../Control/Algorithm/GridBuilder.h"
#if __CUDA_ARCH__ >= 120
#   include "../Control/Algorithm/SAHGridBuilder.h"
#endif
#include "../Control/Algorithm/TwoLevelGridBuilder.h"
#if __CUDA_ARCH__ >= 110
#   include "../Control/Algorithm/SimpleGridBuilder.h"
#endif
#if __CUDA_ARCH__ >= 120
#   include "../Control/Algorithm/FastSimpleGridBuilder.h"
#   include "../Control/Algorithm/FastTwoLevelGridBuilder.h"
#   include "../Control/Algorithm/LazyTwoLevelGridBuilder.h"
#endif

#include "../Control/Algorithm/SimpleGridTraverser.h"
#include "../Control/Algorithm/TwoLevelGridTraverser.h"

#include "../Utils/RandomNumberGenerators.hpp"


///////////////////////////////////////////////////////////////////////////
//Common
///////////////////////////////////////////////////////////////////////////
typedef RegularPixelSampler<SAMPLESPERPIXELX, SAMPLESPERPIXELY>
                                                    t_RegularPixelSampler;
typedef RandomPixelSampler                          t_RandomPixelSampler;

typedef GridParameters                              t_GridParameters;

typedef MaterialContainer                           t_MaterialStorage;

#define Storage_t                                   FaceSoup
//#define Storage_t                                   TriangleSoup
typedef Storage_t<0>                                t_PrimitiveStorage0;
typedef Storage_t<1>                                t_PrimitiveStorage1;

//////////////////////////////////////////////////////////////////////////
//Control Structure & Algorithms
//////////////////////////////////////////////////////////////////////////

#if defined (TWOLEVELGRID) || (__CUDA_ARCH__ < 110)

//Two Level Grid
typedef TwoLevelGrid                                t_Grid;

#   if __CUDA_ARCH__ < 120
typedef TwoLevelGridBuilder< Storage_t<0> >         t_Builder0;
typedef TwoLevelGridBuilder< Storage_t<1> >         t_Builder1;
#   else
#       ifndef LAZYBUILD
typedef FastTwoLevelGridBuilder< Storage_t<0> >     t_Builder0;
typedef FastTwoLevelGridBuilder< Storage_t<1> >     t_Builder1;
#       endif //LAZYBUILD
#   endif //__CUDA_ARCH__ < 120

#   define SimpleTraverser_t                        SimpleTwoLevelGridTraverser
#   define Traverser_t                              TwoLevelGridTraverser
#   define ShadowTraverser_t                        TwoLevelGridShadowTraverser

#else

//Simple Grid
typedef SimpleGrid                                  t_Grid;
#   if __CUDA_ARCH__ < 120
typedef SimpleGridBuilder< Storage_t<0> >           t_Builder0;
typedef SimpleGridBuilder< Storage_t<1> >           t_Builder1;
#   else
typedef FastSimpleGridBuilder< Storage_t<0> >       t_Builder0;
typedef FastSimpleGridBuilder< Storage_t<1> >       t_Builder1;
#   endif //__CUDA_ARCH__ < 120

#   define SimpleTraverser_t                        SimpleGridTraverser
#   define Traverser_t                              UniformGridTraverser
#   define ShadowTraverser_t                        UniformGridShadowTraverser

#endif //(TWOLEVELGRID) || (__CUDA_ARCH__ < 110)


#endif // RAYTRACINGTYPES_H_INCLUDED_328FC2A8_9B26_45B9_8A44_745302E1FE4B
