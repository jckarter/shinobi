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

#ifndef STDAFX_HPP_INCLUDED_06A7A569_C810_4979_9B59_ED80960A84F8
#define STDAFX_HPP_INCLUDED_06A7A569_C810_4979_9B59_ED80960A84F8

//////////////////////////////////////////////////////////////////////////
//C/C++ specific includes
#ifdef _WIN32
//#   define WINDOWS_LEAN_AND_MEAN
#   ifndef NOMINMAX
#     define NOMINMAX 1
#   endif
#   include <windows.h>
#endif

#include <algorithm>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

#define _USE_MATH_DEFINES
#include <float.h>
#include <math.h>

#include <GL/glew.h>

#include <SDL.h>

//////////////////////////////////////////////////////////////////////////
//some windows specific macros
#ifdef _WIN32
#	define RANDNUM rand()/(float)RAND_MAX
#   define SEEDTIME srand(static_cast<unsigned int>(time(0)));
#else
#	define RANDNUM drand48()
#   define SEEDTIME srand48(static_cast<unsigned int>(time(0)));
#endif


#ifdef _WIN32
//prevent console window from closing immediately after job is terminated
#	define SYSTEM_PAUSE system("pause");
#else
#	define SYSTEM_PAUSE
#endif

#ifndef _ASSERT
#define _ASSERT(X)
#endif

//////////////////////////////////////////////////////////////////////////
//utility functions

//do not define in the header if NVCC does not use the same compiler version
std::string itoa(const int a);
//do not define in the header if NVCC does not use the same compiler version
std::string ftoa(const float a);

std::string cutComments(const std::string& aLine, const char* aToken);

#endif // STDAFX_HPP_INCLUDED_06A7A569_C810_4979_9B59_ED80960A84F8
