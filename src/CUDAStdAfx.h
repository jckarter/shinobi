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

#ifndef CUDASTDAFX_H_INCLUDED_87E09C4A_6F21_457B_BF8D_2E5FF22CD899
#define CUDASTDAFX_H_INCLUDED_87E09C4A_6F21_457B_BF8D_2E5FF22CD899

//////////////////////////////////////////////////////////////////////////
//Includes and tools for use with CUDA
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//CUDA specific includes
//////////////////////////////////////////////////////////////////////////
#include <cutil.h> 
#include <vector_types.h>
#include <cuda_runtime_api.h>

#include "Utils/CUDAUtil.h" //std stuff replacements

//////////////////////////////////////////////////////////////////////////
//CUDA specific macros
//////////////////////////////////////////////////////////////////////////
#define WARPSIZE            32
#define HALFWARPSIZE        16
#define LOG2WARPSIZE        5
#define LOG2HALFWARPSIZE    4
#define NUMBANKS            16
#define LOG2NUMBANKS        4

#ifdef __CUDACC__
#   define DEVICE       __device__
#   define HOST         __host__
#   define SHARED       __shared__
#   define GLOBAL       __global__
#   define CONSTANT     __constant__
#   define SYNCTHREADS  __syncthreads()

#   if __CUDA_ARCH__ >= 120
#       define ANY __any
#       define ALL __all
#   else
#       define ANY
#       define ALL
#   endif

#else
#   define DEVICE inline
#   define HOST
#   define SHARED
#   define GLOBAL
#   define CONSTANT
#   define SYNCTHREADS
#   define ANY
#   define ALL
#endif

#ifdef __DEVICE_EMULATION__
#define EMUSYNCTHREADS __syncthreads()
#else
#define EMUSYNCTHREADS
#endif // __DEVICE_EMULATION__

//////////////////////////////////////////////////////////////////////////
//Useful typedefs
typedef unsigned char   byte;
typedef unsigned short  ushort;
typedef unsigned int    uint;
typedef unsigned long   ulong;

//////////////////////////////////////////////////////////////////////////
//constants
#define EPS     0.00001f
#define EPS_RCP 100000.f
#ifndef FLT_MAX
#   define FLT_MAX         3.402823466e+38f
#endif
#ifndef M_PI
#   define M_PI       3.14159265358979323846f
#endif
#ifndef M_E
#   define M_E        2.71828182845904523536f
#endif

#ifdef __CUDACC__
//////////////////////////////////////////////////////////////////////////
//CUDA helper functions
//////////////////////////////////////////////////////////////////////////
__device__ uint numThreads ()
{
    return  blockDim.x * blockDim.y * blockDim.z * 
            gridDim.x * gridDim.y * gridDim.z;
}
__device__ uint numBlocks ()
{
    return  gridDim.x * gridDim.y * gridDim.z;
}
__device__ uint blockSize ()
{
    return  blockDim.x * blockDim.y * blockDim.z;
}

__device__ uint blockId1D ()
{
    return  blockIdx.x + 
            blockIdx.y * gridDim.x + 
            blockIdx.z * gridDim.x * gridDim.y;
}

__device__ uint threadId1D ()
{
    return  threadIdx.x + 
            threadIdx.y * blockDim.x + 
            threadIdx.z * blockDim.x * blockDim.y;
}

__device__ uint globalThreadId1D (const uint aElementsPerThread = 1u)
{
    return threadId1D() + blockId1D() * blockSize() * aElementsPerThread;
}

__device__ uint threadId1DInWarp()
{
    return threadId1D() & (WARPSIZE - 1u);
}

__device__ uint2 globalThreadId2D ()
{
    //NOTE: Assumes 2D grid layout!
    return  make_uint2(
            threadIdx.x + blockIdx.x * blockDim.x,
            threadIdx.y + blockIdx.y * blockDim.y);
}

__device__ uint warpId ()
{
    return (threadId1D() >> LOG2WARPSIZE);
}

__device__ uint isInSecondHalfWarp()
{
    return (threadId1D() & HALFWARPSIZE) >> LOG2HALFWARPSIZE; 
    //return (threadId1DInWarp() < HALFWARPSIZE) ? 0u : 1u;
}

__device__ float fastDivide(float aX, float aY)
{
    return __fdividef(aX,aY);
}

__device__ float intAsFloat(uint aInt)
{
    return __int_as_float(aInt);
}

__device__ uint floatAsInt(float aFloat)
{
    return __float_as_int(aFloat);
}

__device__ float fastSin(float aVal)
{
    return __sinf(aVal);
}

__device__ float fastCos(float aVal)
{
    return __cosf(aVal);
}

__device__ float fastPow(float aVal, float aPow)
{
    return __powf(aVal, aPow);
}

__device__ void sinCos(float aVal, float* aSin, float* aCos)
{
    return __sincosf(aVal, aSin, aCos);
}

//HACK: use with block width 32
__device__ uint threadId1D32 ()
{  return  threadIdx.x + threadIdx.y * 32u;  }
//HACK: use with block width 32
__device__ uint threadId1DInWarp32()
{ return threadIdx.x; }
//HACK: use with block width 32
__device__ uint warpId32 ()
{  return  threadIdx.y;  }

#else

inline uint numThreads (){  return  0;  }
inline uint numBlocks  (){  return  0;  }
inline uint blockSize  (){  return  0;  }
inline uint blockId1D  (){  return  0;  }
inline uint threadId1D (){  return  0;  }
inline uint globalThreadId1D (const uint aElementsPerThread = 1u){ return  0;  }
inline uint threadId1DInWarp(){ return 0; }
inline uint2 globalThreadId2D (){  uint2 retval; retval.x = 0u; return  retval;}
inline uint warpId     (){  return  0;  }
inline uint isInSecondHalfWarp(){   return  0;  }
inline float fastDivide(float aX, float aY){ return aX / aY; }
inline float intAsFloat(uint aInt){ return static_cast<float>(aInt); }
inline uint floatAsInt(float aFloat){ return static_cast<uint>(aFloat); }
inline float fastSin(float aVal){ return sinf(aVal); }
inline float fastCos(float aVal){ return cosf(aVal); }
inline float fastPow(float aVal, float aPow){ return powf(aVal, aPow); }
inline void sinCos(float aVal, float* aSin, float* aCos)
{ *aSin = sinf(aVal); *aCos = cosf(aVal); }
inline uint threadId1D32 (){  return 0u;  }
inline uint threadId1DInWarp32(){ return 0u; }
inline uint warpId32 (){  return  0u;  }
#endif //__CUDACC__

#endif // CUDASTDAFX_H_INCLUDED_87E09C4A_6F21_457B_BF8D_2E5FF22CD899
