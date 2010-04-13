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

#ifndef RANDOMNUMBERGENERATORS_HPP_INCLUDED_BDE88087_89C9_450C_8EC2_2511FF0A99E4
#define RANDOMNUMBERGENERATORS_HPP_INCLUDED_BDE88087_89C9_450C_8EC2_2511FF0A99E4

#include "../CUDAStdAfx.h"

//////////////////////////////////////////////////////////////////////////
//George Marsaglia's KISS (Keep It Simple Stupid) random number generator.
//
// This PRNG combines:
// (1) The congruential generator x(n)=69069*x(n-1)+1327217885 with a period
//     of 2^32,
// (2) A 3-shift shift-register generator with a period of 2^32-1,
// (3) Two 16-bit multiply-with-carry generators with a period of
//     597273182964842497 > 2^59.
// The overall period exceeds 2^123.
//
//
//
// #define UL unsigned long
// #define znew  ((z=36969*(z&65535)+(z>>16))<<16)
// #define wnew  ((w=18000*(w&65535)+(w>>16))&65535)
// #define MWC   (znew+wnew)
// #define SHR3  (jsr=(jsr=(jsr=jsr^(jsr<<17))^(jsr>>13))^(jsr<<5))
// #define CONG  (jcong=69069*jcong+1234567)
// #define KISS  ((MWC^CONG)+SHR3)
// #define LFIB4 (t[c]=t[c]+t[c+58]+t[c+119]+t[++c+178])
// #define SWB   (t[c+237]=(x=t[c+15])-(y=t[++c]+(x<y)))
// #define UNI   (KISS*2.328306e-10)
// #define VNI   ((long) KISS)*4.656613e-10
// /*  Global static variables: */
// static UL z=362436069, w=521288629, jsr=123456789, jcong=380116160;
// static UL t[256];
// static UL x=0,y=0; static unsigned char c=0;
// 
// /* Random seeds must be used to reset z,w,jsr,jcong and
// the table t[256]  Here is an example procedure, using KISS: */
// 
// void settable(UL i1,UL i2,UL i3,UL i4)
// { int i; z=i1;w=i2,jsr=i3; jcong=i4;
// for(i=0;i<256;i++)  t[i]=KISS;        }
//////////////////////////////////////////////////////////////////////////

//class KISSRandomNumberGenerator
//{
//public:
//    uint x,
//        y,//must be non-zero
//        z,
//        w; //doesn't need to be re-seeded but must be < 698769069
//    
//    DEVICE KISSRandomNumberGenerator(
//        const uint aX = 123456789u,
//        const uint aY = 362436069u,
//        const uint aZ = 521288629u,
//        const uint aW = 416191069u) :
//    x(aX), y(aY), z(aZ), w(aW)
//    {}
//
//    DEVICE float operator()()
//    {
//        z = (36969 * (z & 65535) + (z >> 16)) << 16;
//        w = 18000 * (w & 65535) + (w >> 16) & 65535;
//        x = 69069 * x + 1234567;
//        y = (y = (y = y ^ (y << 17)) ^ (y >> 13)) ^ (y << 5);
//        return ((z + w) ^ x + y) * 2.328306E-10f;
//    }
//};

class KISSRandomNumberGenerator
{
public:
    uint data[4];
    //data[0],
    //data[1],//must be zero
    //data[2],
    //data[3]; //doesn't need to be re-seeded but must be < 698769069

    DEVICE KISSRandomNumberGenerator(
        const uint aX = 123456789u,
        const uint aY = 362436069u,
        const uint aZ = 521288629u,
        const uint aW = 416191069u)
    {
        data[0] = (aX); data[1] = (aY); data[2] = (aZ); data[3] = (aW);
    }

    DEVICE float operator()()
    {
        data[2] = (36969 * (data[2] & 65535) + (data[2] >> 16)) << 16;
        data[3] = 18000 * (data[3] & 65535) + (data[3] >> 16) & 65535;
        data[0] = 69069 * data[0] + 1234567;
        data[1] = (data[1] = (data[1] = data[1] ^ (data[1] << 17)) ^ (data[1] >> 13)) ^ (data[1] << 5);
        return ((data[2] + data[3]) ^ data[0] + data[1]) * 2.328306E-10f;
    }
};

// just some simplistic int32 LCG routine (parameters from Numerical Recipes)
// converted to float
class SimpleRandomNumberGenerator
{
public:
    uint seed;
    DEVICE SimpleRandomNumberGenerator(const uint aSeed)
        :seed(aSeed)
    {}

    DEVICE float operator()()
    {
        seed = (seed * 1664525 + 1013904223);
        return (float)seed / 0x100000000;

    }
};

class DummyRandomNumberGenerator
{
    uint x;
public:
    DEVICE DummyRandomNumberGenerator(
        const uint aX = 1u):x(aX)
    {}

    DEVICE float operator()()
    {
        x = (x * x * x * x * x * x * x * x) % 0xFFFF;
        return (float)x / 65535;
    }
};

#endif // RANDOMNUMBERGENERATORS_HPP_INCLUDED_BDE88087_89C9_450C_8EC2_2511FF0A99E4
