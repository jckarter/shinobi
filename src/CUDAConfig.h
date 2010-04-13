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

#ifndef CUDACONFIG_H_INCLUDED_8CD6BBA1_1C5F_4D47_A3F1_CA43170ECE51
#define CUDACONFIG_H_INCLUDED_8CD6BBA1_1C5F_4D47_A3F1_CA43170ECE51

#define NUMIMAGES 1

#define SAMPLESPERPIXELX 1
#define SAMPLESPERPIXELY 1

#define DEFAULTRESX 720
#define DEFAULTRESY 480

extern int gRESX;
extern int gRESY;

#define RENDERTHREADSX 32
#define RENDERTHREADSY 4

#define RENDERBLOCKSX 120
#define RENDERBLOCKSY 1

#define BATCHSIZE 96

#define NUMBUILDBLOCKS  128
#define NUMBUILDTHREADS 256

#define NUMBUILDBLOCKS_TLG  128
#define NUMBUILDTHREADS_TLG 256

#define BACKGROUND_R 0.15f
#define BACKGROUND_G 0.2f
#define BACKGROUND_B 0.3f

#define GAMMA 2.2f

#define TWOLEVELGRID
//#define LAZYBUILD       //only with two level grid and sm_12

//#define GATHERSTATISTICS
#define SCALE_R_NUM_RAYS 0.0f
#define SCALE_G_NUM_ACTIVE 0.0f
#define SCALE_B_NUM_ITESTS 0.005f

//#define SAHGRID
//#define NOWINDOW

#define OUTPUT  "output/output.png"

//#define ANIMATION

#ifdef ANIMATION
#   define CONFIGURATION "animation.cfg"
#else
#   define CONFIGURATION "scene.cfg"
#endif

#endif // CUDACONFIG_H_INCLUDED_8CD6BBA1_1C5F_4D47_A3F1_CA43170ECE51
