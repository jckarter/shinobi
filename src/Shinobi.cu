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

#include "CUDAStdAfx.h"
#include "CUDAConfig.h"

#include "Shinobi.h"

#include "Application/WindowlessApplication.h"

#include "Application/SDLGLApplication.h"

#include "Engine/StaticRTEngine.h"
#include "Engine/DynamicRTEngine.h"

int gRESX = DEFAULTRESX;
int gRESY = DEFAULTRESY;

t_Builder1 DynamicRTEngine::sBuilder1;
t_Builder0 DynamicRTEngine::sBuilder0;
Storage_t<0> DynamicRTEngine::sStorage0;
Storage_t<1> DynamicRTEngine::sStorage1;
cudaStream_t DynamicRTEngine::sRenderEngineSreams[2]; 
uint DynamicRTEngine::sFrameId;
t_Integrator0 DynamicRTEngine::sIntegrator0;
t_Integrator1 DynamicRTEngine::sIntegrator1;
