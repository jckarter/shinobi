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

#ifndef MATERIAL_H_INCLUDED_8F0BEE97_B83C_4424_AC25_57951CD8AC0C
#define MATERIAL_H_INCLUDED_8F0BEE97_B83C_4424_AC25_57951CD8AC0C

#include "../CUDAStdAfx.h"

texture< float4, 1, cudaReadModeElementType >       texDiffuseReflectance;
texture< float4, 1, cudaReadModeElementType >       texSpecularReflectance;

class MaterialContainer
{
public:
    float4* mDiffuseCoefficient;
    float4* mSpecularCoefficient;

    DEVICE float4 getDiffuseReflectance(int aMaterialId) const
    {
        return mDiffuseCoefficient[aMaterialId];
        //return tex1Dfetch(texDiffuseReflectance, aMaterialId);
    }

    DEVICE float4 getSpecularReflectance(int aMaterialId) const
    {
        return mSpecularCoefficient[aMaterialId];
        //return tex1Dfetch(texSpecularReflectance, aMaterialId);
    }

    template<class tMaterial>
    HOST void upload(const tMaterial* aMaterials, const size_t aNumElements) const
    {
        const size_t numElements = aNumElements;
        const size_t containerSize = numElements * sizeof(float4);
        float4* dataHost;

        CUDA_SAFE_CALL( cudaMalloc( (void**)&mDiffuseCoefficient, containerSize) );
        CUDA_SAFE_CALL( cudaMalloc( (void**)&mSpecularCoefficient, containerSize) );
        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHost, containerSize) );

        for (uint i = 0; i < numElements; ++i)
        {
            dataHost[i].x = aMaterials[i].diffuseCoeff.x;
            dataHost[i].y = aMaterials[i].diffuseCoeff.y;
            dataHost[i].z = aMaterials[i].diffuseCoeff.z;
            dataHost[i].w = 0.f;
        }

        CUDA_SAFE_CALL( 
            cudaMemcpy(mDiffuseCoefficient, dataHost, containerSize, cudaMemcpyHostToDevice));

        for (uint i = 0; i < numElements; ++i)
        {
            dataHost[i].x = aMaterials[i].specularCoeff.x;
            dataHost[i].y = aMaterials[i].specularCoeff.y;
            dataHost[i].z = aMaterials[i].specularCoeff.z;
            dataHost[i].w = aMaterials[i].specularExp;
        }

        CUDA_SAFE_CALL( 
            cudaMemcpy(mSpecularCoefficient, dataHost, containerSize, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL( cudaFreeHost(dataHost) );

        cudaChannelFormatDesc chanelFormatDesc =
            cudaCreateChannelDesc<float4>();

        CUDA_SAFE_CALL( cudaBindTexture(NULL, texDiffuseReflectance, (void*) mDiffuseCoefficient,
            chanelFormatDesc, containerSize) );

        CUDA_SAFE_CALL( cudaBindTexture(NULL, texSpecularReflectance, (void*) mSpecularCoefficient,
            chanelFormatDesc, containerSize) );
    }

    HOST void cleanup()
    {
        CUDA_SAFE_CALL( cudaFree(mDiffuseCoefficient) );
        CUDA_SAFE_CALL( cudaFree(mSpecularCoefficient) );
    }
};

#endif // MATERIAL_H_INCLUDED_8F0BEE97_B83C_4424_AC25_57951CD8AC0C
