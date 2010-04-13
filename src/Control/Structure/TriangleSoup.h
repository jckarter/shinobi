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

#ifndef TRIANGLESOUP_H_INCLUDED_AF408B4A_6394_460C_976D_CA685339103E
#define TRIANGLESOUP_H_INCLUDED_AF408B4A_6394_460C_976D_CA685339103E

#include "../../CUDAStdAfx.h"
#include "../../Primitive/Triangle.hpp"
#include "../../Loaders/FWObject.hpp"

texture< float4, 1, cudaReadModeElementType >       tex0TriangleVtx0;
texture< float4, 1, cudaReadModeElementType >       tex0TriangleVtx1;
texture< float4, 1, cudaReadModeElementType >       tex0TriangleVtx2;
texture< float4, 1, cudaReadModeElementType >       tex1TriangleVtx0;
texture< float4, 1, cudaReadModeElementType >       tex1TriangleVtx1;
texture< float4, 1, cudaReadModeElementType >       tex1TriangleVtx2;
texture< float4, 1, cudaReadModeElementType >       texTriAccel;

template <int taFrameId>
struct TriangleSoup
{
    uint *indices;

    float4* dataVtx0;
    float4* dataHostVtx0;

    float4* dataVtx1;
    float4* dataHostVtx1;

    float4* dataVtx2;
    float4* dataHostVtx2;

    float4* triAccel;

    DEVICE Triangle operator[](uint i) const
    {
        Triangle retval;

        float4 vtx0;
        if (taFrameId == 0)//compile-time decision
        {
            vtx0 = tex1Dfetch(tex0TriangleVtx0, indices[i]);
        }
        else
        {
            vtx0 = tex1Dfetch(tex1TriangleVtx0, indices[i]);
        }
        retval.vertices[0].x = vtx0.x;
        retval.vertices[0].y = vtx0.y;
        retval.vertices[0].z = vtx0.z;

        float4 vtx1;
        if (taFrameId == 0)//compile-time decision
        {
            vtx1 = tex1Dfetch(tex0TriangleVtx1, indices[i]);
        }
        else
        {
            vtx1 = tex1Dfetch(tex1TriangleVtx1, indices[i]);
        }
        retval.vertices[1].x = vtx1.x;
        retval.vertices[1].y = vtx1.y;
        retval.vertices[1].z = vtx1.z;

        float4 vtx2;
        if (taFrameId == 0)//compile-time decision
        {
            vtx2 = tex1Dfetch(tex0TriangleVtx2, indices[i]);
        }
        else
        {
            vtx2 = tex1Dfetch(tex1TriangleVtx2, indices[i]);
        }
        retval.vertices[2].x = vtx2.x;
        retval.vertices[2].y = vtx2.y;
        retval.vertices[2].z = vtx2.z;

        return retval;
    }

    DEVICE Triangle operator()(uint i) const
    {
        Triangle retval;

        float4 vtx0;
        if (taFrameId == 0)//compile-time decision
        {
            vtx0 = tex1Dfetch(tex0TriangleVtx0, i);
        }
        else
        {
            vtx0 = tex1Dfetch(tex1TriangleVtx0, i);
        }
        retval.vertices[0].x = vtx0.x;
        retval.vertices[0].y = vtx0.y;
        retval.vertices[0].z = vtx0.z;

        float4 vtx1;
        if (taFrameId == 0)//compile-time decision
        {
            vtx1 = tex1Dfetch(tex0TriangleVtx1, i);
        }
        else
        {
            vtx1 = tex1Dfetch(tex1TriangleVtx1, i);
        }
        retval.vertices[1].x = vtx1.x;
        retval.vertices[1].y = vtx1.y;
        retval.vertices[1].z = vtx1.z;

        float4 vtx2;
        if (taFrameId == 0)//compile-time decision
        {
            vtx2 = tex1Dfetch(tex0TriangleVtx2, i);
        }
        else
        {
            vtx2 = tex1Dfetch(tex1TriangleVtx2, i);
        }
        retval.vertices[2].x = vtx2.x;
        retval.vertices[2].y = vtx2.y;
        retval.vertices[2].z = vtx2.z;

        return retval;
    }

    DEVICE float4 getAccelDataChunck(uint aId) const
    {
        return  tex1Dfetch(texTriAccel, aId);
    }

    DEVICE uint getMaterialId(int i) const
    {
        return 0u;
    }

    HOST void upload(
        const FWObject::t_FaceIterator&       aBegin,
        const FWObject::t_FaceIterator&       aEnd,
        const FWObject&                                     aData)
    {
        Triangle* triangles;
        CUDA_SAFE_CALL( cudaMallocHost((void**)&triangles, aData.getNumFaces() * sizeof(Triangle)) );

        Triangle currentTriangle;
        int triId = 0;
        for(FWObject::t_FaceIterator it = aBegin; it != aEnd; ++it)
        {
            currentTriangle.vertices[0] = aData.getVertex(it->vert1);
            currentTriangle.vertices[1] = aData.getVertex(it->vert2);
            currentTriangle.vertices[2] = aData.getVertex(it->vert3);

            triangles[triId++] = currentTriangle;
        }

        const size_t numTriangles = aData.getNumFaces();
        const size_t size =  numTriangles * sizeof(float4);

        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx0, size) );
        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx0, size) );

        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx1, size) );
        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx1, size) );

        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx2, size) );
        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx2, size) );

        for (uint it = 0; it < numTriangles; ++it)
        {
            dataHostVtx0[it].x = triangles[it].vertices[0].x;
            dataHostVtx0[it].y = triangles[it].vertices[0].y;
            dataHostVtx0[it].z = triangles[it].vertices[0].z;

            dataHostVtx1[it].x = triangles[it].vertices[1].x;
            dataHostVtx1[it].y = triangles[it].vertices[1].y;
            dataHostVtx1[it].z = triangles[it].vertices[1].z;

            dataHostVtx2[it].x = triangles[it].vertices[2].x;
            dataHostVtx2[it].y = triangles[it].vertices[2].y;
            dataHostVtx2[it].z = triangles[it].vertices[2].z;
        }

        CUDA_SAFE_CALL( 
            cudaMemcpy(dataVtx0, dataHostVtx0, size, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL( 
            cudaMemcpy(dataVtx1, dataHostVtx1, size, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL( 
            cudaMemcpy(dataVtx2, dataHostVtx2, size, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaFreeHost(triangles));
        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx0));
        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx1));
        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx2));


        cudaChannelFormatDesc chanelFormatDesc =
            cudaCreateChannelDesc<float4>();

        if (taFrameId == 0)//compile-time decision
        {
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex0TriangleVtx0, (void*) dataVtx0,
                chanelFormatDesc, size) );
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex0TriangleVtx1, (void*) dataVtx1,
                chanelFormatDesc, size) );
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex0TriangleVtx2, (void*) dataVtx2,
                chanelFormatDesc, size) );
        }
        else
        {
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex1TriangleVtx0, (void*) dataVtx0,
                chanelFormatDesc, size) );
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex1TriangleVtx1, (void*) dataVtx1,
                chanelFormatDesc, size) );
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex1TriangleVtx2, (void*) dataVtx2,
                chanelFormatDesc, size) );
        }

        //uploadShevtsovAccelerators(aBegin, aEnd, aData);
    }

    HOST void uploadVertices(
        const FWObject& aKeyFrame1,
        const FWObject& aKeyFrame2,
        const float     aCoeff,
        cudaStream_t    aStream,
        vec3f&          oMinBound,
        vec3f&          oMaxBound
        )
    {
        //////////////////////////////////////////////////////////////////////////
        //cleanup
        //////////////////////////////////////////////////////////////////////////

        if (taFrameId == 0)//compile-time decision
        {
            CUDA_SAFE_CALL( cudaUnbindTexture( &tex0TriangleVtx0) );
            CUDA_SAFE_CALL( cudaUnbindTexture( &tex0TriangleVtx1) );
            CUDA_SAFE_CALL( cudaUnbindTexture( &tex0TriangleVtx2) );
        }
        else
        {
            CUDA_SAFE_CALL( cudaUnbindTexture( &tex1TriangleVtx0) );
            CUDA_SAFE_CALL( cudaUnbindTexture( &tex1TriangleVtx1) );
            CUDA_SAFE_CALL( cudaUnbindTexture( &tex1TriangleVtx2) );
        }

        if (dataVtx0 != NULL)
        {
            CUDA_SAFE_CALL( cudaFree(dataVtx0) );
        }

        if (dataVtx1 != NULL)
        {
            CUDA_SAFE_CALL( cudaFree(dataVtx1) );
        }

        if (dataVtx2 != NULL)
        {
            CUDA_SAFE_CALL( cudaFree(dataVtx2) );
        }

        //////////////////////////////////////////////////////////////////////////
        //Upload Triangles
        //////////////////////////////////////////////////////////////////////////

       
        const size_t numFaces1 = aKeyFrame1.getNumFaces();
        const size_t numFaces2 = aKeyFrame2.getNumFaces();
        const size_t numFaces = numFaces2;
        const size_t size = numFaces * sizeof(float4);

        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx0, size) );
        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx0, size) );

        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx1, size) );
        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx1, size) );

        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx2, size) );
        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx2, size) );


        size_t it = 0;
        for (; it < cudastd::min(numFaces1, numFaces2); ++it)
        {
            dataHostVtx0[it].x = aKeyFrame1.getVertex(aKeyFrame1.getFace(it).vert1).x * (1.f - aCoeff) + aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert1).x * aCoeff;
            dataHostVtx0[it].y = aKeyFrame1.getVertex(aKeyFrame1.getFace(it).vert1).y * (1.f - aCoeff) + aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert1).y * aCoeff;
            dataHostVtx0[it].z = aKeyFrame1.getVertex(aKeyFrame1.getFace(it).vert1).z * (1.f - aCoeff) + aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert1).z * aCoeff;
            dataHostVtx1[it].x = aKeyFrame1.getVertex(aKeyFrame1.getFace(it).vert2).x * (1.f - aCoeff) + aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert2).x * aCoeff;
            dataHostVtx1[it].y = aKeyFrame1.getVertex(aKeyFrame1.getFace(it).vert2).y * (1.f - aCoeff) + aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert2).y * aCoeff;
            dataHostVtx1[it].z = aKeyFrame1.getVertex(aKeyFrame1.getFace(it).vert2).z * (1.f - aCoeff) + aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert2).z * aCoeff;
            dataHostVtx2[it].x = aKeyFrame1.getVertex(aKeyFrame1.getFace(it).vert3).x * (1.f - aCoeff) + aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert3).x * aCoeff;
            dataHostVtx2[it].y = aKeyFrame1.getVertex(aKeyFrame1.getFace(it).vert3).y * (1.f - aCoeff) + aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert3).y * aCoeff;
            dataHostVtx2[it].z = aKeyFrame1.getVertex(aKeyFrame1.getFace(it).vert3).z * (1.f - aCoeff) + aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert3).z * aCoeff;

            oMinBound.x = cudastd::min(dataHostVtx0[it].x, oMinBound.x);
            oMinBound.y = cudastd::min(dataHostVtx0[it].y, oMinBound.y);
            oMinBound.z = cudastd::min(dataHostVtx0[it].z, oMinBound.z);
            oMaxBound.x = cudastd::max(dataHostVtx0[it].x, oMaxBound.x);
            oMaxBound.y = cudastd::max(dataHostVtx0[it].y, oMaxBound.y);
            oMaxBound.z = cudastd::max(dataHostVtx0[it].z, oMaxBound.z);
            oMinBound.x = cudastd::min(dataHostVtx1[it].x, oMinBound.x);
            oMinBound.y = cudastd::min(dataHostVtx1[it].y, oMinBound.y);
            oMinBound.z = cudastd::min(dataHostVtx1[it].z, oMinBound.z);
            oMaxBound.x = cudastd::max(dataHostVtx1[it].x, oMaxBound.x);
            oMaxBound.y = cudastd::max(dataHostVtx1[it].y, oMaxBound.y);
            oMaxBound.z = cudastd::max(dataHostVtx1[it].z, oMaxBound.z);
            oMinBound.x = cudastd::min(dataHostVtx2[it].x, oMinBound.x);
            oMinBound.y = cudastd::min(dataHostVtx2[it].y, oMinBound.y);
            oMinBound.z = cudastd::min(dataHostVtx2[it].z, oMinBound.z);
            oMaxBound.x = cudastd::max(dataHostVtx2[it].x, oMaxBound.x);
            oMaxBound.y = cudastd::max(dataHostVtx2[it].y, oMaxBound.y);
            oMaxBound.z = cudastd::max(dataHostVtx2[it].z, oMaxBound.z);

        }

        for (; it < numFaces2 ; ++it)
        {
            dataHostVtx0[it].x = aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert1).x;
            dataHostVtx0[it].y = aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert1).y;
            dataHostVtx0[it].z = aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert1).z;
            dataHostVtx1[it].x = aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert2).x;
            dataHostVtx1[it].y = aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert2).y;
            dataHostVtx1[it].z = aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert2).z;
            dataHostVtx2[it].x = aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert3).x;
            dataHostVtx2[it].y = aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert3).y;
            dataHostVtx2[it].z = aKeyFrame2.getVertex(aKeyFrame2.getFace(it).vert3).z;

            oMinBound.x = cudastd::min(dataHostVtx0[it].x, oMinBound.x);
            oMinBound.y = cudastd::min(dataHostVtx0[it].y, oMinBound.y);
            oMinBound.z = cudastd::min(dataHostVtx0[it].z, oMinBound.z);
            oMaxBound.x = cudastd::max(dataHostVtx0[it].x, oMaxBound.x);
            oMaxBound.y = cudastd::max(dataHostVtx0[it].y, oMaxBound.y);
            oMaxBound.z = cudastd::max(dataHostVtx0[it].z, oMaxBound.z);
            oMinBound.x = cudastd::min(dataHostVtx1[it].x, oMinBound.x);
            oMinBound.y = cudastd::min(dataHostVtx1[it].y, oMinBound.y);
            oMinBound.z = cudastd::min(dataHostVtx1[it].z, oMinBound.z);
            oMaxBound.x = cudastd::max(dataHostVtx1[it].x, oMaxBound.x);
            oMaxBound.y = cudastd::max(dataHostVtx1[it].y, oMaxBound.y);
            oMaxBound.z = cudastd::max(dataHostVtx1[it].z, oMaxBound.z);
            oMinBound.x = cudastd::min(dataHostVtx2[it].x, oMinBound.x);
            oMinBound.y = cudastd::min(dataHostVtx2[it].y, oMinBound.y);
            oMinBound.z = cudastd::min(dataHostVtx2[it].z, oMinBound.z);
            oMaxBound.x = cudastd::max(dataHostVtx2[it].x, oMaxBound.x);
            oMaxBound.y = cudastd::max(dataHostVtx2[it].y, oMaxBound.y);
            oMaxBound.z = cudastd::max(dataHostVtx2[it].z, oMaxBound.z);
        }

        CUDA_SAFE_CALL( 
            cudaMemcpyAsync(dataVtx0, dataHostVtx0, size, cudaMemcpyHostToDevice, aStream));

        CUDA_SAFE_CALL( 
            cudaMemcpyAsync(dataVtx1, dataHostVtx1, size, cudaMemcpyHostToDevice, aStream));

        CUDA_SAFE_CALL( 
            cudaMemcpyAsync(dataVtx2, dataHostVtx2, size, cudaMemcpyHostToDevice, aStream));

        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx0));
        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx1));
        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx2));


        cudaChannelFormatDesc chanelFormatDesc =
            cudaCreateChannelDesc<float4>();

        if (taFrameId == 0)//compile-time decision
        {
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex0TriangleVtx0, (void*) dataVtx0,
                chanelFormatDesc, size) );
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex0TriangleVtx1, (void*) dataVtx1,
                chanelFormatDesc, size) );
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex0TriangleVtx2, (void*) dataVtx2,
                chanelFormatDesc, size) );
        }
        else
        {
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex1TriangleVtx0, (void*) dataVtx0,
                chanelFormatDesc, size) );
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex1TriangleVtx1, (void*) dataVtx1,
                chanelFormatDesc, size) );
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex1TriangleVtx2, (void*) dataVtx2,
                chanelFormatDesc, size) );
        }

    }

    HOST void cleanup()
    {
        CUDA_SAFE_CALL( cudaFree(indices) );
        CUDA_SAFE_CALL( cudaFree(dataVtx0) );
        CUDA_SAFE_CALL( cudaFree(dataVtx1) );
        CUDA_SAFE_CALL( cudaFree(dataVtx2) );
        //CUDA_SAFE_CALL( cudaFree(triAccel) );
    }

    HOST void uploadShevtsovAccelerators(
        const FWObject::t_FaceIterator&       aBegin,
        const FWObject::t_FaceIterator&       aEnd,
        const FWObject&                                     aData)
    {

        Triangle* triangles;
        CUDA_SAFE_CALL( cudaMallocHost((void**)&triangles, aData.getNumFaces() * sizeof(Triangle)) );

        Triangle currentTriangle;
        int triId = 0;
        for(FWObject::t_FaceIterator it = aBegin; it != aEnd; ++it)
        {
            currentTriangle.vertices[0] = aData.getVertex(it->vert1);
            currentTriangle.vertices[1] = aData.getVertex(it->vert2);
            currentTriangle.vertices[2] = aData.getVertex(it->vert3);

            triangles[triId++] = currentTriangle;
        }

        ShevtsovTriAccel* dataHost;

        const size_t numTriangles = aData.getNumFaces();
        const size_t size =  numTriangles * sizeof(float4) * 3;

        CUDA_SAFE_CALL( cudaMalloc( (void**)&triAccel, size) );
        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHost, size) );


        for (uint it = 0; it < numTriangles; ++it)
        {
            dataHost[it] = ShevtsovTriAccel(triangles[it]);
        }

        CUDA_SAFE_CALL( 
            cudaMemcpy(triAccel, dataHost, size, cudaMemcpyHostToDevice));
        
        CUDA_SAFE_CALL(cudaFreeHost(triangles));
        CUDA_SAFE_CALL(cudaFreeHost(dataHost));

        cudaChannelFormatDesc chanelFormatDesc =
            cudaCreateChannelDesc<float4>();

        CUDA_SAFE_CALL( cudaBindTexture(NULL, texTriAccel, (void*) triAccel,
            chanelFormatDesc, size) );

    }
};

//texture< float, 1, cudaReadModeElementType >        texTriangleVtx0X;
//texture< float, 1, cudaReadModeElementType >        texTriangleVtx0Y;
//texture< float, 1, cudaReadModeElementType >        texTriangleVtx0Z;
//
//texture< float, 1, cudaReadModeElementType >        texTriangleVtx1X;
//texture< float, 1, cudaReadModeElementType >        texTriangleVtx1Y;
//texture< float, 1, cudaReadModeElementType >        texTriangleVtx1Z;
//
//texture< float, 1, cudaReadModeElementType >        texTriangleVtx2X;
//texture< float, 1, cudaReadModeElementType >        texTriangleVtx2Y;
//texture< float, 1, cudaReadModeElementType >        texTriangleVtx2Z;
//
////slower than previous implementation
//struct TriangleSoupFltStorage
//{
//    uint *indices;
//
//    DEVICE Triangle operator[](uint i) const
//    {
//        Triangle retval;
//
//        retval.vertices[0].x = tex1Dfetch(texTriangleVtx0X, (int)indices[i]);
//        retval.vertices[0].y = tex1Dfetch(texTriangleVtx0Y, (int)indices[i]);
//        retval.vertices[0].z = tex1Dfetch(texTriangleVtx0Z, (int)indices[i]);
//
//        retval.vertices[1].x = tex1Dfetch(texTriangleVtx1X, (int)indices[i]);
//        retval.vertices[1].y = tex1Dfetch(texTriangleVtx1Y, (int)indices[i]);
//        retval.vertices[1].z = tex1Dfetch(texTriangleVtx1Z, (int)indices[i]);
//
//        retval.vertices[2].x = tex1Dfetch(texTriangleVtx2X, (int)indices[i]);
//        retval.vertices[2].y = tex1Dfetch(texTriangleVtx2Y, (int)indices[i]);
//        retval.vertices[2].z = tex1Dfetch(texTriangleVtx2Z, (int)indices[i]);
//
//        return retval;
//    }
//
//    DEVICE Triangle operator()(uint i) const
//    {
//        Triangle retval;
//
//        retval.vertices[0].x = tex1Dfetch(texTriangleVtx0X, (int)i);
//        retval.vertices[0].y = tex1Dfetch(texTriangleVtx0Y, (int)i);
//        retval.vertices[0].z = tex1Dfetch(texTriangleVtx0Z, (int)i);
//
//        retval.vertices[1].x = tex1Dfetch(texTriangleVtx1X, (int)i);
//        retval.vertices[1].y = tex1Dfetch(texTriangleVtx1Y, (int)i);
//        retval.vertices[1].z = tex1Dfetch(texTriangleVtx1Z, (int)i);
//
//        retval.vertices[2].x = tex1Dfetch(texTriangleVtx2X, (int)i);
//        retval.vertices[2].y = tex1Dfetch(texTriangleVtx2Y, (int)i);
//        retval.vertices[2].z = tex1Dfetch(texTriangleVtx2Z, (int)i);
//        return retval;
//    }
//
//    HOST void upload(
//        const FWObject::t_faceVector::const_iterator&       aBegin,
//        const FWObject::t_faceVector::const_iterator&       aEnd,
//        const FWObject&                                     aData)
//    {
//        std::vector<Triangle> triangles;
//
//        Triangle currentTriangle;
//        for(FWObject::t_faceVector::const_iterator it = aBegin; it != aEnd; ++it)
//        {
//            currentTriangle.vertices[0] = aData.vertices[it->vert1];
//            currentTriangle.vertices[1] = aData.vertices[it->vert2];
//            currentTriangle.vertices[2] = aData.vertices[it->vert3];
//
//            triangles.push_back(currentTriangle);
//        }
//        float* dataVtx0X = NULL;
//        float* dataVtx0Y = NULL;
//        float* dataVtx0Z = NULL;
//
//        float* dataHostVtx0X = NULL;
//        float* dataHostVtx0Y = NULL;
//        float* dataHostVtx0Z = NULL;
//
//        float* dataVtx1X = NULL;
//        float* dataVtx1Y = NULL;
//        float* dataVtx1Z = NULL;
//
//        float* dataHostVtx1X = NULL;
//        float* dataHostVtx1Y = NULL;
//        float* dataHostVtx1Z = NULL;
//
//        float* dataVtx2X = NULL;
//        float* dataVtx2Y = NULL;
//        float* dataVtx2Z = NULL;
//
//        float* dataHostVtx2X = NULL;
//        float* dataHostVtx2Y = NULL;
//        float* dataHostVtx2Z = NULL;
//
//        const uint numTriangles = triangles.size();
//        const uint size =  numTriangles * sizeof(float);
//
//        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx0X, size) );
//        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx0Y, size) );
//        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx0Z, size) );
//
//        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx0X, size) );
//        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx0Y, size) );
//        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx0Z, size) );
//
//
//        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx1X, size) );
//        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx1Y, size) );
//        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx1Z, size) );
//
//        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx1X, size) );
//        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx1Y, size) );
//        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx1Z, size) );
//
//        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx2X, size) );
//        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx2Y, size) );
//        CUDA_SAFE_CALL( cudaMalloc( (void**)&dataVtx2Z, size) );
//
//        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx2X, size) );
//        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx2Y, size) );
//        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHostVtx2Z, size) );
//
//        for (uint it = 0; it < numTriangles; ++it)
//        {
//            dataHostVtx0X[it] = triangles[it].vertices[0].x;
//            dataHostVtx0Y[it] = triangles[it].vertices[0].y;
//            dataHostVtx0Z[it] = triangles[it].vertices[0].z;
//
//            dataHostVtx1X[it] = triangles[it].vertices[1].x;
//            dataHostVtx1Y[it] = triangles[it].vertices[1].y;
//            dataHostVtx1Z[it] = triangles[it].vertices[1].z;
//
//            dataHostVtx2X[it] = triangles[it].vertices[2].x;
//            dataHostVtx2Y[it] = triangles[it].vertices[2].y;
//            dataHostVtx2Z[it] = triangles[it].vertices[2].z;
//        }
//
//        CUDA_SAFE_CALL(cudaMemcpy(dataVtx0X, dataHostVtx0X, size, cudaMemcpyHostToDevice));
//        CUDA_SAFE_CALL(cudaMemcpy(dataVtx0Y, dataHostVtx0Y, size, cudaMemcpyHostToDevice));
//        CUDA_SAFE_CALL(cudaMemcpy(dataVtx0Z, dataHostVtx0Z, size, cudaMemcpyHostToDevice));
//
//        CUDA_SAFE_CALL(cudaMemcpy(dataVtx1X, dataHostVtx1X, size, cudaMemcpyHostToDevice));
//        CUDA_SAFE_CALL(cudaMemcpy(dataVtx1Y, dataHostVtx1Y, size, cudaMemcpyHostToDevice));
//        CUDA_SAFE_CALL(cudaMemcpy(dataVtx1Z, dataHostVtx1Z, size, cudaMemcpyHostToDevice));
//
//        CUDA_SAFE_CALL(cudaMemcpy(dataVtx2X, dataHostVtx2X, size, cudaMemcpyHostToDevice));
//        CUDA_SAFE_CALL(cudaMemcpy(dataVtx2Y, dataHostVtx2Y, size, cudaMemcpyHostToDevice));
//        CUDA_SAFE_CALL(cudaMemcpy(dataVtx2Z, dataHostVtx2Z, size, cudaMemcpyHostToDevice));
//
//        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx0X));
//        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx0Y));
//        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx0Z));
//        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx1X));
//        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx1Y));
//        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx1Z));
//        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx2X));
//        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx2Y));
//        CUDA_SAFE_CALL(cudaFreeHost(dataHostVtx2Z));
//
//
//        cudaChannelFormatDesc chanelFormatDesc =
//            cudaCreateChannelDesc<float>();
//
//        CUDA_SAFE_CALL( cudaBindTexture(NULL, texTriangleVtx0X, (void*) dataVtx0X, chanelFormatDesc, size) );
//        CUDA_SAFE_CALL( cudaBindTexture(NULL, texTriangleVtx0Y, (void*) dataVtx0Y, chanelFormatDesc, size) );
//        CUDA_SAFE_CALL( cudaBindTexture(NULL, texTriangleVtx0Z, (void*) dataVtx0Z, chanelFormatDesc, size) );
//        CUDA_SAFE_CALL( cudaBindTexture(NULL, texTriangleVtx1X, (void*) dataVtx1X, chanelFormatDesc, size) );
//        CUDA_SAFE_CALL( cudaBindTexture(NULL, texTriangleVtx1Y, (void*) dataVtx1Y, chanelFormatDesc, size) );
//        CUDA_SAFE_CALL( cudaBindTexture(NULL, texTriangleVtx1Z, (void*) dataVtx1Z, chanelFormatDesc, size) );
//        CUDA_SAFE_CALL( cudaBindTexture(NULL, texTriangleVtx2X, (void*) dataVtx2X, chanelFormatDesc, size) );
//        CUDA_SAFE_CALL( cudaBindTexture(NULL, texTriangleVtx2Y, (void*) dataVtx2Y, chanelFormatDesc, size) );
//        CUDA_SAFE_CALL( cudaBindTexture(NULL, texTriangleVtx2Z, (void*) dataVtx2Z, chanelFormatDesc, size) );
//    }
//};

#endif // TRIANGLESOUP_H_INCLUDED_AF408B4A_6394_460C_976D_CA685339103E
