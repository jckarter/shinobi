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

#ifndef FACESOUP_H_INCLUDED_78CD6286_4107_43A0_8A07_539E68823D8C
#define FACESOUP_H_INCLUDED_78CD6286_4107_43A0_8A07_539E68823D8C

#include "../../CUDAStdAfx.h"

#include "../../Primitive/Triangle.hpp"
#include "../../Loaders/FWObject.hpp"

texture< float4, 1, cudaReadModeElementType >       tex0TriangleVertices;
texture< float4, 1, cudaReadModeElementType >       tex1TriangleVertices;
texture< float4, 1, cudaReadModeElementType >       texFaces;
texture< float4, 1, cudaReadModeElementType >       texFaceAccel;

template <int taFrameId>
struct FaceSoup
{
    typedef FWObject::t_FaceIterator    t_FaceIterator;
    typedef FWObject::t_VertexIterator   t_VertexIterator;


    uint*   indices;
    float4* faces;
    float4* vertices;
    float4* verticesHost;
    float4* faceAccel;

    DEVICE Triangle operator[](uint i) const
    {
        Triangle retval;
        float4 face = tex1Dfetch(texFaces, indices[i]);
        retval.vertices[2].x = face.x;
        retval.vertices[2].y = face.y;
        retval.vertices[2].z = face.z;

        float4 vtx0;        
        if (taFrameId == 0)//compile-time decision
        {
            vtx0 = tex1Dfetch(tex0TriangleVertices, floatAsInt(retval.vertices[2].x)); 
        }
        else
        {
            vtx0 = tex1Dfetch(tex1TriangleVertices, floatAsInt(retval.vertices[2].x)); 
        }

        retval.vertices[0].x = vtx0.x;
        retval.vertices[0].y = vtx0.y;
        retval.vertices[0].z = vtx0.z;

        float4 vtx1;
        if (taFrameId == 0)//compile-time decision
        {
            vtx1 = tex1Dfetch(tex0TriangleVertices, floatAsInt(retval.vertices[2].y)); 
        }
        else
        {
            vtx1 = tex1Dfetch(tex1TriangleVertices, floatAsInt(retval.vertices[2].y)); 
        }
       
        retval.vertices[1].x = vtx1.x;
        retval.vertices[1].y = vtx1.y;
        retval.vertices[1].z = vtx1.z;

        float4 vtx2;
        if (taFrameId == 0)//compile-time decision
        {
            vtx2 = tex1Dfetch(tex0TriangleVertices, floatAsInt(retval.vertices[2].z)); 
        }
        else
        {
            vtx2 = tex1Dfetch(tex1TriangleVertices, floatAsInt(retval.vertices[2].z)); 
        }

        retval.vertices[2].x = vtx2.x;
        retval.vertices[2].y = vtx2.y;
        retval.vertices[2].z = vtx2.z;

        return retval;
    }

    DEVICE Triangle operator()(uint i) const
    {
        Triangle retval;
        float4 face = tex1Dfetch(texFaces, i);
        retval.vertices[2].x = face.x;
        retval.vertices[2].y = face.y;
        retval.vertices[2].z = face.z;

        float4 vtx0;        
        if (taFrameId == 0)//compile-time decision
        {
            vtx0 = tex1Dfetch(tex0TriangleVertices, floatAsInt(retval.vertices[2].x)); 
        }
        else
        {
            vtx0 = tex1Dfetch(tex1TriangleVertices, floatAsInt(retval.vertices[2].x)); 
        }

        retval.vertices[0].x = vtx0.x;
        retval.vertices[0].y = vtx0.y;
        retval.vertices[0].z = vtx0.z;

        float4 vtx1;
        if (taFrameId == 0)//compile-time decision
        {
            vtx1 = tex1Dfetch(tex0TriangleVertices, floatAsInt(retval.vertices[2].y)); 
        }
        else
        {
            vtx1 = tex1Dfetch(tex1TriangleVertices, floatAsInt(retval.vertices[2].y)); 
        }

        retval.vertices[1].x = vtx1.x;
        retval.vertices[1].y = vtx1.y;
        retval.vertices[1].z = vtx1.z;

        float4 vtx2;
        if (taFrameId == 0)//compile-time decision
        {
            vtx2 = tex1Dfetch(tex0TriangleVertices, floatAsInt(retval.vertices[2].z)); 
        }
        else
        {
            vtx2 = tex1Dfetch(tex1TriangleVertices, floatAsInt(retval.vertices[2].z)); 
        }

        retval.vertices[2].x = vtx2.x;
        retval.vertices[2].y = vtx2.y;
        retval.vertices[2].z = vtx2.z;

        return retval;
    }

    DEVICE float4 getAccelDataChunck(uint aId) const
    {
        return  tex1Dfetch(texFaceAccel, aId);
    }

    DEVICE int getMaterialId(uint i) const
    {
        float4 face = tex1Dfetch(texFaces, i);
        return floatAsInt(face.w);
    }

    HOST void upload(
        const t_FaceIterator&       aBegin,
        const t_FaceIterator&       aEnd,
        const FWObject&             aData)
    {
        //////////////////////////////////////////////////////////////////////////
        //Upload vertex data
        //////////////////////////////////////////////////////////////////////////
        verticesHost = NULL;

        const size_t numVertices = aData.getNumVertices();
        const size_t verticesSize =  numVertices * sizeof(float4);

        CUDA_SAFE_CALL( cudaMalloc( (void**)&vertices, verticesSize) );
        CUDA_SAFE_CALL( cudaMallocHost((void**)&verticesHost, verticesSize) );

        uint vtxId = 0;
        for (t_VertexIterator it = aData.verticesBegin(); it != aData.verticesEnd(); ++it, ++vtxId)
        {
            verticesHost[vtxId].x = it->x;
            verticesHost[vtxId].y = it->y;
            verticesHost[vtxId].z = it->z;
            verticesHost[vtxId].w = 0.f;
        }

        CUDA_SAFE_CALL( 
            cudaMemcpy(vertices, verticesHost, verticesSize, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaFreeHost(verticesHost));


        cudaChannelFormatDesc chanelFormatDesc =
            cudaCreateChannelDesc<float4>();

        if (taFrameId == 0)//compile-time decision
        {
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex0TriangleVertices, (void*) vertices,
                chanelFormatDesc, verticesSize) );
        }
        else
        {
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex1TriangleVertices, (void*) vertices,
                chanelFormatDesc, verticesSize) );
        }

        //////////////////////////////////////////////////////////////////////////
        //Upload Faces
        //////////////////////////////////////////////////////////////////////////
        const size_t numFaces = aData.getNumFaces();
        const size_t facesSize =  numFaces * sizeof(float4);

        uint2* facesHost = NULL;

        CUDA_SAFE_CALL( cudaMalloc( (void**)&faces, facesSize) );
        CUDA_SAFE_CALL( cudaMallocHost((void**)&facesHost, facesSize) );

        uint faceId = 0;
        for (t_FaceIterator it = aBegin; it != aEnd; ++it, ++faceId)
        {
            facesHost[2 * faceId    ].x = static_cast<uint>(it->vert1);
            facesHost[2 * faceId    ].y = static_cast<uint>(it->vert2);
            facesHost[2 * faceId + 1].x = static_cast<uint>(it->vert3);
            facesHost[2 * faceId + 1].y = static_cast<uint>(it->material);
        }

        CUDA_SAFE_CALL( 
            cudaMemcpy(faces, facesHost, facesSize, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL( cudaFreeHost(facesHost) );

        chanelFormatDesc = cudaCreateChannelDesc<float4>();

        CUDA_SAFE_CALL( cudaBindTexture(NULL, texFaces, (void*) faces,
            chanelFormatDesc, facesSize) );

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
            CUDA_SAFE_CALL( cudaUnbindTexture( &tex0TriangleVertices) );
        }
        else
        {
            CUDA_SAFE_CALL( cudaUnbindTexture( &tex1TriangleVertices) );
        }

        if (vertices != NULL)
        {
            CUDA_SAFE_CALL( cudaFree(vertices) );
        }

        //////////////////////////////////////////////////////////////////////////
        //Upload vertex data
        //////////////////////////////////////////////////////////////////////////

        const size_t numVertices1 = aKeyFrame1.getNumVertices();
        const size_t numVertices2 = aKeyFrame2.getNumVertices();
        const size_t numVertices  = numVertices2;
        const size_t verticesSize = numVertices * sizeof(float4);

        CUDA_SAFE_CALL( cudaMalloc( (void**)&vertices, verticesSize) );        
        CUDA_SAFE_CALL( cudaMallocHost((void**)&verticesHost, verticesSize) );

        size_t it = 0;
        for (; it < cudastd::min(numVertices1, numVertices2); ++it)
        {
            verticesHost[it].x = aKeyFrame1.getVertex(it).x * (1.f - aCoeff) + aKeyFrame2.getVertex(it).x * aCoeff;
            verticesHost[it].y = aKeyFrame1.getVertex(it).y * (1.f - aCoeff) + aKeyFrame2.getVertex(it).y * aCoeff;
            verticesHost[it].z = aKeyFrame1.getVertex(it).z * (1.f - aCoeff) + aKeyFrame2.getVertex(it).z * aCoeff;
            verticesHost[it].w = 0.f;

            oMinBound.x = cudastd::min(verticesHost[it].x, oMinBound.x);
            oMinBound.y = cudastd::min(verticesHost[it].y, oMinBound.y);
            oMinBound.z = cudastd::min(verticesHost[it].z, oMinBound.z);
            oMaxBound.x = cudastd::max(verticesHost[it].x, oMaxBound.x);
            oMaxBound.y = cudastd::max(verticesHost[it].y, oMaxBound.y);
            oMaxBound.z = cudastd::max(verticesHost[it].z, oMaxBound.z);
        }

        for (; it < numVertices2 ; ++it)
        {
            verticesHost[it].x = aKeyFrame2.getVertex(it).x;
            verticesHost[it].y = aKeyFrame2.getVertex(it).y;
            verticesHost[it].z = aKeyFrame2.getVertex(it).z;
            verticesHost[it].w = 0.f;

            oMinBound.x = cudastd::min(verticesHost[it].x, oMinBound.x);
            oMinBound.y = cudastd::min(verticesHost[it].y, oMinBound.y);
            oMinBound.z = cudastd::min(verticesHost[it].z, oMinBound.z);
            oMaxBound.x = cudastd::max(verticesHost[it].x, oMaxBound.x);
            oMaxBound.y = cudastd::max(verticesHost[it].y, oMaxBound.y);
            oMaxBound.z = cudastd::max(verticesHost[it].z, oMaxBound.z);

        }

        CUDA_SAFE_CALL( 
            cudaMemcpyAsync(vertices, verticesHost, verticesSize, cudaMemcpyHostToDevice, aStream));

        CUDA_SAFE_CALL( cudaFreeHost(verticesHost) );

        cudaChannelFormatDesc chanelFormatDesc =
            cudaCreateChannelDesc<float4>();

        if (taFrameId == 0)//compile-time decision
        {
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex0TriangleVertices, (void*) vertices,
                chanelFormatDesc, verticesSize) );
        }
        else
        {
            CUDA_SAFE_CALL( cudaBindTexture(NULL, tex1TriangleVertices, (void*) vertices,
                chanelFormatDesc, verticesSize) );
        }

    }

    HOST void cleanup()
    {
        CUDA_SAFE_CALL( cudaFree(indices) );
        CUDA_SAFE_CALL( cudaFree(vertices) );
        //CUDA_SAFE_CALL( cudaFree(faceAccel) );
    }

    HOST void uploadShevtsovAccelerators(
        const t_FaceIterator&       aBegin,
        const t_FaceIterator&       aEnd,
        const FWObject&             aData)
    {

        Triangle* triangles;
        const size_t numTriangles = aData.getNumFaces();
        const size_t triSize = numTriangles * sizeof(Triangle);

        CUDA_SAFE_CALL( cudaMallocHost((void**)&triangles, triSize) );

        size_t triId = 0;
        for(FWObject::t_FaceIterator it = aBegin; it != aEnd; ++it, ++triId)
        {
            triangles[triId].vertices[0] = aData.getVertex(it->vert1);
            triangles[triId].vertices[1] = aData.getVertex(it->vert2);
            triangles[triId].vertices[2] = aData.getVertex(it->vert3);
        }

        ShevtsovTriAccel* dataHost;

        const size_t size =  numTriangles * sizeof(ShevtsovTriAccel);

        CUDA_SAFE_CALL( cudaMalloc( (void**)&faceAccel, size) );
        CUDA_SAFE_CALL( cudaMallocHost((void**)&dataHost, size) );


        for (uint it = 0; it < numTriangles; ++it)
        {
            dataHost[it] = ShevtsovTriAccel(triangles[it]);
        }

        CUDA_SAFE_CALL( 
            cudaMemcpy(faceAccel, dataHost, size, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaFreeHost(dataHost));
        CUDA_SAFE_CALL(cudaFreeHost(triangles));


        cudaChannelFormatDesc chanelFormatDesc =
            cudaCreateChannelDesc<float4>();

        CUDA_SAFE_CALL( cudaBindTexture(NULL, texFaceAccel, (void*) faceAccel,
            chanelFormatDesc, size) );

    }
};


#endif // FACESOUP_H_INCLUDED_78CD6286_4107_43A0_8A07_539E68823D8C
