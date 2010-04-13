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

#ifndef INTERSECTORTEST_H_INCLUDED_230A85FF_31D4_40A4_958B_C74AB74BE7C0
#define INTERSECTORTEST_H_INCLUDED_230A85FF_31D4_40A4_958B_C74AB74BE7C0

#include "../CUDAStdAfx.h"

#include "../CUDAConfig.h"

#include "../Primitive/Triangle.hpp"
#include "../Primitive/Camera.h"
#include "../Primitive/PixelSampler.h"
#include "../Primitive/FrameBuffer.h"

#include "../Loaders/FWObject.hpp"

#include "../Control/Structure/FaceSoup.h"

#include "../Intersector/SimpleIntersector.hpp"
#include "../Intersector/HybridIntersector.hpp"

#include "../Utils/ImagePNG.hpp"

#define NUMCANDIDATES 400
#define NUMACTIVETHREADS 32  

typedef RegularPixelSampler<SAMPLESPERPIXELX, SAMPLESPERPIXELY> t_RegularPixelSampler;
typedef FaceSoup<0>                                             t_PrimitiveStorage0;

//typedef SimpleIntersector<t_PrimitiveStorage0>                   t_Intersector;
typedef PilotRayIntersector<t_PrimitiveStorage0>                   t_Intersector;


DEVICE CONSTANT Camera                      dcCamera;
DEVICE CONSTANT t_RegularPixelSampler       dcRegularPixelSampler;
DEVICE CONSTANT int                         dcRunId;

DEVICE uint2 getNumCandidates()
{
    uint2 retval = make_uint2(0u, 0u);
    if (threadId1DInWarp() < NUMACTIVETHREADS)
    {
        retval.y = NUMCANDIDATES;
    }

    //if(threadId1DInWarp() % 2 == 0)
    //{
    //    retval.y = NUMCANDIDATES;
    //}
    return retval;
}

GLOBAL void runIntersector(
                   t_PrimitiveStorage0 aStorage,
                   vec3f* oFrameBuffer,
                   int* aGlobalMemoryPtr)
{
    //////////////////////////////////////////////////////////////////////////
    //Ray generation
    //////////////////////////////////////////////////////////////////////////
    extern SHARED uint sharedMem[];
    vec3f* rayOrg =
        (vec3f*)(sharedMem + t_Intersector::SHAREDMEMSIZE);
    vec3f* rayDir = rayOrg + RENDERTHREADSX * RENDERTHREADSY;

    rayOrg[threadId1D()] = dcCamera.getPosition();

    float screenX, screenY;
    dcRegularPixelSampler.getPixelCoords(globalThreadId1D(),
        (float)dcRunId, screenX, screenY);

    rayDir[threadId1D()] = dcCamera.getDirection(screenX, screenY);

    //////////////////////////////////////////////////////////////////////////
    //Intersection tests
    //////////////////////////////////////////////////////////////////////////

    uint  bestHit   = NUMCANDIDATES;
    float rayT      = FLT_MAX;
    uint2 cellRange = getNumCandidates();

    t_Intersector intersect;
    intersect(rayOrg, rayDir, rayT, cellRange, aStorage, sharedMem, bestHit);

    //////////////////////////////////////////////////////////////////////////
    //Output
    //////////////////////////////////////////////////////////////////////////
    if (rayT >= FLT_MAX)
    {
        oFrameBuffer[globalThreadId1D()] =  vec3f::rep(0.2f);
    }
    else
    {
        oFrameBuffer[globalThreadId1D()].x = BACKGROUND_R;
        oFrameBuffer[globalThreadId1D()].y = BACKGROUND_G;
        oFrameBuffer[globalThreadId1D()].z = BACKGROUND_B;
    }

}


class IntersectorTest
{
    void deviceInit(int argc, char* argv[])
    {
        CUT_DEVICE_INIT(argc, argv);
    }

    void intersectorTest(
        dim3                    aBlockGrid,
        dim3                    aThreadBlock,
        const uint              aSharedMemorySize,
        t_PrimitiveStorage0&    aTriangles,
        FrameBuffer&            aFrameBuffer,
        int*&                   aGlobalMemoryPtr,
        Image&                  oResult)
    {
        for(int imageId = 0; imageId < NUMIMAGES; ++imageId)
        {

            CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcRunId", &imageId, sizeof(int)) );

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            runIntersector<<< aBlockGrid, aThreadBlock, aSharedMemorySize >>>(
                aTriangles,
                aFrameBuffer.deviceData,
                aGlobalMemoryPtr);

            CUT_CHECK_ERROR("Kernel Execution failed.\n");

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            cudastd::log::out << "run " << imageId;
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            cudastd::log::out << " run-time: " << elapsedTime << "ms\r";
        }
    }


public:
    void run(int argc, char* argv[])
    {
        deviceInit(argc, argv);

        ////////////////////////////////////////////////////////////////////////////////
        //Load scene data
        ////////////////////////////////////////////////////////////////////////////////
        FWObject scene;
        scene.allocateVertices(NUMCANDIDATES * 3u);
        scene.allocateFaces(NUMCANDIDATES);
        vec3f vtx;
        vtx.x = -2.f;
        vtx.y = -2.f;
        vtx.z = -2.f;
        scene.getVertex(0) = vtx;
        vtx.x =  2.f;
        vtx.y = -2.f;
        vtx.z = -2.f;
        scene.getVertex(1) = vtx;
        vtx.x = -2.f;
        vtx.y = -4.f;
        vtx.z = -2.f;
        scene.getVertex(2) = vtx;

        FWObject::Face face(&scene);
        face.material = 0u;
        face.vert1 = 0u;
        face.vert2 = 1u;
        face.vert3 = 2u;
        scene.getFace(0) = face;

        uint* indices;
        CUDA_SAFE_CALL( cudaMallocHost((void**)&indices, NUMCANDIDATES * sizeof(uint)) );
        indices[0] = 0;

        for (int triId = 1; triId < NUMCANDIDATES; ++triId)
        {
            vtx.x = -2.f;
            vtx.y = -4.f;
            vtx.z = -2.f;
            scene.getVertex(3 * triId + 0) = vtx;
            vtx.x = -2.f;
            vtx.y = -4.f;
            vtx.z = 2.f;
            scene.getVertex(3 * triId + 1) = vtx;
            vtx.x = 2.f;
            vtx.y = -4.f;
            vtx.z = 2.f;
            scene.getVertex(3 * triId + 2) = vtx;

            face.vert1 += 3u;
            face.vert2 += 3u;
            face.vert3 += 3u;

            scene.getFace(triId) = face;
            indices[triId] = triId;
        }

        t_PrimitiveStorage0      triangles;

        CUDA_SAFE_CALL(cudaMalloc((void**)&triangles.indices, NUMCANDIDATES * sizeof(uint)));

        ////////////////////////////////////////////////////////////////////////////////
        //Upload data to device
        ////////////////////////////////////////////////////////////////////////////////
        CUDA_SAFE_CALL( 
            cudaMemcpy(triangles.indices, indices, NUMCANDIDATES * sizeof(uint), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaFreeHost(indices));

        triangles.upload(scene.facesBegin(), scene.facesEnd(), scene);

        Camera cameraHost;
        vec3f position, orientation, up;
        float fov;

        position.x = 0.f;
        position.y = 0.f;
        position.z = 0.f;
        orientation.x = 0.f;
        orientation.y = -1.f;
        orientation.z = 0.f;
        up.x = 0.f;
        up.y = 0.f;
        up.z = 1.f;
        fov = 66.f;
        cameraHost.init(position, orientation, up, fov, gRESX, gRESY);
        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcCamera", &cameraHost, sizeof(Camera)) );


        t_RegularPixelSampler pixelSampler;
        pixelSampler.resX = (float)gRESX;
        pixelSampler.resY = (float)gRESY;

        CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcRegularPixelSampler", &pixelSampler, sizeof(t_RegularPixelSampler)) );

        FrameBuffer frameBuffer;
        frameBuffer.init(gRESX, gRESY);
        frameBuffer.setZero();

        ////////////////////////////////////////////////////////////////////////////////
        //Render
        ////////////////////////////////////////////////////////////////////////////////

        const uint sharedMemorySize =
            RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +                   //rayOrg
            RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +                   //rayDir
            t_Intersector::SHAREDMEMSIZE * 4u +       //Intersector
            0u;

        const uint globalMemorySize = gRESX * gRESY / (RENDERTHREADSX * RENDERTHREADSY) *
            t_Intersector::GLOBALMEMSIZE * 4u;

        int* globalMemoryPtr;
        CUDA_SAFE_CALL( cudaMalloc((void**)&globalMemoryPtr, globalMemorySize));

        cudastd::log::out << "Shared memory size in bytes: " << sharedMemorySize << "\n"
            << "Global memory size in bytes: " << globalMemorySize << "\n";

        cudastd::log::out << "Rendering ...\n";

        dim3 threadBlock( RENDERTHREADSX, RENDERTHREADSY );
        dim3 blockGrid  ( gRESX / RENDERTHREADSX, gRESY / RENDERTHREADSY );

        Image result(gRESX, gRESY);

        intersectorTest(blockGrid, threadBlock, sharedMemorySize, triangles,
            frameBuffer, globalMemoryPtr, result);


        frameBuffer.download(result.getBits());
        result.gammaCorrect(GAMMA);

        cudastd::log::out << "\nWriting " << OUTPUT << " ...\n";
        result.writePNG(OUTPUT);
        cudastd::log::out << "Finished.\n";
    }
};


#endif // INTERSECTORTEST_H_INCLUDED_230A85FF_31D4_40A4_958B_C74AB74BE7C0
