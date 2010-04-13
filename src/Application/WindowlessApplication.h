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

#ifndef WINDOWLESSAPPLICATION_H_INCLUDED_5E534ADF_0419_43F2_84E4_731D7DD67549
#define WINDOWLESSAPPLICATION_H_INCLUDED_5E534ADF_0419_43F2_84E4_731D7DD67549

#include "../CUDAStdAfx.h"
#include "../CUDAConfig.h"

#include "../Shinobi.h"
#include "../Engine/StaticRTEngine.h"

#include "WindowlessApplication.hpp"

#include "../Loaders/SceneLoader.h"

#include "../Utils/ImagePNG.hpp"

class ImageSequenceGenerator
{
public:
    static void renderMultiFrames(
        CameraManager&          aView,
        CameraPath&             aCameraPath,
        Camera&                 aCameraHost,
        t_PrimitiveStorage0&     aTriangles,
        t_MaterialStorage&      aMaterials,
        FrameBuffer&            aFrameBuffer,
        FrameBuffer&            aFinalImage)
    {

        Image frame(gRESX, gRESY);

        for(size_t frameId = 242; frameId < aCameraPath.getNumFrames(); ++frameId)
        {
            cudastd::log::out << "Frame " << frameId << " ...\n";
            aView = aCameraPath.getCamera(frameId);
            aCameraHost.init(aView.getPosition(), aView.getOrientation(), aView.getUp(), aView.getFOV(), gRESX, gRESY);
            CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcCamera", &aCameraHost, sizeof(Camera)) );

            aFrameBuffer.setZero();
            aFinalImage.setZero();

            StaticRTEngine::renderImage(aTriangles,aMaterials,aFrameBuffer,aFinalImage);

            aFinalImage.download(frame.getBits());
            frame.gammaCorrect(GAMMA);

            char* fileNameBase = "output/";
            char frameIdname[7];
#ifdef _WIN32
            //sprintf_s(frameIdname, "%ld", frameId);
            _itoa_s(static_cast<int>(frameId), frameIdname, 10);
#else
            sprintf(frameIdname, "%ld", frameId);
            //itoa(static_cast<int>(frameId), frameIdname, 10);
#endif
            const char* fileName = strcat(strcat(fileNameBase, frameIdname),".png");
            frame.writePNG(fileName);

            cudastd::log::out << "\ndone.\n";
        }//end for(size_t frameId...
    }
};


void WindowlessApplication::deviceInit(int argc, char* argv[])
{
    CUT_DEVICE_INIT(argc, argv);
}

void WindowlessApplication::gatherStatistics(FrameBuffer& aFrameBuffer, FrameBuffer& aFinalImage)
{
    ////////////////////////////////////////////////////////////////////////////
    //Statistics (NOTE: Use DebugIntegrator for the numbers to make sense)
    ////////////////////////////////////////////////////////////////////////////

    Image frameBuffer(gRESX, gRESY);
    aFrameBuffer.download(frameBuffer.getBits());

    float avgPathLength = 0.f;
    float maxPathLength = 0.f;
    float avgActiveLength = 0.f;
    float avgIntersections = 0.f;

    for(int y = 0; y < gRESY; ++y)
    {
        for(int x = 0; x < gRESX; ++x)
        {
            vec3f tmp = frameBuffer(x, y);
            avgPathLength += tmp.x;
            maxPathLength = cudastd::max(tmp.x, maxPathLength);
            avgActiveLength += tmp.y;
            avgIntersections += tmp.z;
            frameBuffer(x,y).x *= SCALE_R_NUM_RAYS;
            frameBuffer(x,y).y *= SCALE_G_NUM_ACTIVE;
            frameBuffer(x,y).z *= SCALE_B_NUM_ITESTS;
        }
    }

    cudastd::log::out << "-----------------STATISTICS-----------------\n";
    cudastd::log::out << "Total number of rays : " << avgPathLength << "\n";
    cudastd::log::out << "Total number of intersection tests: " << avgIntersections << "\n";
    cudastd::log::out << "Intersection tests per ray: " << avgIntersections / avgPathLength << "\n";


    float numRaysRCP = 1.f / static_cast<float>(gRESX * gRESY);
    avgPathLength *= numRaysRCP;
    avgActiveLength *= numRaysRCP;
    avgIntersections *= numRaysRCP;

    cudastd::log::out << "Average path length : " << avgPathLength << "\n";
    cudastd::log::out << "Longest path length : " << maxPathLength << "\n";
    cudastd::log::out << "Average life of a thread (bounces) : " << avgActiveLength << "\n";
    cudastd::log::out << "Intersections per path : " << avgIntersections << "\n";
    cudastd::log::out << "--------------------------------------------\n";

    cudastd::log::out << "\nWriting " << OUTPUT << " ...\n";
    frameBuffer.writePNG(OUTPUT);
}


void WindowlessApplication::generateImage(int argc, char* argv[])
{
    deviceInit(argc, argv);

    ////////////////////////////////////////////////////////////////////////////////
    //Load scene data
    ////////////////////////////////////////////////////////////////////////////////

    FWObject scene;
    CameraPath cameraPath;
    CameraManager view;
    AreaLightSource lightSource;

    loadScene(scene, cameraPath, view, lightSource);

    t_Grid                  grid;
    t_PrimitiveStorage0      triangles;
    t_MaterialStorage       materials;

    cudastd::log::out << "Building acceleration structure...\n";

    cudaEvent_t start_build, end_build;
    cudaEventCreate(&start_build);
    cudaEventCreate(&end_build);
    cudaEventRecord(start_build, 0);

    StaticRTEngine::build(
        grid,
        triangles,
        scene.facesBegin(),
        scene.facesEnd(),
        scene);

    cudastd::log::out << "Finished build.\n";
    fflush(stderr);

    cudaEventRecord(end_build, 0);
    cudaEventSynchronize(end_build);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start_build, end_build);

    cudastd::log::floatPrecision(4);
    cudastd::log::out << "Elapsed time for prepartion: " << elapsedTime << "ms\n";

    ////////////////////////////////////////////////////////////////////////////////
    //Upload data to device
    ////////////////////////////////////////////////////////////////////////////////
    materials.upload(scene.materialsBegin(), scene.getNumMaterials());

    Camera cameraHost;
    cameraHost.init(view.getPosition(), view.getOrientation(), view.getUp(), view.getFOV(), gRESX, gRESY);
    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcCamera", &cameraHost, sizeof(Camera)) );


    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcLightSource", &lightSource, sizeof(AreaLightSource)) );


    t_RegularPixelSampler pixelSampler;
    pixelSampler.resX = (float)gRESX;
    pixelSampler.resY = (float)gRESY;

    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcRegularPixelSampler", &pixelSampler, sizeof(t_RegularPixelSampler)) );

    t_RandomPixelSampler randPixelSampler;
    randPixelSampler.resX = (float)gRESX;
    randPixelSampler.resY = (float)gRESY;

    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcRandomPixelSampler", &randPixelSampler, sizeof(t_RegularPixelSampler)) );

    FrameBuffer frameBuffer;
    frameBuffer.init(gRESX, gRESY);
    frameBuffer.setZero();

    FrameBuffer finalImage;
    finalImage.init(gRESX, gRESY);
    finalImage.setZero();

    t_GridParameters gridParams = grid.getParameters();
    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcGridParameters", &gridParams, sizeof(t_GridParameters)) );


    ////////////////////////////////////////////////////////////////////////////////
    //Render
    ////////////////////////////////////////////////////////////////////////////////

    const uint sharedMemorySize =
        RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +                   //rayOrg
        RENDERTHREADSX * RENDERTHREADSY * sizeof(vec3f) +                   //rayDir
        t_DeviceIntegrator::t_Intersector::SHAREDMEMSIZE * 4u +       //Intersector
        0u;

    const uint globalMemorySize = gRESX * gRESY / (RENDERTHREADSX * RENDERTHREADSY) *
        t_DeviceIntegrator::t_Intersector::GLOBALMEMSIZE * 4u;

    int* globalMemoryPtr;
    CUDA_SAFE_CALL( cudaMalloc((void**)&globalMemoryPtr, globalMemorySize));

    cudastd::log::out << "Shared memory size in bytes: " << sharedMemorySize << "\n"
        << "Global memory size in bytes: " << globalMemorySize << "\n";

    cudastd::log::out << "Rendering ...\n";

    Image result(gRESX, gRESY);

    StaticRTEngine::renderImage(triangles,materials,frameBuffer,finalImage);

    //ImageSequenceGenerator::renderMultiFrames(view, cameraPath, cameraHost, triangles, materials,
    //    frameBuffer, finalImage);

#ifdef GATHERSTATISTICS
    gatherStatistics(frameBuffer, finalImage);
#else
    finalImage.download(result.getBits());
    result.gammaCorrect(GAMMA);
    cudastd::log::out << "\nWriting " << OUTPUT << " ...\n";
    result.writePNG(OUTPUT);
#endif
    cudastd::log::out << "Finished.\n";
}



#endif // WINDOWLESSAPPLICATION_H_INCLUDED_5E534ADF_0419_43F2_84E4_731D7DD67549
