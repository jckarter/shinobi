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

#ifndef SDLGLAPPLICATION_H_INCLUDED_23267079_3E9D_4368_B12B_886CE6D5BE51
#define SDLGLAPPLICATION_H_INCLUDED_23267079_3E9D_4368_B12B_886CE6D5BE51

#include "../CUDAStdAfx.h"
#include "../CUDAConfig.h"

#include "../Shinobi.h"

#include "SDLGLApplication.hpp"

#include "../Loaders/SceneLoader.h"
#include "../Engine/StaticRTEngine.h"
#include "../Engine/DynamicRTEngine.h"

AnimationManager        gAnimation;
FWObject                gScene;
CameraPath              gCameraPath;
AreaLightSource         gLightSource;
t_PrimitiveStorage0     gTriangles;
t_MaterialStorage       gMaterials;
t_Grid                  gGrid;
FrameBuffer             gFinalImage;
FrameBuffer             gFrameBuffer;
t_RegularPixelSampler   gRegularPixelSampler;
t_RandomPixelSampler    gRandomPixelSampler;

void SDLGLApplication::deviceInit(int argc, char* argv[])
{
    CUT_DEVICE_INIT(argc, argv);
}

void SDLGLApplication::initScene()
{

#ifndef ANIMATION

    bool haveScene = loadScene(gScene, gCameraPath, mInitialCamera, gLightSource);

    if (haveScene)
    {
        mCamera = mInitialCamera;
    }
    else
    {
        mQuit = true;
        return;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //Upload data to device
    ////////////////////////////////////////////////////////////////////////////////
    gMaterials.upload(gScene.materialsBegin(), gScene.getNumMaterials());

    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcLightSource", &gLightSource, sizeof(AreaLightSource)) );

    gRegularPixelSampler.resX = (float)gRESX;
    gRegularPixelSampler.resY = (float)gRESY;

    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcRegularPixelSampler", &gRegularPixelSampler, sizeof(t_RegularPixelSampler)) );

    gRandomPixelSampler.resX = (float)gRESX;
    gRandomPixelSampler.resY = (float)gRESY;

    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcRandomPixelSampler", &gRegularPixelSampler, sizeof(t_RegularPixelSampler)) );

    gFrameBuffer.init(gRESX, gRESY);
    gFrameBuffer.setZero();

    gFinalImage.init(gRESX, gRESY);
    gFinalImage.setZero();

    cudastd::log::out << "Building acceleration structure...\n";

    cudaEvent_t start_build, end_build;
    cudaEventCreate(&start_build);
    cudaEventCreate(&end_build);
    cudaEventRecord(start_build, 0);

#ifdef LAZYBUILD
    Camera cameraHost;
    cameraHost.init(mInitialCamera.getPosition(), 
        mInitialCamera.getOrientation(), mInitialCamera.getUp(),
        mInitialCamera.getFOV(), gRESX, gRESY);

    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcCamera", &cameraHost, sizeof(Camera)) );
#endif

    StaticRTEngine::build(
        gGrid,
        gTriangles,
        gScene.facesBegin(),
        gScene.facesEnd(),
        gScene);


    t_GridParameters gridParams = gGrid.getParameters();
    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcGridParameters", &gridParams, sizeof(t_GridParameters)) );

    cudastd::log::out << "Done.\n";

    cudaEventRecord(end_build, 0);
    cudaEventSynchronize(end_build);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start_build, end_build);

    cudastd::log::floatPrecision(4);
    cudastd::log::out << "Elapsed time for preparation: " << elapsedTime << "ms\n";

    cudaEventDestroy(start_build);
    cudaEventDestroy(end_build);

#else

    bool haveAnimation = loadAnimation(gAnimation, gCameraPath, mInitialCamera, gLightSource);

    if (haveAnimation)
    {
        mCamera = mInitialCamera;
    }
    else
    {
        mQuit = true;
        return;
    }

    cudastd::log::out << "Initializing...\n";

    gMaterials.upload(gAnimation.getFrame(gAnimation.getFrameId()).materialsBegin(),
        gAnimation.getFrame(gAnimation.getFrameId()).getNumMaterials());

    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcLightSource", &gLightSource, sizeof(AreaLightSource)) );

    gRegularPixelSampler.resX = (float)gRESX;
    gRegularPixelSampler.resY = (float)gRESY;

    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcRegularPixelSampler", &gRegularPixelSampler, sizeof(t_RegularPixelSampler)) );

    gRandomPixelSampler.resX = (float)gRESX;
    gRandomPixelSampler.resY = (float)gRESY;

    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcRandomPixelSampler", &gRegularPixelSampler, sizeof(t_RegularPixelSampler)) );

    gFrameBuffer.init(gRESX, gRESY);
    gFrameBuffer.setZero();

    gFinalImage.init(gRESX, gRESY);
    gFinalImage.setZero();

#ifdef LAZYBUILD
    Camera cameraHost;
    cameraHost.init(mInitialCamera.getPosition(), 
        mInitialCamera.getOrientation(), mInitialCamera.getUp(),
        mInitialCamera.getFOV(), gRESX, gRESY);

    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcCamera", &cameraHost, sizeof(Camera)) );
#endif

    DynamicRTEngine::init(
        gGrid,
        gAnimation.getFrame(gAnimation.getNumKeyFrames() / 2).facesBegin(),
        gAnimation.getFrame(gAnimation.getNumKeyFrames() / 2).facesEnd(),
        gAnimation.getFrame(gAnimation.getNumKeyFrames() / 2));

#endif
}

void SDLGLApplication::changeCUDAWindowSize(const int aResX, const int aResY)
{
    gRegularPixelSampler.resX = (float)gRESX;
    gRegularPixelSampler.resY = (float)gRESY;

    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcRegularPixelSampler", &gRegularPixelSampler, sizeof(t_RegularPixelSampler)) );

    gRandomPixelSampler.resX = (float)gRESX;
    gRandomPixelSampler.resY = (float)gRESY;

    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcRandomPixelSampler", &gRegularPixelSampler, sizeof(t_RegularPixelSampler)) );

    gFrameBuffer.cleanup();
    gFrameBuffer.init(gRESX, gRESY);
    gFrameBuffer.setZero();

    gFinalImage.cleanup();
    gFinalImage.init(gRESX, gRESY);
    gFinalImage.setZero();
}

void SDLGLApplication::generateFrame(float& oRenderTime, float& oBuildTime)
{
    Camera cameraHost;
    cameraHost.init(mCamera.getPosition(), mCamera.getOrientation(), mCamera.getUp(), mCamera.getFOV(), gRESX, gRESY);
    CUDA_SAFE_CALL( cudaMemcpyToSymbol("dcCamera", &cameraHost, sizeof(Camera)) );

    oRenderTime = oBuildTime = 0.f;
    
#ifdef ANIMATION

    switch ( mRenderMode ) 
    {
    case SINGLEKERNEL:
        DynamicRTEngine::renderFrameSingleKernel(
            gGrid,
            gAnimation.getFrame(gAnimation.getFrameId()),
            gAnimation.getFrame(gAnimation.getNextFrameId()),
            gAnimation.getInterpolationCoefficient(),
            gMaterials,
            gFrameBuffer,
            gFinalImage,
            0,
            oRenderTime,
            oBuildTime);
        break;

    case MULTIKERNEL:
        DynamicRTEngine::renderFrame(
            gGrid,
            gAnimation.getFrame(gAnimation.getFrameId()),
            gAnimation.getFrame(gAnimation.getNextFrameId()),
            gAnimation.getInterpolationCoefficient(),
            gMaterials,
            gFrameBuffer,
            gFinalImage,
            0,
            oRenderTime,
            oBuildTime);
        break;
    default:
        break;
    }//switch ( mRenderMode )
    

    gAnimation.nextFrame();

#else

    switch ( mRenderMode ) 
    {
    case SINGLEKERNEL:
        StaticRTEngine::renderFrameSingleKernel(gTriangles,gMaterials,gFrameBuffer,gFinalImage, 
            mNumImages++, oRenderTime);
        break;
    case MULTIKERNEL:
        StaticRTEngine::renderFrame(gTriangles,gMaterials,gFrameBuffer,gFinalImage, 
            mNumImages++, oRenderTime);
        break;
    default:
        break;
    }//switch ( mRenderMode )

#endif


    gFinalImage.download((vec3f*)frameBufferFloatPtr, gRESX, gRESY);

#ifdef ANIMATION
    gFinalImage.setZero();
#endif

#ifdef GATHERSTATISTICS
    
    switch ( mRenderMode ) 
    {
    case SINGLEKERNEL:
        for(int y = 0; y < gRESY; ++y)
        {
            for(int x = 0; x < gRESX; ++x)
            {
                float itests = 
                    frameBufferFloatPtr[3 * (x + gRESX * y) + 2] * SCALE_B_NUM_ITESTS;
                frameBufferFloatPtr[3 * (x + gRESX * y)] = itests;
                frameBufferFloatPtr[3 * (x + gRESX * y) + 1] = itests;
                frameBufferFloatPtr[3 * (x + gRESX * y) + 2] = itests;
            }
        }
        break;
    case MULTIKERNEL:
        break;
    default:
        break;
    }//switch ( mRenderMode )

#endif

}

void SDLGLApplication::cleanupCUDA(void)
{
    gGrid.cleanup();
    gTriangles.cleanup();
    gFinalImage.cleanup();
    gFrameBuffer.cleanup();
}

#endif // SDLGLAPPLICATION_H_INCLUDED_23267079_3E9D_4368_B12B_886CE6D5BE51
