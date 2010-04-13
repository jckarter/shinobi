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

#ifndef SCENELOADER_H_INCLUDED_4B232D1A_F513_4D82_9A41_3F7E4E6A6BFC
#define SCENELOADER_H_INCLUDED_4B232D1A_F513_4D82_9A41_3F7E4E6A6BFC

#include "SceneConfiguration.hpp"
#include "CameraManager.hpp"
#include "CameraPath.hpp"
#include "FWObject.hpp"
#include "AnimationManager.hpp"
#include "LightSourceLoader.hpp"

#include "../Primitive/LightSource.hpp"

bool loadAnimation(
                   AnimationManager&    oAnimation,
                   CameraPath&         oCameraPath,
                   CameraManager&       oView,
                   AreaLightSource&     oLightSource
                   )
{  
    cudastd::log::out << "Reading configuration file...\n";

    SceneConfiguration sceneConfig = loadSceneConfiguration(CONFIGURATION);

    bool retval = true;

    if (sceneConfig.hasFrameFileNamePrefix &&
        sceneConfig.hasFrameFileNameSuffix &&
        sceneConfig.numFrames != 0)
    {
        cudastd::log::out << "Loading animation...\n";
        oAnimation.read(sceneConfig.frameFileNamePrefix,
            sceneConfig.frameFileNameSuffix, sceneConfig.numFrames);
        oAnimation.setStepSize(sceneConfig.frameStepSize);

        cudastd::log::out << "Number of primitives: " <<
            oAnimation.getFrame(0).getNumFaces() << "\n";
    }
    else
    {
        cudastd::log::out << "No valid key frame animation specified in file "
            << CONFIGURATION << "\n";

        retval = false;
    }

    if (sceneConfig.hasCameraFileName)
    {
        cudastd::log::out << "Loading camera configuration...\n";
        oView.read(sceneConfig.cameraFileName);
    }
    else
    {
        cudastd::log::out << "No camera configuration specified in file "
            << CONFIGURATION << "\n";
    }

    if (sceneConfig.hasCameraPathFileName)
    {
        oCameraPath.read(sceneConfig.cameraPathFileName);
    }

    if(sceneConfig.hasLightsFileName)
    {
        LightSourceLoader loader;
        oLightSource = loader.loadFromFile(sceneConfig.lightsFileName);
    }

    return retval;

}

bool loadScene(
               FWObject&           oScene,
               CameraPath&         oCameraPath,
               CameraManager&      oView,
               AreaLightSource&    oLightSource)
{
    cudastd::log::out << "Reading configuration file...\n";

    SceneConfiguration sceneConfig = loadSceneConfiguration(CONFIGURATION);

    bool retval = true;

    if (sceneConfig.hasObjFileName)
    {
        cudastd::log::out << "Loading scene...\n";
        oScene.read(sceneConfig.objFileName);

        cudastd::log::out << "Number of primitives: " << oScene.getNumFaces() << "\n";

    }
    else
    {
        cudastd::log::out << "No scene specified in file "
            << CONFIGURATION << "\n";

        retval = false;
    }

    if (sceneConfig.hasCameraFileName)
    {
        cudastd::log::out << "Loading camera configuration...\n";
        oView.read(sceneConfig.cameraFileName);
    }
    else
    {
        cudastd::log::out << "No camera configuration specified in file "
            << CONFIGURATION << "\n";
    }

    if (sceneConfig.hasCameraPathFileName)
    {
        oCameraPath.read(sceneConfig.cameraPathFileName);
    }

    if(sceneConfig.hasLightsFileName)
    {
        LightSourceLoader loader;
        oLightSource = loader.loadFromFile(sceneConfig.lightsFileName);
    }

    return retval;
}


#endif // SCENELOADER_H_INCLUDED_4B232D1A_F513_4D82_9A41_3F7E4E6A6BFC
