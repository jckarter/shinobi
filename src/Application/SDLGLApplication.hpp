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

#ifndef SDLGLAPPLICATION_HPP_INCLUDED_2AF01D18_08BE_4F7C_987F_80AD91626F7F
#define SDLGLAPPLICATION_HPP_INCLUDED_2AF01D18_08BE_4F7C_987F_80AD91626F7F

#include <GL/glew.h>

//////////////////////////////////////////////////////////////////////////
//CUDA specific includes (host code only)
//////////////////////////////////////////////////////////////////////////
#include "../CUDAConfig.h"
#include "../Loaders/CameraManager.hpp"

extern float gALPHA;

class SDLGLApplication
{
    //////////////////////////////////////////////////////////////////////////
    //Window
    int mMouseX;
    int mMouseY;

    bool mMinimized, mQuit;
    bool mUpdateMouseCoords;
    bool mHasMouse, mCtrlDown;

    //////////////////////////////////////////////////////////////////////////
    //Camera
    float mMoveStep;
    bool mMoveLeft;
    bool mMoveRight;
    bool mMoveForward;
    bool mMoveBackward;
    bool mMoveUp;
    bool mMoveDown;
    float mVerticalRotationAngle, mHorizontalRotationAngle;
    float mUpOrientationAngleChange;
    CameraManager mInitialCamera;
    CameraManager mCamera;
    //////////////////////////////////////////////////////////////////////////
    //Misc
    uint mNumImages;
    enum RenderMode {SINGLEKERNEL = 0, MULTIKERNEL = 1};
    RenderMode mRenderMode;
    //////////////////////////////////////////////////////////////////////////
    //IO
    const char* mMinimizedWindowName;
    const char* mActiveWindowName;
public:

    //////////////////////////////////////////////////////////////////////////
    // GLSL vars
    //static const char* myGLSLFragmentProgam;
    static GLuint sGLSLProgram;
    static GLuint sGLSLFragmentShader;
    static GLint sGLSLTextureParam;
    static GLint sGLSLAlphaParam;
    //////////////////////////////////////////////////////////////////////////
    //Textures & related
    //GL_TEXTURE_RECTANGLE_ARB allows non-normalized texture coordinates
    static const int TEXTURE_TARGET = GL_TEXTURE_RECTANGLE_ARB;//GL_TEXTURE_2D;
    static const int INTERNAL_FORMAT = GL_RGB32F_ARB;//GL_LUMINANCE32F_ARB;
    static const int TEXTURE_FORMAT = GL_RGB;//GL_LUMINANCE;
    static float* frameBufferFloatPtr;
    static GLuint sFBTextureId;
    static GLuint sFBOId;
    //////////////////////////////////////////////////////////////////////////

    SDLGLApplication():mMinimized(false), mQuit(false), mUpdateMouseCoords(false),
        mHasMouse(false), mCtrlDown(false),
        mMoveStep(2.f),
        mMoveLeft(false), mMoveRight(false), mMoveForward(false),
        mMoveBackward(false), mMoveUp(false), mMoveDown(false),
        mVerticalRotationAngle(0.f), mHorizontalRotationAngle(0.f),
        mUpOrientationAngleChange(0.f), mNumImages(0), mRenderMode(SINGLEKERNEL),
        mMinimizedWindowName("Shinobi"), mActiveWindowName("Fps: ")
    {}

    ~SDLGLApplication();

    void init(int argc, char* argv[]);

    bool dead()
    {
        return mQuit;
    }

    //////////////////////////////////////////////////////////////////////////
    //
    //CUDA specific
    //
    //////////////////////////////////////////////////////////////////////////
    void deviceInit(int argc, char* argv[]);

    void initScene();

    void changeCUDAWindowSize(const int aResX, const int aResY);

    void allocateHostBuffer();

    void generateFrame(float& oRenderTime, float& oBuildTime);

    //////////////////////////////////////////////////////////////////////////
    //
    //SDL specific
    //
    //////////////////////////////////////////////////////////////////////////

    void initVideo(int aWidth, int aHeight);

    void initOpenGL();

    void displayFrame();

    /** Window is active again. **/
    void WindowActive();

    /** Window is inactive. **/
    void WindowInactive	();

    void writeScreenShot();

    void outputCameraParameters();

    /** Keyboard key has been released.
    @param iKeyEnum The key number.
    **/
    void KeyUp		(const int& iKeyEnum);

    /** Keyboard key has been pressed.
    @param iKeyEnum The key number.
    **/
    void KeyDown		(const int& iKeyEnum);


    /** The mouse has been moved.
    @param iButton	Specifies if a mouse button is pressed.
    @param iX	The mouse position on the X-axis in pixels.
    @param iY	The mouse position on the Y-axis in pixels.
    @param iRelX	The mouse position on the X-axis relative to the last position, in pixels.
    @param iRelY	The mouse position on the Y-axis relative to the last position, in pixels.

    @bug The iButton variable is always NULL.
    **/
    void MouseMoved		(const int& iButton, const int& iX, const int& iY, const int& iRelX, const int& iRelY);

    /** A mouse button has been released.
    @param iButton	Specifies if a mouse button is pressed.
    @param iX	The mouse position on the X-axis in pixels.
    @param iY	The mouse position on the Y-axis in pixels.
    @param iRelX	The mouse position on the X-axis relative to the last position, in pixels.
    @param iRelY	The mouse position on the Y-axis relative to the last position, in pixels.
    **/

    void MouseButtonUp	(const int& iButton,  const int& iX, const int& iY, const int& iRelX, const int& iRelY);

    void grabMouse();

    void releaseMouse();

    void fetchEvents();

    void processEvents();

    void resetCamera();

    void cameraChanged();

    void printHelp();

    void getResolution();

    void changeWindowSize();

    void nextRenderMode();

    //////////////////////////////////////////////////////////////////////////
    //
    //OpenGL specific
    //
    //////////////////////////////////////////////////////////////////////////

    static void initFrameBufferTexture(GLuint *aTextureId, const int aResX, const int aResY);

    void initViewPort(void);

    //////////////////////////////////////////////////////////////////////////
    //GLSL
    //////////////////////////////////////////////////////////////////////////

    static void printProgramInfoLog(GLuint obj);
    static void printShaderInfoLog(GLuint obj);

    static void initGLSL(void);

    void runGLSLShader(void);

    static const char* getGLSLProgram();

    //////////////////////////////////////////////////////////////////////////

    void cleanup(void);
    void cleanupCUDA(void);
};

#endif // SDLGLAPPLICATION_HPP_INCLUDED_2AF01D18_08BE_4F7C_987F_80AD91626F7F
