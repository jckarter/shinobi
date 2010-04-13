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

#include "StdAfx.hpp"
#include "SDLGLApplication.hpp"

#include "../Utils/ImagePNG.hpp"

//const char* SDLGLApplication::myGLSLFragmentProgam = "src/Application/FragmentShader.fx";
float gALPHA = 1.f;

GLuint SDLGLApplication::sGLSLProgram;
GLuint SDLGLApplication::sGLSLFragmentShader;
GLint  SDLGLApplication::sGLSLTextureParam;
GLint  SDLGLApplication::sGLSLAlphaParam;
GLuint SDLGLApplication::sFBTextureId;
GLuint SDLGLApplication::sFBOId;
float* SDLGLApplication::frameBufferFloatPtr;

const float MAXSTEPSIZE = 200.f;
const float MINSTEPSIZE = 0.01f;
const float ROTATESCALEFACTOR = 0.003f;
const float ROTATEUPSCALEFACTOR = 0.01f;

SDLGLApplication::~SDLGLApplication()
{
    cleanup();
    SDL_Quit();
}

void SDLGLApplication::init(int argc, char* argv[])
{
    //CUDA specific
    deviceInit(argc, argv);
    allocateHostBuffer();
    initScene();

    initVideo(gRESX, gRESY);
    initOpenGL();
    initGLSL();
    initFrameBufferTexture(&sFBTextureId, gRESX, gRESY);
    initViewPort();
}

void SDLGLApplication::allocateHostBuffer()
{
    frameBufferFloatPtr = new float[gRESX * gRESY * 3];
}

void SDLGLApplication::initVideo(int aWidth, int aHeight) {
    // Load SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr <<  "Unable to initialize SDL: " << SDL_GetError() << "\n";
        return;
    }

    //To use OpenGL, you need to get some information first,
    const SDL_VideoInfo *info = SDL_GetVideoInfo();
    if(!info) {
        /* This should never happen, if it does PANIC! */
        std::cerr << "Video query failed: " << SDL_GetError() << "\n";
        return;
    }

    int bpp = info->vfmt->BitsPerPixel;

    // set bits for red: (5 = 5bits for red channel)
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    // set bits for green:
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    // set bits for blue:
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    // colour depth:
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 8);
    // You want it double buffered?
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, true);

    //SDL_SetVideoMode(w, h, 16, SDL_OPENGL | SDL_RESIZABLE);

    //screen is no longer used, as openGL does all the drawing now!
    if (SDL_SetVideoMode(aWidth, aHeight, bpp, SDL_OPENGL | SDL_SWSURFACE | SDL_RESIZABLE) == 0) {
        std::cerr << "Unable to set video mode: " << SDL_GetError() << "\n";
        return;
    }

    SDL_WM_SetCaption(mActiveWindowName, "");
}


void SDLGLApplication::displayFrame()
{
    if (mMinimized)
    {
        SDL_Delay(1500);
    } 
    else
    {
        float time, renderTime, buildTime;
        generateFrame(renderTime, buildTime);
        time = renderTime + buildTime;
        //display frame rate in window title
        std::string windowName(mActiveWindowName);
        windowName += ftoa(1000.f / time);
        windowName += " (render: ";
        windowName += ftoa(renderTime);
        windowName += " build: ";
        windowName += ftoa(buildTime);
        windowName += ") spp: ";
        windowName += itoa(mNumImages);


        SDL_WM_SetCaption(windowName.c_str(), "");

        runGLSLShader();
    }
}

void SDLGLApplication::WindowActive	()
{
    SDL_WM_SetCaption(mActiveWindowName, "");
}

void SDLGLApplication::WindowInactive	() 
{
    SDL_WM_SetCaption(mMinimizedWindowName, "");
}

void SDLGLApplication::writeScreenShot()
{

    Image img(gRESX, gRESY);

    for(int y = 0; y < gRESY; ++y)
    {
        for(int x = 0; x < gRESX; ++x)
        {
            img(x,y).x = frameBufferFloatPtr[3 * (x + gRESX * y)    ];
            img(x,y).y = frameBufferFloatPtr[3 * (x + gRESX * y) + 1];
            img(x,y).z = frameBufferFloatPtr[3 * (x + gRESX * y) + 2];
        }
    }

    img.gammaCorrect(2.2f);
    img.writePNG(OUTPUT);
    std::cout << "Wrote " << OUTPUT << "\n";
}

void SDLGLApplication::outputCameraParameters()
{
    std::cerr << "position\t\t" 
        << mCamera.getPosition().x << "\t"
        << mCamera.getPosition().y << "\t"
        << mCamera.getPosition().z << "\n"
        << "orientation\t" 
        << mCamera.getOrientation().x << "\t"
        << mCamera.getOrientation().y << "\t"
        << mCamera.getOrientation().z << "\n"
        << "up\t\t\t"
        << mCamera.getUp().x << "\t"
        << mCamera.getUp().y << "\t"
        << mCamera.getUp().z << "\n"
        << "rotation\t\t"
        << mCamera.getRotation().x << "\t"
        << mCamera.getRotation().y << "\t"
        << mCamera.getRotation().z << "\n"
        << "resX\t\t\t" << mCamera.getResX() << "\n"
        << "resY\t\t\t" << mCamera.getResY() << "\n"
        << "FOV\t\t\t"
        << mCamera.getFOV() << "\n";

}

void SDLGLApplication::resetCamera()
{
    mCamera = mInitialCamera;
    cameraChanged();
}

void SDLGLApplication::cameraChanged()
{
    mNumImages = 0;
}

void SDLGLApplication::KeyUp		(const int& iKeyEnum)
{
    switch(iKeyEnum) 
    {
    case SDLK_UP:
    case SDLK_w:
        mMoveForward = false;
        break;
    case SDLK_DOWN:
    case SDLK_s:
        mMoveBackward = false;
        break;
    case SDLK_LEFT:
    case SDLK_a:
        mMoveLeft = false;
        break;
    case SDLK_RIGHT:
    case SDLK_d:
        mMoveRight = false;
        break;
    case SDLK_q:
        mMoveUp = false;
        break;
    case SDLK_z:
        mMoveDown = false;
        break;
    case SDLK_t:
        writeScreenShot();
        break;
    case SDLK_c:
        outputCameraParameters();
        break;
    case SDLK_r:
        getResolution();
        break;
    case SDLK_m:
        nextRenderMode();
        break;
    case SDLK_RCTRL:
    case SDLK_LCTRL:
        mCtrlDown = false;
        break;
    case SDLK_SPACE:
        resetCamera();
        break;
    case SDLK_F1:
        printHelp();
        break;
    default:
        break;
    }
}

void SDLGLApplication::KeyDown		(const int& iKeyEnum)
{
    switch(iKeyEnum) 
    {
    case SDLK_UP:
    case SDLK_w:
        mMoveForward = true;
        break;
    case SDLK_DOWN:
    case SDLK_s:
        mMoveBackward = true;
        break;
    case SDLK_LEFT:
    case SDLK_a:
        mMoveLeft = true;
        break;
    case SDLK_RIGHT:
    case SDLK_d:
        mMoveRight = true;
        break;
    case SDLK_q:
        mMoveUp = true;
        break;
    case SDLK_z:
        mMoveDown = true;
        break;
    case SDLK_t:
        break;
    case SDLK_c:
        break;
    case SDLK_r:
        break;
    case SDLK_m:
        break;
    case SDLK_RCTRL:
    case SDLK_LCTRL:
        mCtrlDown = true;
        break;
    default:
        break;
    }
}

void SDLGLApplication::MouseMoved		(const int& iButton,
                                         const int& iX, 
                                         const int& iY, 
                                         const int& iRelX, 
                                         const int& iRelY)
{
    if(mUpdateMouseCoords)
    {
        mMouseX = iX;
        mMouseY = iY;

        mUpdateMouseCoords = false;
    }
    else
    {
        mMouseX += iRelX;
        mMouseY += iRelY;
    }

    if (mHasMouse)
    {
        if(mCtrlDown)
        {
            mUpOrientationAngleChange +=
                static_cast<float>(iRelX) * ROTATEUPSCALEFACTOR;
        }
        else
        {
            mVerticalRotationAngle +=
                static_cast<float>(iRelY) * ROTATESCALEFACTOR;

            mHorizontalRotationAngle +=
                static_cast<float>(iRelX) * ROTATESCALEFACTOR;
        }
    }
}

void SDLGLApplication::MouseButtonUp	(const int& iButton, 
                                         const int& iX, 
                                         const int& iY, 
                                         const int& iRelX, 
                                         const int& iRelY) 
{
    switch(iButton)
    {
    case SDL_BUTTON_LEFT:
        if (mHasMouse)
        {
            mHasMouse = false;
            releaseMouse();
        }
        else
        {
            mHasMouse = true;
            mMouseX = iX;
            mMouseY = iY;
            grabMouse();
        }
        break;
    case SDL_BUTTON_WHEELUP:
        mMoveStep = std::min(mMoveStep * 2.f, MAXSTEPSIZE);
        break;
    case SDL_BUTTON_WHEELDOWN:
        mMoveStep = std::max(mMoveStep * 0.5f, MINSTEPSIZE);
        break;
    default:
        break;
    }
}

void SDLGLApplication::grabMouse()
{
    SDL_ShowCursor(0);
    SDL_WM_GrabInput(SDL_GRAB_ON);
    mUpdateMouseCoords = true;

}

void SDLGLApplication::releaseMouse()
{
    SDL_WM_GrabInput(SDL_GRAB_OFF);
    SDL_ShowCursor(1);
    mUpdateMouseCoords = true;

}


void SDLGLApplication::fetchEvents()
{
    // Poll for events, and handle the ones we care about.
    SDL_Event event;
    while ( SDL_PollEvent( &event ) ) 
    {
        switch ( event.type ) 
        {
        case SDL_KEYDOWN:
            // If escape is pressed set the Quit-flag
            if (event.key.keysym.sym == SDLK_ESCAPE)
            {
                mQuit = true;
                break;
            }

            KeyDown( event.key.keysym.sym );
            break;

        case SDL_KEYUP:
            KeyUp( event.key.keysym.sym );
            break;

        case SDL_QUIT:
            mQuit = true;
            break;

        case SDL_MOUSEMOTION:
            MouseMoved(
                event.button.button, 
                event.motion.x, 
                event.motion.y, 
                event.motion.xrel, 
                event.motion.yrel);
            break;

        case SDL_MOUSEBUTTONDOWN:
            break;

        case SDL_MOUSEBUTTONUP:
            MouseButtonUp(
                event.button.button, 
                event.motion.x, 
                event.motion.y, 
                event.motion.xrel, 
                event.motion.yrel);
            break;

        case SDL_ACTIVEEVENT:
            if ( event.active.state & SDL_APPACTIVE ) {
                if ( event.active.gain ) {
                    mMinimized = false;
                    WindowActive();
                } else {
                    mMinimized = true;
                    WindowInactive();
                }
            }
            break;

        case SDL_VIDEORESIZE:
            SDL_ResizeEvent* resizeEvent =
                reinterpret_cast<SDL_ResizeEvent*>(&event);
            gRESX = std::max(resizeEvent->w, 128);
            gRESY = std::max(resizeEvent->h, 128);
            changeWindowSize();
            cameraChanged();
            break;
        } // switch
    } // while (handling input)


    processEvents();

}

void SDLGLApplication::processEvents()
{
    bool cameraChangedFlag = mMoveLeft || mMoveRight || mMoveForward ||
        mMoveBackward || mMoveUp || mMoveDown || (mVerticalRotationAngle != 0.f)
        || (mHorizontalRotationAngle != 0.f) ||
        (mUpOrientationAngleChange != 0.f);

    if (cameraChangedFlag)
    {
        cameraChanged();
    }

    float moveLeftAmount     = mMoveLeft     ? mMoveStep : 0.f;
    float moveRightAmount    = mMoveRight    ? mMoveStep : 0.f;
    float moveForwardAmount  = mMoveForward  ? mMoveStep : 0.f;
    float moveBackwardAmount = mMoveBackward ? mMoveStep : 0.f;
    float moveUpAmount       = mMoveUp       ? mMoveStep : 0.f;
    float moveDownAmount     = mMoveDown     ? mMoveStep : 0.f;

    mCamera.moveRight(-moveLeftAmount);
    mCamera.moveRight(moveRightAmount);
    mCamera.moveForward(-moveBackwardAmount);
    mCamera.moveForward(moveForwardAmount);
    mCamera.moveUp(-moveDownAmount);
    mCamera.moveUp(moveUpAmount);

    // rotate direction vector vertically
    const vec3f directionRotatedUp =
        ~(cosf(mVerticalRotationAngle) * mCamera.getOrientation() -
        sinf(mVerticalRotationAngle) * mCamera.getUp());

    // rotate up vector vertically
    const vec3f upRotatedUp =
        ~(directionRotatedUp.cross(
        mCamera.getUp().cross(directionRotatedUp)));

    const vec3f finalDirection = mCamera.rotateVector(
        directionRotatedUp, mInitialCamera.getUp(), -mHorizontalRotationAngle);

    const vec3f finalUp = mCamera.rotateVector(
        upRotatedUp, mInitialCamera.getUp(), -mHorizontalRotationAngle);

    const vec3f finalUp2 = mCamera.rotateVector(
        finalUp, finalDirection, mUpOrientationAngleChange);

    mCamera.setOrientation(finalDirection);
    mCamera.setUp(finalUp2);
    mCamera.setRight(~(finalDirection % finalUp2));

    mVerticalRotationAngle = mHorizontalRotationAngle 
        = mUpOrientationAngleChange = 0.f;

}

void SDLGLApplication::printHelp()
{
    std::cerr << "----------------------------------------------------------\n";
    std::cerr << "Controls                                                  \n";
    std::cerr << "----------------------------------------------------------\n";
    std::cerr << "Start Camera Rotation:\n";
    std::cerr << "    Left click with the mouse inside the window.\n\n";
    std::cerr << "Move Left: a\n"; 
    std::cerr << "Move right: d\n";
    std::cerr << "Move forward: w\n";
    std::cerr << "Move back: s\n";
    std::cerr << "Move up: q\n";
    std::cerr << "Move down: z\n\n";
    std::cerr << "Adjust movement step size: Mouse scroll\n\n";
    std::cerr << "Reset Camera: Space\n\n";
    std::cerr << "Output camera parameters: c\n\n";
    std::cerr << "Set window width and height: r\n\n";
    std::cerr << "Change render mode: m\n\n";
    std::cerr << "Write screen-shot in output/output.png: t\n";

}

void SDLGLApplication::getResolution()
{
    std::cout << "Window width:  ";
    std::cin >> gRESX;
    gRESX = std::max(gRESX, 128);
    std::cout << "Window height: ";
    std::cin >> gRESY;
    gRESY = std::max(gRESY, 128);

    
    changeWindowSize();
}

void SDLGLApplication::changeWindowSize(void)
{
    //host memory
    delete[] frameBufferFloatPtr;
    //texture
    glDeleteTextures(1, &sFBTextureId);
    //shader
    glDetachShader(sGLSLProgram, sGLSLFragmentShader);
    glDeleteShader(sGLSLFragmentShader); 
    glDeleteProgram(sGLSLProgram);

    allocateHostBuffer();

    changeCUDAWindowSize(gRESX, gRESY);

    cameraChanged();

    initVideo(gRESX, gRESY);
    initOpenGL();
    initGLSL();
    initFrameBufferTexture(&sFBTextureId, gRESX, gRESY);
    initViewPort();

}

void SDLGLApplication::nextRenderMode()
{
    switch ( mRenderMode ) 
    {
    case SINGLEKERNEL:
        mRenderMode = MULTIKERNEL;
        break;
    case MULTIKERNEL:
        mRenderMode = SINGLEKERNEL;
        break;
    default:
        break;
    }//switch ( mRenderMode )

    cameraChanged();
}
//////////////////////////////////////////////////////////////////////////
//
//OpenGL specific
//
//////////////////////////////////////////////////////////////////////////

void SDLGLApplication::initOpenGL()
{
    //////////////////////////////////////////////////////////////////////////
    //Various checks
    //////////////////////////////////////////////////////////////////////////
    //Important: call this AFTER an OpenGL context is created
    //std::cerr << "Testing GLEW extensions... ";

    // init GLEW, obtain function pointers
    int err = glewInit();
    // Warning: This does not check if all extensions used 
    // in a given implementation are actually supported. 
    // Function entry points created by glewInit() will be 
    // NULL in that case!
    if (GLEW_OK != err)
    {   
        std::cerr << std::endl << (char*)glewGetErrorString(err) << std::endl;   
        return;
    }

    if (!glewIsSupported("GL_VERSION_2_0"))
    {
        std::cerr << "\nOpenGL 2.0 not supported\n";
        return;
    }

    //std::cerr << "done\n";
    //////////////////////////////////////////////////////////////////////////
    //std::cerr << "Testing Textures... ";

    int maxtexsize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxtexsize);
    if (maxtexsize < gRESX || maxtexsize < gRESY)
    {
        std::cerr << "GL_MAX_TEXTURE_SIZE " << " is " << maxtexsize 
            << std::endl;
        std::cerr << "Required sizes in X and Y are "<< gRESX << " "
            << gRESY << std::endl;

    }

    //std::cerr << "done\n";
}

void SDLGLApplication::initViewPort(void)
{
    // viewport for 1:1 pixel=texel=geometry mapping
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, gRESX, 0.0, gRESY);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, gRESX, gRESY);
}

void SDLGLApplication::initFrameBufferTexture(GLuint *aTextureId, const int aResX, const int aResY)
{   
    // create a new texture name
    glGenTextures (1, aTextureId);
    // bind the texture name to a texture target
    glBindTexture(TEXTURE_TARGET, *aTextureId);
    // turn off filtering and set proper wrap mode 
    // (obligatory for float textures atm)
    glTexParameteri(TEXTURE_TARGET, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(TEXTURE_TARGET, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(TEXTURE_TARGET, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(TEXTURE_TARGET, GL_TEXTURE_WRAP_T, GL_CLAMP);
    // set texenv to replace instead of the default modulate
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    // and allocate graphics memory

    glTexImage2D(TEXTURE_TARGET, 
        0, //not to use any mipmap levels for this texture
        INTERNAL_FORMAT,
        aResX,
        aResY,
        0, //turns off borders
        TEXTURE_FORMAT,
        GL_FLOAT,
        0);
}


//////////////////////////////////////////////////////////////////////////
//GLSL
//////////////////////////////////////////////////////////////////////////
void SDLGLApplication::initGLSL(void)
{
    // create program object
    sGLSLProgram = glCreateProgram();
    // create shader object (fragment shader)
    sGLSLFragmentShader = glCreateShader(GL_FRAGMENT_SHADER_ARB);
    // set source for shader
    const char* source = getGLSLProgram();
    glShaderSource(sGLSLFragmentShader, 1, &source, NULL);
    // compile shader
    glCompileShader(sGLSLFragmentShader);
    // check for errors
    printShaderInfoLog(sGLSLFragmentShader);
    // attach shader to program
    glAttachShader (sGLSLProgram, sGLSLFragmentShader);
    // link into full program, use fixed function vertex pipeline
    glLinkProgram(sGLSLProgram);
    // check for errors
    printProgramInfoLog(sGLSLProgram);
    // Get location of the texture samplers for future use
    sGLSLTextureParam = glGetUniformLocation(sGLSLProgram, "uTexture");
    sGLSLAlphaParam = glGetUniformLocation(sGLSLProgram, "uAlpha");

}

void SDLGLApplication::runGLSLShader(void)
{
    // enable GLSL program
    glUseProgram(sGLSLProgram);
    glEnable(TEXTURE_TARGET);

    // enable texture x (read-only, not changed in the computation loop)
    glActiveTexture(GL_TEXTURE0);	
    glBindTexture(TEXTURE_TARGET, sFBTextureId);
    glUniform1i(sGLSLTextureParam, 0); // texture unit 0
    // enable scalar alpha (same)
    glUniform1f(sGLSLAlphaParam, gALPHA);
    //timing purposes only
    //glFinish();

    //////////////////////////////////////////////////////////////////////////
    //NVidia
    glTexSubImage2D(TEXTURE_TARGET,0,0,0,gRESX,gRESY,
        TEXTURE_FORMAT, GL_FLOAT, frameBufferFloatPtr);
    //////////////////////////////////////////////////////////////////////////

    // make quad filled to hit every pixel/texel
    glPolygonMode(GL_FRONT,GL_FILL);
    // and render quad
    //Note: texture coordinates are flipped in Y
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, static_cast<float>(gRESY));
    glVertex2f(0.0, 0.0);
    glTexCoord2f(static_cast<float>(gRESX), static_cast<float>(gRESY));
    glVertex2f(static_cast<float>(gRESX), 0.0);
    glTexCoord2f(static_cast<float>(gRESX), 0.0); 
    glVertex2f(static_cast<float>(gRESX), static_cast<float>(gRESY));
    glTexCoord2f(0.0, 0.0); 
    glVertex2f(0.0, static_cast<float>(gRESY));
    glEnd();

    //un-bind texture
    glBindTexture(TEXTURE_TARGET, 0);

    glDisable(TEXTURE_TARGET);

    SDL_GL_SwapBuffers();
}

//////////////////////////////////////////////////////////////////////////
//error checking for GLSL
//////////////////////////////////////////////////////////////////////////
void SDLGLApplication::printProgramInfoLog(GLuint obj)
{
    int infologLength = 0;
    int charsWritten  = 0;
    char *infoLog;
    glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infologLength);
    if (infologLength > 1)
    {
        infoLog = (char *)malloc(infologLength);
        glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
        std::cerr << infoLog << std::endl;
        free(infoLog);
    }
}

void SDLGLApplication::printShaderInfoLog(GLuint obj)
{
    int infologLength = 0;
    int charsWritten  = 0;
    char *infoLog;
    glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);
    if (infologLength > 1) 
    {
        infoLog = (char *)malloc(infologLength);
        glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
        std::cerr << infoLog << std::endl;
        free(infoLog);
    }
}

const char* SDLGLApplication::getGLSLProgram()
{
    return
        "                                                                      \
        #extension GL_ARB_texture_rectangle : enable                     \n    \
        \
        uniform sampler2DRect uTexture;                                        \
        \
        vec4 gammaCorrection(float aGamma)                                     \
        {                                                                      \
        float gammaRecip = 1.0 / aGamma;                                       \
        return pow(texture2DRect(uTexture, gl_TexCoord[0].xy),                 \
        vec4(gammaRecip, gammaRecip, gammaRecip, 1.0));                        \
        }                                                                      \
        \
        void main()                                                            \
        {                                                                      \
        gl_FragColor = gammaCorrection(2.2);                                   \
        }                                                                      \
        ";
}

void SDLGLApplication::cleanup(void)
{
    //host memory
    delete[] frameBufferFloatPtr;
    //texture
    glDeleteTextures(1, &sFBTextureId);
    //shader
    glDetachShader(sGLSLProgram, sGLSLFragmentShader);
    glDeleteShader(sGLSLFragmentShader); 
    glDeleteProgram(sGLSLProgram);

    cleanupCUDA();
}
