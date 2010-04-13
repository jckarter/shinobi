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

#ifndef CAMERALOADER_HPP_INCLUDED_7BFAE673_CC91_4440_9070_7499B9F683A5
#define CAMERALOADER_HPP_INCLUDED_7BFAE673_CC91_4440_9070_7499B9F683A5

#include "../Core/Algebra.hpp"

#define _SET_MEMBER(aName, aType)					                           \
    void set##aName (aType aValue)	                                           \
    {												                           \
    m##aName = aValue;								                           \
    }

#define _GET_MEMBER(aName, aType)					                           \
    aType get##aName () const			                                       \
    {												                           \
    return m##aName ;								                           \
    }

class CameraManager
{
    vec3f mPosition, mOrientation, mUp, mRotation, mRight;
    uint mResX, mResY;
    float mFOV;
public:
    CameraManager(const vec3f& aPosition = vec3f::rep(0.f),
         const vec3f& aOrientation = vec3f::rep(0.f),
         const vec3f& aUp = vec3f::rep(0.f),
         const vec3f& aRotation = vec3f::rep(0.f),
         uint aResX = 512u,
         uint aResY = 512u,
         float aFOV = 66.f):
    mPosition(aPosition),
        mOrientation(aOrientation),
        mUp(aUp),
        mRotation(aRotation),
        mRight(~(aOrientation % aUp)),
        mResX(aResX),
        mResY(aResY),
        mFOV(aFOV)
    {
        //Default camera orientation
        mOrientation.z = 1.f;
        mUp.y = -1.f;
        mRight.x = 1.f;
    }

    _GET_MEMBER(Position, vec3f);
    _GET_MEMBER(Orientation, vec3f);
    _GET_MEMBER(Up, vec3f);
    _GET_MEMBER(Right, vec3f);
    _GET_MEMBER(Rotation, vec3f);
    _GET_MEMBER(ResX, uint);
    _GET_MEMBER(ResY, uint);
    _GET_MEMBER(FOV, float);

    _SET_MEMBER(Position, const vec3f&);
    _SET_MEMBER(Orientation, const vec3f&);
    _SET_MEMBER(Up, const vec3f&);
    _SET_MEMBER(Right, const vec3f&);
    _SET_MEMBER(Rotation, const vec3f&);
    _SET_MEMBER(ResX, uint);
    _SET_MEMBER(ResY, uint);
    _SET_MEMBER(FOV, float);

    //Performs rotation around all three axes with the argument angles (in degrees)
    void rotate(const vec3f&);
    //rotate vector around axis with argument angle (in radians)
    static vec3f rotateVector( const vec3f& aVec , const vec3f& aAxis, const float aAngle);
    void moveUp(const float);
    void moveRight(const float);
    void moveForward(const float);

    void read(const char*);
    void write(const char*) const;
};

#undef _GET_MEMBER
#undef _SET_MEMBER

#endif // CAMERALOADER_HPP_INCLUDED_7BFAE673_CC91_4440_9070_7499B9F683A5
