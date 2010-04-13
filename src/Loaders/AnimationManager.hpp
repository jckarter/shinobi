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

#ifndef ANIMATIONMANAGER_HPP_INCLUDED_63663C59_34F4_46CE_91C8_0AAC44521879
#define ANIMATIONMANAGER_HPP_INCLUDED_63663C59_34F4_46CE_91C8_0AAC44521879

#include "FWObject.hpp"

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


class AnimationManager
{
    size_t mNumKeyFrames;
    FWObject* mKeyFrames;
    float mCurrentFrameId; //interpolation coefficient
    float mStepSize;
public:

    AnimationManager(): mNumKeyFrames(0u), mKeyFrames(NULL),
        mCurrentFrameId(0.f), mStepSize(0.8f)
    {}

    ~AnimationManager()
    {
        if (mKeyFrames != NULL)
        {
            delete[] mKeyFrames;
        }
    }

    _GET_MEMBER(NumKeyFrames, size_t);
    _SET_MEMBER(NumKeyFrames, size_t);

    _GET_MEMBER(CurrentFrameId, float);
    _SET_MEMBER(CurrentFrameId, float);

    _GET_MEMBER(StepSize, float);
    _SET_MEMBER(StepSize, float);

    _GET_MEMBER(KeyFrames, FWObject*);
    _SET_MEMBER(KeyFrames, FWObject*);

    size_t getFrameId() const
    {
        return static_cast<size_t>(mCurrentFrameId);
    }

    size_t getNextFrameId() const
    {
        return 
            (static_cast<size_t>(mCurrentFrameId)
            + 1u
            + static_cast<size_t>(mStepSize)) % mNumKeyFrames;
    }

    FWObject& getFrame(size_t aFrameId)
    {
        return mKeyFrames[aFrameId];
    }

    


    float getInterpolationCoefficient() const
    {
        return mCurrentFrameId -
            static_cast<float>(static_cast<size_t>(mCurrentFrameId));
    }

    void nextFrame()
    {
        mCurrentFrameId += mStepSize;

        if (static_cast<size_t>(mCurrentFrameId) >= mNumKeyFrames)
        {
            mCurrentFrameId -=
                static_cast<float>(mNumKeyFrames);
        }
    }

    void allocateFrames(const size_t aSize)
    {
        if (mKeyFrames != NULL)
        {
            delete[] mKeyFrames;
        }

        mKeyFrames = new FWObject[aSize];
        mNumKeyFrames = aSize;

    }

    //frameFileName is aFileNamePrefix::frameIndex::aFileNameSuffix
    void read(const char* aFileNamePrefix,
        const char* aFileNameSuffix,
        size_t aNumFrames);
};

#undef _GET_MEMBER
#undef _SET_MEMBER

#endif // ANIMATIONMANAGER_HPP_INCLUDED_63663C59_34F4_46CE_91C8_0AAC44521879
