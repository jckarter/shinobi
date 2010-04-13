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

#ifndef SCENECONFIGURATION_HPP_INCLUDED_453E4C50_F59A_476A_B8CB_EA50C10655E5
#define SCENECONFIGURATION_HPP_INCLUDED_453E4C50_F59A_476A_B8CB_EA50C10655E5

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


#define _HAS_STRING_MEMBER_DECL(aName)					                           \
    bool hasString_##aName () const			                                       

struct SceneConfiguration
{
    const char* objFileName;
    const char* cameraFileName;
    const char* cameraPathFileName;
    const char* lightsFileName;
    const char* frameFileNamePrefix;
    const char* frameFileNameSuffix;
    int   numFrames;
    float frameStepSize;

    bool hasObjFileName;
    bool hasCameraFileName;
    bool hasCameraPathFileName;
    bool hasLightsFileName;
    bool hasFrameFileNamePrefix;
    bool hasFrameFileNameSuffix;

};

extern "C"
{
    SceneConfiguration loadSceneConfiguration(const char* aFileName);
};

#undef _GET_MEMBER
#undef _SET_MEMBER
#undef _HAS_STRING_MEMBER_DECL

#endif // SCENECONFIGURATION_HPP_INCLUDED_453E4C50_F59A_476A_B8CB_EA50C10655E5
