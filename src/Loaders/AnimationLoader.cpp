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
#include "AnimationManager.hpp"

void AnimationManager::read(const char* aFileNamePrefix,
          const char* aFileNameSuffix,
          size_t aNumFrames)
{
    const std::string fileNamePrefix = aFileNamePrefix;
    const std::string fileNameSuffix = aFileNameSuffix;

    int numDigits = 0;
    int numFrames = static_cast<int>(aNumFrames) - 1;

    for(; numFrames > 0; numFrames /= 10, ++numDigits)
    {}

    if (aNumFrames > 0u)
    {
        allocateFrames(aNumFrames);

        for(size_t frameId = 0; frameId < aNumFrames; ++frameId)
        {
            std::string frameIdStr;
            int currNumDigits = 1;
            int hlpFrameId = static_cast<int>(frameId);
            for(; hlpFrameId > 9; hlpFrameId /= 10, ++currNumDigits)
            {}

            for (; currNumDigits < numDigits; ++currNumDigits)
            {
                frameIdStr.append(std::string("0"));
            }

            frameIdStr.append(itoa(static_cast<int>(frameId)));

            const std::string fileName =
                fileNamePrefix +
                frameIdStr +
                fileNameSuffix;

            mKeyFrames[frameId].read(fileName.c_str());
        }
    }
}

