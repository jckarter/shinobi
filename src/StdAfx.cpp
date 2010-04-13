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

std::string itoa(const int a)
{
    std::stringstream ss;
    ss<<a;
    return ss.str();
}

std::string ftoa(const float a)
{
    std::stringstream ss;
    ss.precision(2);
    ss.setf(std::ios::fixed,std::ios::floatfield);
    ss<<a;
    return ss.str();
}

std::string cutComments(const std::string& aLine,
                        const char* aToken)
{
    std::string token(aToken);
    const size_t commentStart = aLine.find(token, 0);

    if (commentStart == std::string::npos)
    {
        //line does not contain comment token
        const size_t lineBegin = aLine.find_first_not_of(" \t", 0);
        if (lineBegin < aLine.size())
        {
            return aLine.substr(lineBegin, aLine.size());
        }
        else
        {
            //line contains only whitespace
            return std::string("");
        }
    }
    else
    {
        //line contains comment token
        const size_t lineBegin = aLine.find_first_not_of(" \t", 0);
        return aLine.substr(lineBegin, commentStart - lineBegin);
    }
}
