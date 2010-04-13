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

#include "Sort.h"
#include "Utils/chag/pp/sort.cuh"


void Sort::operator()(uint2 *pData0, 
                      uint2 *pData1,
                      uint aNumElements,
                      uint aNumBits, 
                      cudaStream_t aStream
                      ) const
{
    if (aNumBits <= 16)
    {
        chag::pp::sort<0u, 16u, BitUnsetForPairs, uint2>
            (pData0, (pData0 + aNumElements), pData1, pData0, aStream);
    }
    else if(aNumBits <= 20)
    {
        chag::pp::sort<0u, 20u, BitUnsetForPairs, uint2>
            (pData0, (pData0 + aNumElements), pData1, pData0, aStream);
    }
    else if(aNumBits <= 24)
    {
        chag::pp::sort<0u, 24u, BitUnsetForPairs, uint2>
            (pData0, (pData0 + aNumElements), pData1, pData0, aStream);
    }
    else
    {
        chag::pp::sort<0u, 32u, BitUnsetForPairs, uint2>
            (pData0, (pData0 + aNumElements), pData1, pData0, aStream);
    }

}
