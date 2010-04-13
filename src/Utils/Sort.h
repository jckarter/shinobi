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

#ifndef SORT_H_INCLUDED_F2BF97B8_CD57_47F1_BFAD_26D1687E3917
#define SORT_H_INCLUDED_F2BF97B8_CD57_47F1_BFAD_26D1687E3917

#include "../CUDAStdAfx.h"

class Sort
{
public:
    template< unsigned taBit, typename T >
    struct BitUnsetForPairs
    {
        HOST DEVICE bool operator () ( T aPair ) const
        {
            return !(aPair.x & (1u << taBit));
        }
    };

    void operator()(uint2 *pData0, uint2 *pData1, uint aNumElements,
        uint aNumBits, cudaStream_t aStream = 0
        ) const;
};


#endif // SORT_H_INCLUDED_F2BF97B8_CD57_47F1_BFAD_26D1687E3917
