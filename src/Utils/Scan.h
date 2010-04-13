/****************************************************************************/
/* Copyright (c) 2009, Stefan Popov, Javor Kalojanov
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

#ifndef PREFIXSUM_H_INCLUDED_5FEA3B3A_E2F1_49B2_9483_55F751905C38
#define PREFIXSUM_H_INCLUDED_5FEA3B3A_E2F1_49B2_9483_55F751905C38

#include "../CUDAStdAfx.h"
#include "CUDAUtil.h"


class ExclusiveScan
{
public:
    void operator()(
        uint*& aIn, 
        const uint aNumElements, 
        cudaStream_t aStream = 0u
        ) const;
};


class InclusiveScan
{
public:
    class OperatorPlus
    {
    public:
        OperatorPlus()
        {}

        DEVICE uint operator()(uint a, uint b)
        {
            return a + b;
        }
    };

    //aIn must point to device memory
    //currently limited to 31 million elements because of grid size restrictions
    void operator()(
        uint*& aIn, 
        const uint aNumElements,
        cudaStream_t aStream = 0u
        ) const;
};

#endif // PREFIXSUM_H_INCLUDED_5FEA3B3A_E2F1_49B2_9483_55F751905C38
