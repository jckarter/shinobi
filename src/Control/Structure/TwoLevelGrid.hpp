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

#ifndef TWOLEVELGRID_HPP_INCLUDED_1684EB4C_F1F8_4A10_8415_2D6EE02CE0D7
#define TWOLEVELGRID_HPP_INCLUDED_1684EB4C_F1F8_4A10_8415_2D6EE02CE0D7

#include "../../CUDAStdAfx.h"
#include "Grid.hpp"

struct TwoLevelGridCell
{
    typedef uint2                   t_data;

    //////////////////////////////////////////////////////////////////////////
    //8 bytes:
    //1st, 2nd, 3rd and 4th - begin of leaf array
    //5th                   - dummy | isEmpty flag | isValid flag | isLeaf flag
    //6th, 7th and 8th      - resolutions in z, y and x
    //////////////////////////////////////////////////////////////////////////
    t_data  data;

    TwoLevelGridCell()
    {
        data.x = 0u;
        data.y = 0u;
    }

    TwoLevelGridCell(t_data aData): data(aData)
    {}

    enum{
        SHIFTX              =   0,
        SHIFTY              =   8,
        SHIFTZ              =   16,
        SHIFTW              =   24,
        VALIDMASK           =   0x20,
        NOTEMPTYMASK        =   0x40,
        LEAFMASK            =   0x80,
        MASK                =   0xFF,
        LARGELEAFMASK       =   0x10000000,
        LARGEVALIDMASK      =   0x20000000,
        LARGENOTEMPTYMASK   =   0x40000000,
        LARGENOTEMPTYMASKNEG=   0x3FFFFFFF,
        LARGEVALIDMASKNEG   =   0x5FFFFFFF,
        LARGELEAFMASKNEG    =   0x6FFFFFFF,
    };

        HOST DEVICE uint notEmpty() const
        {
            return data.y & LARGENOTEMPTYMASK;
        }

        HOST DEVICE uint operator [] (const uint aId) const
        {
            return (data.y >> (aId * 8)) & MASK;
        }

        HOST DEVICE void clear()
        {
            data.y = 0u;
        }

        //NOTE: Buggy, calling it twice for the same cell yields wrong data
        HOST DEVICE void setX(const uint aVal)
        {
            data.y |= (aVal & MASK) << SHIFTX;
        }
        //NOTE: Buggy, calling it twice for the same cell yields wrong data
        HOST DEVICE void setY(const uint aVal)
        {
            data.y |= (aVal & MASK) << SHIFTY;
        }
        //NOTE: Buggy, calling it twice for the same cell yields wrong data
        HOST DEVICE void setZ(const uint aVal)
        {
            data.y |= (aVal & MASK) << SHIFTZ;
        }
        //NOTE: Buggy, calling it twice for the same cell yields wrong data
        HOST DEVICE void setW(const uint aVal)
        {
            data.y |= (aVal & MASK) << SHIFTW;
        }

        HOST DEVICE static uint get(const uint aId, const uint aVal)
        {
            return (aVal >> (aId * 8)) & MASK;
        }

        HOST DEVICE void setEmpty()
        {
            data.y &= LARGENOTEMPTYMASKNEG;
        }

        HOST DEVICE void setNotEmpty()
        {
            data.y |= LARGENOTEMPTYMASK;
        }

        HOST DEVICE void setValid()
        {
            data.y |= LARGEVALIDMASK;
        }

        HOST DEVICE void setNotValid()
        {
            data.y &= LARGEVALIDMASKNEG;
        }

        HOST DEVICE void setLeaf()
        {
            data.y |= LARGELEAFMASK;
        }

        HOST DEVICE void setNotLeaf()
        {
            data.y &= LARGELEAFMASKNEG;
        }

        HOST DEVICE void setLeafRangeBegin(const uint aVal)
        {
            data.x = aVal;
        }

        HOST DEVICE uint getLeafRangeBegin() const
        {
            return data.x;
        }
};

#endif // TWOLEVELGRID_HPP_INCLUDED_1684EB4C_F1F8_4A10_8415_2D6EE02CE0D7
