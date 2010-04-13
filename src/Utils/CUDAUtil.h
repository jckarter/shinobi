/****************************************************************************/
/* Copyright (c) 2009, Lukas Marsalek, Javor Kalojanov
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

#ifndef CUDAUTIL_H_INCLUDED_6566107B_240D_4B9A_B102_260A46A1B2BA
#define CUDAUTIL_H_INCLUDED_6566107B_240D_4B9A_B102_260A46A1B2BA

/*****************************************************************************
* Logger class to roughly simulate I/O Streams with printf. 
* For use in SPU code
* 
* Lukas Marsalek, 01.07.2008
****************************************************************************/

#include <stdio.h>

namespace cudastd 
{	
    class NewLine
    {
    public:
        static NewLine manip_endl;
    };	
    class Hex
    {
    public:
        static Hex manip_hex;
    };	

    class InfoSwitch
    {
    public:
        static InfoSwitch manip_infoSwitch;
    };

    class ErrorSwitch
    {
    public:
        static ErrorSwitch manip_errorSwitch;
    };

    class FloatPrecision
    {
    private:

        unsigned int precision;

    public:

        static FloatPrecision manip_precision;

        FloatPrecision& setPrecision(unsigned int prec)
        {
            precision = prec;
            return manip_precision;
        }

        unsigned int getPrecision() const
        {
            return precision;
        }
    };


    class log
    {

    public:	

        static log out;		

        log()
        { 
            hexOutput = false;
            loggingOff = false;
#ifdef _WIN32
            _snprintf_s(fpString, 256, "%%f");
#else
            snprintf(fpString, 256, "%%f");			
#endif
        }

        static NewLine endl()
        {
            return cudastd::NewLine::manip_endl;
        }

        static FloatPrecision floatPrecision(unsigned int precision)
        {
            return cudastd::FloatPrecision::manip_precision.setPrecision(precision);
        }

        static Hex hexFormat()
        {
            return cudastd::Hex::manip_hex;
        }

        void loggingSilence(bool state)
        {
            loggingOff = state;
        }

        log& operator << (const InfoSwitch&)
        {			
            return out;
        }

        log& operator << (const ErrorSwitch&)
        {			
            return out;
        }

        log& operator << (const NewLine&)
        {
            if(loggingOff) return out;
#ifdef _WIN32
            printf_s("\n");
#else
            printf("\n");
#endif
            return out;
        }	

        log& operator << (const FloatPrecision& fp)
        {			
            if(loggingOff) return out;
#ifdef _WIN32
            _snprintf_s(fpString, 256, "%%.%df", fp.getPrecision());
#else
            snprintf(fpString, 256, "%%.%df", fp.getPrecision());			
#endif

            return out;
        }	

        log& operator << (const Hex&)
        {
            if(loggingOff) return out;

            hexOutput = true;
            return out;
        }	


        log& operator << (const char* str)
        {
            if(loggingOff) return out;

#ifdef _WIN32
            printf_s("%s", str);
#else
            printf("%s", str);			
#endif            		
            return out;
        }		

        log& operator << (const unsigned long long value)
        {
            if(loggingOff) return out;

            if(hexOutput)
            {
                printf("0x%llx", value);
            }
            else
            {
                printf("%lld", value);
            }
            hexOutput = false;
            return out;
        }		

        log& operator << (const unsigned int value)
        {
            if(loggingOff) return out;

            if(hexOutput)
            {
#ifdef _WIN32
                printf_s("0x%x", value);
#else
                printf("0x%x", value);
#endif
            }
            else
            {
#ifdef _WIN32
                printf_s("%d", value);
#else
                printf("%d", value);
#endif
            }
            hexOutput = false;
            return out;
        }	

        log& operator << (const int value)
        {
            if(loggingOff) return out;

            if(hexOutput)
            {
#ifdef _WIN32
                printf_s("0x%x", value);
#else
                printf("0x%x", value);
#endif
            }
            else
            {
#ifdef _WIN32
                printf_s("%d", value);
#else
                printf("%d", value);
#endif
            }
            hexOutput = false;
            return out;
        }

#ifndef _WIN32

        log& operator << (const size_t value)
        {
            if(loggingOff) return out;

            if(hexOutput)
            {
#ifdef _WIN32
                printf_s("0x%lx", value);
#else
                printf("0x%lx", value);
#endif
            }
            else
            {
#ifdef _WIN32
                printf_s("%ld", value);
#else
                printf("%ld", value);
#endif
            }			
            hexOutput = false;
            return out;
        }

#endif		
        log& operator << (const long int value)
        {
            if(loggingOff) return out;

            if(hexOutput)
            {
#ifdef _WIN32
                printf_s("0x%lx", value);
#else
                printf("0x%lx", value);
#endif
            }
            else
            {
#ifdef _WIN32
                printf_s("%ld", value);
#else
                printf("%ld", value);
#endif
            }			
            hexOutput = false;
            return out;
        }


        log& operator << (const float value)
        {			
            if(loggingOff) return out;
#ifdef _WIN32
            printf_s((const char*)fpString, value);
            _snprintf_s(fpString, 256, "%%f");
#else
            printf((const char*)fpString, value);
            snprintf(fpString, 256, "%%f");
#endif
            return out;
        }	

    private:

        bool hexOutput;
        bool infoOn;
        bool loggingOff;		
        char fpString[256];

    };
};//namespace cudastd

/*******************************************************************************
*Classes and function to mimic implementations in the <xutility> header
*
*
*
*******************************************************************************/

namespace cudastd 
{	
    template<class tType> inline
        const tType& max(const tType& aLeft, const tType& aRight)
    {
        return ((aLeft < aRight) ? aRight : aLeft);
    }

    template<class tType> inline
        const tType& min(const tType& aLeft, const tType& aRight)
    {
        return ((aRight < aLeft) ? aRight : aLeft);
    }

};//namespace cudastd

#endif // CUDAUTIL_H_INCLUDED_6566107B_240D_4B9A_B102_260A46A1B2BA
