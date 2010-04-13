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

#ifndef SCANTEST_H_INCLUDED_46FB6973_01EF_4701_A39C_76F43EFDBACC
#define SCANTEST_H_INCLUDED_46FB6973_01EF_4701_A39C_76F43EFDBACC

#include "../CUDAStdAfx.h"
#include "../Utils/Scan.h"

class ScanTest
{
public:
    void run(int argc, char* argv[])
    {
        const uint numElements = 31 * (1024 * 1024) - 1;
        uint* hostData1;
        uint* hostData2;
        uint* deviceData;

        CUDA_SAFE_CALL(cudaMallocHost((void**)&hostData1, numElements * sizeof(uint)) );
        CUDA_SAFE_CALL(cudaMallocHost((void**)&hostData2, numElements * sizeof(uint)) );
        CUDA_SAFE_CALL(cudaMalloc((void**)&deviceData, numElements * sizeof(uint)) );

        for (uint i = 0; i < numElements; ++i)
        {
            hostData1[i] = 1u;
        }

        CUDA_SAFE_CALL(cudaMemcpy(deviceData, hostData1, numElements*sizeof(uint), cudaMemcpyHostToDevice) );

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        InclusiveScan()(deviceData, numElements);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudastd::log::out << "Performed inclusive scan of " << numElements << " elements in " << elapsedTime << "ms\n";

        CUDA_SAFE_CALL( cudaMemcpy(hostData2, deviceData, numElements*sizeof(uint), cudaMemcpyDeviceToHost) );

        uint sum = 0;
        for(uint i = 0; i < numElements; ++i)
        {
            sum += hostData1[i];
            hostData1[i] = sum;

            if (hostData1[i] != hostData2[i])
            {
                cudastd::log::out << "Error in inclusive scan\n";
                cudastd::log::out << "Element index : "<< i << "\n";
                cudastd::log::out << "Host value :" << hostData1[i] << "device value : " << hostData2[i] << "\n";
            }
        }

        for (uint i = 0; i < numElements; ++i)
        {
            hostData1[i] = 1u;
        }

        CUDA_SAFE_CALL(cudaMemcpy(deviceData, hostData1, numElements*sizeof(uint), cudaMemcpyHostToDevice) );

        cudaEventRecord(start, 0);

        ExclusiveScan()(deviceData, numElements);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudastd::log::out << "Performed exclusive scan of " << numElements << " elements in " << elapsedTime << "ms\n";

        CUDA_SAFE_CALL( cudaMemcpy(hostData2, deviceData, numElements*sizeof(uint), cudaMemcpyDeviceToHost) );

        sum = 0;
        for(uint i = 0; i < numElements; ++i)
        {
            uint tmp = hostData1[i];
            hostData1[i] = sum;
            sum += tmp;

            if (hostData1[i] != hostData2[i])
            {
                cudastd::log::out << "Error in exclusive scan\n";
                cudastd::log::out << "Element index : "<< i << "\n";
                cudastd::log::out << "Host value :" << hostData1[i] << "device value : " << hostData2[i] << "\n";
            }
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        CUDA_SAFE_CALL( cudaFree(deviceData) );
        CUDA_SAFE_CALL( cudaFreeHost( hostData1) );
        CUDA_SAFE_CALL( cudaFreeHost( hostData2) );

    }
};

#endif // SCANTEST_H_INCLUDED_46FB6973_01EF_4701_A39C_76F43EFDBACC
