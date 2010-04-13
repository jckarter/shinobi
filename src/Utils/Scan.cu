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

#include "Scan.h"
#include "Utils/chag/pp/prefix.cuh"

void ExclusiveScan::operator()(
        uint*& aIn, 
        const uint aNumElements, 
        cudaStream_t aStream
        ) const
{
     chag::pp::prefix(aIn, (aIn + aNumElements), aIn, (uint*)0, aStream);
}


#define PREFIXSUMTHREADS 512u

//Assumes that all threads participate
template<class ta_operator, class ta_type>
DEVICE ta_type blockScan32bit(
                              ta_type aValue, ta_operator aOperator, ta_type aZeroValue, 
                              unsigned aThreadIDLinear, unsigned aShMemOffsetInWords, int aBlockSize,
                              bool aOpDirectionIsMSBtoLSB = false, bool aReduceInsteadOfScan = false
                              )
{
    extern SHARED ta_type shMem[];

    enum {warpBits = 5};
    enum {warpSize = 1 << warpBits};
    enum {laneMask = warpSize - 1};

    unsigned warpNum = aThreadIDLinear >> warpBits;
    unsigned warpLane = aThreadIDLinear & laneMask;

    int directionSign = aOpDirectionIsMSBtoLSB ? 1 : -1;

    //First do it at a warp level
    unsigned threadValueOffsetBase = aShMemOffsetInWords + (warpNum << (warpBits + 1)) + warpLane;
    int threadValueOffset = threadValueOffsetBase + (aOpDirectionIsMSBtoLSB ? 0 : warpSize);
    {
        shMem[threadValueOffsetBase] = aZeroValue; shMem[threadValueOffsetBase + warpSize] = aZeroValue;
        shMem[threadValueOffset] = aValue;

        unsigned _tmpVal;
        EMUSYNCTHREADS; _tmpVal = aOperator(shMem[threadValueOffset], shMem[threadValueOffset + 1  * directionSign]);
        EMUSYNCTHREADS; shMem[threadValueOffset] = _tmpVal;
        EMUSYNCTHREADS; _tmpVal = aOperator(shMem[threadValueOffset], shMem[threadValueOffset + 2  * directionSign]);
        EMUSYNCTHREADS; shMem[threadValueOffset] = _tmpVal;
        EMUSYNCTHREADS; _tmpVal = aOperator(shMem[threadValueOffset], shMem[threadValueOffset + 4  * directionSign]);
        EMUSYNCTHREADS; shMem[threadValueOffset] = _tmpVal;
        EMUSYNCTHREADS; _tmpVal = aOperator(shMem[threadValueOffset], shMem[threadValueOffset + 8  * directionSign]);
        EMUSYNCTHREADS; shMem[threadValueOffset] = _tmpVal;
        EMUSYNCTHREADS; _tmpVal = aOperator(shMem[threadValueOffset], shMem[threadValueOffset + 16 * directionSign]);
        EMUSYNCTHREADS; shMem[threadValueOffset] = _tmpVal;
    }
    SYNCTHREADS;


    //Now run warp 0, threads 0..ta_blocksize / 32 to do block level scan
    unsigned blockScratchPadOffset = aShMemOffsetInWords + aBlockSize * 2;
    unsigned warpsInBlock = aBlockSize >> warpBits; //Note: block size MUST be a multiple of warpSize
    {
        unsigned blockValueOffsetBase, blockValueOffset;

        if(warpNum == 0)
        {
            blockValueOffsetBase = blockScratchPadOffset + warpLane;
            blockValueOffset = blockValueOffsetBase + (aOpDirectionIsMSBtoLSB ? 0 : warpSize);

            bool laneActive = warpLane < warpsInBlock;
            shMem[blockValueOffsetBase] = aZeroValue; shMem[blockValueOffsetBase + warpSize] = aZeroValue;
            if(laneActive) shMem[blockValueOffset] = shMem[aShMemOffsetInWords + (warpLane << (warpBits + 1)) + (aOpDirectionIsMSBtoLSB ? 0 : warpSize * 2 - 1)];
        }

        unsigned _tmpVal;

        EMUSYNCTHREADS; if(warpNum == 0 && warpsInBlock >  1) _tmpVal = aOperator(shMem[blockValueOffset], shMem[blockValueOffset + 1  * directionSign]);
        EMUSYNCTHREADS; if(warpNum == 0 && warpsInBlock >  1) shMem[blockValueOffset] = _tmpVal;
        EMUSYNCTHREADS; if(warpNum == 0 && warpsInBlock >  2) _tmpVal = aOperator(shMem[blockValueOffset], shMem[blockValueOffset + 2  * directionSign]);
        EMUSYNCTHREADS; if(warpNum == 0 && warpsInBlock >  2) shMem[blockValueOffset] = _tmpVal;
        EMUSYNCTHREADS; if(warpNum == 0 && warpsInBlock >  4) _tmpVal = aOperator(shMem[blockValueOffset], shMem[blockValueOffset + 4  * directionSign]);
        EMUSYNCTHREADS; if(warpNum == 0 && warpsInBlock >  4) shMem[blockValueOffset] = _tmpVal;
        EMUSYNCTHREADS; if(warpNum == 0 && warpsInBlock >  8) _tmpVal = aOperator(shMem[blockValueOffset], shMem[blockValueOffset + 8  * directionSign]);
        EMUSYNCTHREADS; if(warpNum == 0 && warpsInBlock >  8) shMem[blockValueOffset] = _tmpVal;
        EMUSYNCTHREADS; if(warpNum == 0 && warpsInBlock > 16) _tmpVal = aOperator(shMem[blockValueOffset], shMem[blockValueOffset + 16 * directionSign]);
        EMUSYNCTHREADS; if(warpNum == 0 && warpsInBlock > 16) shMem[blockValueOffset] = _tmpVal;
    }

    SYNCTHREADS;

    //Now propagate the block partial results back to the warp levels
    unsigned retValue;
    if(aReduceInsteadOfScan)
        retValue = shMem[blockScratchPadOffset + (aOpDirectionIsMSBtoLSB ? 0 : warpSize + warpsInBlock - 1)];
    else
        retValue = aOperator(shMem[threadValueOffset], shMem[blockScratchPadOffset + warpNum + (aOpDirectionIsMSBtoLSB ? 0 : warpSize) + directionSign]);

    SYNCTHREADS;

    return retValue;
}

GLOBAL void prefixSumKernel(
                            uint *aOut,                 //output array
                            const uint *aIn,            //input array
                            const uint aNumOutElements, //output size
                            const uint aNumInElements,  //input size
                            const uint aOutOffset,      //consider every N-th output element
                            const uint aInOffset)       //consider every N-th input element
{
    const uint inputPos = aInOffset * globalThreadId1D() + aInOffset - 1;
    uint val = 0u;
    if(inputPos < aNumInElements)
    {
        val = aIn[inputPos];
    }

    InclusiveScan::OperatorPlus op;
    const uint result = blockScan32bit(val, op, 0u, threadIdx.x, 0u, PREFIXSUMTHREADS);

    const uint outputPos = aOutOffset * globalThreadId1D() + aOutOffset - 1;
    if (outputPos < aNumOutElements)
    {
        aOut[outputPos] = result;
    }

}



GLOBAL void sumKernel(
                      uint *aOut,  
                      const uint *aIn
                      )

{

    if (blockId1D() > 0)
    {
        const uint inputPos     = blockId1D() - 1;
        const uint blockSum     = aIn[inputPos];
        const uint outputPos    = globalThreadId1D();
        aOut[outputPos]         += blockSum;
    }
}

void InclusiveScan::operator()(
                       uint*& aIn, 
                       const uint aNumElements,
                       cudaStream_t aStream
                       ) const
{

    dim3 block(PREFIXSUMTHREADS);
    dim3 grid0((aNumElements + PREFIXSUMTHREADS - 1) / PREFIXSUMTHREADS );
    const uint sharedMemorySize = (64 + PREFIXSUMTHREADS * 2) * sizeof(uint);

    if (grid0.x == 1u)
    {
        //////////////////////////////////////////////////////////////////////////
        //in-place prefix sum
        //////////////////////////////////////////////////////////////////////////

        prefixSumKernel<<<grid0, block, sharedMemorySize, aStream >>>
            (aIn, aIn, aNumElements, aNumElements, 1, 1);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");
    }
    else
    {
        ////////////////////////////////////////////////////////////////////////////
        //subdivide input into bins and scan each
        ////////////////////////////////////////////////////////////////////////////

        prefixSumKernel<<<grid0, block, sharedMemorySize, aStream >>>
            (aIn, aIn, aNumElements, aNumElements, 1, 1);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        ////////////////////////////////////////////////////////////////////////////
        //scan the partial sums
        ////////////////////////////////////////////////////////////////////////////

        dim3 grid1((grid0.x + PREFIXSUMTHREADS - 1) / PREFIXSUMTHREADS);
        uint* blockSums1;
        CUDA_SAFE_CALL( cudaMalloc((void**)&blockSums1, grid0.x * sizeof(uint)) );

        prefixSumKernel<<<grid1, block, sharedMemorySize, aStream >>>
            (blockSums1, aIn, grid0.x, aNumElements, 1, PREFIXSUMTHREADS);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        if (grid1.x > 1u)
        {
            ////////////////////////////////////////////////////////////////////////////
            //scan the partial sums
            ////////////////////////////////////////////////////////////////////////////

            dim3 grid2((grid1.x + PREFIXSUMTHREADS - 1) / PREFIXSUMTHREADS);
            uint* blockSums2;
            CUDA_SAFE_CALL( cudaMalloc((void**)&blockSums2, grid1.x * sizeof(uint)) );

            prefixSumKernel<<<grid2, block, sharedMemorySize, aStream >>>
                (blockSums2, blockSums1, grid1.x, grid0.x, 1, PREFIXSUMTHREADS);

            CUT_CHECK_ERROR("Kernel Execution failed.\n");

            //////////////////////////////////////////////////////////////////////////
            //add scanned block sums to each element
            //////////////////////////////////////////////////////////////////////////

            sumKernel<<<grid1, block, 0, aStream >>>(blockSums1, blockSums2);

            CUT_CHECK_ERROR("Kernel Execution failed.\n");

            CUDA_SAFE_CALL( cudaFree(blockSums2) );
        }

        //////////////////////////////////////////////////////////////////////////
        //add scanned block sums to each element
        //////////////////////////////////////////////////////////////////////////

        sumKernel<<<grid0, block, 0, aStream >>>(aIn, blockSums1);

        CUT_CHECK_ERROR("Kernel Execution failed.\n");

        CUDA_SAFE_CALL( cudaFree(blockSums1) );


    }
}

#undef PREFIXSUMTHREADS

////computes prefix sum via exclusive scan
//GLOBAL void prefixSumKernel(
//                            uint *aOut,                 //output array
//                            const uint *aIn,            //input array
//                            const uint aNumOutElements, //output size
//                            const uint aNumInElements,  //input size
//                            const uint aOutOffset,      //consider every N-th output element
//                            const uint aInOffset)       //consider every N-th input element
//
//{
//
//    extern SHARED uint shMem[]; 
//
//    //upload 2 elements to shared memory
//    //pad after every NUMBANKS elements
//    const uint shMemPos1 = threadId1D() + (threadId1D() >> LOG2NUMBANKS);
//    //pad after every NUMBANKS elements
//    const uint shMemPos2 = threadId1D() + blockSize() + 
//        ((threadId1D() + blockSize()) >> LOG2NUMBANKS);
//    const uint inputPos1 = aInOffset * globalThreadId1D(2u) + aInOffset - 1;
//    const uint inputPos2 = inputPos1 + aInOffset * blockSize();
//
//    shMem[shMemPos1] = 0.f;
//    shMem[shMemPos2] = 0.f;
//
//    if(inputPos1 < aNumInElements)
//    {
//        shMem[shMemPos1] = aIn[inputPos1];
//    }
//    if(inputPos2 < aNumInElements)
//    {
//        shMem[shMemPos2] = aIn[inputPos2];
//    }
//
//    //build sum tree
//    uint stride = 1u;
//    for (; stride <= blockSize(); stride <<= 1)
//    {
//        SYNCTHREADS;
//
//        if (threadId1D() < blockSize() / stride)
//        {
//            uint inPos  = stride * (2 * threadId1D() + 1) - 1;
//            uint outPos = stride * (2 * threadId1D() + 2) - 1;
//            inPos  += (inPos  >> LOG2NUMBANKS); //padding
//            outPos += (outPos >> LOG2NUMBANKS); //padding
//            shMem[outPos] += shMem[inPos];
//        }
//    }
//
//    SYNCTHREADS;
//
//    //zero the last element
//    if(threadId1D() == 0)
//    {
//        uint lastElemId = 2 * blockSize() - 1 + ((2 * blockSize() - 1) >> LOG2NUMBANKS);
//        shMem[lastElemId] = 0.f;
//    }
//
//    //use partial sums to compute the prefix sum array
//    for(; stride >= 1u; stride >>= 1)
//    {
//        SYNCTHREADS;
//
//        if (threadId1D() < blockSize() / stride)
//        {
//            uint inPos  = 2 * blockSize() - 1 - stride * (2 * threadId1D() + 1);
//            uint outPos = 2 * blockSize() - 1 - stride *  2 * threadId1D();
//            inPos  += (inPos  >> LOG2NUMBANKS); //padding
//            outPos += (outPos >> LOG2NUMBANKS); //padding
//            const float tmp = shMem[outPos];
//            shMem[outPos] += shMem[inPos];
//            shMem[inPos]  = tmp; 
//        }
//    }
//
//    SYNCTHREADS;
//
//    const uint outputPos1 = aOutOffset * globalThreadId1D(2u) + aOutOffset - 1;
//    const uint outputPos2 = outputPos1 + aOutOffset * blockSize();
//
//    //write out the result
//    if(outputPos1 < aNumOutElements)
//    {
//        aOut[outputPos1] = shMem[shMemPos1];
//    }
//    if(outputPos2 < aNumOutElements)
//    {
//        aOut[outputPos2] = shMem[shMemPos2];
//    }
//}
