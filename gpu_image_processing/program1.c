#include <stdio.h>

#define R 0
#define G 1
#define B 2

__global__ void doIt(int* dest, int* a, int width, int height, int rowStride, int colStride){
    /* const int blockNum = gridDim.x*blockIdx.y + blockIdx.x; */
    /* const int threadNum = blockDim.x*threadIdx.y + threadIdx.x; */
    /* const int blockBase = blockDim.x*blockDim.y*blockNum; */

    const int numThreadsPerRow = blockDim.x*gridDim.x;

    const int rowIndex = blockDim.y*blockIdx.y + threadIdx.y;
    const int colIndex = blockDim.x*blockIdx.x + threadIdx.x;

    if(rowIndex >= height) return;
    if(colIndex >= width) return;

    const int thread_id = rowIndex * numThreadsPerRow + colIndex;

    const int rIndex = rowIndex * rowStride + colIndex * colStride + R;
    const int gIndex = rowIndex * rowStride + colIndex * colStride + G;
    const int bIndex = rowIndex * rowStride + colIndex * colStride + B;

    int average = 0.1989*a[rIndex] + 0.7870*a[gIndex] + 0.0140 * a[bIndex];
    dest[rIndex] = average;
    dest[gIndex] = average;
    dest[bIndex] = average;

}
