#include <stdio.h>

#define R 0
#define G 1
#define B 2

__device__ int getIndex(int width, int height, int rowStride, int colStride, int rowIndex, int colIndex){
    if(rowIndex >= height) rowIndex = height-1;
    if(colIndex >= width) colIndex = width-1;

    if(colIndex < 0) rowIndex = 0;
    if(rowIndex < 0) colIndex = 0;;
    
    return rowIndex*rowStride + colIndex*colStride;
}

__global__ void doIt(int* dest, int* a, int width, int height, int rowStride, int colStride){
    /* const int blockNum = gridDim.x*blockIdx.y + blockIdx.x; */
    /* const int threadNum = blockDim.x*threadIdx.y + threadIdx.x; */
    /* const int blockBase = blockDim.x*blockDim.y*blockNum; */

    const int numThreadsPerRow = blockDim.x*gridDim.x;

    const int rowIndex = blockDim.y*blockIdx.y + threadIdx.y;
    const int colIndex = blockDim.x*blockIdx.x + threadIdx.x;

    const int index = getIndex(width, height, rowStride, colStride, rowIndex, colIndex);

    if(rowIndex >= height) return;
    if(colIndex >= width) return;

    const int thread_id = rowIndex * numThreadsPerRow + colIndex;

    int convolution_matrix[9] = {
        -4, -2,  5, 
        -1,  1,  2, 
        -5,  2,  3
    };
    /* int convolution_matrix[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0}; */
    
    int result = 0;

    int r, c;
    int count = 0;
    for(r=-1; r <= 1; r++){
        for(c=-1; c <= 1; c++){
            int multiplier = convolution_matrix[count];
            int neighbor_index = getIndex(width, height, rowStride, colStride, rowIndex+r, colIndex+c);
            int neighbor = 0;
            if(neighbor_index > 0) neighbor = a[neighbor_index+blockIdx.z];
            count += 1;
            result += multiplier * neighbor;
        }
    }

    if(result < 0) result = 0;
    if(result > 255) result = 255;
    
    dest[index+blockIdx.z] = result;
}
