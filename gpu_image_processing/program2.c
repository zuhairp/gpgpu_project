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

    int red[3] = { 0xFC, 0x44, 0x4D };
    int white[3] = { 255, 255, 255 };
    int blue[3] = { 0x46, 0x27, 0xF5 };
    int val[3] = { a[rIndex], a[gIndex], a[bIndex] };

    int redDistance = (red[0]-val[0])*(red[0]-val[0]) + (red[1]-val[1])*(red[1]-val[1]) + (red[2]-val[2])*(red[2]-val[2]);
    int blueDistance = (blue[0]-val[0])*(blue[0]-val[0]) + (blue[1]-val[1])*(blue[1]-val[1]) + (blue[2]-val[2])*(blue[2]-val[2]);
    int whiteDistance = (white[0]-val[0])*(white[0]-val[0]) + (white[1]-val[1])*(white[1]-val[1]) + (white[2]-val[2])*(white[2]-val[2]);

    int* result; 
    if(redDistance <= blueDistance && redDistance <= whiteDistance) result = red;
    if(blueDistance <= redDistance && blueDistance <= whiteDistance) result = blue;
    if(whiteDistance <= redDistance && whiteDistance <= blueDistance) result = white;


    dest[rIndex] = result[0];
    dest[gIndex] = result[1];
    dest[bIndex] = result[2];
}
