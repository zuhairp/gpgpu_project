import pycuda.autoinit
import pycuda.driver as drv
import numpy
import itertools
import time
from pycuda.compiler import SourceModule
import csv


mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b, int stride_dest, int stride_a, int stride_b)
{
    extern __shared__ float temp[];
    
    temp[threadIdx.x] = a[blockIdx.x * stride_a + threadIdx.x] * b[threadIdx.x * stride_b + blockIdx.y];
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            sum += temp[i];
        }
        atomicAdd(&(dest[blockIdx.x * stride_dest + blockIdx.y]), sum);
    }
}
""")
multiply_them = mod.get_function("multiply_them")

data = []

npvals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
for n, p, m in itertools.product(npvals, npvals, [1, 2, 4, 8, 16, 32]):
    a = numpy.random.randn(n, m).astype(numpy.float32)
    b = numpy.random.randn(m, p).astype(numpy.float32)
    
    dest = numpy.zeros((n, p)).astype(numpy.float32)
    print m
    start = time.clock()
    multiply_them(
            drv.Out(dest), drv.In(a), drv.In(b), numpy.int32(dest.strides[0])/4, numpy.int32(a.strides[0])/4, numpy.int32(b.strides[0])/4,
            block=(m,1,1), grid=(n,p), shared=m)
    end = time.clock()
    data.append((n*p*m, end - start))
print "done"

with open('data_gpu.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(data)
