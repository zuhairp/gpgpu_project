import sys
import math
import argparse
from time import time
from os import walk

import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule

import numpy
from skimage import io

parser = argparse.ArgumentParser(description='Benchmark a CUDA kernel against images')
parser.add_argument('kernel_file', type=str, help='File path to the kernel to benchmark')
parser.add_argument('-n', '--iterations', type=int, default=1, help='The number of times to iterate')
parser.add_argument('-f', '--filename', type=str, default='', help='The file in pictures/ to run against')
parser.add_argument('-p', '--percolor', action='store_true', help='If set, then kernel will be run seperately for R, G, and B')

arguments = parser.parse_args()

kernel = open(arguments.kernel_file, 'r').read()

mod = SourceModule(kernel)
program = mod.get_function("doIt")

if arguments.filename is '':
    for _, _, fnames in walk('pictures'):
        filenames = fnames
        break
else:
    filenames = [arguments.filename]

total_time = 0
times = []
for i in range(arguments.iterations):
    count = 0
    for filename in filenames:
        count += 1
        sys.stdout.write("%02d %03d %s    \r" % (i, count, filename))
        sys.stdout.flush()
        
        a = io.imread('pictures/'+filename).astype(numpy.int32)
        rowStride, colStride, _ = a.strides
        print(a.strides)
        dest = numpy.zeros_like(a)
        height, width, _ = a.shape
        print(a.shape)
        grid=(
            int(math.ceil(width/32)),
            int(math.ceil(height/32)),
            3 if arguments.percolor else 1
        )
        print(grid)
        start = time()
        # program(drv.Out(dest), drv.In(a), block=(16, 21, 3), grid=(240,103))
        program(
            drv.Out(dest), drv.In(a), 
            numpy.int32(width), numpy.int32(height),
            numpy.int32(rowStride/4), numpy.int32(colStride/4), 
            block=(32, 32, 1), 
            grid=grid,
        )
        end = time()
        total_time += (end - start)
        times.append(end-start)
        io.imsave('output/'+filename, dest)

print("")
print(total_time/count)

import pprint
pprint.pprint(times)
