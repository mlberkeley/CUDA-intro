import numpy as np
import numba
from numba import cuda

print(np.__version__)
print(numba.__version__)

# cuda.detect()

@cuda.jit
def add_scalars(a, b, c):
    c[0] = a + b

dev_c = cuda.device_array((1,), np.float32)

add_scalars[1, 1](2.0, 7.0, dev_c)

c = dev_c.copy_to_host()
print(f"2.0 + 7.0 = {c[0]}")

# add_array
# Useful variables:
#  cuda.threadIdx.x
#  cuda.blockDim.x
#  cuda.blockIdx.x
@cuda.jit
def add_array(a, b, c):
    # YOUR CODE HERE
    pass

N = 20
a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32)
dev_c = cuda.device_array_like(a)

add_array[4, 8](a, b, dev_c)

c = dev_c.copy_to_host()
print(c)

# ANSWER:
# 
#    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
#    if i < a.size:
#        c[i] = a[i] + b[i]
