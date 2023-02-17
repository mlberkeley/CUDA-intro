# CUDA_intro

## Setup

Add the following lines to your ~/.profile to add `nvcc` to the path:

```sh
export PATH="/usr/local/cuda-11.6/bin:$PATH"
export PATH="~/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH"
```

Create a virtual env and install numba:

```sh
python -m venv venv
source venv/bin/activate
pip install numba cuda-python numpy
```

## vec_add

Implement a vector add function in CUDA C++

Build: `make build`

Run: `make run`

## numba

Implement a vector add function in Python using Numba

Run: `make run`

