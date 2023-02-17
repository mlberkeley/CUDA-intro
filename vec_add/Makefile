INCLUDES := -I$(CURDIR)/../vendor/cuda-samples/Common
NVCCFLAGS := -G -g

build:
	nvcc $(INCLUDES) $(NVCCFLAGS) main.cu -o main

run: main
	./main