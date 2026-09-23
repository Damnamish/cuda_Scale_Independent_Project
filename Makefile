NVCC      := nvcc
CXXSTD    := -std=c++17
INCLUDES  := -Isrc -I/usr/local/cuda/include
LIBDIRS   := -L/usr/local/cuda/lib64
LIBS      := -lcudart

SRC_CU    := src/kernels.cu
SRC_CPP   := src/main.cpp src/ppm_io.cpp
TARGET    := bin/cuda_batch_filter

.PHONY: build run clean

build: $(TARGET)

$(TARGET): $(SRC_CU) $(SRC_CPP)
	mkdir -p bin
	$(NVCC) $(CXXSTD) $(INCLUDES) $(SRC_CPP) $(SRC_CU) -o $(TARGET) $(LIBDIRS) $(LIBS)

run: build
	./run.sh

clean:
	rm -f $(TARGET)
	rm -rf data/output
