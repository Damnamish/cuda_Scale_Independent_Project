#pragma once
//
// Host-callable entry point into the CUDA kernels. Deliberately kept free
// of any CUDA types (dim3, cudaError_t, __global__, etc.) so this header
// -- and anything that only includes this header, like main.cpp -- can be
// compiled with a plain C++17 host compiler. Only kernels.cu needs nvcc.
//
#include <cstdint>

// Converts an interleaved-RGB image to grayscale and applies a
// (2*blurRadius+1)^2 box blur, entirely on the GPU.
//
//   hostRGBIn / hostRGBOut : host buffers, width*height*3 bytes each
//                            (RGB in, grayscale-as-RGB out)
//   width, height          : image dimensions in pixels
//   blurRadius              : box blur radius; window size is (2r+1)x(2r+1)
//
// Returns the GPU kernel execution time in milliseconds for this image
// (grayscale + blur kernels combined), or a negative value on failure.
float processImageOnGPU(const unsigned char *hostRGBIn, unsigned char *hostRGBOut,
                         int width, int height, int blurRadius);
