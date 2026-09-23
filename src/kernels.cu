#include "kernels.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Minimal local stand-in for the checkCudaErrors() macro normally pulled in
// from NVIDIA's cuda-samples helper_cuda.h, so this file has no external
// dependency beyond the CUDA toolkit itself.
inline void checkCuda(cudaError_t result, const char *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

//
// One thread per pixel: standard luma-weighted RGB -> grayscale.
//
__global__ void rgbToGrayscaleKernel(const unsigned char *rgbIn, unsigned char *grayOut,
                                      int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int pixelIdx = y * width + x;
    int rgbIdx = pixelIdx * 3;
    float r = rgbIn[rgbIdx + 0];
    float g = rgbIn[rgbIdx + 1];
    float b = rgbIn[rgbIdx + 2];
    grayOut[pixelIdx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
}

//
// One thread per output pixel: (2*radius+1)^2 box blur over the grayscale
// image, with edge-clamped (border pixels shrink the window rather than
// sampling out of bounds) sampling. Writes into an RGB buffer (R=G=B) so
// the result is directly viewable as a normal image.
//
__global__ void boxBlurKernel(const unsigned char *grayIn, unsigned char *rgbOut,
                               int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int sum = 0;
    int count = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        int sy = y + dy;
        if (sy < 0 || sy >= height) {
            continue;
        }
        for (int dx = -radius; dx <= radius; dx++) {
            int sx = x + dx;
            if (sx < 0 || sx >= width) {
                continue;
            }
            sum += grayIn[sy * width + sx];
            count++;
        }
    }
    unsigned char avg = static_cast<unsigned char>(sum / count);

    int outIdx = (y * width + x) * 3;
    rgbOut[outIdx + 0] = avg;
    rgbOut[outIdx + 1] = avg;
    rgbOut[outIdx + 2] = avg;
}

float processImageOnGPU(const unsigned char *hostRGBIn, unsigned char *hostRGBOut,
                         int width, int height, int blurRadius) {
    const size_t rgbBytes = static_cast<size_t>(width) * height * 3;
    const size_t grayBytes = static_cast<size_t>(width) * height;

    unsigned char *d_rgbIn = nullptr;
    unsigned char *d_gray = nullptr;
    unsigned char *d_rgbOut = nullptr;

    checkCudaErrors(cudaMalloc(&d_rgbIn, rgbBytes));
    checkCudaErrors(cudaMalloc(&d_gray, grayBytes));
    checkCudaErrors(cudaMalloc(&d_rgbOut, rgbBytes));

    checkCudaErrors(cudaMemcpy(d_rgbIn, hostRGBIn, rgbBytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t startEvent, stopEvent;
    checkCudaErrors(cudaEventCreate(&startEvent));
    checkCudaErrors(cudaEventCreate(&stopEvent));
    checkCudaErrors(cudaEventRecord(startEvent));

    rgbToGrayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_rgbIn, d_gray, width, height);
    checkCudaErrors(cudaGetLastError());

    boxBlurKernel<<<blocksPerGrid, threadsPerBlock>>>(d_gray, d_rgbOut, width, height, blurRadius);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stopEvent));
    checkCudaErrors(cudaEventSynchronize(stopEvent));

    float millisecondsElapsed = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&millisecondsElapsed, startEvent, stopEvent));
    checkCudaErrors(cudaEventDestroy(startEvent));
    checkCudaErrors(cudaEventDestroy(stopEvent));

    checkCudaErrors(cudaMemcpy(hostRGBOut, d_rgbOut, rgbBytes, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_rgbIn));
    checkCudaErrors(cudaFree(d_gray));
    checkCudaErrors(cudaFree(d_rgbOut));

    return millisecondsElapsed;
}
