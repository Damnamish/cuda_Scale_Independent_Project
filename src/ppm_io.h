#pragma once
//
// Minimal binary PPM (P6) reader/writer.
//
// Kept dependency-free (no stb_image, no OpenCV) so the project builds
// with nothing beyond a C++17 compiler + the CUDA toolkit. PPM is a
// trivial, uncompressed format: a short text header followed by raw
// interleaved RGB bytes, which makes it easy to generate sample data
// and to eyeball-verify correctness.
//
#include <cstdint>
#include <string>
#include <vector>

struct Image {
    int width = 0;
    int height = 0;
    // Interleaved RGB, 3 bytes per pixel, row-major, size == width*height*3
    std::vector<unsigned char> pixels;
};

// Reads a binary PPM (P6) file. Returns false (and leaves img untouched)
// on any I/O or format error.
bool readPPM(const std::string &path, Image &img);

// Writes img out as a binary PPM (P6) file. Returns false on I/O error.
bool writePPM(const std::string &path, const Image &img);
