//
// cuda_batch_filter
//
// Batch-processes every .ppm image in an input directory through a GPU
// pipeline (RGB -> grayscale -> box blur, both stages as CUDA kernels)
// and writes the results to an output directory. Prints a per-image and
// summary log to stdout, intended to double as the "proof of execution"
// artifact for the assignment when redirected to a file, e.g.:
//
//   ./bin/cuda_batch_filter data/input data/output 2 | tee proof_of_execution.log
//
// This file is plain C++17 (no CUDA-specific syntax) and only talks to
// the GPU through processImageOnGPU(), declared in kernels.h and
// implemented in kernels.cu.
//
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "kernels.h"
#include "ppm_io.h"

namespace fs = std::filesystem;

namespace {

void printUsage(const char *prog) {
    std::cerr << "Usage: " << prog << " <input_dir> <output_dir> [blur_radius] [max_images]\n"
              << "  input_dir    directory of .ppm images to process\n"
              << "  output_dir   directory to write processed .ppm images to (created if missing)\n"
              << "  blur_radius  box blur radius; window is (2r+1)x(2r+1) (default: 2)\n"
              << "  max_images   process at most this many images (default: all)\n";
}

std::vector<fs::path> collectInputImages(const fs::path &inputDir) {
    std::vector<fs::path> files;
    for (const auto &entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".ppm") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    const fs::path inputDir = argv[1];
    const fs::path outputDir = argv[2];
    const int blurRadius = (argc >= 4) ? std::stoi(argv[3]) : 2;
    const size_t maxImages = (argc >= 5) ? static_cast<size_t>(std::stoul(argv[4]))
                                          : std::numeric_limits<size_t>::max();

    if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
        std::cerr << "ERROR: input directory '" << inputDir.string() << "' does not exist\n";
        return 1;
    }
    fs::create_directories(outputDir);

    std::vector<fs::path> inputFiles = collectInputImages(inputDir);
    if (inputFiles.empty()) {
        std::cerr << "ERROR: no .ppm files found in '" << inputDir.string() << "'\n";
        return 1;
    }
    if (inputFiles.size() > maxImages) {
        inputFiles.resize(maxImages);
    }

    std::cout << "cuda_batch_filter: processing " << inputFiles.size()
              << " image(s) from '" << inputDir.string() << "'\n"
              << "  blur radius: " << blurRadius << " (window "
              << (2 * blurRadius + 1) << "x" << (2 * blurRadius + 1) << ")\n"
              << "  output dir : '" << outputDir.string() << "'\n\n";

    size_t processedCount = 0;
    size_t failedCount = 0;
    double totalKernelMs = 0.0;

    const auto wallStart = std::chrono::steady_clock::now();

    for (size_t i = 0; i < inputFiles.size(); i++) {
        const fs::path &inPath = inputFiles[i];

        Image img;
        if (!readPPM(inPath.string(), img)) {
            std::cerr << "  [" << (i + 1) << "/" << inputFiles.size() << "] SKIP  " << inPath.filename().string()
                      << " (read failed)\n";
            failedCount++;
            continue;
        }

        Image outImg;
        outImg.width = img.width;
        outImg.height = img.height;
        outImg.pixels.resize(img.pixels.size());

        float kernelMs = processImageOnGPU(img.pixels.data(), outImg.pixels.data(),
                                            img.width, img.height, blurRadius);

        const fs::path outPath = outputDir / inPath.filename();
        if (!writePPM(outPath.string(), outImg)) {
            std::cerr << "  [" << (i + 1) << "/" << inputFiles.size() << "] SKIP  " << inPath.filename().string()
                      << " (write failed)\n";
            failedCount++;
            continue;
        }

        totalKernelMs += kernelMs;
        processedCount++;

        std::cout << "  [" << (i + 1) << "/" << inputFiles.size() << "] " << inPath.filename().string()
                  << " (" << img.width << "x" << img.height << ") -> " << outPath.filename().string()
                  << "  [" << kernelMs << " ms GPU]\n";
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    const double wallMs = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();

    std::cout << "\nDone. Processed " << processedCount << "/" << inputFiles.size() << " image(s)";
    if (failedCount > 0) {
        std::cout << " (" << failedCount << " failed)";
    }
    std::cout << ".\n"
              << "  Total GPU kernel time : " << totalKernelMs << " ms\n"
              << "  Average per image     : " << (processedCount ? totalKernelMs / processedCount : 0.0) << " ms\n"
              << "  Total wall-clock time : " << wallMs << " ms\n";

    return failedCount > 0 ? 1 : 0;
}
