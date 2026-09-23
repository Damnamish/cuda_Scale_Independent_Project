# CUDA Batch Image Filter

A CUDA program that batch-processes a directory of images through a
two-stage GPU pipeline -- RGB-to-grayscale, then a box blur -- and writes
the results back out. Built for the CUDA at Scale for the Enterprise
independent project.

## Overview

Given a folder of `.ppm` images, `cuda_batch_filter` loads each one,
uploads it to the GPU, runs two custom CUDA kernels over it (grayscale
conversion, then an NxN box blur), copies the result back, and writes
it to an output folder -- looping over every image in the input
directory in a single execution. The included sample dataset
(`data/input/`) has 200 procedurally generated 64x64 images (gradients,
checkerboards, rings, stripes, noise) so the pipeline has real bulk
data to run against out of the box.

Both the grayscale and blur stages run entirely on the GPU; no CPU-side
image processing happens except reading/writing the raw pixel bytes on
either side of the pipeline.

## Code Organization

```
bin/            Build output (cuda_batch_filter binary). Empty until `make build`.
data/input/     200 sample input images (.ppm, 64x64), procedurally generated.
data/output/    Where processed images land. Empty until the program is run.
lib/            Unused -- no third-party libraries needed (kept for template parity).
src/
  main.cpp      Host driver: walks the input directory, calls into the GPU
                pipeline per image, logs timing, writes results. Plain
                C++17, no CUDA-specific syntax.
  kernels.cu    The two CUDA kernels (grayscale, box blur) plus the host-side
                wrapper (cudaMalloc/cudaMemcpy/launch/cudaFree) that main.cpp
                calls through kernels.h.
  kernels.h     Declares processImageOnGPU() -- the only symbol main.cpp
                needs from the CUDA side.
  ppm_io.h/.cpp Minimal dependency-free binary PPM (P6) reader/writer.
gen_images.py   Standalone script (stdlib-only) used to (re)generate the
                sample dataset in data/input/. Not part of the build.
Makefile        `make build` / `make run` / `make clean`.
run.sh          Runs the built binary over data/input and tees the output
                into a timestamped proof_of_execution_*.log file.
INSTALL         Toolchain requirements and build/run instructions.
```

## Algorithm / Kernels

**1. RGB to grayscale** (`rgbToGrayscaleKernel`) -- one CUDA thread per
pixel. Each thread reads its pixel's R, G, B bytes and writes a single
luma-weighted grayscale byte:

```
gray = 0.299*R + 0.587*G + 0.114*B
```

**2. Box blur** (`boxBlurKernel`) -- one CUDA thread per output pixel.
Each thread averages the `(2*radius+1) x (2*radius+1)` window of
grayscale values centered on its pixel, clamping the window at image
borders (rather than sampling out of bounds or wrapping around), and
writes the averaged value into all three channels of the output image
so the result is directly viewable as a normal RGB image. `radius` is
a runtime CLI argument (default 2, i.e. a 5x5 window).

Both kernels use a 2D grid of 16x16 thread blocks sized to cover the
image dimensions, which is a natural fit for 2D pixel data.

Per-image GPU execution time is measured with `cudaEvent` timestamps
around the two kernel launches and printed as part of the run log;
`main.cpp` also totals this across the whole batch.

## Building and Running

See `INSTALL` for toolchain requirements. Short version:

```
make build
./run.sh
```

`run.sh` processes every image in `data/input/`, writes results to
`data/output/`, and saves a timestamped log
(`proof_of_execution_*.log`) of the run -- useful as an execution
record for grading.

To process a different folder, or change the blur radius:

```
./bin/cuda_batch_filter <input_dir> <output_dir> [blur_radius]
```

## Regenerating / replacing the sample dataset

`data/input/` ships with 200 procedurally generated PPM images so the
project runs immediately after cloning. To regenerate them (or change
the count/size), edit the constants at the top of `gen_images.py` and
re-run it:

```
python3 gen_images.py
```

Any other `.ppm` files (converted from JPG/PNG with e.g. ImageMagick's
`convert in.jpg out.ppm`) can also be dropped into an input directory
and processed the same way.

## Design notes / what I'd do differently

- The box blur re-reads overlapping pixels from global memory for
  every thread rather than staging a tile into shared memory first;
  for these small (64x64) sample images that's negligible, but on
  larger images a shared-memory tiled blur would cut redundant global
  memory traffic significantly.
- Grayscale and blur are two separate kernel launches (with an
  intermediate buffer) rather than fused into one kernel. Keeping them
  separate made both easier to reason about and test independently,
  at the cost of one extra global memory round-trip per pixel.
- PPM was chosen over a compressed format (PNG/JPEG) specifically to
  avoid pulling in an external image library -- it keeps the build
  dependency-free, at the cost of larger files on disk.

_(This section is a starting point -- swap in your own observations
once you've actually built and run this against your GPU, including
timings you saw and anything that surprised you.)_
