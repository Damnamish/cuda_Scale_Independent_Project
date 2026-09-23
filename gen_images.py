#!/usr/bin/env python3
"""Generate a batch of small synthetic PPM (P6) images as sample input data.

No external image libraries required (pure standard library), so this
can be re-run on any machine with Python 3 to regenerate/expand the
dataset. Images are procedurally generated (gradients, checkerboards,
rings, stripes, noise) purely to provide bulk, varied pixel data for
the CUDA batch image filter to chew on -- not sourced from anywhere.
"""
import os
import random
import struct

WIDTH = 64
HEIGHT = 64
COUNT = 200
OUT_DIR = "data/input"

random.seed(42)


def write_ppm(path, width, height, pixels):
    with open(path, "wb") as f:
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        f.write(header)
        f.write(pixels)


def clamp(v):
    return max(0, min(255, int(v)))


def gen_gradient(w, h, c1, c2, horizontal=True):
    buf = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            t = (x / (w - 1)) if horizontal else (y / (h - 1))
            r = clamp(c1[0] * (1 - t) + c2[0] * t)
            g = clamp(c1[1] * (1 - t) + c2[1] * t)
            b = clamp(c1[2] * (1 - t) + c2[2] * t)
            i = (y * w + x) * 3
            buf[i] = r
            buf[i + 1] = g
            buf[i + 2] = b
    return bytes(buf)


def gen_checkerboard(w, h, c1, c2, tile):
    buf = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            on = ((x // tile) + (y // tile)) % 2 == 0
            col = c1 if on else c2
            i = (y * w + x) * 3
            buf[i], buf[i + 1], buf[i + 2] = col
    return bytes(buf)


def gen_rings(w, h, c1, c2, freq):
    buf = bytearray(w * h * 3)
    cx, cy = w / 2.0, h / 2.0
    maxd = (cx ** 2 + cy ** 2) ** 0.5
    for y in range(h):
        for x in range(w):
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / maxd
            t = (0.5 + 0.5 * __import__("math").sin(d * freq * 6.28318))
            r = clamp(c1[0] * (1 - t) + c2[0] * t)
            g = clamp(c1[1] * (1 - t) + c2[1] * t)
            b = clamp(c1[2] * (1 - t) + c2[2] * t)
            i = (y * w + x) * 3
            buf[i] = r
            buf[i + 1] = g
            buf[i + 2] = b
    return bytes(buf)


def gen_stripes(w, h, c1, c2, thickness, diagonal=False):
    buf = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            k = (x + y) if diagonal else x
            on = (k // thickness) % 2 == 0
            col = c1 if on else c2
            i = (y * w + x) * 3
            buf[i], buf[i + 1], buf[i + 2] = col
    return bytes(buf)


def gen_noise(w, h, base):
    buf = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 3
            buf[i] = clamp(base[0] + random.randint(-60, 60))
            buf[i + 1] = clamp(base[1] + random.randint(-60, 60))
            buf[i + 2] = clamp(base[2] + random.randint(-60, 60))
    return bytes(buf)


def random_color():
    return (random.randint(20, 235), random.randint(20, 235), random.randint(20, 235))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    generators = ["gradient_h", "gradient_v", "checker", "rings", "stripes", "diag_stripes", "noise"]

    for idx in range(COUNT):
        kind = generators[idx % len(generators)]
        c1 = random_color()
        c2 = random_color()

        if kind == "gradient_h":
            px = gen_gradient(WIDTH, HEIGHT, c1, c2, horizontal=True)
        elif kind == "gradient_v":
            px = gen_gradient(WIDTH, HEIGHT, c1, c2, horizontal=False)
        elif kind == "checker":
            tile = random.choice([4, 8, 16])
            px = gen_checkerboard(WIDTH, HEIGHT, c1, c2, tile)
        elif kind == "rings":
            freq = random.uniform(2.0, 6.0)
            px = gen_rings(WIDTH, HEIGHT, c1, c2, freq)
        elif kind == "stripes":
            thick = random.choice([2, 4, 6, 8])
            px = gen_stripes(WIDTH, HEIGHT, c1, c2, thick, diagonal=False)
        elif kind == "diag_stripes":
            thick = random.choice([2, 4, 6, 8])
            px = gen_stripes(WIDTH, HEIGHT, c1, c2, thick, diagonal=True)
        else:  # noise
            px = gen_noise(WIDTH, HEIGHT, c1)

        fname = f"img_{idx:03d}_{kind}.ppm"
        write_ppm(os.path.join(OUT_DIR, fname), WIDTH, HEIGHT, px)

    print(f"Wrote {COUNT} images ({WIDTH}x{HEIGHT}) to {OUT_DIR}/")


if __name__ == "__main__":
    main()
