#include "ppm_io.h"

#include <fstream>
#include <iostream>

namespace {

// Reads whitespace-separated PPM header tokens, skipping "#" comments
// that run to end-of-line, exactly as the PPM spec requires.
bool readToken(std::istream &in, std::string &token) {
    token.clear();
    int c = in.get();

    // Skip whitespace and comment lines.
    while (true) {
        while (c != EOF && std::isspace(c)) {
            c = in.get();
        }
        if (c == '#') {
            while (c != EOF && c != '\n') {
                c = in.get();
            }
            continue;
        }
        break;
    }

    if (c == EOF) {
        return false;
    }

    while (c != EOF && !std::isspace(c)) {
        token.push_back(static_cast<char>(c));
        c = in.get();
    }
    return !token.empty();
}

} // namespace

bool readPPM(const std::string &path, Image &img) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "ERROR: could not open '" << path << "' for reading\n";
        return false;
    }

    std::string magic;
    if (!readToken(in, magic) || magic != "P6") {
        std::cerr << "ERROR: '" << path << "' is not a binary PPM (P6) file\n";
        return false;
    }

    std::string widthTok, heightTok, maxvalTok;
    if (!readToken(in, widthTok) || !readToken(in, heightTok) || !readToken(in, maxvalTok)) {
        std::cerr << "ERROR: malformed PPM header in '" << path << "'\n";
        return false;
    }

    int width = std::stoi(widthTok);
    int height = std::stoi(heightTok);
    int maxval = std::stoi(maxvalTok);
    if (width <= 0 || height <= 0) {
        std::cerr << "ERROR: invalid dimensions in '" << path << "'\n";
        return false;
    }
    if (maxval != 255) {
        std::cerr << "ERROR: '" << path << "' has maxval " << maxval
                   << "; only 8-bit-per-channel PPMs (maxval 255) are supported\n";
        return false;
    }

    // Per the PPM spec exactly one whitespace byte separates the header from
    // the raw pixel data -- but readToken() above already consumed it while
    // detecting the end of the maxval token, so the stream is correctly
    // positioned at the start of pixel data already; nothing more to skip.

    const size_t numBytes = static_cast<size_t>(width) * height * 3;
    std::vector<unsigned char> pixels(numBytes);
    in.read(reinterpret_cast<char *>(pixels.data()), static_cast<std::streamsize>(numBytes));
    if (static_cast<size_t>(in.gcount()) != numBytes) {
        std::cerr << "ERROR: '" << path << "' is truncated (expected " << numBytes
                   << " bytes of pixel data)\n";
        return false;
    }

    img.width = width;
    img.height = height;
    img.pixels = std::move(pixels);
    return true;
}

bool writePPM(const std::string &path, const Image &img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "ERROR: could not open '" << path << "' for writing\n";
        return false;
    }

    out << "P6\n" << img.width << ' ' << img.height << "\n255\n";
    out.write(reinterpret_cast<const char *>(img.pixels.data()),
               static_cast<std::streamsize>(img.pixels.size()));
    return static_cast<bool>(out);
}
