// Golden reference model for edge_accel algorithms.
// Pure standard C++ (no HLS types) so it can be shared between the
// Vitis HLS testbench and the plain-g++ verification harness.
#pragma once

#include <cstddef>
#include <vector>
#include <algorithm>

struct GrayImg {
    int rows;
    int cols;
    std::vector<unsigned char> d;

    GrayImg() : rows(0), cols(0) {}
    GrayImg(int r, int c) : rows(r), cols(c), d((size_t)r * (size_t)c, 0) {}

    unsigned char& at(int r, int c) { return d[(size_t)r * (size_t)cols + (size_t)c]; }
    unsigned char  at(int r, int c) const { return d[(size_t)r * (size_t)cols + (size_t)c]; }
};

static inline int clip255(int v) {
    if (v < 0)   return 0;
    if (v > 255) return 255;
    return v;
}

// Project default gamma: hardware-friendly sqrt(x/255)*255 piecewise map.
static inline int gamma_val(int xi) {
    if (xi < 16)  return clip255(xi * 3);
    if (xi < 64)  return clip255(32 + ((xi - 16) * 2));
    if (xi < 144) return clip255(128 + ((xi - 64) * 3) / 4);
    return clip255(188 + ((xi - 144) * 67) / 111);
}

// Gaussian discrete sigma quantization (Q4.8 thresholds; MUST match edge_accel.cpp).
static inline int gauss_sigma_idx_3x3(int sigma_q48) { return (sigma_q48 < 256) ? 0 : 1; }
static inline int gauss_sigma_idx_5x5(int sigma_q48) { return (sigma_q48 < 320) ? 0 : (sigma_q48 < 448) ? 1 : 2; }

static GrayImg golden_gaussian3x3(const GrayImg& in, int sigma_idx) {
    int corner = 1;
    int edge   = (sigma_idx == 0) ? 2 : 1;
    int center = (sigma_idx == 0) ? 4 : 1;
    int div    = (sigma_idx == 0) ? 16 : 9;
    GrayImg o(in.rows, in.cols);
    for (int r = 0; r < in.rows; ++r)
        for (int c = 0; c < in.cols; ++c) {
            if (r == 0 || c == 0 || r == in.rows - 1 || c == in.cols - 1) {
                o.at(r, c) = in.at(r, c); // replicate border
            } else {
                int sum =
                    corner * ((int)in.at(r - 1, c - 1) + (int)in.at(r - 1, c + 1)
                              + (int)in.at(r + 1, c - 1) + (int)in.at(r + 1, c + 1)) +
                    edge   * ((int)in.at(r - 1, c) + (int)in.at(r, c - 1)
                              + (int)in.at(r, c + 1) + (int)in.at(r + 1, c)) +
                    center * (int)in.at(r, c);
                o.at(r, c) = (unsigned char)(sum / div);
            }
        }
    return o;
}

static GrayImg golden_gaussian5x5(const GrayImg& in, int sigma_idx) {
    static const int coef[3][25] = {
        { 1, 4, 6, 4, 1,  4,16,24,16, 4,  6,24,36,24, 6,  4,16,24,16, 4,  1, 4, 6, 4, 1 },
        { 1, 2, 3, 2, 1,  2, 4, 6, 4, 2,  3, 6, 9, 6, 3,  2, 4, 6, 4, 2,  1, 2, 3, 2, 1 },
        { 1, 1, 1, 1, 1,  1, 2, 2, 2, 1,  1, 2, 4, 2, 1,  1, 2, 2, 2, 1,  1, 1, 1, 1, 1 },
    };
    int div = (sigma_idx == 0) ? 256 : (sigma_idx == 1) ? 81 : 36;
    GrayImg o(in.rows, in.cols);
    for (int r = 0; r < in.rows; ++r)
        for (int c = 0; c < in.cols; ++c) {
            if (r < 2 || c < 2 || r >= in.rows - 2 || c >= in.cols - 2) {
                o.at(r, c) = in.at(r, c); // 2px replicate border
            } else {
                int sum = 0;
                for (int dr = -2; dr <= 2; ++dr)
                    for (int dc = -2; dc <= 2; ++dc)
                        sum += coef[sigma_idx][(dr + 2) * 5 + (dc + 2)] * (int)in.at(r + dr, c + dc);
                o.at(r, c) = (unsigned char)(sum / div);
            }
        }
    return o;
}

static GrayImg golden_gamma(const GrayImg& in) {
    GrayImg o(in.rows, in.cols);
    for (int r = 0; r < in.rows; ++r)
        for (int c = 0; c < in.cols; ++c)
            o.at(r, c) = (unsigned char)gamma_val(in.at(r, c));
    return o;
}

static GrayImg golden_sharpen(const GrayImg& in, int k) {
    // Unsharp, strength k in Q0.8: out = ((256 + 4k)*center - k*sum(4 neighbors)) >> 8.
    GrayImg o(in.rows, in.cols);
    for (int r = 0; r < in.rows; ++r)
        for (int c = 0; c < in.cols; ++c) {
            if (r == 0 || c == 0 || r == in.rows - 1 || c == in.cols - 1) {
                o.at(r, c) = in.at(r, c); // replicate border
            } else {
                int v = ((256 + 4 * k) * (int)in.at(r, c)
                         - k * ((int)in.at(r - 1, c) + (int)in.at(r + 1, c)
                                + (int)in.at(r, c - 1) + (int)in.at(r, c + 1))) >> 8;
                o.at(r, c) = (unsigned char)clip255(v);
            }
        }
    return o;
}

static GrayImg golden_sobel(const GrayImg& in, int thr) {
    GrayImg o(in.rows, in.cols);
    for (int r = 0; r < in.rows; ++r)
        for (int c = 0; c < in.cols; ++c) {
            if (r == 0 || c == 0 || r == in.rows - 1 || c == in.cols - 1) {
                o.at(r, c) = 0;
            } else {
                int gx = -(int)in.at(r - 1, c - 1) + (int)in.at(r - 1, c + 1)
                         - 2 * (int)in.at(r, c - 1) + 2 * (int)in.at(r, c + 1)
                         - (int)in.at(r + 1, c - 1) + (int)in.at(r + 1, c + 1);
                int gy = -(int)in.at(r - 1, c - 1) - 2 * (int)in.at(r - 1, c) - (int)in.at(r - 1, c + 1)
                         + (int)in.at(r + 1, c - 1) + 2 * (int)in.at(r + 1, c) + (int)in.at(r + 1, c + 1);
                int mag = (gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy);
                o.at(r, c) = (mag >= thr) ? 255 : 0;
            }
        }
    return o;
}

static GrayImg golden_roberts(const GrayImg& in, int thr) {
    GrayImg o(in.rows, in.cols);
    for (int r = 0; r < in.rows; ++r)
        for (int c = 0; c < in.cols; ++c) {
            if (r == in.rows - 1 || c == in.cols - 1) {
                o.at(r, c) = 0;
            } else {
                int gx = (int)in.at(r, c) - (int)in.at(r + 1, c + 1);
                int gy = (int)in.at(r + 1, c) - (int)in.at(r, c + 1);
                int mag = (gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy);
                o.at(r, c) = (mag >= thr) ? 255 : 0;
            }
        }
    return o;
}

static GrayImg golden_dilate(const GrayImg& in, int k) {
    int rad = k / 2;
    GrayImg o(in.rows, in.cols);
    for (int r = 0; r < in.rows; ++r)
        for (int c = 0; c < in.cols; ++c) {
            int m = 0;
            for (int dr = -rad; dr <= rad; ++dr)
                for (int dc = -rad; dc <= rad; ++dc) {
                    int rr = r + dr, cc = c + dc;
                    int v = 0;
                    if (rr >= 0 && rr < in.rows && cc >= 0 && cc < in.cols) v = in.at(rr, cc);
                    if (v > m) m = v;
                }
            o.at(r, c) = (unsigned char)m;
        }
    return o;
}

static GrayImg golden_erode(const GrayImg& in, int k) {
    int rad = k / 2;
    GrayImg o(in.rows, in.cols);
    for (int r = 0; r < in.rows; ++r)
        for (int c = 0; c < in.cols; ++c) {
            int m = 255;
            for (int dr = -rad; dr <= rad; ++dr)
                for (int dc = -rad; dc <= rad; ++dc) {
                    int rr = r + dr, cc = c + dc;
                    int v = 255;
                    if (rr >= 0 && rr < in.rows && cc >= 0 && cc < in.cols) v = in.at(rr, cc);
                    if (v < m) m = v;
                }
            o.at(r, c) = (unsigned char)m;
        }
    return o;
}

static GrayImg golden_closing(const GrayImg& in, int k, int iters) {
    // Closing = (dilate then erode) x iters, kernel k, identity OOB padding.
    GrayImg cur = in;
    for (int i = 0; i < iters; ++i) {
        cur = golden_dilate(cur, k);
        cur = golden_erode(cur, k);
    }
    return cur;
}

static GrayImg golden_opening(const GrayImg& in, int k, int iters) {
    // Opening = (erode then dilate) x iters, kernel k, identity OOB padding.
    GrayImg cur = in;
    for (int i = 0; i < iters; ++i) {
        cur = golden_erode(cur, k);
        cur = golden_dilate(cur, k);
    }
    return cur;
}

static GrayImg golden_median3x3(const GrayImg& in) {
    GrayImg o(in.rows, in.cols);
    for (int r = 0; r < in.rows; ++r)
        for (int c = 0; c < in.cols; ++c) {
            if (r == 0 || c == 0 || r == in.rows - 1 || c == in.cols - 1) {
                o.at(r, c) = in.at(r, c); // 1px replicate border
            } else {
                int v[9], k = 0;
                for (int dr = -1; dr <= 1; ++dr)
                    for (int dc = -1; dc <= 1; ++dc)
                        v[k++] = (int)in.at(r + dr, c + dc);
                std::sort(v, v + 9);
                o.at(r, c) = (unsigned char)v[4];
            }
        }
    return o;
}

static GrayImg golden_canny(const GrayImg& in, int low, int high) {
    // Full Canny, zero-padding at every stage (OOB = 0): Gaussian 3x3 -> Gx/Gy
    // -> |Gx|+|Gy| + 4-sector dir -> NMS -> double threshold -> forward hysteresis.
    const int R = in.rows, C = in.cols;

    std::vector<unsigned char> blur((size_t)R * C, 0);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            int sum = 0;
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc) {
                    int rr = r + dr, cc = c + dc, v = 0;
                    if (rr >= 0 && rr < R && cc >= 0 && cc < C) v = in.at(rr, cc);
                    int g = (dr == 0 && dc == 0) ? 4 : ((dr == 0 || dc == 0) ? 2 : 1);
                    sum += g * v;
                }
            blur[r * C + c] = (unsigned char)(sum >> 4);
        }

    std::vector<int> mag((size_t)R * C, 0), dir((size_t)R * C, 0);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            int gx = 0, gy = 0;
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc) {
                    int rr = r + dr, cc = c + dc, v = 0;
                    if (rr >= 0 && rr < R && cc >= 0 && cc < C) v = blur[rr * C + cc];
                    if (dr == -1)      { if (dc == -1) { gx -= v; gy -= v; } else if (dc == 1) { gx += v; gy -= v; } else gy -= 2 * v; }
                    else if (dr == 0)  { if (dc == -1) gx -= 2 * v; else if (dc == 1) gx += 2 * v; }
                    else               { if (dc == -1) { gx -= v; gy += v; } else if (dc == 1) { gx += v; gy += v; } else gy += 2 * v; }
                }
            int ax = gx < 0 ? -gx : gx;
            int ay = gy < 0 ? -gy : gy;
            int s;
            if (ay > 2 * ax)      s = 0;
            else if (ax > 2 * ay) s = 2;
            else                  s = (gx * gy > 0) ? 1 : 3;
            mag[r * C + c] = ax + ay;
            dir[r * C + c] = s;
        }

    std::vector<int> thinned((size_t)R * C, 0);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            int m = mag[r * C + c], s = dir[r * C + c];
            int n1 = 0, n2 = 0;
            if (s == 0)      { n1 = (r > 0) ? mag[(r - 1) * C + c] : 0;     n2 = (r < R - 1) ? mag[(r + 1) * C + c] : 0; }
            else if (s == 2) { n1 = (c > 0) ? mag[r * C + c - 1] : 0;       n2 = (c < C - 1) ? mag[r * C + c + 1] : 0; }
            else if (s == 1) { n1 = (r > 0 && c > 0) ? mag[(r - 1) * C + c - 1] : 0; n2 = (r < R - 1 && c < C - 1) ? mag[(r + 1) * C + c + 1] : 0; }
            else             { n1 = (r > 0 && c < C - 1) ? mag[(r - 1) * C + c + 1] : 0; n2 = (r < R - 1 && c > 0) ? mag[(r + 1) * C + c - 1] : 0; }
            thinned[r * C + c] = (m >= n1 && m >= n2) ? m : 0;
        }

    GrayImg o(R, C);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            // Match the streaming HLS: right 3 columns and bottom 3 rows are
            // not produced by the 3-stage-delay pipeline (zeroed).
            if (r >= R - 3 || c >= C - 3) { o.at(r, c) = 0; continue; }
            int m = thinned[r * C + c];
            int edge;
            if (m >= high)      edge = 1;
            else if (m < low)   edge = 0;
            else {
                int left = (c > 0) ? (o.at(r, c - 1) == 255 ? 1 : 0) : 0;
                int tl   = (r > 0 && c > 0) ? (o.at(r - 1, c - 1) == 255 ? 1 : 0) : 0;
                int top  = (r > 0) ? (o.at(r - 1, c) == 255 ? 1 : 0) : 0;
                int tr   = (r > 0 && c < C - 3) ? (o.at(r - 1, c + 1) == 255 ? 1 : 0) : 0;
                edge = (left || tl || top || tr) ? 1 : 0;
            }
            o.at(r, c) = edge ? 255 : 0;
        }
    return o;
}

static GrayImg golden_clahe(const GrayImg& in, int clip_limit) {
    // CLAHE, fixed 8x8 tiles, clip_limit in Q0.8, bilinear interpolation between
    // 4 neighbouring tiles (fixed-point Q8 weights).
    const int TILES = 8;
    const int R = in.rows, C = in.cols;
    const int tile_h = R / TILES, tile_w = C / TILES;

    int cdf[TILES][TILES][256];
    for (int ty = 0; ty < TILES; ++ty)
        for (int tx = 0; tx < TILES; ++tx) {
            int hist[256] = {0};
            for (int r = ty * tile_h; r < (ty + 1) * tile_h; ++r)
                for (int c = tx * tile_w; c < (tx + 1) * tile_w; ++c)
                    hist[in.at(r, c)]++;
            int npix = tile_h * tile_w;
            int clip = (clip_limit * npix) >> 8;
            if (clip < 1) clip = 1;
            int excess = 0;
            for (int b = 0; b < 256; ++b)
                if (hist[b] > clip) { excess += hist[b] - clip; hist[b] = clip; }
            int redist = excess / 256;
            int leftover = excess % 256;
            for (int b = 0; b < 256; ++b) { hist[b] += redist; if (b < leftover) hist[b] += 1; }
            int cum = 0, cummin = npix + 1;
            for (int b = 0; b < 256; ++b) { cum += hist[b]; if (cum > 0 && cum < cummin) cummin = cum; }
            cum = 0;
            for (int b = 0; b < 256; ++b) {
                cum += hist[b];
                int val = (cummin > npix) ? b : ((cum - cummin) * 255 / (npix - cummin));
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                cdf[ty][tx][b] = val;
            }
        }

    GrayImg o(R, C);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            int p = in.at(r, c);
            int tx = c / tile_w, rx = c % tile_w;
            int ty = r / tile_h;
            int tx2 = (tx + 1 < TILES) ? tx + 1 : tx;
            int wx = (rx * 256) / tile_w;   // Q8
            int c0 = cdf[ty][tx][p], c1 = cdf[ty][tx2][p];
            int v = ((256 - wx) * c0 + wx * c1) / 256;
            o.at(r, c) = (unsigned char)v;
        }
    return o;
}

static GrayImg golden_median5x5(const GrayImg& in) {
    GrayImg o(in.rows, in.cols);
    for (int r = 0; r < in.rows; ++r)
        for (int c = 0; c < in.cols; ++c) {
            if (r < 2 || c < 2 || r >= in.rows - 2 || c >= in.cols - 2) {
                o.at(r, c) = in.at(r, c); // 2px replicate border
            } else {
                int v[25], k = 0;
                for (int dr = -2; dr <= 2; ++dr)
                    for (int dc = -2; dc <= 2; ++dc)
                        v[k++] = (int)in.at(r + dr, c + dc);
                std::sort(v, v + 25);
                o.at(r, c) = (unsigned char)v[12];
            }
        }
    return o;
}
