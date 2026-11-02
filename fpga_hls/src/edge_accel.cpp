#include "edge_accel.hpp"

/*
 * Programmable multi-stage streaming pipeline.
 *
 *   s_axis(axis8_t) -> PRE0 -> PRE1 -> PRE2 -> EDGE -> POST0 -> POST1 -> m_axis
 *
 * Every stage is a streaming kernel with II=1 and line-buffer + sliding-window
 * for the 3x3 operators.  Internal stages exchange `pixel_t` (data + TUSER/
 * TLAST); only the top-level ports use `axis8_t`, converted by adapters.
 *
 * Phase 1 implements: GAMMA (writable LUT), GAUSSIAN (3x3), SHARPEN (3x3),
 * ROBERTS, SOBEL, OPENING/CLOSING (3x3, 1 iteration).  MEDIAN / CLAHE / CANNY
 * are reserved and rejected by validation with ERR_UNSUPPORTED.
 */

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */
static ap_uint<8> clip_u8(int v)
{
#pragma HLS INLINE
    if (v < 0)   return 0;
    if (v > 255) return 255;
    return (ap_uint<8>)v;
}

static inline void emit(
    pixel_stream_t& out,
    ap_uint<8> data,
    ap_uint<1> user,
    ap_uint<1> last
)
{
#pragma HLS INLINE
    pixel_t q;
    q.data = data;
    q.user = user;
    q.last = last;
    out.write(q);
}

/* -------------------------------------------------------------------------
 * Adapters (axis8_t <-> pixel_t)
 * ---------------------------------------------------------------------- */
static void axis_to_pixel(
    hls::stream<axis8_t>& in,
    pixel_stream_t& out,
    int rows,
    int cols
)
{
    ap_uint<12> R = (ap_uint<12>)rows;
    ap_uint<12> C = (ap_uint<12>)cols;
    ap_uint<12> Cm1 = C - 1;
    for (ap_uint<12> r = 0; r < R; ++r) {
        for (ap_uint<12> c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            axis8_t p = in.read();
            emit(out, p.data, (r == 0 && c == 0) ? 1 : 0, (c == Cm1) ? 1 : 0);
        }
    }
}

static void pixel_to_axis(
    pixel_stream_t& in,
    hls::stream<axis8_t>& out,
    int rows,
    int cols
)
{
    ap_uint<12> R = (ap_uint<12>)rows;
    ap_uint<12> C = (ap_uint<12>)cols;
    for (ap_uint<12> r = 0; r < R; ++r) {
        for (ap_uint<12> c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            pixel_t p = in.read();

            axis8_t q;
            q.data = p.data;
            q.keep = -1;
            q.strb = -1;
            // Positional sideband: TUSER=frame start, TLAST=frame end (single
            // full-frame AXI-DMA packet; do NOT forward the row-end TLAST that
            // the internal stages carry, or the S2MM channel would truncate at
            // the first row boundary).
            q.user = (r == 0 && c == 0) ? 1 : 0;
            q.last = (r == R - 1 && c == C - 1) ? 1 : 0;
            q.id = 0;
            q.dest = 0;

            out.write(q);
        }
    }
}

/* -------------------------------------------------------------------------
 * Bypass (identity) stage 鈥?regenerates TUSER/TLAST positionally.
 * ---------------------------------------------------------------------- */
static void passthrough(pixel_stream_t& in, pixel_stream_t& out, int rows, int cols)
{
    ap_uint<12> R = (ap_uint<12>)rows;
    ap_uint<12> C = (ap_uint<12>)cols;
    ap_uint<12> Cm1 = C - 1;
    for (ap_uint<12> r = 0; r < R; ++r) {
        for (ap_uint<12> c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            pixel_t p = in.read();
            emit(out, p.data, (r == 0 && c == 0) ? 1 : 0, (c == Cm1) ? 1 : 0);
        }
    }
}

/* -------------------------------------------------------------------------
 * GAMMA 鈥?point op through the PS-writable 256-entry LUT.
 * ---------------------------------------------------------------------- */
static void gamma_stream(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols,
    const ap_uint<8> lut[256]
)
{
    ap_uint<12> R = (ap_uint<12>)rows;
    ap_uint<12> C = (ap_uint<12>)cols;
    ap_uint<12> Cm1 = C - 1;
    for (ap_uint<12> r = 0; r < R; ++r) {
        for (ap_uint<12> c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            pixel_t p = in.read();
            emit(out, lut[p.data], (r == 0 && c == 0) ? 1 : 0, (c == Cm1) ? 1 : 0);
        }
    }
}

/* -------------------------------------------------------------------------
 * GAUSSIAN 3x3 (border-replicated), II=1.
 * `sigma_idx` selects a discrete preset:
 *   0 -> {1,2,1;2,4,2;1,2,1}/16  (sigma ~0.8)
 *   1 -> {1,1,1;1,1,1;1,1,1}/9   (sigma ~1.5)
 * ---------------------------------------------------------------------- */
static void gaussian_stream(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols,
    int sigma_idx
)
{
    static ap_uint<8> line0[MAX_WIDTH];
    static ap_uint<8> line1[MAX_WIDTH];
    static ap_uint<8> out_line[MAX_WIDTH];

#pragma HLS BIND_STORAGE variable=line0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=line1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=out_line type=ram_2p impl=bram

    /* Top border: replicate input row 0. */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();

        line0[c] = 0;
        line1[c] = p.data;

        emit(out, p.data, (c == 0) ? 1 : 0, (c == cols - 1) ? 1 : 0);
    }

    /* Read row 1 into the line buffers (no output yet). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();

        line0[c] = line1[c];
        line1[c] = p.data;
    }

    /* Interior rows: reading row r produces output row r-1. */
    for (int r = 2; r < rows; ++r) {

        ap_uint<8> w00 = 0, w01 = 0, w02 = 0;
        ap_uint<8> w10 = 0, w11 = 0, w12 = 0;
        ap_uint<8> w20 = 0, w21 = 0, w22 = 0;

        ap_uint<8> left_border  = 0;
        ap_uint<8> right_border = 0;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1

            pixel_t p = in.read();
            ap_uint<8> pixel = p.data;

            ap_uint<8> top = line0[c];
            ap_uint<8> mid = line1[c];

            line0[c] = mid;
            line1[c] = pixel;

            if (c == 0)        left_border  = mid;
            if (c == cols - 1) right_border = mid;

            if (c == 0) {
                w00 = 0; w01 = 0;
                w10 = 0; w11 = 0;
                w20 = 0; w21 = 0;
            } else {
                w00 = w01; w01 = w02;
                w10 = w11; w11 = w12;
                w20 = w21; w21 = w22;
            }

            w02 = top;
            w12 = mid;
            w22 = pixel;

            out_line[c] = 0;
            if (c >= 2) {
                int corner = 1;
                int edge   = (sigma_idx == 0) ? 2 : 1;
                int center = (sigma_idx == 0) ? 4 : 1;
                int sum =
                    corner * ((int)w00 + (int)w02 + (int)w20 + (int)w22) +
                    edge   * ((int)w01 + (int)w10 + (int)w12 + (int)w21) +
                    center * (int)w11;
                out_line[c - 1] = (sigma_idx == 0)
                    ? (ap_uint<8>)(sum >> 4)
                    : (ap_uint<8>)(sum / 9);
            }
        }

        out_line[0] = left_border;
        out_line[cols - 1] = right_border;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
            emit(out, out_line[c], 0, (c == cols - 1) ? 1 : 0);
        }
    }

    /* Bottom border: replicate input row rows-1 (stored in line1). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        emit(out, line1[c], 0, (c == cols - 1) ? 1 : 0);
    }
}

/* -------------------------------------------------------------------------
 * GAUSSIAN 5x5 (2px border-replicated), II=1.
 * `sigma_idx` selects a discrete preset (integer kernels, divisor = sum):
 *   0 -> {1,4,6,4,1;4,16,24,16,4;6,24,36,24,6;4,16,24,16,4;1,4,6,4,1}/256 (sigma ~1.0)
 *   1 -> {1,2,3,2,1;2,4,6,4,2;3,6,9,6,3;2,4,6,4,2;1,2,3,2,1}/81      (sigma ~1.5)
 *   2 -> {1,1,1,1,1;1,2,2,2,1;1,2,4,2,1;1,2,2,2,1;1,1,1,1,1}/36      (sigma ~2.0)
 * ---------------------------------------------------------------------- */
static void gaussian5x5_stream(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols,
    int sigma_idx
)
{
    static ap_uint<8> line0[MAX_WIDTH];
    static ap_uint<8> line1[MAX_WIDTH];
    static ap_uint<8> line2[MAX_WIDTH];
    static ap_uint<8> line3[MAX_WIDTH];
    static ap_uint<8> out_line[MAX_WIDTH];

#pragma HLS BIND_STORAGE variable=line0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=line1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=line2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=line3 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=out_line type=ram_2p impl=bram

    static const int coef[3][25] = {
        { 1, 4, 6, 4, 1,  4,16,24,16, 4,  6,24,36,24, 6,  4,16,24,16, 4,  1, 4, 6, 4, 1 },
        { 1, 2, 3, 2, 1,  2, 4, 6, 4, 2,  3, 6, 9, 6, 3,  2, 4, 6, 4, 2,  1, 2, 3, 2, 1 },
        { 1, 1, 1, 1, 1,  1, 2, 2, 2, 1,  1, 2, 4, 2, 1,  1, 2, 2, 2, 1,  1, 1, 1, 1, 1 },
    };
#pragma HLS ARRAY_PARTITION variable=coef complete dim=1

    /* Top border row 0 (replicate): output unchanged, seed all line buffers. */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();
        line0[c] = p.data; line1[c] = p.data; line2[c] = p.data; line3[c] = p.data;
        emit(out, p.data, (c == 0) ? 1 : 0, (c == cols - 1) ? 1 : 0);
    }
    /* Top border row 1 (replicate): output unchanged. */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();
        line0[c] = line1[c]; line1[c] = line2[c]; line2[c] = line3[c]; line3[c] = p.data;
        emit(out, p.data, 0, (c == cols - 1) ? 1 : 0);
    }
    /* Read row 2 (no output). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();
        line0[c] = line1[c]; line1[c] = line2[c]; line2[c] = line3[c]; line3[c] = p.data;
    }
    /* Read row 3 (no output). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();
        line0[c] = line1[c]; line1[c] = line2[c]; line2[c] = line3[c]; line3[c] = p.data;
    }

    ap_uint<8> w[5][5];
#pragma HLS ARRAY_PARTITION variable=w complete dim=0

    /* Interior rows: read row r (r=4..rows-1), output row r-2. */
    for (int r = 4; r < rows; ++r) {

        for (int i = 0; i < 5; ++i) {
#pragma HLS UNROLL
            for (int j = 0; j < 5; ++j) {
#pragma HLS UNROLL
                w[i][j] = 0;
            }
        }

        ap_uint<8> left0 = 0, left1 = 0, right1 = 0, right0 = 0;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1

            pixel_t p = in.read();
            ap_uint<8> pixel = p.data;

            ap_uint<8> r4 = line0[c];   // row r-4
            ap_uint<8> r3 = line1[c];   // row r-3
            ap_uint<8> r2 = line2[c];   // row r-2 (output row)
            ap_uint<8> r1 = line3[c];   // row r-1
            ap_uint<8> r0 = pixel;      // row r

            line0[c] = r3; line1[c] = r2; line2[c] = r1; line3[c] = r0;

            if (c == 0)        left0 = r2;
            if (c == 1)        left1 = r2;
            if (c == cols - 2) right1 = r2;
            if (c == cols - 1) right0 = r2;

            /* Shift the 5x5 window left; new column = {r4,r3,r2,r1,r0}. */
            w[0][0]=w[0][1]; w[0][1]=w[0][2]; w[0][2]=w[0][3]; w[0][3]=w[0][4]; w[0][4]=r4;
            w[1][0]=w[1][1]; w[1][1]=w[1][2]; w[1][2]=w[1][3]; w[1][3]=w[1][4]; w[1][4]=r3;
            w[2][0]=w[2][1]; w[2][1]=w[2][2]; w[2][2]=w[2][3]; w[2][3]=w[2][4]; w[2][4]=r2;
            w[3][0]=w[3][1]; w[3][1]=w[3][2]; w[3][2]=w[3][3]; w[3][3]=w[3][4]; w[3][4]=r1;
            w[4][0]=w[4][1]; w[4][1]=w[4][2]; w[4][2]=w[4][3]; w[4][3]=w[4][4]; w[4][4]=r0;

            out_line[c] = 0;
            if (c >= 4) {
                int sum = 0;
                for (int i = 0; i < 5; ++i) {
#pragma HLS UNROLL
                    for (int j = 0; j < 5; ++j) {
#pragma HLS UNROLL
                        sum += coef[sigma_idx][i * 5 + j] * (int)w[i][j];
                    }
                }
                int v = (sigma_idx == 0) ? (sum >> 8)
                      : (sigma_idx == 1) ? (sum / 81)
                      : (sum / 36);
                out_line[c - 2] = (ap_uint<8>)v;
            }
        }

        out_line[0] = left0;
        out_line[1] = left1;
        out_line[cols - 2] = right1;
        out_line[cols - 1] = right0;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
            emit(out, out_line[c], 0, (c == cols - 1) ? 1 : 0);
        }
    }

    /* Bottom border rows rows-2 and rows-1 (replicate). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        emit(out, line2[c], 0, (c == cols - 1) ? 1 : 0);
    }
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        emit(out, line3[c], 0, (c == cols - 1) ? 1 : 0);
    }
}

/* -------------------------------------------------------------------------
 * Median 鈥?bitonic sorting network (ascending), returns the middle element.
 * Input is padded to the next power of two with the maximum value (255) so
 * the median index a[N/2] is unaffected by the padding.
 * ---------------------------------------------------------------------- */
template <int N>
static ap_uint<8> median_n(const ap_uint<8> w[N])
{
    const int P = (N <= 16) ? 16 : 32;
    ap_uint<8> a[P];
#pragma HLS ARRAY_PARTITION variable=a complete dim=1

    for (int i = 0; i < P; ++i) {
#pragma HLS UNROLL
        a[i] = (i < N) ? w[i] : (ap_uint<8>)255;
    }

    /* Bitonic sort, ascending. */
    for (int k = 2; k <= P; k <<= 1) {
#pragma HLS UNROLL
        for (int j = k >> 1; j > 0; j >>= 1) {
#pragma HLS UNROLL
            for (int i = 0; i < P; ++i) {
#pragma HLS UNROLL
                int l = i ^ j;
                if (l > i) {
                    if (((i & k) == 0 && a[i] > a[l]) ||
                        ((i & k) != 0 && a[i] < a[l])) {
                        ap_uint<8> t = a[i];
                        a[i] = a[l];
                        a[l] = t;
                    }
                }
            }
        }
    }
    return a[N / 2];
}

/* -------------------------------------------------------------------------
 * MEDIAN 3x3 (1px border-replicated), II=1.
 * ---------------------------------------------------------------------- */
static void median3x3_stream(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols
)
{
    static ap_uint<8> line0[MAX_WIDTH];
    static ap_uint<8> line1[MAX_WIDTH];
    static ap_uint<8> out_line[MAX_WIDTH];

#pragma HLS BIND_STORAGE variable=line0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=line1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=out_line type=ram_2p impl=bram

    /* Top border row 0 (replicate). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();
        line0[c] = 0;
        line1[c] = p.data;
        emit(out, p.data, (c == 0) ? 1 : 0, (c == cols - 1) ? 1 : 0);
    }
    /* Read row 1 (no output). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();
        line0[c] = line1[c];
        line1[c] = p.data;
    }

    /* Interior rows: read row r, output row r-1. */
    for (int r = 2; r < rows; ++r) {
        ap_uint<8> w00 = 0, w01 = 0, w02 = 0;
        ap_uint<8> w10 = 0, w11 = 0, w12 = 0;
        ap_uint<8> w20 = 0, w21 = 0, w22 = 0;
        ap_uint<8> left_border = 0, right_border = 0;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
            pixel_t p = in.read();
            ap_uint<8> pixel = p.data;

            ap_uint<8> top = line0[c];
            ap_uint<8> mid = line1[c];
            line0[c] = mid;
            line1[c] = pixel;

            if (c == 0)        left_border  = mid;
            if (c == cols - 1) right_border = mid;

            if (c == 0) {
                w00 = 0; w01 = 0;
                w10 = 0; w11 = 0;
                w20 = 0; w21 = 0;
            } else {
                w00 = w01; w01 = w02;
                w10 = w11; w11 = w12;
                w20 = w21; w21 = w22;
            }
            w02 = top;
            w12 = mid;
            w22 = pixel;

            out_line[c] = 0;
            if (c >= 2) {
                ap_uint<8> win[9] = { w00, w01, w02, w10, w11, w12, w20, w21, w22 };
                out_line[c - 1] = median_n<9>(win);
            }
        }

        out_line[0] = left_border;
        out_line[cols - 1] = right_border;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
            emit(out, out_line[c], 0, (c == cols - 1) ? 1 : 0);
        }
    }

    /* Bottom border row rows-1 (replicate). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        emit(out, line1[c], 0, (c == cols - 1) ? 1 : 0);
    }
}

/* -------------------------------------------------------------------------
 * MEDIAN 5x5 (2px border-replicated), II=1.
 * ---------------------------------------------------------------------- */
static void median5x5_stream(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols
)
{
    static ap_uint<8> line0[MAX_WIDTH];
    static ap_uint<8> line1[MAX_WIDTH];
    static ap_uint<8> line2[MAX_WIDTH];
    static ap_uint<8> line3[MAX_WIDTH];
    static ap_uint<8> out_line[MAX_WIDTH];

#pragma HLS BIND_STORAGE variable=line0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=line1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=line2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=line3 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=out_line type=ram_2p impl=bram

    /* Top border rows 0,1 (replicate): output unchanged, seed buffers. */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();
        line0[c] = p.data; line1[c] = p.data; line2[c] = p.data; line3[c] = p.data;
        emit(out, p.data, (c == 0) ? 1 : 0, (c == cols - 1) ? 1 : 0);
    }
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();
        line0[c] = line1[c]; line1[c] = line2[c]; line2[c] = line3[c]; line3[c] = p.data;
        emit(out, p.data, 0, (c == cols - 1) ? 1 : 0);
    }
    /* Read rows 2,3 (no output). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();
        line0[c] = line1[c]; line1[c] = line2[c]; line2[c] = line3[c]; line3[c] = p.data;
    }
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();
        line0[c] = line1[c]; line1[c] = line2[c]; line2[c] = line3[c]; line3[c] = p.data;
    }

    ap_uint<8> w[5][5];
#pragma HLS ARRAY_PARTITION variable=w complete dim=0

    /* Interior rows: read row r (r=4..rows-1), output row r-2. */
    for (int r = 4; r < rows; ++r) {
        for (int i = 0; i < 5; ++i) {
#pragma HLS UNROLL
            for (int j = 0; j < 5; ++j) {
#pragma HLS UNROLL
                w[i][j] = 0;
            }
        }
        ap_uint<8> left0 = 0, left1 = 0, right1 = 0, right0 = 0;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
            pixel_t p = in.read();
            ap_uint<8> pixel = p.data;

            ap_uint<8> r4 = line0[c], r3 = line1[c], r2 = line2[c], r1 = line3[c], r0 = pixel;
            line0[c] = r3; line1[c] = r2; line2[c] = r1; line3[c] = r0;

            if (c == 0)        left0 = r2;
            if (c == 1)        left1 = r2;
            if (c == cols - 2) right1 = r2;
            if (c == cols - 1) right0 = r2;

            w[0][0]=w[0][1]; w[0][1]=w[0][2]; w[0][2]=w[0][3]; w[0][3]=w[0][4]; w[0][4]=r4;
            w[1][0]=w[1][1]; w[1][1]=w[1][2]; w[1][2]=w[1][3]; w[1][3]=w[1][4]; w[1][4]=r3;
            w[2][0]=w[2][1]; w[2][1]=w[2][2]; w[2][2]=w[2][3]; w[2][3]=w[2][4]; w[2][4]=r2;
            w[3][0]=w[3][1]; w[3][1]=w[3][2]; w[3][2]=w[3][3]; w[3][3]=w[3][4]; w[3][4]=r1;
            w[4][0]=w[4][1]; w[4][1]=w[4][2]; w[4][2]=w[4][3]; w[4][3]=w[4][4]; w[4][4]=r0;

            out_line[c] = 0;
            if (c >= 4) {
                ap_uint<8> flat[25];
#pragma HLS ARRAY_PARTITION variable=flat complete dim=1
                for (int i = 0; i < 5; ++i) {
#pragma HLS UNROLL
                    for (int j = 0; j < 5; ++j) {
#pragma HLS UNROLL
                        flat[i * 5 + j] = w[i][j];
                    }
                }
                out_line[c - 2] = median_n<25>(flat);
            }
        }

        out_line[0] = left0;
        out_line[1] = left1;
        out_line[cols - 2] = right1;
        out_line[cols - 1] = right0;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
            emit(out, out_line[c], 0, (c == cols - 1) ? 1 : 0);
        }
    }

    /* Bottom border rows rows-2, rows-1 (replicate). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        emit(out, line2[c], 0, (c == cols - 1) ? 1 : 0);
    }
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        emit(out, line3[c], 0, (c == cols - 1) ? 1 : 0);
    }
}

/* -------------------------------------------------------------------------
 * SHARPEN 3x3 (border-replicated), II=1.
 * `strength` = Q0.8 unsharp amount k (0..255):
 *   out = ((256 + 4k)*center - k*sum(4 neighbors)) >> 8
 * k=0 -> identity; k=255 -> full sharpening.
 * ---------------------------------------------------------------------- */
static void sharpen_stream(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols,
    ap_uint<8> strength
)
{
    static ap_uint<8> line0[MAX_WIDTH];
    static ap_uint<8> line1[MAX_WIDTH];
    static ap_uint<8> out_line[MAX_WIDTH];

#pragma HLS BIND_STORAGE variable=line0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=line1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=out_line type=ram_2p impl=bram

    /* Top border: replicate input row 0. */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();

        line0[c] = 0;
        line1[c] = p.data;

        emit(out, p.data, (c == 0) ? 1 : 0, (c == cols - 1) ? 1 : 0);
    }

    /* Read row 1 into the line buffers (no output yet). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();

        line0[c] = line1[c];
        line1[c] = p.data;
    }

    /* Interior rows: reading row r produces output row r-1. */
    for (int r = 2; r < rows; ++r) {

        ap_uint<8> w00 = 0, w01 = 0, w02 = 0;
        ap_uint<8> w10 = 0, w11 = 0, w12 = 0;
        ap_uint<8> w20 = 0, w21 = 0, w22 = 0;

        ap_uint<8> left_border  = 0;
        ap_uint<8> right_border = 0;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1

            pixel_t p = in.read();
            ap_uint<8> pixel = p.data;

            ap_uint<8> top = line0[c];
            ap_uint<8> mid = line1[c];

            line0[c] = mid;
            line1[c] = pixel;

            if (c == 0)        left_border  = mid;
            if (c == cols - 1) right_border = mid;

            if (c == 0) {
                w00 = 0; w01 = 0;
                w10 = 0; w11 = 0;
                w20 = 0; w21 = 0;
            } else {
                w00 = w01; w01 = w02;
                w10 = w11; w11 = w12;
                w20 = w21; w21 = w22;
            }

            w02 = top;
            w12 = mid;
            w22 = pixel;

            out_line[c] = 0;
            if (c >= 2) {
                int k = (int)strength;   // Q0.8
                int v =
                    ((256 + 4 * k) * (int)w11
                     - k * ((int)w01 + (int)w10 + (int)w12 + (int)w21)) >> 8;
                out_line[c - 1] = clip_u8(v);
            }
        }

        out_line[0] = left_border;
        out_line[cols - 1] = right_border;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
            emit(out, out_line[c], 0, (c == cols - 1) ? 1 : 0);
        }
    }

    /* Bottom border: replicate input row rows-1 (stored in line1). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        emit(out, line1[c], 0, (c == cols - 1) ? 1 : 0);
    }
}

/* -------------------------------------------------------------------------
 * SOBEL (zero border), II=1.  Threshold from p0.
 * ---------------------------------------------------------------------- */
static void sobel_stream(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols,
    ap_uint<8> threshold
)
{
    static ap_uint<8> line0[MAX_WIDTH];
    static ap_uint<8> line1[MAX_WIDTH];
    static ap_uint<8> out_line[MAX_WIDTH];

#pragma HLS BIND_STORAGE variable=line0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=line1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=out_line type=ram_2p impl=bram

    /* Read the first row: boundary row -> output zeros. */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();

        line1[c] = p.data;
        line0[c] = 0;

        emit(out, 0, (c == 0) ? 1 : 0, (c == cols - 1) ? 1 : 0);
    }

    /* Process rows 1 ... rows-1. */
    for (int r = 1; r < rows; ++r) {

        ap_uint<8> w00 = 0, w01 = 0, w02 = 0;
        ap_uint<8> w10 = 0, w11 = 0, w12 = 0;
        ap_uint<8> w20 = 0, w21 = 0, w22 = 0;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1

            pixel_t p = in.read();
            ap_uint<8> pixel = p.data;

            ap_uint<8> top = line0[c];
            ap_uint<8> mid = line1[c];

            line0[c] = mid;
            line1[c] = pixel;

            if (c == 0) {
                w00 = 0; w01 = 0;
                w10 = 0; w11 = 0;
                w20 = 0; w21 = 0;
            } else {
                w00 = w01; w01 = w02;
                w10 = w11; w11 = w12;
                w20 = w21; w21 = w22;
            }

            w02 = top;
            w12 = mid;
            w22 = pixel;

            out_line[c] = 0;
            if (r >= 2 && c >= 2) {
                int gx =
                    -(int)w00 + (int)w02
                    - 2 * (int)w10 + 2 * (int)w12
                    -(int)w20 + (int)w22;

                int gy =
                    -(int)w00
                    - 2 * (int)w01
                    - (int)w02
                    + (int)w20
                    + 2 * (int)w21
                    + (int)w22;

                int mag =
                    (gx < 0 ? -gx : gx) +
                    (gy < 0 ? -gy : gy);

                out_line[c - 1] = (mag >= (int)threshold) ? 255 : 0;
            }
        }

        if (r >= 2) {
            for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
                emit(out, out_line[c], 0, (c == cols - 1) ? 1 : 0);
            }
        }
    }

    /* Final bottom boundary row. */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        emit(out, 0, 0, (c == cols - 1) ? 1 : 0);
    }
}

/* -------------------------------------------------------------------------
 * ROBERTS (zero bottom/right border), II=1.  Threshold from p0.
 * ---------------------------------------------------------------------- */
static void roberts_stream(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols,
    ap_uint<8> threshold
)
{
    static ap_uint<8> line0[MAX_WIDTH];
    static ap_uint<8> out_line[MAX_WIDTH];

#pragma HLS BIND_STORAGE variable=line0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=out_line type=ram_2p impl=bram

    /* Read row 0 into the line buffer (no output yet). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();
        line0[c] = p.data;
    }

    /* Reading row r produces output row r-1. */
    for (int r = 1; r < rows; ++r) {

        ap_uint<8> left    = 0;  // I(r,   c-1)
        ap_uint<8> topleft = 0;  // I(r-1, c-1)

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1

            pixel_t p = in.read();
            ap_uint<8> cur = p.data;   // I(r, c)

            ap_uint<8> top = line0[c]; // I(r-1, c)

            line0[c] = cur;

            out_line[c] = 0;
            if (c >= 1) {
                int gx = (int)topleft - (int)cur; // A - D
                int gy = (int)left    - (int)top; // C - B
                int mag = (gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy);
                out_line[c - 1] = (mag >= (int)threshold) ? 255 : 0;
            }

            topleft = top;
            left    = cur;
        }

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
            emit(out, out_line[c], (r == 1 && c == 0) ? 1 : 0, (c == cols - 1) ? 1 : 0);
        }
    }

    /* Bottom zero border row. */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        emit(out, 0, 0, (c == cols - 1) ? 1 : 0);
    }
}

/* -------------------------------------------------------------------------
 * CANNY 鈥?full streaming implementation, fused into a single process (no
 * nested DATAFLOW).  Zero-padding at every stage:
 *   Gaussian(3x3) -> Gx/Gy -> |Gx|+|Gy| + 4-sector dir -> NMS -> double
 *   threshold -> forward-scan hysteresis.
 * Packed mag+dir: bit[17:16]=dir, bit[15:0]=mag.
 * ---------------------------------------------------------------------- */
static void canny_stream(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols,
    ap_uint<8> low_thr,
    ap_uint<8> high_thr
)
{
    static ap_uint<8>  gl0[MAX_WIDTH], gl1[MAX_WIDTH];   // image rows r-2, r-1
    static ap_uint<8>  bl0[MAX_WIDTH], bl1[MAX_WIDTH];   // blurred rows r-3, r-2
    static ap_uint<18> nm0[MAX_WIDTH], nm1[MAX_WIDTH];   // mag+dir rows r-4, r-3
    static ap_uint<1>  hrow[MAX_WIDTH];                  // hysteresis prev-row edge
    static ap_uint<8>  out_line[MAX_WIDTH];

#pragma HLS BIND_STORAGE variable=gl0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=gl1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=bl0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=bl1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=nm0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=nm1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=hrow type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=out_line type=ram_2p impl=bram

    /* Windows */
    ap_uint<8>  gw[3][3];
    ap_uint<8>  bw[3][3];
    ap_uint<18> nw[3][3];
#pragma HLS ARRAY_PARTITION variable=gw complete dim=0
#pragma HLS ARRAY_PARTITION variable=bw complete dim=0
#pragma HLS ARRAY_PARTITION variable=nw complete dim=0

    /* hysteresis shift registers */
    ap_uint<1> hleft    = 0;   // current row edge at c'-1
    ap_uint<1> h_topleft = 0;  // prev-row edge at c'-1 (saved before overwrite)

    for (int c = 0; c < cols; ++c) {
        gl0[c] = 0; gl1[c] = 0; bl0[c] = 0; bl1[c] = 0;
        nm0[c] = 0; nm1[c] = 0; hrow[c] = 0; out_line[c] = 0;
    }

    for (int r = 0; r < rows; ++r) {
        /* reset horizontal windows / shift registers at row start (zero-pad) */
        for (int i = 0; i < 3; ++i) {
#pragma HLS UNROLL
            for (int j = 0; j < 3; ++j) {
#pragma HLS UNROLL
                gw[i][j] = 0; bw[i][j] = 0; nw[i][j] = 0;
            }
        }
        hleft = 0; h_topleft = 0;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1

            pixel_t p = in.read();
            ap_uint<8> pixel = p.data;

            /* ---- Stage 1: Gaussian 3x3 ---- */
            ap_uint<8> g_top = gl0[c];
            ap_uint<8> g_mid = gl1[c];
            gl0[c] = g_mid;
            gl1[c] = pixel;

            gw[0][0]=gw[0][1]; gw[0][1]=gw[0][2]; gw[0][2]=g_top;
            gw[1][0]=gw[1][1]; gw[1][1]=gw[1][2]; gw[1][2]=g_mid;
            gw[2][0]=gw[2][1]; gw[2][1]=gw[2][2]; gw[2][2]=pixel;

            ap_uint<8> blurred = (ap_uint<8>)(
                ((int)gw[0][0] + 2*(int)gw[0][1] + (int)gw[0][2] +
                 2*(int)gw[1][0] + 4*(int)gw[1][1] + 2*(int)gw[1][2] +
                 (int)gw[2][0] + 2*(int)gw[2][1] + (int)gw[2][2]) >> 4);

            /* Virtual border (row -1 / col -1) must be zero-padded for the
             * gradient stage, matching golden_canny's OOB=0 convention. */
            if (r == 0 || c == 0) blurred = 0;

            /* ---- Stage 2: Gradient (Sobel) ---- */
            ap_uint<8> b_top = (c >= 1) ? bl0[c - 1] : (ap_uint<8>)0;
            ap_uint<8> b_mid = (c >= 1) ? bl1[c - 1] : (ap_uint<8>)0;
            if (c >= 1) { bl0[c - 1] = b_mid; bl1[c - 1] = blurred; }

            bw[0][0]=bw[0][1]; bw[0][1]=bw[0][2]; bw[0][2]=b_top;
            bw[1][0]=bw[1][1]; bw[1][1]=bw[1][2]; bw[1][2]=b_mid;
            bw[2][0]=bw[2][1]; bw[2][1]=bw[2][2]; bw[2][2]=blurred;

            int gx = -(int)bw[0][0] + (int)bw[0][2] - 2*(int)bw[1][0] + 2*(int)bw[1][2]
                     - (int)bw[2][0] + (int)bw[2][2];
            int gy = -(int)bw[0][0] - 2*(int)bw[0][1] - (int)bw[0][2]
                     + (int)bw[2][0] + 2*(int)bw[2][1] + (int)bw[2][2];

            ap_uint<16> ax = (gx < 0) ? (ap_uint<16>)(-gx) : (ap_uint<16>)gx;
            ap_uint<16> ay = (gy < 0) ? (ap_uint<16>)(-gy) : (ap_uint<16>)gy;
            ap_uint<16> mval = ax + ay;

            ap_uint<2> dval;
            if (ay > 2 * ax)          dval = 0;
            else if (ax > 2 * ay)     dval = 2;
            else                      dval = (gx * gy > 0) ? (ap_uint<2>)1 : (ap_uint<2>)3;

            ap_uint<18> md = ((ap_uint<18>)dval << 16) | (ap_uint<18>)mval;

            /* ---- Stage 3: NMS ---- */
            ap_uint<18> n_top = (c >= 2) ? nm0[c - 2] : (ap_uint<18>)0;
            ap_uint<18> n_mid = (c >= 2) ? nm1[c - 2] : (ap_uint<18>)0;
            if (c >= 2) { nm0[c - 2] = n_mid; nm1[c - 2] = md; }

            nw[0][0]=nw[0][1]; nw[0][1]=nw[0][2]; nw[0][2]=n_top;
            nw[1][0]=nw[1][1]; nw[1][1]=nw[1][2]; nw[1][2]=n_mid;
            nw[2][0]=nw[2][1]; nw[2][1]=nw[2][2]; nw[2][2]=md;

            ap_uint<16> cmag = nw[1][1](15, 0);
            ap_uint<2>  cdir = nw[1][1](17, 16);
            ap_uint<16> nb1 = 0, nb2 = 0;
            if (cdir == 0)      { nb1 = nw[0][1](15, 0); nb2 = nw[2][1](15, 0); }
            else if (cdir == 2) { nb1 = nw[1][0](15, 0); nb2 = nw[1][2](15, 0); }
            else if (cdir == 1) { nb1 = nw[0][0](15, 0); nb2 = nw[2][2](15, 0); }
            else                { nb1 = nw[0][2](15, 0); nb2 = nw[2][0](15, 0); }

            ap_uint<16> thinned = (cmag >= nb1 && cmag >= nb2) ? cmag : (ap_uint<16>)0;

            /* ---- Stage 4: double threshold + forward hysteresis ---- */
            ap_uint<1> strong = (thinned >= high_thr) ? (ap_uint<1>)1 : (ap_uint<1>)0;
            ap_uint<1> weak   = (thinned >= low_thr && thinned < high_thr) ? (ap_uint<1>)1 : (ap_uint<1>)0;

            ap_uint<1> top       = (c >= 3) ? hrow[c - 3] : (ap_uint<1>)0;
            ap_uint<1> top_right = (c >= 2) ? hrow[c - 2] : (ap_uint<1>)0;
            ap_uint<1> top_left  = h_topleft;
            ap_uint<1> left      = hleft;

            ap_uint<1> edge = strong.to_bool() || (weak.to_bool() &&
                              (left.to_bool() || top_left.to_bool() || top.to_bool() || top_right.to_bool()))
                              ? (ap_uint<1>)1 : (ap_uint<1>)0;

            h_topleft = top;
            if (c >= 3) hrow[c - 3] = edge;
            hleft = edge;

            if (c >= 3) out_line[c - 3] = edge ? (ap_uint<8>)255 : (ap_uint<8>)0;
        }

        /* emit output row r-3 (valid for r >= 3) */
        if (r >= 3) {
            for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
                emit(out, out_line[c], (r == 3 && c == 0) ? 1 : 0, (c == cols - 1) ? 1 : 0);
            }
        }
    }

    /* bottom 3 zero rows (zero-padding) */
    for (int k = 0; k < 3; ++k) {
        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
            emit(out, 0, 0, (c == cols - 1) ? 1 : 0);
        }
    }
}

/* -------------------------------------------------------------------------
 * CLAHE 鈥?contrast-limited adaptive histogram equalization, fixed 8x8 tiles,
 * horizontal bilinear interpolation between left/right tiles.  clip_limit Q0.8.
 * One tile-row of pixels is buffered while the histograms are accumulated.
 * ---------------------------------------------------------------------- */
static void clahe_stream(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols,
    ap_uint<8> clip_limit
)
{
    const int TILES = 8;
    int tile_h = rows / TILES;
    int tile_w = cols / TILES;

    static ap_uint<8>  tilebuf[(MAX_HEIGHT / TILES) * MAX_WIDTH];
    static ap_uint<16> hist[TILES * 256];
    static ap_uint<8>  cdf[TILES * 256];
    static ap_uint<9>  wtab[MAX_WIDTH / TILES];

#pragma HLS BIND_STORAGE variable=tilebuf type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=hist type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=cdf type=ram_1p impl=bram
#pragma HLS ARRAY_PARTITION variable=wtab complete dim=1

    /* Precompute interpolation weights (Q8) once; avoids per-pixel division. */
    for (int i = 0; i < tile_w; ++i) wtab[i] = (i * 256) / tile_w;

    for (int ty = 0; ty < TILES; ++ty) {

        /* Phase 1: read one tile-row, buffer pixels, accumulate histograms. */
        for (int b = 0; b < TILES * 256; ++b) hist[b] = 0;

        for (int r = 0; r < tile_h; ++r) {
            int t = 0, rx = 0;
            for (int c = 0; c < cols; ++c) {
                pixel_t p = in.read();
                tilebuf[r * cols + c] = p.data;
                ap_uint<16> h = hist[t * 256 + p.data];
                hist[t * 256 + p.data] = h + 1;
                ++rx;
                if (rx == tile_w) { rx = 0; ++t; }
            }
        }

        /* Phase 2: clip + CDF. */
        int npix = tile_h * tile_w;
        int clip = ((int)clip_limit * npix) >> 8;
        if (clip < 1) clip = 1;
        for (int t = 0; t < TILES; ++t) {
            int excess = 0;
            for (int b = 0; b < 256; ++b) {
                int idx = t * 256 + b;
                if ((int)hist[idx] > clip) { excess += (int)hist[idx] - clip; hist[idx] = clip; }
            }
            int redist = excess / 256;
            int leftover = excess % 256;
            for (int b = 0; b < 256; ++b) {
                int idx = t * 256 + b;
                hist[idx] = hist[idx] + redist + ((b < leftover) ? 1 : 0);
            }
            int cum = 0, cummin = npix + 1;
            for (int b = 0; b < 256; ++b) { cum += (int)hist[t * 256 + b]; if (cum > 0 && cum < cummin) cummin = cum; }
            cum = 0;
            for (int b = 0; b < 256; ++b) {
                cum += (int)hist[t * 256 + b];
                int val = (cummin > npix) ? b : ((cum - cummin) * 255 / (npix - cummin));
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                cdf[t * 256 + b] = (ap_uint<8>)val;
            }
        }

        /* Phase 3: remap (horizontal bilinear interpolation) + emit. */
        for (int r = 0; r < tile_h; ++r) {
            int t = 0, rx = 0;
            for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
                ap_uint<8> p = tilebuf[r * cols + c];
                int t2 = (t + 1 < TILES) ? (t + 1) : t;
                int wx = (int)wtab[rx];
                int v = ((256 - wx) * (int)cdf[t * 256 + p] + wx * (int)cdf[t2 * 256 + p]) / 256;
                emit(out, (ap_uint<8>)v,
                     (ty == 0 && r == 0 && c == 0) ? 1 : 0,
                     (c == cols - 1) ? 1 : 0);
                ++rx;
                if (rx == tile_w) { rx = 0; ++t; }
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * 3x3 min/max (morphology) with identity border padding.
 *   IS_MAX == true  -> dilation (max, pad = 0)
 *   IS_MAX == false -> erosion  (min, pad = 255)
 * ---------------------------------------------------------------------- */
static ap_uint<8> max9(
    ap_uint<8> a, ap_uint<8> b, ap_uint<8> c,
    ap_uint<8> d, ap_uint<8> e, ap_uint<8> f,
    ap_uint<8> g, ap_uint<8> h, ap_uint<8> i
)
{
#pragma HLS INLINE
    ap_uint<8> m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    if (d > m) m = d;
    if (e > m) m = e;
    if (f > m) m = f;
    if (g > m) m = g;
    if (h > m) m = h;
    if (i > m) m = i;
    return m;
}

static ap_uint<8> min9(
    ap_uint<8> a, ap_uint<8> b, ap_uint<8> c,
    ap_uint<8> d, ap_uint<8> e, ap_uint<8> f,
    ap_uint<8> g, ap_uint<8> h, ap_uint<8> i
)
{
#pragma HLS INLINE
    ap_uint<8> m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    if (d < m) m = d;
    if (e < m) m = e;
    if (f < m) m = f;
    if (g < m) m = g;
    if (h < m) m = h;
    if (i < m) m = i;
    return m;
}

template <bool IS_MAX>
static void minmax3x3_stream(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols
)
{
    const ap_uint<8> PAD = IS_MAX ? (ap_uint<8>)0 : (ap_uint<8>)255;

    static ap_uint<8> line0[MAX_WIDTH];
    static ap_uint<8> line1[MAX_WIDTH];
    static ap_uint<8> out_line[MAX_WIDTH];

#pragma HLS BIND_STORAGE variable=line0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=line1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=out_line type=ram_2p impl=bram

    ap_uint<8> w00, w01, w02, w10, w11, w12, w20, w21, w22;

    for (int c = 0; c < cols; ++c) {
        line0[c] = PAD;
        line1[c] = PAD;
    }

    /* Read row 0 (no output yet). */
    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        pixel_t p = in.read();
        ap_uint<8> cur = p.data;
        line0[c] = line1[c];
        line1[c] = cur;
    }

    /* Main rows: reading row r produces output row r-1. */
    for (int r = 1; r < rows; ++r) {

        w00 = PAD; w01 = PAD; w02 = PAD;
        w10 = PAD; w11 = PAD; w12 = PAD;
        w20 = PAD; w21 = PAD; w22 = PAD;

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1

            pixel_t p = in.read();
            ap_uint<8> cur = p.data;

            ap_uint<8> top = line0[c];
            ap_uint<8> mid = line1[c];

            line0[c] = mid;
            line1[c] = cur;

            if (c == 0) {
                w00 = PAD; w01 = PAD;
                w10 = PAD; w11 = PAD;
                w20 = PAD; w21 = PAD;
            } else {
                w00 = w01; w01 = w02;
                w10 = w11; w11 = w12;
                w20 = w21; w21 = w22;
            }

            w02 = top;
            w12 = mid;
            w22 = cur;

            if (c >= 1) {
                out_line[c - 1] = IS_MAX
                    ? max9(w00, w01, w02, w10, w11, w12, w20, w21, w22)
                    : min9(w00, w01, w02, w10, w11, w12, w20, w21, w22);
            }
        }

        /* Right border: center (r-1, cols-1) with a PAD column. */
        w00 = w01; w01 = w02;
        w10 = w11; w11 = w12;
        w20 = w21; w21 = w22;
        w02 = PAD; w12 = PAD; w22 = PAD;
        out_line[cols - 1] = IS_MAX
            ? max9(w00, w01, w02, w10, w11, w12, w20, w21, w22)
            : min9(w00, w01, w02, w10, w11, w12, w20, w21, w22);

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
            emit(out, out_line[c], (r == 1 && c == 0) ? 1 : 0, (c == cols - 1) ? 1 : 0);
        }
    }

    /* Bottom border: center row rows-1 with a PAD row. */
    w00 = PAD; w01 = PAD; w02 = PAD;
    w10 = PAD; w11 = PAD; w12 = PAD;
    w20 = PAD; w21 = PAD; w22 = PAD;

    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1

        ap_uint<8> cur = PAD;

        ap_uint<8> top = line0[c];
        ap_uint<8> mid = line1[c];

        line0[c] = mid;
        line1[c] = cur;

        if (c == 0) {
            w00 = PAD; w01 = PAD;
            w10 = PAD; w11 = PAD;
            w20 = PAD; w21 = PAD;
        } else {
            w00 = w01; w01 = w02;
            w10 = w11; w11 = w12;
            w20 = w21; w21 = w22;
        }

        w02 = top;
        w12 = mid;
        w22 = cur;

        if (c >= 1) {
            out_line[c - 1] = IS_MAX
                ? max9(w00, w01, w02, w10, w11, w12, w20, w21, w22)
                : min9(w00, w01, w02, w10, w11, w12, w20, w21, w22);
        }
    }

    w00 = w01; w01 = w02;
    w10 = w11; w11 = w12;
    w20 = w21; w21 = w22;
    w02 = PAD; w12 = PAD; w22 = PAD;
    out_line[cols - 1] = IS_MAX
        ? max9(w00, w01, w02, w10, w11, w12, w20, w21, w22)
        : min9(w00, w01, w02, w10, w11, w12, w20, w21, w22);

    for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
        emit(out, out_line[c], 0, (c == cols - 1) ? 1 : 0);
    }
}

/* -------------------------------------------------------------------------
 * Morphology 鈥?one pass of a POST slot (kernel=3 only), run as a separate
 * DATAFLOW process (no nesting).  A slot runs exactly two passes (single
 * iteration):
 *   closing = dilate(max) then erode(min)
 *   opening = erode(min) then dilate(max)
 * `pass` = 0 (first) or 1 (second).
 * ---------------------------------------------------------------------- */
static void post_pass(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols,
    ap_uint<8> op,
    int pass
)
{
    if (op != OP_CLOSING && op != OP_OPENING) {
        passthrough(in, out, rows, cols);
        return;
    }

    /* closing = max,min ; opening = min,max */
    bool is_max = ((pass & 1) == 0) == (op == OP_CLOSING);

    if (is_max) minmax3x3_stream<true>(in, out, rows, cols);
    else        minmax3x3_stream<false>(in, out, rows, cols);
}

/* -------------------------------------------------------------------------
 * Slot dispatchers.  Each consumes `in` fully and produces `out` fully,
 * so they are valid DATAFLOW processes regardless of the selected branch.
 * ---------------------------------------------------------------------- */
static void pre_slot(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols,
    ap_uint<8> op,
    const ap_uint<32> p[SLOT_PARAMS],
    const ap_uint<8> lut[256]
)
{
    if (op == OP_GAMMA) {
        gamma_stream(in, out, rows, cols, lut);
    } else if (op == OP_GAUSSIAN) {
        ap_uint<12> sigma = (ap_uint<12>)(p[1] & 0xFFF);   // Q4.8
        /*
         * XC7Z010 resource cut:
         * kernel=3 and kernel=5 both execute on the existing 3x3 datapath.
         * Validation/interface compatibility is unchanged, so PS software may
         * still request kernel=5 without requiring any protocol change.
         */
        int idx = (sigma < 256) ? 0 : 1;                  // 3x3: 0.8 / 1.5
        gaussian_stream(in, out, rows, cols, idx);
    } else if (op == OP_SHARPEN) {
        sharpen_stream(in, out, rows, cols, (ap_uint<8>)(p[0] & 0xFF));
    } else if (op == OP_MEDIAN) {
        /*
         * XC7Z010 resource cut:
         * kernel=3 and kernel=5 both execute on the existing 3x3 datapath.
         */
        median3x3_stream(in, out, rows, cols);
    } else if (op == OP_CLAHE) {
        clahe_stream(in, out, rows, cols, (ap_uint<8>)(p[0] & 0xFF));
    } else {
        /* BYPASS and any op that passed validation as unsupported-but-safe. */
        passthrough(in, out, rows, cols);
    }
}

static void edge_slot(
    pixel_stream_t& in,
    pixel_stream_t& out,
    int rows,
    int cols,
    ap_uint<8> op,
    const ap_uint<32> p[SLOT_PARAMS]
)
{
    ap_uint<8> threshold = (ap_uint<8>)(p[0] & 0xFF);

    if (op == OP_ROBERTS) {
        roberts_stream(in, out, rows, cols, threshold);
    } else if (op == OP_SOBEL) {
        sobel_stream(in, out, rows, cols, threshold);
    } else if (op == OP_CANNY) {
        canny_stream(in, out, rows, cols, (ap_uint<8>)(p[0] & 0xFF), (ap_uint<8>)(p[1] & 0xFF));
    } else {
        passthrough(in, out, rows, cols);
    }
}

/* -------------------------------------------------------------------------
 * Validation 鈥?hardware protection before the pipeline runs.
 * ---------------------------------------------------------------------- */
static bool is_op_implemented(ap_uint<8> op)
{
    return ((CAP_OPS_BITS >> op.to_uint()) & 1u) != 0;
}

static bool op_legal_in_slot(ap_uint<8> op, SlotKind kind)
{
    switch (kind) {
    case SLOT_PRE:
        return op == OP_BYPASS || op == OP_GAMMA || op == OP_GAUSSIAN ||
               op == OP_MEDIAN || op == OP_CLAHE || op == OP_SHARPEN;
    case SLOT_EDGE:
        return op == OP_BYPASS || op == OP_ROBERTS || op == OP_SOBEL || op == OP_CANNY;
    case SLOT_POST:
        return op == OP_BYPASS || op == OP_OPENING || op == OP_CLOSING;
    default:
        return false;
    }
}

/* Returns 0 on success, or an ErrCode. */
static ap_uint<8> validate_config(
    int rows,
    int cols,
    ap_uint<8> pre0_op, ap_uint<8> pre1_op, ap_uint<8> pre2_op,
    ap_uint<8> edge_op, ap_uint<8> post0_op, ap_uint<8> post1_op,
    const ap_uint<32> p[6][SLOT_PARAMS]
)
{
    if (rows <= 0 || cols <= 0 || rows > MAX_HEIGHT || cols > MAX_WIDTH)
        return ERR_DIMS;

    ap_uint<8>  ops[6]   = { pre0_op, pre1_op, pre2_op, edge_op, post0_op, post1_op };
    SlotKind    kinds[6] = { SLOT_PRE, SLOT_PRE, SLOT_PRE, SLOT_EDGE, SLOT_POST, SLOT_POST };

    for (int i = 0; i < 6; ++i) {
        ap_uint<8> op = ops[i];

        if (op == OP_BYPASS)
            continue;

        if (op >= OP_COUNT)
            return ERR_BAD_OP;

        if (!op_legal_in_slot(op, kinds[i]))
            return ERR_SLOT_MISMATCH;

        if (!is_op_implemented(op))
            return ERR_UNSUPPORTED;

        /* Per-op parameter range checks. */
        if (op == OP_GAUSSIAN) {
            ap_uint<8> k = (ap_uint<8>)(p[i][0] & 0xFF);
            if (k != 3 && k != 5) return ERR_PARAM;   // kernel must be 3 or 5
        }
        if (op == OP_MEDIAN) {
            ap_uint<8> k = (ap_uint<8>)(p[i][0] & 0xFF);
            if (k != 3 && k != 5) return ERR_PARAM;   // kernel must be 3 or 5
        }
        if (op == OP_CANNY) {
            ap_uint<8> low  = (ap_uint<8>)(p[i][0] & 0xFF);
            ap_uint<8> high = (ap_uint<8>)(p[i][1] & 0xFF);
            if (low > high) return ERR_PARAM;   // low must be <= high
        }
        if (op == OP_OPENING || op == OP_CLOSING) {
            ap_uint<8> k  = (ap_uint<8>)(p[i][0] & 0xFF);
            ap_uint<8> it = (ap_uint<8>)(p[i][1] & 0xFF);
            if (k != 3)    return ERR_PARAM;   // kernel must be 3
            if (it != 1)   return ERR_PARAM;   // iterations must be 1
        }
        /* Sobel/Roberts threshold is 8-bit -> any value is in range. */
    }

    return ERR_NONE;
}

/* -------------------------------------------------------------------------
 * Top-level edge_accel.
 * ---------------------------------------------------------------------- */
void edge_accel(
    hls::stream<axis8_t>& s_axis,
    hls::stream<axis8_t>& m_axis,
    int rows,
    int cols,
    ap_uint<8> pre0_op, ap_uint<32> pre0_p0, ap_uint<32> pre0_p1, ap_uint<32> pre0_p2, ap_uint<32> pre0_p3,
    ap_uint<8> pre1_op, ap_uint<32> pre1_p0, ap_uint<32> pre1_p1, ap_uint<32> pre1_p2, ap_uint<32> pre1_p3,
    ap_uint<8> pre2_op, ap_uint<32> pre2_p0, ap_uint<32> pre2_p1, ap_uint<32> pre2_p2, ap_uint<32> pre2_p3,
    ap_uint<8> edge_op, ap_uint<32> edge_p0, ap_uint<32> edge_p1, ap_uint<32> edge_p2, ap_uint<32> edge_p3,
    ap_uint<8> post0_op, ap_uint<32> post0_p0, ap_uint<32> post0_p1, ap_uint<32> post0_p2, ap_uint<32> post0_p3,
    ap_uint<8> post1_op, ap_uint<32> post1_p0, ap_uint<32> post1_p1, ap_uint<32> post1_p2, ap_uint<32> post1_p3,
    ap_uint<8> gamma_lut[256],
    ap_uint<32>& status,
    ap_uint<32>& capability
)
{
#pragma HLS INTERFACE axis port=s_axis
#pragma HLS INTERFACE axis port=m_axis

#pragma HLS INTERFACE s_axilite port=rows       bundle=CTRL
#pragma HLS INTERFACE s_axilite port=cols       bundle=CTRL
#pragma HLS INTERFACE s_axilite port=capability bundle=CTRL
#pragma HLS INTERFACE s_axilite port=status     bundle=CTRL

#pragma HLS INTERFACE s_axilite port=pre0_op bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pre0_p0 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pre0_p1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pre0_p2 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pre0_p3 bundle=CTRL

#pragma HLS INTERFACE s_axilite port=pre1_op bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pre1_p0 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pre1_p1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pre1_p2 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pre1_p3 bundle=CTRL

#pragma HLS INTERFACE s_axilite port=pre2_op bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pre2_p0 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pre2_p1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pre2_p2 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pre2_p3 bundle=CTRL

#pragma HLS INTERFACE s_axilite port=edge_op bundle=CTRL
#pragma HLS INTERFACE s_axilite port=edge_p0 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=edge_p1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=edge_p2 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=edge_p3 bundle=CTRL

#pragma HLS INTERFACE s_axilite port=post0_op bundle=CTRL
#pragma HLS INTERFACE s_axilite port=post0_p0 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=post0_p1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=post0_p2 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=post0_p3 bundle=CTRL

#pragma HLS INTERFACE s_axilite port=post1_op bundle=CTRL
#pragma HLS INTERFACE s_axilite port=post1_p0 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=post1_p1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=post1_p2 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=post1_p3 bundle=CTRL

#pragma HLS INTERFACE s_axilite port=gamma_lut bundle=GAMMA
#pragma HLS INTERFACE s_axilite port=return    bundle=CTRL

    capability = CAPABILITY_VALUE;

    ap_uint<32> p[6][SLOT_PARAMS];
#pragma HLS ARRAY_PARTITION variable=p complete dim=0
    p[0][0] = pre0_p0;  p[0][1] = pre0_p1;  p[0][2] = pre0_p2;  p[0][3] = pre0_p3;
    p[1][0] = pre1_p0;  p[1][1] = pre1_p1;  p[1][2] = pre1_p2;  p[1][3] = pre1_p3;
    p[2][0] = pre2_p0;  p[2][1] = pre2_p1;  p[2][2] = pre2_p2;  p[2][3] = pre2_p3;
    p[3][0] = edge_p0;  p[3][1] = edge_p1;  p[3][2] = edge_p2;  p[3][3] = edge_p3;
    p[4][0] = post0_p0; p[4][1] = post0_p1; p[4][2] = post0_p2; p[4][3] = post0_p3;
    p[5][0] = post1_p0; p[5][1] = post1_p1; p[5][2] = post1_p2; p[5][3] = post1_p3;

    ap_uint<8> err = validate_config(
        rows, cols,
        pre0_op, pre1_op, pre2_op, edge_op, post0_op, post1_op,
        p);

    if (err != ERR_NONE) {
        status = ((ap_uint<32>)err << 4) | 1u;
        return;
    }
    status = 0;

        // ARCH_FIX_ALL_BYPASS_DIRECT
    const bool all_bypass =
        (pre0_op  == OP_BYPASS) &&
        (pre1_op  == OP_BYPASS) &&
        (pre2_op  == OP_BYPASS) &&
        (edge_op  == OP_BYPASS) &&
        (post0_op == OP_BYPASS) &&
        (post1_op == OP_BYPASS);

    if (all_bypass) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
                axis8_t a = s_axis.read();

                a.keep = -1;
                a.strb = -1;
                a.user = (r == 0 && c == 0) ? 1 : 0;
                a.last = (r == rows - 1 && c == cols - 1) ? 1 : 0;
                a.id   = 0;
                a.dest = 0;

                m_axis.write(a);
            }
        }

        return;
    }

pixel_stream_t s0, s1, s2, s3, s4, s5, s6, s7, s8;
#pragma HLS STREAM variable=s0 depth=4096
#pragma HLS BIND_STORAGE variable=s0 type=fifo impl=bram
#pragma HLS STREAM variable=s1 depth=4096
#pragma HLS BIND_STORAGE variable=s1 type=fifo impl=bram
#pragma HLS STREAM variable=s2 depth=4096
#pragma HLS BIND_STORAGE variable=s2 type=fifo impl=bram
#pragma HLS STREAM variable=s3 depth=4096
#pragma HLS BIND_STORAGE variable=s3 type=fifo impl=bram
#pragma HLS STREAM variable=s4 depth=4096
#pragma HLS BIND_STORAGE variable=s4 type=fifo impl=bram
#pragma HLS STREAM variable=s5 depth=4096
#pragma HLS BIND_STORAGE variable=s5 type=fifo impl=bram
#pragma HLS STREAM variable=s6 depth=4096
#pragma HLS BIND_STORAGE variable=s6 type=fifo impl=bram
#pragma HLS STREAM variable=s7 depth=4096
#pragma HLS BIND_STORAGE variable=s7 type=fifo impl=bram
#pragma HLS STREAM variable=s8 depth=4096
#pragma HLS BIND_STORAGE variable=s8 type=fifo impl=bram
#pragma HLS DATAFLOW
    axis_to_pixel(s_axis, s0, rows, cols);
    pre_slot(s0, s1, rows, cols, pre0_op, p[0], gamma_lut);
    pre_slot(s1, s2, rows, cols, pre1_op, p[1], gamma_lut);
    pre_slot(s2, s3, rows, cols, pre2_op, p[2], gamma_lut);
    edge_slot(s3, s4, rows, cols, edge_op, p[3]);
    post_pass(s4, s5, rows, cols, post0_op, 0);
    post_pass(s5, s6, rows, cols, post0_op, 1);
    post_pass(s6, s7, rows, cols, post1_op, 0);
    post_pass(s7, s8, rows, cols, post1_op, 1);
    pixel_to_axis(s8, m_axis, rows, cols);
}
