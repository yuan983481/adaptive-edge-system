#include "edge_accel.hpp"
#include <iostream>

struct Cfg {
    ap_uint<8> op[6];
    ap_uint<32> p[6][4];
};

static void clear_cfg(Cfg& c)
{
    for (int i = 0; i < 6; ++i) {
        c.op[i] = OP_BYPASS;
        for (int j = 0; j < 4; ++j) c.p[i][j] = 0;
    }
}

static void feed(hls::stream<axis8_t>& s, int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            axis8_t a;
            a.data = (ap_uint<8>)((r * 29 + c * 13 + ((r ^ c) * 7)) & 0xFF);
            a.keep = -1;
            a.strb = -1;
            a.user = (r == 0 && c == 0) ? 1 : 0;
            /* AXI DMA packet = one complete frame. */
            a.last = (r == rows - 1 && c == cols - 1) ? 1 : 0;
            a.id = 0;
            a.dest = 0;
            s.write(a);
        }
    }
}

static bool drain(hls::stream<axis8_t>& m, int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (m.empty()) return false;

            axis8_t a = m.read();

            const bool first = (r == 0 && c == 0);
            const bool last  = (r == rows - 1 && c == cols - 1);

            /* TUSER only on the first pixel of the frame. */
            if (first) {
                if (a.user != 1) return false;
            } else {
                if (a.user != 0) return false;
            }

            /* TLAST only on the final pixel of the complete frame. */
            if (last) {
                if (a.last != 1) return false;
            } else {
                if (a.last != 0) return false;
            }
        }
    }

    /* No extra pixels should remain after one complete frame. */
    return m.empty();
}

static bool run_case(const char* name, Cfg cfg)
{
    const int rows = 16;
    const int cols = 16;

    hls::stream<axis8_t> s, m;
    ap_uint<8> gamma_lut[256];
    for (int i = 0; i < 256; ++i) gamma_lut[i] = (ap_uint<8>)i;

    feed(s, rows, cols);

    ap_uint<32> status = 0, capability = 0;

    edge_accel(
        s, m, rows, cols,

        cfg.op[0], cfg.p[0][0], cfg.p[0][1], cfg.p[0][2], cfg.p[0][3],
        cfg.op[1], cfg.p[1][0], cfg.p[1][1], cfg.p[1][2], cfg.p[1][3],
        cfg.op[2], cfg.p[2][0], cfg.p[2][1], cfg.p[2][2], cfg.p[2][3],
        cfg.op[3], cfg.p[3][0], cfg.p[3][1], cfg.p[3][2], cfg.p[3][3],
        cfg.op[4], cfg.p[4][0], cfg.p[4][1], cfg.p[4][2], cfg.p[4][3],
        cfg.op[5], cfg.p[5][0], cfg.p[5][1], cfg.p[5][2], cfg.p[5][3],

        gamma_lut,
        status, capability
    );

    bool ok = (status == 0) &&
              drain(m, rows, cols);

    std::cout << (ok ? "PASS " : "FAIL ") << name
              << " status=0x" << std::hex << status.to_uint()
              << " capability=0x" << capability.to_uint()
              << std::dec << "\n";
    return ok;
}

int main()
{
    bool ok = true;
    Cfg c;

    clear_cfg(c);
    ok &= run_case("Bypass", c);

    clear_cfg(c);
    c.op[0] = OP_GAMMA;
    ok &= run_case("Gamma", c);

    clear_cfg(c);
    c.op[0] = OP_CLAHE; c.p[0][0] = 64;
    ok &= run_case("CLAHE", c);

    clear_cfg(c);
    c.op[1] = OP_GAUSSIAN; c.p[1][0] = 3; c.p[1][1] = 256;
    ok &= run_case("Gaussian3x3", c);

    clear_cfg(c);
    c.op[1] = OP_GAUSSIAN; c.p[1][0] = 5; c.p[1][1] = 256;
    ok &= run_case("Gaussian5->3 fallback", c);

    clear_cfg(c);
    c.op[1] = OP_MEDIAN; c.p[1][0] = 3;
    ok &= run_case("Median3x3", c);

    clear_cfg(c);
    c.op[1] = OP_MEDIAN; c.p[1][0] = 5;
    ok &= run_case("Median5->3 fallback", c);

    clear_cfg(c);
    c.op[2] = OP_SHARPEN; c.p[2][0] = 128; c.p[2][1] = 256;
    ok &= run_case("PRE2 Sharpen", c);

    clear_cfg(c);
    c.op[3] = OP_CANNY; c.p[3][0] = 32; c.p[3][1] = 96;
    ok &= run_case("Canny", c);

    clear_cfg(c);
    c.op[4] = OP_OPENING; c.p[4][0] = 3; c.p[4][1] = 1;
    ok &= run_case("POST0 Opening", c);

    clear_cfg(c);
    c.op[5] = OP_CLOSING; c.p[5][0] = 3; c.p[5][1] = 1;
    ok &= run_case("POST1 Closing", c);

    clear_cfg(c);
    c.op[4] = OP_OPENING; c.p[4][0] = 3; c.p[4][1] = 1;
    c.op[5] = OP_CLOSING; c.p[5][0] = 3; c.p[5][1] = 1;
    ok &= run_case("Multi-POST Opening->Closing", c);

    clear_cfg(c);
    c.op[0] = OP_GAMMA;
    c.op[1] = OP_GAUSSIAN; c.p[1][0] = 3; c.p[1][1] = 256;
    c.op[2] = OP_SHARPEN; c.p[2][0] = 128; c.p[2][1] = 256;
    c.op[3] = OP_CANNY; c.p[3][0] = 32; c.p[3][1] = 96;
    c.op[5] = OP_CLOSING; c.p[5][0] = 3; c.p[5][1] = 1;
    ok &= run_case("Gamma->Gaussian->Sharpen->Canny->Closing", c);

    clear_cfg(c);
    c.op[0] = OP_CLAHE; c.p[0][0] = 64;
    c.op[1] = OP_MEDIAN; c.p[1][0] = 3;
    c.op[3] = OP_SOBEL; c.p[3][0] = 32;
    c.op[4] = OP_OPENING; c.p[4][0] = 3; c.p[4][1] = 1;
    c.op[5] = OP_CLOSING; c.p[5][0] = 3; c.p[5][1] = 1;
    ok &= run_case("CLAHE->Median->Sobel->Opening->Closing", c);

    std::cout << (ok ? "RESULT: ORIGINAL-ARCH NO5X5 TESTS PASS\n"
                     : "RESULT: ORIGINAL-ARCH NO5X5 TESTS FAIL\n");
    return ok ? 0 : 1;
}
