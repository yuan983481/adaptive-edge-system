#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

/*
 * Edge accelerator — programmable multi-stage streaming pipeline.
 *
 * Datapath:
 *   s_axis(axis8_t) -> PRE0 -> PRE1 -> PRE2 -> EDGE -> POST0 -> POST1 -> m_axis(axis8_t)
 *
 * Internal streams use `pixel_t` (data + TUSER/TLAST sideband) so the
 * sideband is preserved end-to-end without violating the HLS rule that
 * ap_axiu / axis8_t may only appear on the top-level AXI-Stream ports.
 *
 * This header is the single source of truth for the software <-> hardware
 * contract: operator IDs, register offsets, fixed-point conventions, and
 * the error/status protocol.  The Python side mirrors these constants.
 */

static const int MAX_WIDTH  = 640;
static const int MAX_HEIGHT = 480;

static const int NUM_SLOTS  = 6;
static const int SLOT_PARAMS = 4;   // p0..p3 per slot

/* Top-level AXI-Stream pixel (8-bit gray). */
typedef ap_axiu<8, 1, 1, 1> axis8_t;

/* Internal streaming pixel: data + frame-start + row-end sideband. */
struct pixel_t {
    ap_uint<8> data;
    ap_uint<1> user;   // 1 = first pixel of frame
    ap_uint<1> last;   // 1 = last pixel of row
};

typedef hls::stream<pixel_t> pixel_stream_t;

/* -------------------------------------------------------------------------
 * Operator IDs (shared with software).  `op` field of every slot register.
 * ---------------------------------------------------------------------- */
enum OpId {
    OP_BYPASS   = 0,
    OP_GAMMA    = 1,
    OP_GAUSSIAN = 2,
    OP_MEDIAN   = 3,
    OP_CLAHE    = 4,
    OP_SHARPEN  = 5,
    OP_ROBERTS  = 6,
    OP_SOBEL    = 7,
    OP_CANNY    = 8,
    OP_OPENING  = 9,
    OP_CLOSING  = 10,
    OP_COUNT    = 11
};

/* Which slot family a given op is legal in. */
enum SlotKind {
    SLOT_PRE  = 0,
    SLOT_EDGE = 1,
    SLOT_POST = 2
};

/* -------------------------------------------------------------------------
 * Error / status protocol (STATUS register, RO).
 *   [0]   = error
 *   [7:4] = err_code
 * ---------------------------------------------------------------------- */
enum ErrCode {
    ERR_NONE          = 0,
    ERR_BAD_OP        = 1,   // op id out of range
    ERR_SLOT_MISMATCH = 2,   // op not legal in this slot
    ERR_UNSUPPORTED   = 3,   // op not implemented in this build
    ERR_PARAM         = 4,   // parameter out of range
    ERR_DIMS          = 5    // rows/cols out of range
};

/* -------------------------------------------------------------------------
 * CAPABILITY register (RO).  [31:24]=major [23:16]=minor [15:0]=op bitmask
 * (bit i == 1  ->  op i is implemented in this bitstream).
 *
 * Phase 1 implements: BYPASS, GAMMA, GAUSSIAN(3x3), SHARPEN(3x3),
 * ROBERTS, SOBEL, OPENING(3x3), CLOSING(3x3).  MEDIAN / CLAHE / CANNY are
 * reserved but return ERR_UNSUPPORTED until Phase 2/3.
 * ---------------------------------------------------------------------- */
#define CAP_VERSION_MAJOR 1
#define CAP_VERSION_MINOR 0
#define CAP_OPS_BITS \
    ((1u << OP_BYPASS)   | (1u << OP_GAMMA)   | (1u << OP_GAUSSIAN) | \
     (1u << OP_MEDIAN)   | (1u << OP_CLAHE)   | (1u << OP_SHARPEN)  | \
     (1u << OP_ROBERTS)  | (1u << OP_SOBEL)   | (1u << OP_CANNY)    | \
     (1u << OP_OPENING)  | (1u << OP_CLOSING))
#define CAPABILITY_VALUE \
    ((ap_uint<32>)((CAP_VERSION_MAJOR << 24) | (CAP_VERSION_MINOR << 16) | CAP_OPS_BITS))

/* -------------------------------------------------------------------------
 * AXI-Lite register map.
 *
 * Offsets are AUTO-ASSIGNED by Vitis HLS (scalar s_axilite ports; output
 * ports `capability`/`status` each get an extra `*_ctrl` ap_vld register).
 * Do NOT hard-code offsets here — the generated driver header
 * `xedge_accel_hw.h` (from IP export) is the authoritative source, and the
 * actual Phase-1 offsets are also documented in `register_map.md`.
 *
 * 0x00-0x0F are reserved for the block-level CTRL (AP_START/AP_DONE/AP_IDLE/
 * AP_READY/INTERRUPT) and GIER/IP_IER/IP_ISR.  `start`=AP_START, `done`=AP_DONE,
 * `idle`=AP_IDLE.
 * ---------------------------------------------------------------------- */

/*
 * Gamma LUT: separate AXI-Lite bundle `GAMMA`, 256 x 8-bit (byte address
 * 0..255).  Software writes the full 256-entry curve before asserting start.
 *
 * Fixed-point conventions (Phase 2+): sigma=Q4.8, strength=Q0.8,
 * clip_limit=Q0.8, threshold=u8, kernel={3,5}, iterations=1 (fixed).
 */

void edge_accel(
    hls::stream<axis8_t>& s_axis,
    hls::stream<axis8_t>& m_axis,
    int rows,
    int cols,
    /* PRE0 */ ap_uint<8> pre0_op, ap_uint<32> pre0_p0, ap_uint<32> pre0_p1, ap_uint<32> pre0_p2, ap_uint<32> pre0_p3,
    /* PRE1 */ ap_uint<8> pre1_op, ap_uint<32> pre1_p0, ap_uint<32> pre1_p1, ap_uint<32> pre1_p2, ap_uint<32> pre1_p3,
    /* PRE2 */ ap_uint<8> pre2_op, ap_uint<32> pre2_p0, ap_uint<32> pre2_p1, ap_uint<32> pre2_p2, ap_uint<32> pre2_p3,
    /* EDGE */ ap_uint<8> edge_op, ap_uint<32> edge_p0, ap_uint<32> edge_p1, ap_uint<32> edge_p2, ap_uint<32> edge_p3,
    /* POST0 */ ap_uint<8> post0_op, ap_uint<32> post0_p0, ap_uint<32> post0_p1, ap_uint<32> post0_p2, ap_uint<32> post0_p3,
    /* POST1 */ ap_uint<8> post1_op, ap_uint<32> post1_p0, ap_uint<32> post1_p1, ap_uint<32> post1_p2, ap_uint<32> post1_p3,
    /* Gamma LUT (PS-writable) */ ap_uint<8> gamma_lut[256],
    /* Status / capability (RO) */ ap_uint<32>& status, ap_uint<32>& capability
);
