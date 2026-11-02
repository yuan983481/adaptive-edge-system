// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// CTRL
// 0x000 : Control signals
//         bit 0  - ap_start (Read/Write/COH)
//         bit 1  - ap_done (Read/COR)
//         bit 2  - ap_idle (Read)
//         bit 3  - ap_ready (Read/COR)
//         bit 7  - auto_restart (Read/Write)
//         bit 9  - interrupt (Read)
//         others - reserved
// 0x004 : Global Interrupt Enable Register
//         bit 0  - Global Interrupt Enable (Read/Write)
//         others - reserved
// 0x008 : IP Interrupt Enable Register (Read/Write)
//         bit 0 - enable ap_done interrupt (Read/Write)
//         bit 1 - enable ap_ready interrupt (Read/Write)
//         others - reserved
// 0x00c : IP Interrupt Status Register (Read/TOW)
//         bit 0 - ap_done (Read/TOW)
//         bit 1 - ap_ready (Read/TOW)
//         others - reserved
// 0x010 : Data signal of rows
//         bit 31~0 - rows[31:0] (Read/Write)
// 0x014 : reserved
// 0x018 : Data signal of cols
//         bit 31~0 - cols[31:0] (Read/Write)
// 0x01c : reserved
// 0x020 : Data signal of pre0_op
//         bit 7~0 - pre0_op[7:0] (Read/Write)
//         others  - reserved
// 0x024 : reserved
// 0x028 : Data signal of pre0_p0
//         bit 31~0 - pre0_p0[31:0] (Read/Write)
// 0x02c : reserved
// 0x030 : Data signal of pre0_p1
//         bit 31~0 - pre0_p1[31:0] (Read/Write)
// 0x034 : reserved
// 0x038 : Data signal of pre0_p2
//         bit 31~0 - pre0_p2[31:0] (Read/Write)
// 0x03c : reserved
// 0x040 : Data signal of pre0_p3
//         bit 31~0 - pre0_p3[31:0] (Read/Write)
// 0x044 : reserved
// 0x048 : Data signal of pre1_op
//         bit 7~0 - pre1_op[7:0] (Read/Write)
//         others  - reserved
// 0x04c : reserved
// 0x050 : Data signal of pre1_p0
//         bit 31~0 - pre1_p0[31:0] (Read/Write)
// 0x054 : reserved
// 0x058 : Data signal of pre1_p1
//         bit 31~0 - pre1_p1[31:0] (Read/Write)
// 0x05c : reserved
// 0x060 : Data signal of pre1_p2
//         bit 31~0 - pre1_p2[31:0] (Read/Write)
// 0x064 : reserved
// 0x068 : Data signal of pre1_p3
//         bit 31~0 - pre1_p3[31:0] (Read/Write)
// 0x06c : reserved
// 0x070 : Data signal of pre2_op
//         bit 7~0 - pre2_op[7:0] (Read/Write)
//         others  - reserved
// 0x074 : reserved
// 0x078 : Data signal of pre2_p0
//         bit 31~0 - pre2_p0[31:0] (Read/Write)
// 0x07c : reserved
// 0x080 : Data signal of pre2_p1
//         bit 31~0 - pre2_p1[31:0] (Read/Write)
// 0x084 : reserved
// 0x088 : Data signal of pre2_p2
//         bit 31~0 - pre2_p2[31:0] (Read/Write)
// 0x08c : reserved
// 0x090 : Data signal of pre2_p3
//         bit 31~0 - pre2_p3[31:0] (Read/Write)
// 0x094 : reserved
// 0x098 : Data signal of edge_op
//         bit 7~0 - edge_op[7:0] (Read/Write)
//         others  - reserved
// 0x09c : reserved
// 0x0a0 : Data signal of edge_p0
//         bit 31~0 - edge_p0[31:0] (Read/Write)
// 0x0a4 : reserved
// 0x0a8 : Data signal of edge_p1
//         bit 31~0 - edge_p1[31:0] (Read/Write)
// 0x0ac : reserved
// 0x0b0 : Data signal of edge_p2
//         bit 31~0 - edge_p2[31:0] (Read/Write)
// 0x0b4 : reserved
// 0x0b8 : Data signal of edge_p3
//         bit 31~0 - edge_p3[31:0] (Read/Write)
// 0x0bc : reserved
// 0x0c0 : Data signal of post0_op
//         bit 7~0 - post0_op[7:0] (Read/Write)
//         others  - reserved
// 0x0c4 : reserved
// 0x0c8 : Data signal of post0_p0
//         bit 31~0 - post0_p0[31:0] (Read/Write)
// 0x0cc : reserved
// 0x0d0 : Data signal of post0_p1
//         bit 31~0 - post0_p1[31:0] (Read/Write)
// 0x0d4 : reserved
// 0x0d8 : Data signal of post0_p2
//         bit 31~0 - post0_p2[31:0] (Read/Write)
// 0x0dc : reserved
// 0x0e0 : Data signal of post0_p3
//         bit 31~0 - post0_p3[31:0] (Read/Write)
// 0x0e4 : reserved
// 0x0e8 : Data signal of post1_op
//         bit 7~0 - post1_op[7:0] (Read/Write)
//         others  - reserved
// 0x0ec : reserved
// 0x0f0 : Data signal of post1_p0
//         bit 31~0 - post1_p0[31:0] (Read/Write)
// 0x0f4 : reserved
// 0x0f8 : Data signal of post1_p1
//         bit 31~0 - post1_p1[31:0] (Read/Write)
// 0x0fc : reserved
// 0x100 : Data signal of post1_p2
//         bit 31~0 - post1_p2[31:0] (Read/Write)
// 0x104 : reserved
// 0x108 : Data signal of post1_p3
//         bit 31~0 - post1_p3[31:0] (Read/Write)
// 0x10c : reserved
// 0x110 : Data signal of status
//         bit 31~0 - status[31:0] (Read)
// 0x114 : Control signal of status
//         bit 0  - status_ap_vld (Read/COR)
//         others - reserved
// 0x120 : Data signal of capability
//         bit 31~0 - capability[31:0] (Read)
// 0x124 : Control signal of capability
//         bit 0  - capability_ap_vld (Read/COR)
//         others - reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XEDGE_ACCEL_CTRL_ADDR_AP_CTRL         0x000
#define XEDGE_ACCEL_CTRL_ADDR_GIE             0x004
#define XEDGE_ACCEL_CTRL_ADDR_IER             0x008
#define XEDGE_ACCEL_CTRL_ADDR_ISR             0x00c
#define XEDGE_ACCEL_CTRL_ADDR_ROWS_DATA       0x010
#define XEDGE_ACCEL_CTRL_BITS_ROWS_DATA       32
#define XEDGE_ACCEL_CTRL_ADDR_COLS_DATA       0x018
#define XEDGE_ACCEL_CTRL_BITS_COLS_DATA       32
#define XEDGE_ACCEL_CTRL_ADDR_PRE0_OP_DATA    0x020
#define XEDGE_ACCEL_CTRL_BITS_PRE0_OP_DATA    8
#define XEDGE_ACCEL_CTRL_ADDR_PRE0_P0_DATA    0x028
#define XEDGE_ACCEL_CTRL_BITS_PRE0_P0_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_PRE0_P1_DATA    0x030
#define XEDGE_ACCEL_CTRL_BITS_PRE0_P1_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_PRE0_P2_DATA    0x038
#define XEDGE_ACCEL_CTRL_BITS_PRE0_P2_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_PRE0_P3_DATA    0x040
#define XEDGE_ACCEL_CTRL_BITS_PRE0_P3_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_PRE1_OP_DATA    0x048
#define XEDGE_ACCEL_CTRL_BITS_PRE1_OP_DATA    8
#define XEDGE_ACCEL_CTRL_ADDR_PRE1_P0_DATA    0x050
#define XEDGE_ACCEL_CTRL_BITS_PRE1_P0_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_PRE1_P1_DATA    0x058
#define XEDGE_ACCEL_CTRL_BITS_PRE1_P1_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_PRE1_P2_DATA    0x060
#define XEDGE_ACCEL_CTRL_BITS_PRE1_P2_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_PRE1_P3_DATA    0x068
#define XEDGE_ACCEL_CTRL_BITS_PRE1_P3_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_PRE2_OP_DATA    0x070
#define XEDGE_ACCEL_CTRL_BITS_PRE2_OP_DATA    8
#define XEDGE_ACCEL_CTRL_ADDR_PRE2_P0_DATA    0x078
#define XEDGE_ACCEL_CTRL_BITS_PRE2_P0_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_PRE2_P1_DATA    0x080
#define XEDGE_ACCEL_CTRL_BITS_PRE2_P1_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_PRE2_P2_DATA    0x088
#define XEDGE_ACCEL_CTRL_BITS_PRE2_P2_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_PRE2_P3_DATA    0x090
#define XEDGE_ACCEL_CTRL_BITS_PRE2_P3_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_EDGE_OP_DATA    0x098
#define XEDGE_ACCEL_CTRL_BITS_EDGE_OP_DATA    8
#define XEDGE_ACCEL_CTRL_ADDR_EDGE_P0_DATA    0x0a0
#define XEDGE_ACCEL_CTRL_BITS_EDGE_P0_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_EDGE_P1_DATA    0x0a8
#define XEDGE_ACCEL_CTRL_BITS_EDGE_P1_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_EDGE_P2_DATA    0x0b0
#define XEDGE_ACCEL_CTRL_BITS_EDGE_P2_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_EDGE_P3_DATA    0x0b8
#define XEDGE_ACCEL_CTRL_BITS_EDGE_P3_DATA    32
#define XEDGE_ACCEL_CTRL_ADDR_POST0_OP_DATA   0x0c0
#define XEDGE_ACCEL_CTRL_BITS_POST0_OP_DATA   8
#define XEDGE_ACCEL_CTRL_ADDR_POST0_P0_DATA   0x0c8
#define XEDGE_ACCEL_CTRL_BITS_POST0_P0_DATA   32
#define XEDGE_ACCEL_CTRL_ADDR_POST0_P1_DATA   0x0d0
#define XEDGE_ACCEL_CTRL_BITS_POST0_P1_DATA   32
#define XEDGE_ACCEL_CTRL_ADDR_POST0_P2_DATA   0x0d8
#define XEDGE_ACCEL_CTRL_BITS_POST0_P2_DATA   32
#define XEDGE_ACCEL_CTRL_ADDR_POST0_P3_DATA   0x0e0
#define XEDGE_ACCEL_CTRL_BITS_POST0_P3_DATA   32
#define XEDGE_ACCEL_CTRL_ADDR_POST1_OP_DATA   0x0e8
#define XEDGE_ACCEL_CTRL_BITS_POST1_OP_DATA   8
#define XEDGE_ACCEL_CTRL_ADDR_POST1_P0_DATA   0x0f0
#define XEDGE_ACCEL_CTRL_BITS_POST1_P0_DATA   32
#define XEDGE_ACCEL_CTRL_ADDR_POST1_P1_DATA   0x0f8
#define XEDGE_ACCEL_CTRL_BITS_POST1_P1_DATA   32
#define XEDGE_ACCEL_CTRL_ADDR_POST1_P2_DATA   0x100
#define XEDGE_ACCEL_CTRL_BITS_POST1_P2_DATA   32
#define XEDGE_ACCEL_CTRL_ADDR_POST1_P3_DATA   0x108
#define XEDGE_ACCEL_CTRL_BITS_POST1_P3_DATA   32
#define XEDGE_ACCEL_CTRL_ADDR_STATUS_DATA     0x110
#define XEDGE_ACCEL_CTRL_BITS_STATUS_DATA     32
#define XEDGE_ACCEL_CTRL_ADDR_STATUS_CTRL     0x114
#define XEDGE_ACCEL_CTRL_ADDR_CAPABILITY_DATA 0x120
#define XEDGE_ACCEL_CTRL_BITS_CAPABILITY_DATA 32
#define XEDGE_ACCEL_CTRL_ADDR_CAPABILITY_CTRL 0x124

// GAMMA
// 0x100 ~
// 0x1ff : Memory 'gamma_lut' (256 * 8b)
//         Word n : bit [ 7: 0] - gamma_lut[4n]
//                  bit [15: 8] - gamma_lut[4n+1]
//                  bit [23:16] - gamma_lut[4n+2]
//                  bit [31:24] - gamma_lut[4n+3]
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XEDGE_ACCEL_GAMMA_ADDR_GAMMA_LUT_BASE 0x100
#define XEDGE_ACCEL_GAMMA_ADDR_GAMMA_LUT_HIGH 0x1ff
#define XEDGE_ACCEL_GAMMA_WIDTH_GAMMA_LUT     8
#define XEDGE_ACCEL_GAMMA_DEPTH_GAMMA_LUT     256

