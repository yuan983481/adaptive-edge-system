# FPGA 自适应边缘检测加速器 — 已实现功能工作报告

日期：2026-11-01
版本：Phase 3（11 算子全实现）+ 7010 资源裁剪
器件：Xilinx Zynq-7000 **xc7z010clg400-1**（Linux 平台）
工具：Vitis HLS / Vivado 2025.2

---

## 1. 概述

本工作完成了 FPGA（PL）侧的**可编程、可串联、运行时动态切换**的图像边缘检测加速器 `edge_accel`。软件端通过 Linux UIO + dma-proxy 写入算子选择与参数、通过 AXI DMA 传入灰度图像，即可在不重新综合、不重新生成位流的前提下，动态执行任意 Pipeline 组合。

在完成六个基础算子（Phase 1/2）并补齐 **Canny** 与 **CLAHE**（Phase 3，共 11 个算子）之后，因开发板由 xc7z020 更换为 xc7z010，对 HLS 实现做了资源裁剪，并在 7010 上完成 Vivado 实现（时序 MET）与 Linux 上板联调。

---

## 2. 硬件架构

```
s_axis(8bit) → PRE0 → PRE1 → PRE2 → EDGE → POST0 → POST1 → m_axis(8bit)
```

- 6 个处理 slot 固定顺序，每级可独立配置为 `BYPASS` 或某个算子，并携带 4 个 32-bit 参数（`p0..p3`）。
- 顶层 `#pragma HLS DATAFLOW` 展开为多个并行进程，级间用 `hls::stream` 连接，稳态吞吐 II=1（每周期 1 像素）。
- 内部流 `pixel_t{data,user,last}`；顶层 AXI4-Stream `axis8_t = ap_axiu<8,1,1,1>`，由 adapter 做 AXIS↔内部流转换。
- 3×3/5×5 窗口算子用行缓冲（line buffer，BRAM）+ 滑动窗口实现，无整帧中间缓冲。

### slot 与算子分配

| 槽位 | 允许的算子 |
|---|---|
| PRE0/PRE1/PRE2 | BYPASS, GAMMA, GAUSSIAN, MEDIAN, CLAHE, SHARPEN |
| EDGE | BYPASS, ROBERTS, SOBEL, CANNY |
| POST0/POST1 | BYPASS, OPENING, CLOSING |

---

## 3. 已实现算子（11 个）

| OpId | 名称 | 槽位 | 参数 | 说明 |
|---|---|---|---|---|
| 0 | BYPASS | 任意 | — | 直通 |
| 1 | GAMMA | PRE | 256×8 PS 可写 LUT | 独立 AXI-Lite `GAMMA` 从口 |
| 2 | GAUSSIAN | PRE | kernel(3), sigma(Q4.8) | 3×3 离散 sigma；kernel=5 按 3×3 执行 |
| 3 | MEDIAN | PRE | kernel(3) | bitonic 排序网络取中值；kernel=5 按 3×3 执行 |
| 4 | CLAHE | PRE | clip_limit(Q0.8) | 固定 8×8 tile，水平双线性插值 |
| 5 | SHARPEN | PRE | strength(Q0.8) | unsharp mask |
| 6 | ROBERTS | EDGE | threshold(u8) | 2×2 梯度 + 阈值 |
| 7 | SOBEL | EDGE | threshold(u8) | 3×3 梯度 + 阈值 |
| 8 | CANNY | EDGE | low(u8), high(u8) | 完整 4 阶段 |
| 9 | OPENING | POST | kernel(3), iterations(1) | 3×3 腐蚀+膨胀 |
| 10 | CLOSING | POST | kernel(3), iterations(1) | 3×3 膨胀+腐蚀 |

`capability` 只读寄存器 = **0x010007ff**（bit 0..10 全 1，11 算子全实现）。

### 关键算子细节

- **Canny**：Gaussian(3×3) → Sobel Gx/Gy → L1 幅值(|Gx|+|Gy|) + 4 扇区方向 → 非极大值抑制(NMS) → 双阈值 → hysteresis。hysteresis 为**前向扫描近似**（左/左上/上/右上四邻域传播）；输出 0/255 二值边缘。
- **CLAHE**：固定 8×8 tile，逐 tile 统计直方图 → clip → 重分配 → CDF 均衡映射 → **水平**双线性插值（无垂直插值）。
- **硬件保护**：非法 op_id / op 与槽位不匹配 / 参数越界 / 尺寸越限 → 对应 err_code，pipeline 不启动。

### 7010 资源裁剪

为适配 xc7z010（LUT 17600 / BRAM 60 / DSP 80），相对 xc7z020 版做了：GAUSSIAN/MEDIAN 的 5×5 裁为 3×3（接口不变）、全 BYPASS 直通快路径、内部流 FIFO 落 BRAM。详见 `FPGA_STATUS_REPORT.md` 第四节。

---

## 4. 接口契约

### 4.1 寄存器（CTRL bundle，8 字节对齐）

| 偏移 | 寄存器 |
|---|---|
| 0x000 | CTRL（bit0=ap_start, bit1=ap_done, bit2=ap_idle, bit3=ap_ready, bit9=interrupt） |
| 0x004 / 0x008 / 0x00c | GIER / IP_IER / IP_ISR |
| 0x010 / 0x018 | rows / cols |
| 0x020–0x040 | pre0_op / pre0_p0..p3 |
| 0x048–0x068 | pre1_op / pre1_p0..p3 |
| 0x070–0x090 | pre2_op / pre2_p0..p3 |
| 0x098–0x0b8 | edge_op / edge_p0..p3 |
| 0x0c0–0x0e0 | post0_op / post0_p0..p3 |
| 0x0e8–0x108 | post1_op / post1_p0..p3 |
| 0x110 / 0x114 | status（[0]=error, [7:4]=err_code）/ status_ap_vld |
| 0x120 / 0x124 | capability / capability_ap_vld |

GAMMA bundle：`gamma_lut` 基址 0x100，范围 0x100..0x1ff，256×8-bit，**每 32-bit 字打包 4 项**。

### 4.2 错误码

`0=OK, 1=ERR_BAD_OP, 2=ERR_SLOT_MISMATCH, 3=ERR_UNSUPPORTED, 4=ERR_PARAM, 5=ERR_DIMS`。

### 4.3 图像 IO 格式

- 顶层 AXI4-Stream 8-bit 灰度；**TUSER=帧首**，**输出 TLAST=帧尾**（单帧单包）。
- 输入侧按 rows/cols 位置式重建 TUSER/TLAST，DMA 一次发 `rows×cols` 字节即可。
- 约束：1 ≤ rows ≤ 480，1 ≤ cols ≤ 640；CLAHE 还要求 rows/cols 为 8 的倍数。
- 运行时单次事务 64×64，大图经 overlap+crop 分块。

---

## 5. Vivado 系统集成

```
PS7(M_AXI_GP0) → axi_smc(NUM_MI=3) → edge_accel_0/s_axi_CTRL
                                  → edge_accel_0/s_axi_GAMMA
                                  → axi_dma_0/S_AXI_LITE
axi_dma_0/M_AXIS_MM2S → edge_accel_0/s_axis
edge_accel_0/m_axis → axi_dma_0/S_AXIS_S2MM
axi_dma_0(M_AXI_MM2S/S2MM/SG) → axi_mem_intercon → PS7/S_AXI_HP0
```

- 时钟：`clk_fpga_0` = 50 MHz（PS FCLK_CLK0），驱动全部。
- AXI DMA：SG 模式，MM2S/S2MM 流宽 8-bit，store-and-forward 开。
- **地址映射**（各 64K）：CTRL `0x40000000` / GAMMA `0x40010000` / DMA-Lite `0x41E00000`。
- 软件通过 `/dev/uio0`（map0=CTRL、map1=GAMMA）+ `/dev/dma_proxy_tx/rx` 访问。
- 中断：`edge_accel/interrupt`（ap_done）本版未接 PS IRQ，软件轮询 `ap_done` 完成。

---

## 6. 验证结果

| 验证项 | 结果 |
|---|---|
| C-Simulation | 35 用例全 PASS（单算子 + 组合链 + 错误保护 + TUSER/TLAST） |
| RTL Co-Simulation | **PASS** — 36 项全过，最大流深 4096 无死锁 |
| 覆盖范围 | Canny、Canny(hard)、CLAHE、CLAHE→Canny、Gaussian→Canny、全组合链、错误保护 |
| Vivado 综合/布局布线/DRC/write_bitstream（7010） | 全部通过 |
| 实现后时序（7010） | **MET**（见下） |
| Linux 上板联调 | UIO + dma-proxy + 64×64 分块收发验证通过 |

### 6.1 资源（xc7z010，Vivado 实现后，整系统）

| 资源 | 用量 | 占比 |
|---|---|---|
| Slice LUT | 15620 | 88.75% |
| FF | 17449 | 49.57% |
| BRAM Tile | 47 | 78.33% |
| DSP | 26 | 32.50% |

### 6.2 时序（实现后，50 MHz）

| 项 | 值 |
|---|---|
| 建立 slack (WNS) | **+4.986 ns**（0 失败） |
| 保持 slack (WHS) | +0.016 ns（0 失败） |
| TNS / THS | 0 / 0 |
| DRC | 0 error / 0 critical |

---

## 7. 已知限制

1. **单次处理规模 64×64**：当前稳定验证为 64×64（4096 像素），大图经 overlap+crop 分块（960×540 = 180 Tile/帧）。
2. **分辨率上限 640×480**（HLS MAX_WIDTH/MAX_HEIGHT，为 CLAHE tile 缓冲放得下而降）。
3. **CLAHE**：固定 8×8 tile、仅水平插值、clip_limit Q0.8。
4. **Canny**：hysteresis 为前向扫描近似；分块 Canny 与整幅 Canny 理论上不完全等价。
5. **GAUSSIAN/MEDIAN**：7010 上 5×5 裁为 3×3。
6. **形态学 OPENING/CLOSING**：仅 kernel=3、iterations=1。
7. 中断未接 PS IRQ_F2P（软件用轮询完成）。

---

## 8. 交付物清单

- **位流/硬件**：`bitstream/edge_system_wrapper.{bit,hwh}`
- **HLS 源码**：`fpga_hls/src/edge_accel.{cpp,hpp}`（7010 裁剪版）、`fpga_hls/tb/golden_ref.hpp`
- **接口契约**：`docs/FPGA_INTERFACE_CONTRACT.md`
- **证据**：`docs/FPGA_STATUS_REPORT.md`（资源/时序/验证）、本报告
