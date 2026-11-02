# FPGA Capability / Interface Contract

版本：Phase 3（2026-10-11，含 Canny + CLAHE）。软件侧据此实现 `hardware_capability.py` / `PipelineValidator` / `pipeline_compiler.py`。
**状态标记**：✅ 已实现且通过 C-Sim/RTL Co-Sim；⛔ 未实现（硬件校验返回 ERR_UNSUPPORTED）。

> **2026-10-12 变更**：输出 `m_axis` 的 **TLAST 语义由「行尾」改为「帧尾」**（单帧单包，适配 AXI DMA S2MM 包边界）。寄存器映射、OpId、capability、像素数据**均未变**；仅影响 DMA 传输粒度（整帧一次，非逐行）。见第 6 节。

> **重要**：当前最大分辨率为 **640×480**（`MAX_WIDTH=640, MAX_HEIGHT=480`），这是为了让 CLAHE 的 tile 缓冲能放进有限 BRAM。全链路所有算子都在此分辨率下工作。

---

## 1. 已实现算子（已测试）

| OpId | 名称 | 状态 | 说明 |
|---|---|---|---|
| 0 | BYPASS | ✅ | 直通，逐像素复制 |
| 1 | GAMMA | ✅ | 256×8 PS 可写 LUT（独立 AXI-Lite bundle `GAMMA`） |
| 2 | GAUSSIAN | ✅ | 3×3 + 离散 sigma（7010 上 kernel=5 按 3×3 执行） |
| 3 | MEDIAN | ✅ | 3×3（bitonic 排序网络；7010 上 kernel=5 按 3×3 执行） |
| 4 | CLAHE | ✅ | 固定 8×8 tile + clip_limit，水平双线性插值 |
| 5 | SHARPEN | ✅ | 3×3 + 定点 strength |
| 6 | ROBERTS | ✅ | 2×2 梯度 + 阈值 |
| 7 | SOBEL | ✅ | 3×3 梯度 + 阈值 |
| 8 | CANNY | ✅ | 完整：Gaussian→Gx/Gy→mag/dir→NMS→双阈值→hysteresis |
| 9 | OPENING | ✅ | 3×3，1 次迭代 |
| 10 | CLOSING | ✅ | 3×3，1 次迭代 |

全部 11 个算子已实现。capability 位掩码 = **0x010007ff**（见第 4 节）。

## 2. 每个算子的参数格式 / 范围 / 默认值

每个 slot 有 `op`（u8）+ `p0..p3`（各 u32，低字节有效）。定点约定：sigma=Q4.8，strength=Q0.8，clip_limit=Q0.8，threshold=u8。

| 算子 | p0 | p1 | p2 | p3 | 校验范围 |
|---|---|---|---|---|---|
| BYPASS | — | — | — | — | 无 |
| GAMMA | — | — | — | — | 无标量参数（用 LUT） |
| GAUSSIAN | kernel | sigma(Q4.8) | — | — | kernel∈{3,5}（7010 上 5 按 3×3 执行）；sigma 量化见下 |
| MEDIAN | kernel | — | — | — | kernel∈{3,5}（7010 上 5 按 3×3 执行） |
| CLAHE | clip_limit(Q0.8) | (tile_x=8) | (tile_y=8) | — | clip_limit 0..255；tile 固定 8×8 |
| SHARPEN | strength(Q0.8) | — | — | — | 0..255 |
| ROBERTS | threshold(u8) | — | — | — | 0..255 |
| SOBEL | threshold(u8) | — | — | — | 0..255 |
| CANNY | low(u8) | high(u8) | — | — | low≤high；否则 ERR_PARAM |
| OPENING | kernel | iterations | — | — | kernel==3，iterations==1 |
| CLOSING | kernel | iterations | — | — | kernel==3，iterations==1 |

**Gaussian sigma 量化（Q4.8 阈值，软硬件必须一致）：**
- 3×3：sigma < 256(1.0) → 预设0（σ≈0.8）；否则 → 预设1（σ≈1.5）
- 5×5：sigma < 320(1.25) → 预设0（σ≈1.0）；sigma < 448(1.75) → 预设1（σ≈1.5）；否则 → 预设2（σ≈2.0）

**CLAHE 说明：**
- tile_grid 固定 8×8（p1/p2 传 8，硬件忽略），`clip_limit` 为 Q0.8（实际 clip = clip_limit×tile_pixels/256）。
- 只做**水平**双线性插值（左/右相邻 tile 的 CDF 加权），无垂直插值——因此 tile 行边界处会有轻微不连续（流式实现的取舍）。
- `rows`/`cols` 必须是 8 的倍数（8×8 tile 整除）。

**Canny 说明：**
- 完整 4 阶段：Gaussian(3×3)→Sobel Gx/Gy→L1 幅值(|Gx|+|Gy|)+4 扇区方向→NMS→双阈值→hysteresis。
- hysteresis 是**前向扫描近似**（left/top-left/top/top-right 传播），非全 2-pass 连通域 flood-fill；这是流式实现的常见取舍。
- 输出为 0/255 二值边缘，右 3 列 / 底 3 行为 0（流式 3 级延迟的天然边界）。

## 3. 固定执行顺序与各 slot 支持的 op_id

```
s_axis → PRE0 → PRE1 → PRE2 → EDGE → POST0 → POST1 → m_axis
```

- **PRE0/PRE1/PRE2**：BYPASS, GAMMA, GAUSSIAN, MEDIAN, CLAHE, SHARPEN
- **EDGE**：BYPASS, ROBERTS, SOBEL, CANNY
- **POST0/POST1**：BYPASS, OPENING, CLOSING

每个 slot 的 op 独立配置；op=BYPASS 时该 slot 直通。全链为固定 9 进程 DATAFLOW，运行时按 op_id 动态选择算子。

## 4. Operator ID 与 capability 位掩码

- `OpId`：0..10（见第 1 节表）。
- **capability 寄存器 = 0x010007ff**（只读）：`[31:24]=major(1)`，`[23:16]=minor(0)`，`[15:0]=已实现 op 位掩码`（bit i=1 表示 op_i 已实现）。当前所有 11 个算子均已实现（bit 0..10 全 1）。

## 5. AXI-Lite 寄存器（8 字节对齐；偏移在 Phase 4 IP export 时已确认，端口未变）

IP 名称 `edge_accel_v1_0`。两个 AXI-Lite 从接口：`s_axi_CTRL`（控制+槽位）与 `s_axi_GAMMA`（Gamma LUT）。

**CTRL bundle（相对基址，寄存器每 8 字节对齐，中间 4 字节 reserved）：**

| 偏移 | 寄存器 | 说明 |
|---|---|---|
| 0x000 | CTRL | bit0=ap_start, bit1=ap_done, bit2=ap_idle, bit3=ap_ready, bit9=interrupt |
| 0x004 | GIER | 全局中断使能 |
| 0x008 | IP_IER | 中断使能 |
| 0x00c | IP_ISR | 中断状态 |
| 0x010 | rows | W，int |
| 0x018 | cols | W，int |
| 0x020 | pre0_op | W，u8 |
| 0x028 | pre0_p0 | W，u32 |
| 0x030 | pre0_p1 | W，u32 |
| 0x038 | pre0_p2 | W，u32 |
| 0x040 | pre0_p3 | W，u32 |
| 0x048..0x068 | pre1_op/p0/p1/p2/p3 | 同上（0x048, 0x050, 0x058, 0x060, 0x068） |
| 0x070..0x090 | pre2_op/p0/p1/p2/p3 | 0x070, 0x078, 0x080, 0x088, 0x090 |
| 0x098..0x0b8 | edge_op/p0/p1/p2/p3 | 0x098, 0x0a0, 0x0a8, 0x0b0, 0x0b8 |
| 0x0c0..0x0e0 | post0_op/p0/p1/p2/p3 | 0x0c0, 0x0c8, 0x0d0, 0x0d8, 0x0e0 |
| 0x0e8..0x108 | post1_op/p0/p1/p2/p3 | 0x0e8, 0x0f0, 0x0f8, 0x100, 0x108 |
| 0x110 | status | R，[0]=error, [7:4]=err_code |
| 0x114 | status_ctrl | R，bit0 = status_ap_vld（读清除） |
| 0x120 | capability | R，见第 4 节 |
| 0x124 | capability_ctrl | R，bit0 = capability_ap_vld（读清除） |

**GAMMA bundle（相对基址）：**
- `gamma_lut` 基址 0x100，范围 0x100..0x1ff，256×8-bit，**每个 32-bit 字打包 4 项**：word n 的 bit[7:0]=gamma_lut[4n]，bit[15:8]=gamma_lut[4n+1]，bit[23:16]=gamma_lut[4n+2]，bit[31:24]=gamma_lut[4n+3]。

> 软件端用驱动头 `xedge_accel_hw.h` 的 `XEDGE_ACCEL_CTRL_ADDR_*` / `XEDGE_ACCEL_GAMMA_ADDR_*` 宏，或按上表镜像为 Python 常量；不要自行假设 4 字节连续布局。
> **已用当前（含 Canny+CLAHE）版本重新 export IP 复核**：上表偏移与生成的 `xedge_accel_hw.h` 完全一致（CTRL ADDR_WIDTH=9/512B，GAMMA ADDR_WIDTH=9/512B）。`status`/`capability` 是输出端口（ap_vld），各自多一个 `*_ctrl` 寄存器（0x114/0x124）。GAMMA `gamma_lut` 基址 0x100（见下）。

## 6. 图像 I/O 格式与约束

- 顶层 AXI4-Stream，`axis8_t = ap_axiu<8,1,1,1>`（8-bit gray，TKEEP/TSTRB/TUSER/TLAST 各 1 bit）。
- **TUSER=1**：帧首像素（r=0 且 c=0）。**输出 TLAST=1**：帧尾像素（r=rows-1 且 c=cols-1，即整帧唯一一次）——与 AXI DMA 单帧单包传输对齐（S2MM 以 TLAST 作为包结束，行级 TLAST 会在首行就截断）。
- **输入侧**：`axis_to_pixel` 只取 `TDATA`，按 rows/cols **位置式重建** TUSER/TLAST，忽略上游（DMA MM2S）送来的 TUSER/TLAST；因此 MM2S 一次发 `rows×cols` 字节即可，无需按行发 TLAST。
- **内部流** `pixel_t.last` 仍为行尾标记（算子位置式重建），但仅 `pixel_to_axis` 输出时改为帧尾 TLAST。
- `rows`/`cols` 约束：1 ≤ rows ≤ **480**，1 ≤ cols ≤ **640**（MAX_HEIGHT/MAX_WIDTH），越界 → ERR_DIMS。**CLAHE 还要求 rows/cols 是 8 的倍数**。
- 运行期间 LUT 与配置寄存器不可写（PS 在 start 前写全，done 后读 status）。

## 7. start/done/idle/error 状态与错误码

- `start`=AP_START，`done`=AP_DONE，`idle`=AP_IDLE（CTRL 寄存器，HLS 自动管理）。
- `status`（R）：`[0]=error`，`[7:4]=err_code`；0 = 成功。
- 错误码：`0=OK`，`1=ERR_BAD_OP`（op_id 越界），`2=ERR_SLOT_MISMATCH`（op 与 slot 不匹配），`3=ERR_UNSUPPORTED`（未实现算子），`4=ERR_PARAM`（参数越界），`5=ERR_DIMS`（尺寸越限）。出错时 pipeline 不启动，done 仍会置位（ap_ctrl 协议）。

## 8. 当前资源 / 时序数据与未解决限制

**Vivado 实现后（xc7z010clg400-1，50 MHz，布局布线后）：**

| 指标 | 值 |
|---|---|
| Slice LUT | 15620 / 17600（88.75%） |
| FF | 17449 / 35200（49.57%） |
| BRAM Tile | 47 / 60（78.33%） |
| DSP | 26 / 80（32.50%） |
| 建立 slack (WNS) | +4.986 ns |
| 保持 slack (WHS) | +0.016 ns |
| DRC | 0 error / 0 critical |

**验证状态（2026-11-01 复核）：**
- ✅ C-Simulation 通过（35 用例 + capability）。
- ✅ **RTL Co-Simulation 通过**（36 项全 PASS，最大流深 4096 无死锁）。
- ✅ **IP export 完成**。
- ✅ **Vivado Block Design + 实现（xc7z010）**：时序 MET（WNS +4.986 ns）。
- ✅ **Linux 上板联调**：UIO + dma-proxy 收发验证通过。

**IP 导出路径（当前版本，7010 裁剪版）：**
- 工程：`hls_source/edge_accel_final_build/`（Vitis HLS 2025.2，`edge_accel_final_1_5.cfg`，器件 xc7z010clg400-1）。
- IP 归档：`fpga_hls/ip/edge_accel.zip`（version 1.5）。
- 驱动头：`fpga_hls/ip/xedge_accel_hw.h`（权威寄存器表，见上）。
- IP 名称：`edge_accel_v1_5`；VLNV `xilinx.com:hls:edge_accel:1.5`。

**未解决限制 / 明确不支持：**
- **分辨率上限 640×480**（为 CLAHE tile 缓冲放得下而降）。
- CLAHE：固定 8×8 tile、只水平插值（无垂直）、clip_limit Q0.8。
- Canny：hysteresis 为前向扫描近似（非全 2-pass flood-fill）。
- 形态学 OPENING/CLOSING 仅 kernel=3、iterations=1。
- Gaussian sigma 为离散量化（3×3 两档、5×5 三档），非连续任意值。
- GAUSSIAN/MEDIAN 的 5×5 在 7010 上裁为 3×3（kernel=5 请求仍接受、按 3×3 执行）。
- 单次安全事务 64×64（4096 像素），大图经 overlap+crop 分块执行。
