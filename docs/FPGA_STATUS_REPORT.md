# FPGA 硬件侧状态报告

日期：2026-11-01
状态：**Phase 3 完成（11 算子）→ 7010（xc7z010）资源裁剪 → Vivado 实现（时序 MET）→ Linux 上板联调**

---

## 一、总体状态

FPGA 侧已完成可编程、可串联 Pipeline 与 11 个算子（含 Canny/CLAHE）的实现与验证；因开发板由 xc7z020 更换为 **xc7z010（7010）**，对 HLS 实现做了资源裁剪，并在 7010 上完成 Vivado 实现（时序 MET）与 Linux 上板联调。

- 顶层 `edge_accel`：`s_axis → PRE0 → PRE1 → PRE2 → EDGE → POST0 → POST1 → m_axis`，6 个 slot 每个都支持 BYPASS / 算子选择 / 参数配置，固定顺序 + 顶层 `#pragma HLS DATAFLOW`。
- 内部流用 `pixel_t{data,user,last}`（8-bit 像素 + TUSER/TLAST sideband），顶层用 `axis8_t`（ap_axiu）+ adapter 转换。
- 软件通过 **Linux UIO + dma-proxy** 写 AXI-Lite 配置（op_id + 参数 + rows/cols + Gamma LUT）+ DMA 发图，FPGA 即可在不重综合、不重生成 bitstream 的情况下动态执行任意 Pipeline 组合。

## 二、器件与运行平台

| 项 | 值 |
|---|---|
| 器件 | Xilinx Zynq-7000 **xc7z010clg400-1** |
| 时钟 | `clk_fpga_0` = 50 MHz（PS FCLK_CLK0） |
| 运行时 | Linux（UIO + dma-proxy，非 PYNQ） |
| 位流 | `bitstream/edge_system_wrapper.bit` |

## 三、已实现功能（11 算子）

| OpId | 名称 | 槽位 | 参数 |
|---|---|---|---|
| 0 | BYPASS | 任意 | — |
| 1 | GAMMA | PRE | 256×8 PS 可写 LUT |
| 2 | GAUSSIAN | PRE | kernel(3) + sigma(Q4.8，离散量化)；kernel=5 按 3×3 执行 |
| 3 | MEDIAN | PRE | kernel(3)；kernel=5 按 3×3 执行 |
| 4 | CLAHE | PRE | clip_limit(Q0.8)，固定 8×8 tile，水平插值 |
| 5 | SHARPEN | PRE | strength(Q0.8) |
| 6 | ROBERTS | EDGE | threshold(u8) |
| 7 | SOBEL | EDGE | threshold(u8) |
| 8 | CANNY | EDGE | low(u8) + high(u8) |
| 9 | OPENING | POST | kernel(3) + iterations(1) |
| 10 | CLOSING | POST | kernel(3) + iterations(1) |

`capability` 只读寄存器 = **0x010007ff**（bit 0..10 全 1，11 算子全实现）。

- **Canny** 完整 4 阶段：Gaussian(3×3)→Sobel Gx/Gy→L1 幅值+4 扇区方向→NMS→双阈值→hysteresis（前向扫描近似）。
- **CLAHE**：直方图均衡 + clip + CDF + 水平双线性插值（无垂直插值）。
- 硬件保护：非法 op_id / op 与槽位不匹配 / 参数越界 / 尺寸越限 → 对应 err_code，pipeline 不启动。

## 四、7010 资源裁剪（相对 xc7z020 版）

为适配 xc7z010 更小的资源（LUT 17600 vs 53200、BRAM 60 vs 140、DSP 80 vs 220），`edge_accel.cpp` 做了以下裁剪（**接口与 OpId/capability 完全不变**）：

1. **GAUSSIAN 5×5 → 3×3**：kernel=3/5 请求都走 3×3 数据通路（PS 仍可发 kernel=5，协议不变，按 3×3 执行）。
2. **MEDIAN 5×5 → 3×3**：同上。
3. **全 BYPASS 直通快路径**：6 个 slot 全为 BYPASS 时，s_axis→m_axis 直接转发，不进入 DATAFLOW（省 9 路流）。
4. **内部流 FIFO 落 BRAM**：9 个 `hls::stream` 加 `BIND_STORAGE type=fifo impl=bram`，用 BRAM 换 LUT。

## 五、资源与时序（xc7z010，2026-11-01 实现后）

**Vivado 实现后（整系统 edge_system，50 MHz，布局布线后）：**

| 指标 | 值 |
|---|---|
| Slice LUT | 15620 / 17600（**88.75%**） |
| FF（Slice Registers） | 17449 / 35200（49.57%） |
| BRAM Tile | 47 / 60（78.33%） |
| DSP | 26 / 80（32.50%） |
| 建立 slack (WNS) | **+4.986 ns**（0 失败） |
| 保持 slack (WHS) | +0.016 ns（0 失败） |
| TNS / THS | 0 / 0 |
| DRC | 0 error / 0 critical（仅 advisory/warning） |

> 时序 **MET**、DRC 干净。LUT 88.75% / BRAM 78.33% 表明 7010 上资源已较紧张，这也是运行时采用 64×64 分块执行的原因之一。

## 六、关键决策与限制

1. **最大分辨率 640×480**（HLS `MAX_WIDTH`/`MAX_HEIGHT`，为 CLAHE tile 缓冲放得下而降）。
2. **64×64 单次处理规模**：上板稳定验证的单次安全事务为 64×64（4096 像素），大图经 overlap+crop 分块（TILE 64×64、HALO 4、有效 CORE 56×56）逐 Tile 处理；960×540 = 180 Tile/帧、640×360 = 84 Tile/帧。
3. **CLAHE**：固定 8×8 tile、只水平插值（tile 行边界有轻微不连续）。
4. **Canny**：hysteresis 为前向扫描近似（非全 2-pass flood-fill）；分块 Canny 与整幅 Canny 理论上不完全等价。
5. **GAUSSIAN/MEDIAN**：7010 上 5×5 裁为 3×3（接口不变）。
6. **形态学 OPENING/CLOSING**：仅 kernel=3、iterations=1。

## 七、验证状态

- ✅ **C-Simulation**：35 用例全过（单算子 + 组合链 + 错误保护 + TUSER/TLAST sideband）。
- ✅ **RTL Co-Simulation**：36 项全过（`C/RTL co-simulation finished: PASS`），覆盖 Canny、Canny(hard)、CLAHE、CLAHE→Canny、Gaussian→Canny、全部组合链、参数/槽位/尺寸错误保护；最大流深 4096，无死锁。
- ✅ **IP export**：寄存器表与契约一致（CTRL + GAMMA 两个 AXI-Lite）。
- ✅ **Vivado Block Design + 实现（7010）**：新 IP 集成、综合、布局布线、DRC、write_bitstream 全通过；实现后时序 **MET**（WNS +4.986 ns）。
- ✅ **Linux 上板联调**：UIO + dma-proxy 收发与 64×64 分块执行验证通过。

### 7.1 Vivado 集成结果（Block Design `edge_system`）

- 结构：**PS7 → axi_smc → {edge_accel_0/s_axi_CTRL, edge_accel_0/s_axi_GAMMA, axi_dma_0/S_AXI_LITE}**；**axi_dma_0 M_AXIS_MM2S → edge_accel_0/s_axis**，**edge_accel_0/m_axis → axi_dma_0/S_AXIS_S2MM**；DMA 三个 master 经 `axi_mem_intercon` → PS7 `S_AXI_HP0`。
- 时钟/复位：`clk_fpga_0`（PS FCLK_CLK0 = 50 MHz）驱动全部；`rst_ps7_0_50M/peripheral_aresetn` 驱动 edge_accel `ap_rst_n` + axi_smc + DMA。
- **中断**：edge_accel `interrupt`（ap_done）本版未接 PS IRQ_F2P——软件用轮询 `ap_done`（CTRL 0x00 bit1）或读 `status` 完成。
- **DMA 数据通路**：`axi_dma` 为 SG 模式、MM2S/S2MM 流宽均 8-bit、store-and-forward 开；一次传输 = 一整帧（或一个 64×64 Tile）`rows×cols` 字节。

### 7.2 最终地址映射（.hwh 权威值，全部 64K）

| 从口 | 基址 | 高址 | 参数名 |
|---|---|---|---|
| edge_accel_0 `s_axi_CTRL` | **0x40000000** | 0x4000FFFF | C_S_AXI_CTRL_BASEADDR |
| edge_accel_0 `s_axi_GAMMA` | **0x40010000** | 0x4001FFFF | C_S_AXI_GAMMA_BASEADDR |
| axi_dma_0 `S_AXI_LITE` | **0x41E00000** | 0x41E0FFFF | — |

- IP/DMA 实例名：**`edge_accel_0`**、**`axi_dma_0`**。
- 软件端通过 `/dev/uio0`（map0=CTRL、map1=GAMMA）映射 AXI-Lite，通过 `/dev/dma_proxy_tx/rx` 走 DMA。

### 7.3 TUSER/TLAST 与 DMA 包边界适配（已采用）

- **冲突**：原契约 TLAST=行尾；AXI DMA S2MM 把 TLAST 当「包结束」，整帧传输会在首行就被截断。
- **采用方案（单帧单包）**：输出适配器 `pixel_to_axis` 的 TLAST 仅在**帧尾**置 1，TUSER 仍为帧首；输入 `axis_to_pixel` 按 rows/cols 位置式重建、忽略 DMA 送来的 TLAST。算子（Canny/CLAHE 等）零改动。

## 八、工程文件清单

**源码**：`fpga_hls/src/edge_accel.cpp`（HLS 实现，7010 裁剪版）、`fpga_hls/src/edge_accel.hpp`（接口/枚举）、`fpga_hls/src/gamma_lut.h`。

**测试**：`fpga_hls/tb/golden_ref.hpp`（软件参考）、`fpga_hls/tb/tb_edge_accel.cpp`（主测试）。

**Vivado 工程**：`D:\FPGA\adaptive_edge_py_last\adaptive_edge_vivado\`（Block Design `edge_system`，器件 xc7z010clg400-1）；位流 `impl_1/edge_system_wrapper.bit`。

**文档**：`FPGA_INTERFACE_CONTRACT.md`（接口契约）、`FPGA_STATUS_REPORT.md`（本报告）、`WORK_REPORT.md`（功能工作报告）。

## 九、下一步

1. ✅ 软件侧按契约实现 `hardware_capability.py` / `PipelineValidator` / `pipeline_compiler.py`（软件侧已完成并冻结）。
2. ✅ 7010 Vivado Block Design → 综合/布局布线/DRC/时序 → 导出 `.bit`/`.hwh`（时序 MET）。
3. ✅ Linux 上板联调（UIO + dma-proxy + 64×64 分块收发验证）。
4. ⏳ 扩大单次 Tile 规模（64×64→128×128/256×256）、重构 DATAFLOW 与 FIFO 深度，向连续 AXI-Stream 整帧处理演进。
