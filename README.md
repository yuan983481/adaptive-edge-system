# 大模型协同自适应边缘检测系统
## LLM-Assisted Adaptive Edge Detection on Zynq-7000

本项目面向复杂图像场景下固定边缘检测算法适应性不足的问题，构建了一套融合 **CNN 图像状态感知、Rule/LLM Pipeline 规划、硬件约束校验、FPGA 动态执行与闭环反馈** 的自适应边缘检测系统。

系统基于 **Xilinx Zynq-7000 XC7Z010-CLG400-1**，FPGA 端采用单一 bitstream 的可编程流水线结构，通过运行时寄存器配置动态切换不同图像处理算子；软件端提供 RulePlanner、LLMPlanner 和异步快慢双环 Agent，并支持 Software / FPGA 两种执行后端。

> 项目核心流程：  
> **图像/视频输入 → 8 维图像状态分析 → Pipeline 规划 → PipelineValidator → FPGA/Software 执行 → 反馈评估 → Fast/Slow Loop 调整**

---

## 1. 主要特性

- **8 维图像状态感知**
  - CNN Image Analyzer V2：预测噪声、脉冲噪声、模糊、低照度、低对比度 5 个退化状态；
  - 传统特征：边缘密度、图像复杂度、纹理复杂度；
  - 使用独立校准阈值完成状态判定。

- **三种规划模式**
  - `rule`：确定性 RulePlanner，稳定且无需网络；
  - `llm`：DeepSeek/兼容 Chat Completions API 的 LLMPlanner；
  - `agent`：硬件感知闭环 Agent，执行“规划—执行—评估—再规划”。

- **V4 异步快慢双环**
  - Fast Loop：逐帧执行本地轻量参数调整；
  - Slow Loop：场景变化或持续异常时异步触发 DeepSeek 重规划；
  - Pipeline Cache：缓存已验证的场景级 Pipeline；
  - LLM 后台规划不直接操作 FPGA、DMA 或 MMIO，不阻塞当前帧硬件执行。

- **FPGA 可重构流水线**
  - 6 个处理槽位：`PRE0 → PRE1 → PRE2 → EDGE → POST0 → POST1`；
  - 支持 **10 种图像处理算子 + 1 种 BYPASS 模式**；
  - 在固定槽位约束下支持多种合法 Pipeline 组合；
  - Pipeline 经 Validator 校验后编译为 FPGA `op_id + p0~p3` 寄存器配置。

- **两种执行后端**
  - `software`：使用 OpenCV 等软件算子；
  - `fpga`：Linux UIO + dma-proxy + AXI DMA 调用 FPGA 加速器。

- **浏览器 Live GUI**
  - 显示原始帧、FPGA 边缘结果、场景状态、Pipeline、Fast/Slow Loop、Cache、延迟与运行状态；
  - 板卡 HTTP 服务默认监听 `8080` 端口。

---

## 2. FPGA 算子

| OpId | 模式/算子 | 槽位 | 主要参数 |
|---:|---|---|---|
| 0 | BYPASS | 任意 | — |
| 1 | GAMMA | PRE | 256×8-bit LUT |
| 2 | GAUSSIAN | PRE | 3×3，sigma 量化 |
| 3 | MEDIAN | PRE | 3×3 |
| 4 | CLAHE | PRE | 8×8 Tile，量化 clip limit |
| 5 | SHARPEN | PRE | strength |
| 6 | ROBERTS | EDGE | threshold |
| 7 | SOBEL | EDGE | threshold |
| 8 | CANNY | EDGE | low/high threshold |
| 9 | OPENING | POST | 3×3，iterations=1 |
| 10 | CLOSING | POST | 3×3，iterations=1 |

说明：

- Canny 的 hysteresis 采用**流式前向扫描近似**，因此 FPGA Canny 与 OpenCV Canny 不要求逐像素完全一致；
- CLAHE 采用固定 8×8 Tile，并使用有限插值策略；
- XC7Z010 版本中 Gaussian / Median 的 5×5 请求按 3×3 硬件路径执行；
- OPENING / CLOSING 当前仅支持 `kernel=3, iterations=1`。

---

## 3. 系统结构

### PS（ARM/Linux）

```text
Image / Video
      │
      ├── CNN Image Analyzer V2
      └── Classical Feature Extractor
                │
          8-D Image State
                │
      ┌─────────┴─────────┐
      │                   │
 RulePlanner          LLMPlanner
      │               (DeepSeek)
      └─────────┬─────────┘
                │
        PipelineValidator
                │
        Closed-loop Agent
                │
        Pipeline Compiler
                │
       AXI-Lite + AXI DMA
```

### PL（FPGA）

```text
s_axis
  │
PRE0 → PRE1 → PRE2 → EDGE → POST0 → POST1
                                      │
                                   m_axis
```

大尺寸图像和视频帧采用：

```text
TILE = 64 × 64
HALO = 4
CORE = 56 × 56
```

即每个 Tile 独立完成 DMA + FPGA 执行，裁除 overlap 后重新拼接为原始尺寸。

---

## 4. 仓库目录

```text
.
├── README.md
├── bitstream/
│   ├── edge_system_wrapper.bit
│   ├── edge_system_wrapper.hwh
│   └── edge_system_wrapper.xsa
├── display/
│   ├── dashboard.html
│   └── dashboard_live.html
├── docs/
│   ├── FPGA_INTERFACE_CONTRACT.md
│   ├── FPGA_STATUS_REPORT.md
│   └── WORK_REPORT.md
├── fpga_hls/
│   ├── src/
│   │   ├── edge_accel.cpp
│   │   ├── edge_accel.hpp
│   │   └── gamma_lut.h
│   ├── tb/
│   │   ├── golden_ref.hpp
│   │   └── tb_edge_accel.cpp
│   └── ip/
│       └── edge_accel.zip
├── reports/
│   ├── edge_system_wrapper_timing_summary_routed.rpt
│   └── edge_system_wrapper_utilization_placed.rpt
├── software/
│   ├── main.py
│   ├── config.yaml
│   ├── requirements.txt
│   ├── models/
│   ├── skills/
│   ├── src/
│   ├── training/
│   └── scripts/
└── vivado_bd/
    ├── adaptive_edge_vivado.xpr
    └── edge_system/
```

---

## 5. 环境要求

### PC / Software Backend

推荐：

- Windows 或 Linux
- Python **3.10 / 3.11**
- `pip`
- OpenCV、PyTorch、NumPy 等依赖

安装：

```bash
cd software
python -m pip install -r requirements.txt
```

> 不建议使用过新的 Python 版本直接部署本项目；推荐 Python 3.10 或 3.11，以获得更稳定的 OpenCV / NumPy / PyTorch 兼容性。

### FPGA Backend

需要：

- Xilinx Zynq-7000 XC7Z010-CLG400-1
- Linux
- 已加载 `edge_system_wrapper.bit`
- `/dev/uio0`
- `/dev/dma_proxy_tx`
- `/dev/dma_proxy_rx`
- `uio_pdrv_genirq`
- `dma-proxy.ko`

主要地址：

| 接口 | 基址 |
|---|---|
| CTRL AXI-Lite | `0x40000000` |
| GAMMA AXI-Lite | `0x40010000` |
| DMA AXI-Lite | `0x41E00000` |

---

## 6. 快速开始

### 6.1 Software Backend + RulePlanner

```bash
cd software

python main.py \
  --image <image_path> \
  --backend software \
  --planner rule \
  --cnn-checkpoint models/image_analyzer_v2_best.pth \
  --out outputs/result.png
```

如果不指定 `--cnn-checkpoint`，系统会回退到传统图像特征分析路径。

### 6.2 Software Backend + LLMPlanner

先配置环境变量。

Linux / macOS：

```bash
export LLM_BASE_URL="YOUR_API_BASE_URL"
export LLM_MODEL="YOUR_MODEL_NAME"
export LLM_API_KEY="YOUR_API_KEY"
```

PowerShell：

```powershell
$env:LLM_BASE_URL="YOUR_API_BASE_URL"
$env:LLM_MODEL="YOUR_MODEL_NAME"
$env:LLM_API_KEY="YOUR_API_KEY"
```

运行：

```bash
python main.py \
  --image <image_path> \
  --backend software \
  --planner llm \
  --cnn-checkpoint models/image_analyzer_v2_best.pth
```

LLM 只能在预定义算子空间内生成 Pipeline，最终结果必须通过 `PipelineValidator`。

### 6.3 FPGA Backend

在 Zynq Linux 板卡上：

```bash
cd software

sudo -E python3 main.py \
  --image <image_path> \
  --backend fpga \
  --planner rule \
  --cnn-checkpoint models/image_analyzer_v2_best.pth \
  --out outputs/fpga_result.png
```

### 6.4 V4 视频异步双环 Agent

```bash
sudo -E python3 main.py \
  --video <video_path> \
  --backend fpga \
  --planner agent \
  --cnn-checkpoint models/image_analyzer_v2_best.pth \
  --out outputs/demo.mp4
```

摄像头输入：

```bash
sudo -E python3 main.py \
  --camera 0 \
  --backend fpga \
  --planner agent \
  --cnn-checkpoint models/image_analyzer_v2_best.pth
```

---

## 7. Live GUI

视频模式运行后，板卡端 `LiveMonitor` 默认监听：

```text
0.0.0.0:8080
```

PC 端打开：

```text
display/dashboard_live.html
```

在页面中将 Board URL 设置为：

```text
http://<BOARD_IP>:8080
```

即可查看：

- 原始视频帧；
- FPGA Edge Output；
- Scene / Image State；
- 当前 Pipeline；
- Fast Loop / Slow Loop 状态；
- DeepSeek 重规划事件；
- Pipeline Cache；
- FPGA / DMA 处理延迟。

---

## 8. FPGA 验证结果

当前 XC7Z010 版本已完成：

- HLS C-Simulation：35 个用例通过；
- RTL Co-Simulation：36 项通过；
- Vivado 综合、布局布线、DRC、Bitstream 生成通过；
- Linux UIO + dma-proxy 上板联调通过；
- 64×64 Tile DMA/FPGA 执行验证通过。

### 实现后资源

| 资源 | 用量 | 占比 |
|---|---:|---:|
| Slice LUT | 15620 / 17600 | 88.75% |
| FF | 17449 / 35200 | 49.57% |
| BRAM Tile | 47 / 60 | 78.33% |
| DSP | 26 / 80 | 32.50% |

### 时序

| 指标 | 结果 |
|---|---:|
| Clock | 50 MHz |
| WNS | +4.986 ns |
| WHS | +0.016 ns |
| TNS / THS | 0 / 0 |
| Timing | MET |

详细报告见：

```text
docs/FPGA_STATUS_REPORT.md
reports/
```

---

## 9. CNN 模型

运行 CNN Image Analyzer V2 主要需要：

```text
software/models/image_analyzer_v2_best.pth
software/models/image_analyzer_thresholds.json
```

其中：

- `image_analyzer_v2_best.pth`：当前 V2 最佳模型；
- `image_analyzer_thresholds.json`：5 个退化状态对应的独立校准阈值。

训练与阈值校准脚本位于：

```text
software/training/
```

---

## 10. 已知限制

1. FPGA 当前稳定验证的单次事务为 **64×64**；大图采用 overlap + crop 分块。
2. HLS 内部最大图像尺寸配置为 **640×480**；大尺寸输入通过软件 Tile 机制处理。
3. Canny hysteresis 为流式前向扫描近似，不与 OpenCV Canny 追求逐像素一致。
4. CLAHE 为 8×8 Tile，并采用有限插值策略。
5. XC7Z010 资源较紧张，Gaussian / Median 当前采用 3×3 硬件实现。
6. OPENING / CLOSING 当前固定为 3×3、1 次迭代。
7. FPGA `ap_done` 中断当前未接入 PS IRQ，软件采用轮询方式确认完成。
8. 系统级性能同时受到 DMA、Tile 调度和软件控制开销影响，因此 FPGA 内核吞吐与整机帧率应分别评价。

---

## 11. 关键文档

- `docs/FPGA_INTERFACE_CONTRACT.md`：FPGA 软件/硬件接口契约；
- `docs/FPGA_STATUS_REPORT.md`：资源、时序、验证和上板状态；
- `docs/WORK_REPORT.md`：功能工作报告；
- `software/V4_CHANGELOG.md`：V4 异步双环改动；
- `software/V4_TEST_REPORT.md`：V4 软件验证记录。

---

## 12. 安全说明

请勿将真实 API Key 写入源码或提交到 GitHub。

建议仅通过环境变量配置：

```text
LLM_BASE_URL
LLM_MODEL
LLM_API_KEY
```

`.env`、密钥文件、运行输出和本地缓存应保持在 Git 忽略列表中。

---

## 13. 项目状态

当前版本已经完成：

**CNN/传统图像感知 → Rule/LLM/Agent 规划 → PipelineValidator → FPGA Pipeline 编译 → UIO/DMA 执行 → 输出反馈 → V4 Fast/Slow Loop → Live GUI**

该仓库用于项目代码提交、硬件设计复现与比赛展示。
