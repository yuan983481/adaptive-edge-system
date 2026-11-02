# Adaptive Edge System V4 - Async Dual-Loop Agent
## 1. 项目简介
本项目实现一个面向 FPGA 的自适应边缘检测系统。
系统首先分析输入图像的噪声、模糊、亮度、对比度和纹理等状态，然后根据图像状态动态生成图像处理 Pipeline。Pipeline 经过合法性检查后，可以在软件端执行，也可以编译成 FPGA 可识别的寄存器配置，由同一个 FPGA bitstream 动态执行不同的算子组合。
项目的主要目标不是实现一个固定的 Sobel 或 Canny 加速器，而是实现：**图像状态分析 → 自适应 Pipeline 规划 → Pipeline 校验 → FPGA 动态执行。**
当前系统支持三种 Planner：
- `RulePlanner`：基于规则生成 Pipeline；
- `LLMPlanner`：通过大语言模型根据图像状态生成 Pipeline。
- `Agent`：采用本地 Fast Loop + DeepSeek 异步 Slow Loop 的双时间尺度控制，支持场景 Pipeline 缓存、反馈微调以及异步大模型重规划。
同时支持两种执行后端：
- `software`：使用 OpenCV 等软件算子执行；
- `fpga`：通过 AXI-Lite 和 AXI DMA 调用 FPGA 加速器。

# 2. 图像状态
系统使用 8 维图像状态描述输入图像：
         状态         |       含义       
    `noise_level`     |    一般噪声程度   
    `impulse_noise`   | 椒盐/脉冲噪声程度 
     `blur_level`     |    图像模糊程度   
     `low_light`      |    低照度程度     
    `low_contrast`    |    低对比度程度   
    `edge_density`    |     边缘密度      
  `image_complexity`  |   图像整体复杂度 
 `texture_complexity` |    纹理复杂度 

系统支持 CNN Image Analyzer V2 与传统图像特征分析两种感知方式。
使用 CNN 模型时，前 5 个退化状态由 CNN Image Analyzer V2 预测，并结合剩下 3 个传统图像特征，形成 8 维图像状态。
CNN 输出的不同维度不能直接通过固定的 0.5 阈值统一判断，因此项目使用 models/image_analyzer_thresholds.json 保存校准后的独立阈值。
未加载 CNN 模型时，系统可使用传统图像分析方法生成对应的图像状态，保证 RulePlanner、Agent 与 FPGA 执行流程仍可正常运行。


# 3. 自适应 Pipeline
Pipeline 分为三个阶段：
    阶段     | 最大数量 | 用途 
 Preprocess  | 3       | 图像增强、去噪、锐化 
 Edge        | 1       | 边缘检测 
 Postprocess | 2       | 边缘后处理

FPGA 上的实际 Slot 为：PRE0 / PRE1 / PRE2 / EDGE / POST0 / POST1
Planner 生成 `PipelinePlan` 后，`PipelineValidator` 会检查：
- 算子是否存在；
- 参数是否合法；
- Pipeline 数量是否超限；
- 算子是否适合当前 Slot；
- FPGA Backend 下算子是否受硬件支持；
- 图像尺寸是否满足硬件要求。
通过 Validator 后，FPGA Backend 会进一步通过 `PipelineCompiler` 将 Pipeline 转换为 FPGA 的 `op_id + p0~p3` 配置。

# 4. FPGA Phase 3
当前 FPGA 版本为：Xilinx Zynq-7000 XC7Z010-CLG400-1
当前共实现 10 种图像处理算子和 BYPASS 模式，共 11 种可配置操作模式。
 OpId | 算子     | 可用位置 
 0    | BYPASS   | 任意 
 1    | GAMMA    | PRE 
 2    | GAUSSIAN | PRE 
 3    | MEDIAN   | PRE 
 4    | CLAHE    | PRE 
 5    | SHARPEN  | PRE 
 6    | ROBERTS  | EDGE 
 7    | SOBEL    | EDGE 
 8    | CANNY    | EDGE 
 9    | OPENING  | POST 
 10   | CLOSING  | POST 

FPGA capability 寄存器值为：0x010007FF，bit 0~10 分别对应上述 11 种可配置操作模式。

# 5. FPGA 算子限制
## 5.1 图像尺寸
FPGA 单次稳定验证事务： 64 × 64 = 4096 pixels
大尺寸图像：采用 TILE=64、HALO=4、CORE=56 的 overlap+crop 分块执行。

## 5.2 CLAHE
对于单次直接 FPGA 事务，CLAHE 要求输入行列尺寸为 8 的整数倍。
对于大尺寸图像，系统采用 64×64 分块执行，每个硬件 Tile 均满足 CLAHE 的尺寸要求，因此原始整帧尺寸无需满足 8 的整数倍。

## 5.3 Gaussian
Gaussian 支持：kernel = 3 
sigma 在 FPGA 内部采用离散档位，而不是任意连续值。
软件的 `hardware_capability.py` 与 `pipeline_compiler.py` 已按照硬件量化规则处理参数。

## 5.4 Canny
当前 FPGA Canny 包括：
Gaussian 3×3
Sobel
L1 Gradient Magnitude
NMS
Double Threshold
Forward Hysteresis
其中 hysteresis 为流式前向扫描近似实现，并非完整的 flood-fill hysteresis。
因此 FPGA Canny 与 OpenCV Canny 不要求逐像素完全一致。

## 5.5 Opening / Closing
FPGA 当前只支持：kernel = 3 & iterations = 1
软件 Backend 可以支持更多参数，但使用 FPGA Backend 时 Validator 会按照硬件能力进行限制。

# 6. FPGA 数据接口
FPGA 输入输出均采用 8-bit AXI4-Stream。
对于不超过 4096 像素的小尺寸图像，可采用单次 DMA 事务完成处理。
对于大尺寸图像和视频帧，系统采用 64×64 重叠分块，每个 Tile 独立完成一次 DMA 传输和 FPGA 执行，处理完成后仅保留中心有效区域并重新拼接成原始尺寸结果。

# 7. FPGA 地址和实例
Vivado 最终地址映射：
 接口           | 地址 
 CTRL AXI-Lite  | `0x40000000` 
 GAMMA AXI-Lite | `0x40010000` 
 DMA AXI-Lite   | `0x41E00000` 

Vivado 实例名：edge_accel_0/axi_dma_0
软件当前主要通过：axi_dma_0/CTRL MMIO/GAMMA MMIO与硬件通信。

# 8. CTRL 寄存器
主要寄存器如下：
 Offset        | 内容 
 `0x000`       | CTRL 
 `0x010`       | rows 
 `0x018`       | cols 
 `0x020~0x040` | PRE0 
 `0x048~0x068` | PRE1 
 `0x070~0x090` | PRE2 
 `0x098~0x0B8` | EDGE 
 `0x0C0~0x0E0` | POST0 
 `0x0E8~0x108` | POST1 
 `0x110`       | status 
 `0x114`       | status_ap_vld 
 `0x120`       | capability 
 `0x124`       | capability_ap_vld 

每个 Pipeline Slot 包含：op/p0/p1/p2/p3

# 9. Gamma LUT
Gamma 使用独立 AXI-Lite 接口。
LUT 地址：0x100 ~ 0x1FF
共有：256 × uint8
每个 32-bit word 打包 4 个 LUT 值，因此软件实际写：64 × uint32
`pipeline_compiler.py` 会负责生成 Gamma LUT，`fpga_runtime.py` 负责写入硬件。

# 10. FPGA 错误码
 错误码 | 含义 
 0     | OK 
 1     | ERR_BAD_OP 
 2     | ERR_SLOT_MISMATCH 
 3     | ERR_UNSUPPORTED 
 4     | ERR_PARAM 
 5     | ERR_DIMS 

FPGA 如果检测到非法配置，会设置 `status`。
`fpga_runtime.py` 会读取 status，并转换成 Python 异常。

# 11. 项目目录
adaptive_edge_system_python/
│
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
│
├── src/
│   ├── agent_tools.py
│   ├── cnn_infer.py
│   ├── config.py
│   ├── evaluation.py
│   ├── fpga_agent.py
│   ├── fpga_runtime.py
│   ├── hardware_capability.py
│   ├── hed_runner.py
│   ├── hls_report_analyzer.py
│   ├── image_features.py
│   ├── live_monitor.py
│   ├── llm_contract.py
│   ├── llm_planner.py
│   ├── operator_library.py
│   ├── pipeline_compiler.py
│   ├── pipeline_executor.py
│   ├── pipeline_planner.py
│   ├── pipeline_validator.py
│   ├── schemas.py
│   ├── software_ops.py
│   ├── temporal_controller.py
│   ├── video_agent.py
│   └── visualize.py
│
├── skills/
│   └── image_pipeline_skill.json
│
├── scripts/
│   ├── analyze_hls_report.py
│   └── diagnose_gopro_blur.py
│
├── models/
│   ├── image_analyzer_v2_best.pth
│   └── image_analyzer_thresholds.json
│
├── overlays/
│   ├── edge_system_wrapper.bit
│   └── edge_system_wrapper.hwh
│
├── prepare_training_data_v2.py
├── fine_tune_image_analyzer.py
├── calibrate_image_analyzer.py
├── evaluate_adaptive_system.py
└── compare_fixed_vs_adaptive.py

# 12. 模型文件
使用 CNN Image Analyzer V2 时需要：
models/image_analyzer_v2_best.pth
models/image_analyzer_thresholds.json
未加载 CNN 模型时，系统可使用传统图像分析方法运行，因此 CNN 模型文件不是 Software/FPGA 主流程的强制依赖。
其中：
`image_analyzer_v2_best.pth`是当前 Image Analyzer V2 的最佳模型。
`image_analyzer_thresholds.json`保存五个 CNN degradation 输出对应的校准阈值。

# 13. 安装依赖
PC 软件环境：
pip install -r requirements.txt
主要依赖包括：
numpy/opencv-python/PyYAML/requests/torch/torchvision/scikit-image/matplotlib
FPGA Backend 需要运行在带 Linux + dma-proxy + UIO 驱动的 Zynq 板卡上。
普通 PC 只运行 Software Backend 时不需要这些内核驱动。

# 14. Software Backend
使用 CNN V2 + RulePlanner：
python main.py `
  --image test.png `
  --backend software `
  --planner rule `
  --cnn-checkpoint models/image_analyzer_v2_best.pth

指定输出：
python main.py `
  --image test.png `
  --backend software `
  --planner rule `
  --cnn-checkpoint models/image_analyzer_v2_best.pth `
  --out outputs/test_edge.png

软件 Backend 主要用于：
- 功能验证；
- 算法基线；
- 与 FPGA 输出进行比较；
- Fixed / Adaptive 实验。

# 15. LLM Planner
LLM Planner 使用兼容 OpenAI Chat Completions 格式的 API。
需要配置环境变量：LLM_BASE_URL/LLM_MODEL
如果接口需要 Key：LLM_API_KEY

PowerShell 示例：
```
$env:LLM_BASE_URL="YOUR_API_BASE_URL"
$env:LLM_MODEL="YOUR_MODEL_NAME"
$env:LLM_API_KEY="YOUR_API_KEY"
```

运行：
python main.py `
  --image test.png `
  --backend software `
  --planner llm `
  --cnn-checkpoint models/image_analyzer_v2_best.pth

LLM Planner 生成的 Pipeline 同样必须经过 `PipelineValidator`，因此 LLM 不能绕过软件和 FPGA 的执行约束。如果 LLM 请求失败，系统可以回退到 RulePlanner。

# 16. FPGA Backend 配置
当前 `config.yaml`：
```yaml
fpga:
  runtime: edf_dma_proxy
  uio_device: /dev/uio0
  ctrl_uio_map: 0
  gamma_uio_map: 1
  mmio_span: 0x10000
  dma_tx_device: /dev/dma_proxy_tx
  dma_rx_device: /dev/dma_proxy_rx
  dma_buffer_size: 524288
  expected_capability: 0x010007ff
  timeout_s: 5.0
```

# 17. FPGA Backend 运行
在 Linux 板卡上运行（需 dma-proxy.ko + UIO，用 sudo）：
sudo -E python3 main.py \
  --image test.png \
  --backend fpga \
  --planner rule \
  --cnn-checkpoint models/image_analyzer_v2_best.pth

FPGA Runtime（EdfDmaProxyRuntime）会完成：
1. 映射 `/dev/uio0`（CTRL/GAMMA AXI-Lite）；
2. 打开 `/dev/dma_proxy_tx`、`/dev/dma_proxy_rx`；
3. 编译 Pipeline；
4. 配置 FPGA Pipeline 寄存器；
5. 如果使用 Gamma，则写入 Gamma LUT；
6. 大图按 64×64 Tile 分块；
7. 发送图像数据；
8. 启动 FPGA；
9. 检查 FPGA status；
10. 接收输出图像。
FPGA Backend 不支持 HED，因此 FPGA 模式下应禁用 HED。

# 18. CNN V2 数据准备
生成 Image Analyzer V2 训练数据：
python prepare_training_data_v2.py
该脚本可以使用 DIV2K/SIDD/LOL/GoPro 构建训练和验证数据。
主要生成：
train_labels.json
val_labels.json
dataset_info.json

# 19. CNN V2 训练
运行：
python fine_tune_image_analyzer.py
主要输出：
models/image_analyzer_v2_best.pth
models/image_analyzer_v2_latest.pth
models/training_curve_v2.png
程序正式运行时主要使用：
models/image_analyzer_v2_best.pth
`image_analyzer_v2_latest.pth` 主要用于训练过程恢复或调试，不是部署必需文件。

# 20. CNN 阈值校准
训练完成后运行：
python calibrate_image_analyzer.py
生成：
models/image_analyzer_thresholds.json
该文件分别保存noise_level/impulse_noise/blur_level/low_light/low_contrast对应的校准阈值。
Planner 判断退化时应使用校准阈值，而不是简单统一使用 `0.5`。

# 21. 批量评估
`evaluate_adaptive_system.py` 用于批量测试 Adaptive Pipeline。
运行：
python evaluate_adaptive_system.py
默认测试数据集：
SIDD/LOL/GoPro
结果中会记录：
8-D image state/Pipeline/Planner reason
analyzer_ms/planner_ms/validator_ms/executor_ms/total_ms
该脚本适合检查不同数据集上的 Pipeline 选择情况和软件阶段耗时。

# 22. Fixed 与 Adaptive 对比实验
正式实验推荐使用：python compare_fixed_vs_adaptive.py
该脚本会比较：固定 Pipeline vs Adaptive Pipeline
并统计：
Precision/Recall/F1/IoU
Tolerance Precision/Tolerance Recall/Tolerance F1
Delta F1/Edge Pixel Count/Runtime
如果需要保存结果图：
python compare_fixed_vs_adaptive.py --save-images
该实验是评价 Adaptive Pipeline 是否真正优于固定 Pipeline 的主要实验之一。

# 23. 性能测试说明
软件会记录不同阶段的运行时间：analyzer_ms/planner_ms/validator_ms/executor_ms/total_ms
FPGA 性能分析时，不应直接把 `total_ms` 当作 FPGA 处理时间。
特别是 LLM Planner 可能包含网络请求，其耗时远高于图像处理本身。
因此性能报告建议分别统计：
 指标                      | 含义 
 Analyzer latency          | 图像状态分析时间 
 Planner latency           | Rule/LLM 规划时间 
 Validator latency         | Pipeline 校验时间 
 Software executor latency | 软件图像处理时间 
 FPGA executor latency     | DMA + FPGA 执行时间 
 FPS                       | 连续处理吞吐率 

真实系统中 Planner 不需要每帧运行。
可以对一段连续图像复用已经生成的 Pipeline，从而让 FPGA 作为高频数据处理模块运行。

# 24. HED
HED 当前仅用于 Software Backend。
FPGA Backend 不支持 HED。

# 25. V4 异步快慢双环 Agent
连续视频模式采用双时间尺度控制架构：
- Fast Loop：逐帧运行，根据 FPGA 输出反馈进行轻量参数微调；
- Slow Loop：在场景变化或持续异常时触发 DeepSeek 异步重规划；
- Pipeline Cache：缓存已验证的场景处理策略，场景再次出现时可直接复用；
- 异步规划：DeepSeek 在后台线程运行，不直接操作 FPGA，也不会阻塞当前帧的硬件执行。

完整流程为：
视频帧
→ 图像状态分析
→ 场景状态判断
→ Fast Loop 本地调整
→ 必要时触发 DeepSeek Slow Loop
→ PipelineValidator
→ FPGA 分块执行
→ 输出反馈
→ Pipeline 缓存与后续调整

# 26. 项目最终目标
本项目最终希望验证三个方面。
### 自适应能力
不同图像状态能够生成不同 Pipeline，而不是所有图像固定执行同一套处理流程。
### FPGA 动态执行能力
同一个 FPGA bitstream 可以在运行时通过寄存器切换：Gamma/Gaussian/Median/CLAHE/Sharpen/Roberts/Sobel/Canny/Opening/Closing
以及不同的算子组合。
### 性能与质量
最终需要从两个角度进行评价：
Fixed Pipeline vs Adaptive Pipeline用于评价自适应策略的边缘检测质量；
Software Executor vs FPGA Executor 用于评价 FPGA 硬件执行性能、DMA/分块开销以及与软件基线之间的差异。

# 27. 实时可视化界面
系统提供浏览器端 Live GUI，用于实时显示：
- 原始视频帧；
- FPGA 边缘检测结果；
- 当前场景状态；
- 当前 Pipeline；
- Fast Loop / Slow Loop 状态；
- DeepSeek 重规划事件；
- Pipeline Cache 命中情况；
- FPGA 处理延迟及运行状态。
板卡负责图像处理、Agent 控制和 FPGA 执行，PC 浏览器通过以太网进行实时监视，实现计算与显示解耦。
相关文件：
dashboard_live.html
dashboard.html