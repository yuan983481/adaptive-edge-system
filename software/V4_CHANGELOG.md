# V4 Stable Changelog

## Main changes

- Asynchronous DeepSeek slow loop: network planning runs in one background worker.
- FPGA safety: the background worker never executes FPGA pipelines or touches DMA/MMIO.
- Non-blocking bootstrap: frame 0 immediately uses a validated RulePlanner plan.
- Scene -> Pipeline cache for immediate reuse when a scene reappears.
- Hysteresis-confirmed scene transitions bypass the generic slow-loop cooldown.
- Fast-loop persistence: abnormal feedback must persist for multiple frames.
- Fast-loop cooldown: threshold changes cannot happen on every frame.
- Expanded JSONL and summary telemetry for V3/V4 A/B evaluation.

## Compatibility

- Existing image mode remains unchanged, including the original DeepSeek closed-loop agent.
- Existing `software` and `fpga` backends remain available.
- Existing operator schema, validator, compiler and FPGA register contract remain unchanged.

## Recommended release policy

Keep V3 as a frozen validation baseline. Use V4 as the active development/competition branch.
