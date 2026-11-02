# V4 Validation Report

## 1. Static validation

- `python -m compileall -q .`: PASS

## 2. Non-blocking slow-loop simulation

A fake LLM planner was configured to sleep for 250 ms before returning a valid plan.
Observed behavior:

- Frame 0 used the deterministic RulePlanner bootstrap plan immediately (~33 ms in the test run).
- Frames 1-4 continued processing while the fake LLM was still pending.
- The asynchronous plan became ready later and was applied by the video/main thread.
- The background worker did not execute the pipeline and did not access FPGA runtime.

Result: PASS (`ASYNC_TEST_OK`).

## 3. Fast-loop persistence/cooldown

Test configuration:

- abnormal feedback on every frame
- `fast_adjust_bad_frames = 3`
- `fast_adjust_cooldown_frames = 5`
- Canny starts at 50/120

Observed adjustments:

- frame 2: 50/120 -> 46/112
- frame 7: 46/112 -> 42/104

There were no frame-by-frame threshold changes.

Result: PASS (`FAST_COOLDOWN_OK`).

## 4. Software video smoke test

The uploaded night-sky test clip was processed successfully with the V4 software path using the deterministic planner fallback.

Result: PASS.

## 5. Still required on the user's machine/board

- Run with the real DeepSeek endpoint and confirm `ASYNC-RESULT` messages while video continues.
- Run the FPGA backend and verify DMA/MMIO behavior with V4.
- Compare V3 and V4 summaries on the same video.
