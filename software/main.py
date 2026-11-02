import argparse
import json
import time
from pathlib import Path

import cv2

from src.config import load_config
from src.image_features import classical_image_state
from src.pipeline_planner import RulePlanner
from src.pipeline_validator import PipelineValidator
from src.hed_runner import HEDRunner
from src.pipeline_executor import PipelineExecutor
from src.visualize import make_side_by_side, save_image


def parse_args():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--video")
    source.add_argument("--camera", type=int)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--backend", choices=["software", "fpga"], default="software")
    ap.add_argument("--planner", choices=["rule", "llm", "agent"], default=None)
    ap.add_argument("--cnn-checkpoint", default=None)
    ap.add_argument("--gt-edge", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--agent-max-iterations", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=0, help="0 means process until end/quit")
    ap.add_argument("--display", action="store_true", help="show live side-by-side preview")
    return ap.parse_args()


def build_hed_if_available(cfg: dict):
    h = cfg["hed"]
    p = Path(h["prototxt"])
    m = Path(h["caffemodel"])
    if p.exists() and m.exists():
        return HEDRunner(p, m, (int(h["input_width"]), int(h["input_height"])))
    return None


def load_gt(path: str | None):
    if not path:
        return None
    gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"Cannot read GT edge image: {path}")
    return gt


def build_common(args, cfg, rows: int, cols: int):
    planner_cfg = dict(cfg["planner"])
    planner_cfg["backend"] = args.backend
    planner_cfg["image_rows"] = int(rows)
    planner_cfg["image_cols"] = int(cols)
    planner_cfg.setdefault("allow_hed", True)
    if args.backend == "fpga":
        planner_cfg["allow_hed"] = False

    skill_path = Path(args.config).parent / "skills" / "image_pipeline_skill.json"
    validator = PipelineValidator(skill_path)

    fpga_runtime = None
    if args.backend == "fpga":
        from src.fpga_runtime import EdfDmaProxyRuntime
        fpga_runtime = EdfDmaProxyRuntime(cfg)

    hed_runner = build_hed_if_available(cfg)
    executor = PipelineExecutor(
        backend=args.backend,
        fpga_runtime=fpga_runtime,
        hed_runner=hed_runner,
    )
    return planner_cfg, skill_path, validator, executor


def build_analyzer(args):
    if args.cnn_checkpoint:
        from src.cnn_infer import CNNImageAnalyzer
        analyzer = CNNImageAnalyzer(args.cnn_checkpoint)
        return analyzer.analyze, "cnn"
    return classical_image_state, "classical"


def build_llm_planner(skill_path, validator):
    from src.llm_planner import LLMPlanner
    return LLMPlanner(skill_path, validator=validator)


def run_image(args, cfg):
    total_t0 = time.perf_counter()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    analyzer, analyzer_name = build_analyzer(args)
    analyzer_t0 = time.perf_counter()
    state = analyzer(image)
    analyzer_ms = (time.perf_counter() - analyzer_t0) * 1000.0

    planner_cfg, skill_path, validator, executor = build_common(args, cfg, image.shape[0], image.shape[1])
    planner_mode = args.planner or planner_cfg.get("mode", "agent")
    gt = load_gt(args.gt_edge)
    agent_trace = None
    agent_stop_reason = None

    planner_t0 = time.perf_counter()
    if planner_mode == "agent":
        from src.agent_tools import AgentTools
        from src.fpga_agent import DeepSeekFPGAAgent
        try:
            llm_planner = build_llm_planner(skill_path, validator)
            tools = AgentTools(validator=validator, executor=executor, constraints=planner_cfg, gt_edge=gt)
            agent_cfg = dict(cfg.get("agent") or {})
            max_iterations = args.agent_max_iterations or int(agent_cfg.get("max_iterations", 2))
            agent = DeepSeekFPGAAgent(
                planner=llm_planner,
                tools=tools,
                max_iterations=max_iterations,
                edge_density_min=float(agent_cfg.get("edge_density_min", 0.015)),
                edge_density_max=float(agent_cfg.get("edge_density_max", 0.25)),
                max_tiny_component_ratio=float(agent_cfg.get("max_tiny_component_ratio", 0.65)),
                min_gt_f1=float(agent_cfg.get("min_gt_f1", 0.80)),
            )
            agent_result = agent.run(image, state, planner_cfg)
            plan, result = agent_result.plan, agent_result.result
            agent_trace, agent_stop_reason = agent_result.trace, agent_result.stop_reason
            planner_used = "deepseek-agent"
            executor_ms = sum(float(x["executor_ms"]) for x in agent_trace)
            validator_ms = 0.0
        except Exception as exc:
            print(f"[agent warning] DeepSeek agent failed: {exc}")
            print("[agent warning] falling back to deterministic RulePlanner")
            plan = RulePlanner().plan(state, planner_cfg)
            planner_used = "rule-fallback"
            vt0 = time.perf_counter()
            validator.validate(plan, max_pre=int(planner_cfg["max_pre_ops"]), max_post=int(planner_cfg["max_post_ops"]), allow_hed=bool(planner_cfg["allow_hed"]), target_backend=args.backend, image_rows=image.shape[0], image_cols=image.shape[1])
            validator_ms = (time.perf_counter() - vt0) * 1000.0
            et0 = time.perf_counter()
            result = executor.run(image, plan)
            executor_ms = (time.perf_counter() - et0) * 1000.0
    else:
        if planner_mode == "llm":
            try:
                plan = build_llm_planner(skill_path, validator).plan(state, planner_cfg)
                planner_used = "llm"
            except Exception as exc:
                print(f"[planner warning] LLM planner failed: {exc}")
                plan = RulePlanner().plan(state, planner_cfg)
                planner_used = "rule-fallback"
        else:
            plan = RulePlanner().plan(state, planner_cfg)
            planner_used = "rule"
        vt0 = time.perf_counter()
        validator.validate(plan, max_pre=int(planner_cfg["max_pre_ops"]), max_post=int(planner_cfg["max_post_ops"]), allow_hed=bool(planner_cfg["allow_hed"]), target_backend=args.backend, image_rows=image.shape[0], image_cols=image.shape[1])
        validator_ms = (time.perf_counter() - vt0) * 1000.0
        et0 = time.perf_counter()
        result = executor.run(image, plan)
        executor_ms = (time.perf_counter() - et0) * 1000.0

    planner_ms = (time.perf_counter() - planner_t0) * 1000.0
    total_ms = (time.perf_counter() - total_t0) * 1000.0
    output_dir = Path(cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else output_dir / f"{Path(args.image).stem}_edge.png"
    save_image(out_path, result)
    vis_path = out_path.with_name(out_path.stem + "_compare.jpg")
    save_image(vis_path, make_side_by_side(image, result, title=f"{planner_used} | exec={executor_ms:.2f} ms"))

    print("\n=== Image State ===")
    print(json.dumps(state.to_dict(), indent=2))
    print("\n=== Pipeline ===")
    print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
    if agent_trace is not None:
        print("\n=== Agent Trace ===")
        print(json.dumps(agent_trace, indent=2, ensure_ascii=False))
        print(f"agent_stop_reason={agent_stop_reason}")
    print("\n=== Runtime ===")
    print(f"analyzer={analyzer_name}")
    print(f"planner={planner_used}")
    print(f"backend={args.backend}")
    print(f"analyzer_ms={analyzer_ms:.3f}")
    print(f"planner_or_agent_ms={planner_ms:.3f}")
    print(f"validator_ms={validator_ms:.3f}")
    print(f"executor_ms_total={executor_ms:.3f}")
    print(f"total_ms={total_ms:.3f}")
    print(f"result={out_path}")
    print(f"comparison={vis_path}")


def run_video(args, cfg):
    from src.agent_tools import AgentTools
    from src.temporal_controller import FastLoopController
    from src.video_agent import DualLoopVideoAgent, VideoRunStats, make_demo_frame, append_jsonl

    source = args.camera if args.camera is not None else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-3 or fps > 240:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        ok, probe = cap.read()
        if not ok:
            raise RuntimeError("Cannot read first video frame")
        height, width = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    planner_cfg, skill_path, validator, executor = build_common(args, cfg, height, width)
    analyzer, analyzer_name = build_analyzer(args)
    tools = AgentTools(validator=validator, executor=executor, constraints=planner_cfg, gt_edge=None)
    agent_cfg = dict(cfg.get("agent") or {})
    video_cfg = dict(cfg.get("video_agent") or {})

    llm_planner = None
    if (args.planner or planner_cfg.get("mode", "agent")) == "agent":
        try:
            llm_planner = build_llm_planner(skill_path, validator)
        except Exception as exc:
            print(f"[video agent warning] DeepSeek unavailable: {exc}")
            print("[video agent warning] continuing with local fast loop + RulePlanner bootstrap/cache")

    fast = FastLoopController(
        ema_alpha=float(video_cfg.get("ema_alpha", 0.25)),
        enter_factor=float(video_cfg.get("hysteresis_enter_factor", 1.05)),
        exit_factor=float(video_cfg.get("hysteresis_exit_factor", 0.85)),
        scene_change_threshold=float(video_cfg.get("scene_change_threshold", 0.18)),
        min_replan_interval_frames=int(video_cfg.get("min_replan_interval_frames", max(15, int(fps)))),
        periodic_replan_frames=int(video_cfg.get("periodic_replan_frames", max(90, int(fps * 10)))),
        feedback_bad_frames=int(video_cfg.get("feedback_bad_frames", 5)),
        edge_density_min=float(agent_cfg.get("edge_density_min", 0.015)),
        edge_density_max=float(agent_cfg.get("edge_density_max", 0.25)),
        max_tiny_component_ratio=float(agent_cfg.get("max_tiny_component_ratio", 0.65)),
        threshold_step=int(video_cfg.get("threshold_step", 4)),
        fast_adjust_bad_frames=int(video_cfg.get("fast_adjust_bad_frames", 3)),
        fast_adjust_cooldown_frames=int(video_cfg.get("fast_adjust_cooldown_frames", 5)),
    )
    controller = DualLoopVideoAgent(
        analyzer=analyzer,
        validator=validator,
        tools=tools,
        constraints=planner_cfg,
        fast_controller=fast,
        llm_planner=llm_planner,
        enable_scene_cache=bool(video_cfg.get("enable_scene_cache", True)),
        refine_on_feedback=bool(video_cfg.get("refine_on_feedback", True)),
    )

    output_dir = Path(cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    elif args.video:
        out_path = output_dir / f"{Path(args.video).stem}_agent_demo.mp4"
    else:
        out_path = output_dir / "camera_agent_demo.mp4"
    log_path = out_path.with_suffix(".jsonl")
    if log_path.exists():
        log_path.unlink()

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {out_path}")

    stats = VideoRunStats()
    frame_index = 0
    print(f"[video] source={source} fps={fps:.2f} size={width}x{height} analyzer={analyzer_name}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            t0 = time.perf_counter()
            # LIVE_GUI_V1_BEGIN
            if not hasattr(run_video, "_live_monitor"):
                from src.live_monitor import LiveMonitor
                run_video._live_monitor = LiveMonitor(
                    host="0.0.0.0",
                    port=8080,
                    jpeg_quality=82,
                    tile_core=56,
                )
                run_video._live_monitor.start()

            _live_t0 = __import__("time").perf_counter()
            edge, meta = controller.process_frame(frame, frame_index)
            _live_processing_ms = (
                __import__("time").perf_counter() - _live_t0
            ) * 1000.0

            try:
                run_video._live_monitor.publish(
                    frame,
                    edge,
                    meta,
                    frame_index,
                    processing_ms=_live_processing_ms,
                )
            except Exception as _live_exc:
                if frame_index == 0:
                    print("[live-gui] publish warning:", _live_exc)
            # LIVE_GUI_V1_END
            processing_ms = (time.perf_counter() - t0) * 1000.0
            meta["processing_ms"] = processing_ms
            meta["source_fps"] = fps

            demo = make_demo_frame(frame, edge, meta)
            writer.write(demo)
            append_jsonl(log_path, meta)

            stats.frames += 1
            stats.processing_ms_sum += processing_ms
            if meta["slow_trigger"]:
                stats.slow_replans += 1
            if meta.get("slow_requested"):
                stats.async_requests += 1
            if meta.get("fast_adjusted"):
                stats.fast_adjustments += 1
            if meta.get("cache_hit"):
                stats.cache_hits += 1
            if meta.get("cache_switch"):
                stats.cache_switches += 1
            if meta.get("slow_pending"):
                stats.pending_frames += 1

            async_result = meta.get("async_result") or {}
            if async_result.get("ready"):
                stats.llm_ms_sum += float(async_result.get("llm_ms", 0.0))
                async_source = str(async_result.get("source", ""))
                if async_source.startswith("deepseek"):
                    stats.deepseek_replans += 1
                if async_source == "deepseek-error":
                    stats.fallback_replans += 1
                if async_result.get("applied"):
                    stats.async_applied += 1
                if async_result.get("rejected"):
                    stats.async_rejected += 1

            stats.plan_generation = max(stats.plan_generation, int(meta.get("plan_generation", 0)))

            if meta["slow_trigger"]:
                action = "queued" if meta.get("slow_requested") else ("cache" if meta.get("cache_hit") else "pending/skip")
                print(f"[frame {frame_index}] SLOW-EVENT: {meta['slow_reason']} | action={action} | scene={meta['scene']}")
            if async_result.get("ready"):
                print(
                    f"[frame {frame_index}] ASYNC-RESULT: source={async_result.get('source')} "
                    f"applied={async_result.get('applied')} scene={async_result.get('scene')} "
                    f"llm_ms={float(async_result.get('llm_ms', 0.0)):.1f}"
                )

            if args.display:
                cv2.imshow("EdgeAgent-FPGA Dual Loop", demo)
                if (cv2.waitKey(1) & 0xFF) in {27, ord('q')}:
                    break
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()
        controller.close(wait=False)

    summary = stats.to_dict()
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Dual-loop Video Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"video={out_path}")
    print(f"trace={log_path}")
    print(f"summary={summary_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.video is not None or args.camera is not None:
        run_video(args, cfg)
    else:
        run_image(args, cfg)


if __name__ == "__main__":
    main()
