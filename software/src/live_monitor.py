from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe(to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {
                str(k): _json_safe(v)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass
    return str(value)


class LiveMonitor:
    def __init__(self, host="0.0.0.0", port=8080, jpeg_quality=82, tile_core=56):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.tile_core = int(tile_core)

        self._lock = threading.Lock()
        self._server = None
        self._thread = None
        self._source_jpg = None
        self._edge_jpg = None

        self._status = {
            "online": True,
            "frame": -1,
            "published_at": time.time(),
            "scene": "waiting",
            "planner_source": "—",
            "processing_ms": 0.0,
            "effective_fps": 0.0,
            "tiles": 0,
            "counters": {
                "slow_replans": 0,
                "deepseek_replans": 0,
                "cache_hits": 0,
                "fast_adjustments": 0,
            },
        }

        self._seen_slow_frames = set()
        self._seen_deepseek_frames = set()
        self._seen_cache_frames = set()
        self._seen_fast_frames = set()

    def start(self):
        if self._server is not None:
            return

        monitor = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "AdaptiveEdgeLive/1.0"

            def log_message(self, format, *args):
                return

            def _headers(self, code, content_type, length=None):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                if length is not None:
                    self.send_header("Content-Length", str(length))
                self.end_headers()

            def do_OPTIONS(self):
                self._headers(204, "text/plain", 0)

            def do_GET(self):
                path = urlparse(self.path).path

                if path == "/health":
                    payload = json.dumps({
                        "ok": True,
                        "service": "adaptive-edge-live",
                        "port": monitor.port,
                    }).encode("utf-8")
                    self._headers(200, "application/json; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                    return

                if path == "/status":
                    with monitor._lock:
                        snapshot = dict(monitor._status)
                        snapshot["counters"] = dict(monitor._status.get("counters", {}))
                    payload = json.dumps(
                        snapshot, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    self._headers(200, "application/json; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                    return

                if path in ("/source.jpg", "/edge.jpg"):
                    with monitor._lock:
                        blob = monitor._source_jpg if path == "/source.jpg" else monitor._edge_jpg
                    if blob is None:
                        payload = b"frame not available"
                        self._headers(404, "text/plain; charset=utf-8", len(payload))
                        self.wfile.write(payload)
                        return
                    self._headers(200, "image/jpeg", len(blob))
                    self.wfile.write(blob)
                    return

                payload = b"Adaptive Edge Live Monitor\n/health\n/status\n/source.jpg\n/edge.jpg\n"
                self._headers(200, "text/plain; charset=utf-8", len(payload))
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="adaptive-edge-live-http",
            daemon=True,
        )
        self._thread.start()
        print(f"[live-gui] HTTP monitor listening on http://0.0.0.0:{self.port}")

    def stop(self):
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()

    def _encode_jpg(self, image):
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.ndim not in (2, 3):
            return None
        ok, encoded = cv2.imencode(
            ".jpg",
            arr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        return encoded.tobytes() if ok else None

    @staticmethod
    def _contains(text, *needles):
        s = str(text or "").lower()
        return any(n.lower() in s for n in needles)

    def _update_counters(self, frame_index, meta):
        counters = self._status["counters"]
        slow_trigger = bool(meta.get("slow_trigger"))
        planner_source = str(meta.get("planner_source", ""))
        fast_reason = str(meta.get("fast_reason", ""))

        if slow_trigger and frame_index not in self._seen_slow_frames:
            self._seen_slow_frames.add(frame_index)
            counters["slow_replans"] += 1

        if self._contains(planner_source, "deepseek") and frame_index not in self._seen_deepseek_frames:
            self._seen_deepseek_frames.add(frame_index)
            counters["deepseek_replans"] += 1

        if self._contains(planner_source, "cache") and frame_index not in self._seen_cache_frames:
            self._seen_cache_frames.add(frame_index)
            counters["cache_hits"] += 1

        neutral_fast = (
            not fast_reason
            or self._contains(
                fast_reason,
                "no fast-loop correction",
                "plan reuse",
                "reuse",
                "stable",
                "none",
            )
        )
        if not neutral_fast and frame_index not in self._seen_fast_frames:
            self._seen_fast_frames.add(frame_index)
            counters["fast_adjustments"] += 1

    def publish(self, source_frame, edge_frame, meta, frame_index, processing_ms=None):
        meta_json = _json_safe(meta)
        if not isinstance(meta_json, dict):
            meta_json = {"meta": meta_json}

        rows, cols = source_frame.shape[:2]
        tiles = math.ceil(cols / self.tile_core) * math.ceil(rows / self.tile_core)

        if processing_ms is None:
            processing_ms = float(meta_json.get("processing_ms", 0.0) or 0.0)

        fps = 1000.0 / processing_ms if processing_ms and processing_ms > 0 else 0.0

        source_jpg = self._encode_jpg(source_frame)
        edge_jpg = self._encode_jpg(edge_frame)

        with self._lock:
            self._update_counters(int(frame_index), meta_json)
            counters = dict(self._status["counters"])

            status = {
                "online": True,
                "frame": int(frame_index),
                "published_at": time.time(),
                "processing_ms": float(processing_ms or 0.0),
                "effective_fps": float(fps),
                "tiles": int(tiles),
                "source_width": int(cols),
                "source_height": int(rows),
                "counters": counters,
            }
            status.update(meta_json)
            status["frame"] = int(frame_index)
            status["published_at"] = time.time()
            status["processing_ms"] = float(processing_ms or 0.0)
            status["effective_fps"] = float(fps)
            status["tiles"] = int(tiles)
            status["counters"] = counters

            self._status = status
            if source_jpg is not None:
                self._source_jpg = source_jpg
            if edge_jpg is not None:
                self._edge_jpg = edge_jpg
