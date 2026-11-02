import json
import xml.etree.ElementTree as ET
from pathlib import Path

def _text(node, path: str, default=None):
    x = node.find(path)
    if x is not None and x.text is not None:
        return x.text
    return default

def parse_csynth_xml(path: str | Path) -> dict:
    root = ET.parse(path).getroot()
    summary = root.find(".//PerformanceEstimates/SummaryOfOverallLatency")
    timing = root.find(".//PerformanceEstimates/SummaryOfTimingAnalysis")
    resources = root.find(".//AreaEstimates/Resources")
    available = root.find(".//AreaEstimates/AvailableResources")
    out = {
        "latency_cycles_min": None,
        "latency_cycles_max": None,
        "interval_min": None,
        "interval_max": None,
        "estimated_clock_ns": None,
        "target_clock_ns": None,
        "resources": {},
        "available_resources": {},
    }
    if summary is not None:
        for key, xml_name in [
            ("latency_cycles_min", "Best-caseLatency"),
            ("latency_cycles_max", "Worst-caseLatency"),
            ("interval_min", "Interval-min"),
            ("interval_max", "Interval-max"),
        ]:
            v = _text(summary, xml_name)
            if v not in (None, "undef"):
                try:
                    out[key] = int(float(v))
                except ValueError:
                    pass
    if timing is not None:
        for key, xml_name in [
            ("estimated_clock_ns", "EstimatedClockPeriod"), 
            ("target_clock_ns", "TargetClockPeriod")
        ]:
            v = _text(timing, xml_name)
            if v:
                try:
                    out[key] = float(v)
                except ValueError:
                    pass
    for target, node in [
        ("resources", resources),
        ("available_resources", available)
    ]:
        if node is not None:
            for child in list(node):
                try:
                    out[target][child.tag] = int(float(child.text))
                except Exception:
                    pass
    return out

def optimization_advice(report: dict, skill_path: str | Path) -> list[str]:
    skill = json.loads(Path(skill_path).read_text(encoding="utf-8"))
    advice: list[str] = []
    ii = report.get("interval_max") or report.get("interval_min")
    if ii is not None and ii > 1:
        for rule in skill["rules"]:
            if rule["condition"] == "II > 1":
                advice.extend(rule["advice"])
    r = report.get("resources", {})
    a = report.get("available_resources", {})
    for key, condition in [("LUT", "LUT_high"), ("BRAM_18K", "BRAM_high"), ("DSP", "DSP_high")]:
        used = r.get(key)
        avail = a.get(key)
        if used is not None and avail:
            if used / avail >= 0.75:
                for rule in skill["rules"]:
                    if rule["condition"] == condition:
                        advice.extend(rule["advice"])
    if report.get("latency_cycles_max") and report["latency_cycles_max"] > 1_000_000:
        for rule in skill["rules"]:
            if rule["condition"] == "latency_high":
                advice.extend(rule["advice"])
    seen = set()
    result = []
    for item in advice:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result