import argparse
import json

from src.hls_report_analyzer import (parse_csynth_xml, optimization_advice)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csynth_xml")
    ap.add_argument("--skill", default="skills/hls_optimization_skill.json")
    args = ap.parse_args()
    report = parse_csynth_xml(args.csynth_xml)
    print(json.dumps(report, indent=2))
    advice = optimization_advice(report, args.skill)
    print("\nOptimization advice:")
    if not advice:
        print("- No obvious rule-based issue detected.")
    else:
        for item in advice:
            print(f"- {item}")

if __name__ == "__main__":
    main()