from pathlib import Path
import re
import shutil

MAIN = Path("main.py")
BACKUP = Path("main.py.before_live_gui_v1")

if not MAIN.exists():
    raise SystemExit("ERROR: run this script from the project root (main.py not found)")

text = MAIN.read_text()

if "# LIVE_GUI_V1_BEGIN" in text:
    raise SystemExit("LIVE GUI V1 patch already present; no changes made")

if not BACKUP.exists():
    shutil.copy2(MAIN, BACKUP)
    print(f"backup: {BACKUP}")

pattern = re.compile(
    r'^(?P<indent>[ \t]*)edge,\s*meta\s*=\s*controller\.process_frame'
    r'\(frame,\s*frame_index\)\s*$',
    re.MULTILINE,
)

match = pattern.search(text)
if not match:
    raise SystemExit(
        "ERROR: could not find exact video line:\n"
        "    edge, meta = controller.process_frame(frame, frame_index)\n"
        "No file changed."
    )

indent = match.group("indent")
original = match.group(0)

replacement = f'''{indent}# LIVE_GUI_V1_BEGIN
{indent}if not hasattr(run_video, "_live_monitor"):
{indent}    from src.live_monitor import LiveMonitor
{indent}    run_video._live_monitor = LiveMonitor(
{indent}        host="0.0.0.0",
{indent}        port=8080,
{indent}        jpeg_quality=82,
{indent}        tile_core=56,
{indent}    )
{indent}    run_video._live_monitor.start()

{indent}_live_t0 = __import__("time").perf_counter()
{original}
{indent}_live_processing_ms = (
{indent}    __import__("time").perf_counter() - _live_t0
{indent}) * 1000.0

{indent}try:
{indent}    run_video._live_monitor.publish(
{indent}        frame,
{indent}        edge,
{indent}        meta,
{indent}        frame_index,
{indent}        processing_ms=_live_processing_ms,
{indent}    )
{indent}except Exception as _live_exc:
{indent}    if frame_index == 0:
{indent}        print("[live-gui] publish warning:", _live_exc)
{indent}# LIVE_GUI_V1_END'''

MAIN.write_text(text[:match.start()] + replacement + text[match.end():])

print("LIVE GUI V1 PATCH OK")
print("next:")
print("  python3 -m py_compile main.py src/live_monitor.py")
