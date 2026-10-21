import json
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

SEP = "=================================================="


def _section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def _check(cond: bool, msg: str) -> None:
    _ok(msg) if cond else _fail(msg)


_section("Import")
try:
    from mcp_gpu_server.server import (
        _all_stats, gpu_info, gpu_stats,
        gpu_temperature, gpu_utilization, gpu_vram,
    )
    _ok("mcp_gpu_server imported")
except ImportError as e:
    _fail(f"import failed: {e}")


_section("_all_stats()")
stats = _all_stats()
print(json.dumps(stats, indent=2))
_check(isinstance(stats, dict), "returns dict")
_check("count" in stats and "gpus" in stats, "has count and gpus keys")
_check(stats["count"] > 0, f"at least 1 GPU (found {stats['count']})")

gpu = stats["gpus"][0]
for field in ("index", "name", "driver", "cuda", "temp_c", "gpu_pct", "mem_pct", "vram"):
    _check(field in gpu, f"gpu[0] has {field}")

vram = gpu["vram"]
for field in ("total_mib", "used_mib", "free_mib", "pct"):
    _check(field in vram, f"vram has {field}")

_check(vram["total_mib"] > 0, f"VRAM total > 0 ({vram['total_mib']} MiB)")
_check(vram["used_mib"] <= vram["total_mib"], "used_mib <= total_mib")
_check(vram["free_mib"] <= vram["total_mib"], "free_mib <= total_mib")
_check(0 <= vram["pct"] <= 100, f"vram pct in range (got {vram['pct']})")
_check(0 <= gpu["gpu_pct"] <= 100, f"gpu_pct in range (got {gpu['gpu_pct']})")
_check(0 <= gpu["mem_pct"] <= 100, f"mem_pct in range (got {gpu['mem_pct']})")
_check(0 <= gpu["temp_c"] <= 120, f"temp_c in range (got {gpu['temp_c']})")


_section("gpu_info()")
info = gpu_info()
print(json.dumps(info, indent=2))
_check(info["count"] == stats["count"], "count matches")
_check(all(k in info["gpus"][0] for k in ("index", "name", "driver", "cuda")), "required fields present")
_check("temp_c" not in info["gpus"][0], "no temp_c (correct scope)")


_section("gpu_utilization()")
util = gpu_utilization()
print(json.dumps(util, indent=2))
_check(all(k in util["gpus"][0] for k in ("index", "gpu_pct", "mem_pct")), "required fields present")
_check("name" not in util["gpus"][0], "no name (correct scope)")


_section("gpu_vram()")
vram_tool = gpu_vram()
print(json.dumps(vram_tool, indent=2))
_check(all(k in vram_tool["gpus"][0] for k in ("index", "total_mib", "used_mib", "free_mib", "pct")), "required fields present")


_section("gpu_temperature()")
temp_tool = gpu_temperature()
print(json.dumps(temp_tool, indent=2))
_check(all(k in temp_tool["gpus"][0] for k in ("index", "temp_c")), "required fields present")
_check(len(temp_tool["gpus"][0]) == 2, "only index and temp_c")


_section("gpu_stats()")
full = gpu_stats()
print(json.dumps(full, indent=2))
_check(full == stats, "gpu_stats() equals _all_stats()")


_section("nvidia-smi fallback")
try:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    _ok(f"nvidia-smi available: {r.stdout.strip()}")
except FileNotFoundError:
    print("  [SKIP] nvidia-smi not on PATH")


_section("pynvml direct")
try:
    import pynvml
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    _ok(f"pynvml init OK, {count} device(s)")
    for i in range(count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(h)
        _ok(f"  device {i}: {name if isinstance(name, str) else name.decode()}")
    pynvml.nvmlShutdown()
    _ok("pynvml shutdown OK")
except Exception as e:
    print(f"  [SKIP] pynvml unavailable: {e}")


_section("CLI entrypoint")
result = subprocess.run(
    [sys.executable, "-c", "from mcp_gpu_server.server import main"],
    capture_output=True, text=True,
)
_check(result.returncode == 0, "entrypoint importable")


_section("All tests passed")
print(f"\n  GPU: {gpu['name']}")
print(f"  VRAM: {vram['total_mib']} MiB total, {vram['used_mib']} MiB used ({vram['pct']}%)")
print(f"  Temp: {gpu['temp_c']}C   GPU util: {gpu['gpu_pct']}%\n")
