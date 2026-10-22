import subprocess
from typing import Any
from mcp.server import CacheHint, MCPServer
from mcp.types import ToolAnnotations

mcp = MCPServer(
    "mcp-gpu-server",
    cache_hints={"tools/list": CacheHint(ttl_ms=86_400_000, scope="public")},
)
READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)


def _all_stats() -> dict[str, Any]:
    """Read stats via NVML, fall back to nvidia-smi (no CUDA version there)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        drv = pynvml.nvmlSystemGetDriverVersion()
        cuda = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        cuda_s = f"{cuda // 1000}.{(cuda % 1000) // 10}"
        gpus = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            gpus.append({
                "index": i,
                "name": name.decode() if isinstance(name, bytes) else name,
                "driver": drv.decode() if isinstance(drv, bytes) else drv,
                "cuda": cuda_s,
                "temp_c": temp,
                "gpu_pct": util.gpu,
                "mem_pct": util.memory,
                "vram": {
                    "total_mib": mem.total >> 20,
                    "used_mib": mem.used >> 20,
                    "free_mib": mem.free >> 20,
                    "pct": round(mem.used / mem.total * 100, 1),
                },
            })
        pynvml.nvmlShutdown()
        return {"count": count, "gpus": gpus}
    except Exception:
        pass

    try:
        rows = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,compute_cap,temperature.gpu,"
                "utilization.gpu,utilization.memory,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip().splitlines()
    except Exception:
        return {"count": 0, "gpus": [], "error": "No NVIDIA GPU or nvidia-smi not found"}

    gpus = []
    for row in rows:
        idx, name, drv, cap, temp, gu, mu, mtot, mused, mfree = (v.strip() for v in row.split(","))
        mtot_i, mused_i, mfree_i = int(mtot), int(mused), int(mfree)
        gpus.append({
            "index": int(idx),
            "name": name,
            "driver": drv,
            "cuda": None,
            "compute_cap": cap,
            "temp_c": int(temp),
            "gpu_pct": int(gu),
            "mem_pct": int(mu),
            "vram": {
                "total_mib": mtot_i,
                "used_mib": mused_i,
                "free_mib": mfree_i,
                "pct": round(mused_i / mtot_i * 100, 1),
            },
        })
    return {"count": len(gpus), "gpus": gpus}


@mcp.tool(annotations=READ_ONLY)
def gpu_info() -> dict[str, Any]:
    """GPU name, driver version, and CUDA version for each device."""
    s = _all_stats()
    return {"count": s["count"], "gpus": [
        {k: g[k] for k in ("index", "name", "driver", "cuda")} for g in s["gpus"]
    ]}


@mcp.tool(annotations=READ_ONLY)
def gpu_utilization() -> dict[str, Any]:
    """Busy % of the GPU cores (gpu_pct) and memory controller (mem_pct) for each device.

    mem_pct is memory bandwidth activity, not how full VRAM is; use gpu_vram for that.
    """
    s = _all_stats()
    return {"count": s["count"], "gpus": [
        {k: g[k] for k in ("index", "gpu_pct", "mem_pct")} for g in s["gpus"]
    ]}


@mcp.tool(annotations=READ_ONLY)
def gpu_vram() -> dict[str, Any]:
    """VRAM total, used, free (MiB) and utilization % for each device."""
    s = _all_stats()
    return {"count": s["count"], "gpus": [
        {"index": g["index"]} | g["vram"] for g in s["gpus"]
    ]}


@mcp.tool(annotations=READ_ONLY)
def gpu_temperature() -> dict[str, Any]:
    """Core temperature in °C for each device."""
    s = _all_stats()
    return {"count": s["count"], "gpus": [
        {k: g[k] for k in ("index", "temp_c")} for g in s["gpus"]
    ]}


@mcp.tool(annotations=READ_ONLY)
def gpu_stats() -> dict[str, Any]:
    """Full snapshot: info, VRAM, utilization, and temperature for every device."""
    return _all_stats()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
