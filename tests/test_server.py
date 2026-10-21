import subprocess
import sys

import anyio
import pynvml
from mcp import Client

from mcp_gpu_server import server

CSV = "0, NVIDIA L40S, 580.126.09, 8.9, 29, 3, 1, 46068, 610, 45457\n"


def _no_nvml(monkeypatch):
    def fail():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_LIBRARY_NOT_FOUND)

    monkeypatch.setattr(pynvml, "nvmlInit", fail)


def test_nvidia_smi_fallback_serves_every_tool(monkeypatch):
    _no_nvml(monkeypatch)
    monkeypatch.setattr(
        server.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=CSV)
    )
    assert server.gpu_info() == {
        "count": 1,
        "gpus": [{"index": 0, "name": "NVIDIA L40S", "driver": "580.126.09", "cuda": None}],
    }
    assert server.gpu_vram()["gpus"][0] == {
        "index": 0, "total_mib": 46068, "used_mib": 610, "free_mib": 45457, "pct": 1.3,
    }
    assert server.gpu_utilization()["gpus"][0] == {"index": 0, "gpu_pct": 3, "mem_pct": 1}
    assert server.gpu_temperature()["gpus"][0] == {"index": 0, "temp_c": 29}


def test_no_gpu(monkeypatch):
    _no_nvml(monkeypatch)

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(server.subprocess, "run", missing)
    assert server.gpu_info() == {"count": 0, "gpus": []}
    assert server.gpu_stats()["error"]


def test_tools_over_mcp(monkeypatch):
    _no_nvml(monkeypatch)
    monkeypatch.setattr(
        server.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=CSV)
    )

    async def run():
        async with Client(server.mcp) as client:
            tools = (await client.list_tools()).tools
            assert {t.name for t in tools} == {
                "gpu_info", "gpu_utilization", "gpu_vram", "gpu_temperature", "gpu_stats",
            }
            assert all(t.annotations.read_only_hint for t in tools)
            res = await client.call_tool("gpu_info", {})
            assert not res.is_error
            assert res.structured_content["gpus"][0]["name"] == "NVIDIA L40S"

    anyio.run(run)


def test_stdio_entrypoint_starts():
    # stdout must stay clean for JSON-RPC; closing stdin should exit cleanly
    r = subprocess.run(
        [sys.executable, "-c", "from mcp_gpu_server.server import main; main()"],
        input="", capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == ""
