mcp-name: io.github.moezdil/mcp-gpu-server

# mcp-gpu-server

[![PyPI](https://img.shields.io/pypi/v/gpu-server)](https://pypi.org/project/gpu-server/)

An MCP server that exposes NVIDIA GPU metrics as tools. Once connected, any MCP-compatible client can query your GPU status in real time directly from a conversation.

## What it does

Instead of running nvidia-smi manually, you ask your AI assistant and it calls these tools automatically:

    gpu_info         GPU name, driver version, CUDA version
    gpu_utilization  core utilization % and memory bandwidth %
    gpu_vram         total, used, free VRAM in MiB and usage %
    gpu_temperature  GPU core temperature in Celsius
    gpu_stats        everything above in one call

Example response from gpu_stats:

    {
      "count": 1,
      "gpus": [{
        "index": 0,
        "name": "NVIDIA L40S",
        "driver": "580.126.09",
        "cuda": "13.0",
        "temp_c": 29,
        "gpu_pct": 0,
        "mem_pct": 0,
        "vram": {
          "total_mib": 46068,
          "used_mib": 610,
          "free_mib": 45457,
          "pct": 1.3
        }
      }]
    }

## How it works

Queries NVML (pynvml) directly when available. Falls back to nvidia-smi subprocess if NVML is not accessible. Returns clean JSON in both cases. The nvidia-smi query does not include the CUDA version, so `cuda` is `null` on that path.

## Install

    pip install gpu-server

The package moved from `mcp-gpu-server` to `gpu-server` on PyPI in 0.2.1. The old name is frozen at 0.1.3, which does not start with mcp 2.x. The `mcp-gpu-server` command still works, and since 0.2.2 `gpu-server` does too.

Listed on the official [MCP Registry](https://registry.modelcontextprotocol.io) as `io.github.moezdil/mcp-gpu-server`.

## Connect to your MCP client

Add this to your MCP client config file:

    {
      "mcpServers": {
        "gpu": {
          "command": "mcp-gpu-server"
        }
      }
    }

Or without installing anything, using [uv](https://docs.astral.sh/uv/):

    {
      "mcpServers": {
        "gpu": {
          "command": "uvx",
          "args": ["gpu-server"]
        }
      }
    }


## Run tests

    uv run pytest tests/test_server.py   # no GPU needed
    uv run python tests/test_gpu.py      # on a GPU host

## Tested on real hardware

Release 0.2.2 was installed from PyPI into a clean virtual environment on a Linux host and tested against a real GPU on 2026-10-21.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GB |
| Driver / CUDA | 580.178.04 / 13.0 |
| Python | 3.14.4 |
| Packages | gpu-server 0.2.2, mcp 2.3.0, nvidia-ml-py 13.615.71 |

Install the published package and run the GPU integration test:

    python3 -m venv venv
    venv/bin/pip install gpu-server==0.2.2
    git clone --depth 1 https://github.com/moezdil/mcp-gpu-server.git repo
    venv/bin/python repo/tests/test_gpu.py

Result: all 37 checks passed, for both the NVML path and nvidia-smi.

Then call every tool through a real MCP client over stdio:

    import anyio, json, sys
    from mcp import Client, StdioServerParameters

    async def main():
        async with Client(StdioServerParameters(command=sys.argv[1])) as c:
            print("protocol:", c.protocol_version)
            for t in (await c.list_tools()).tools:
                r = await c.call_tool(t.name, {})
                print(t.name, t.annotations.read_only_hint, json.dumps(r.structured_content))

    anyio.run(main)

Saved as `e2e.py` and run with `venv/bin/python e2e.py venv/bin/gpu-server`. Result: the client and server negotiated protocol `2026-07-28`, all five tools are listed as read-only, and every call returned structured content without errors:

    gpu_stats {"count": 1, "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3090",
      "driver": "580.178.04", "cuda": "13.0", "temp_c": 42, "gpu_pct": 0, "mem_pct": 0,
      "vram": {"total_mib": 24576, "used_mib": 451, "free_mib": 24124, "pct": 1.8}}]}

## Requirements

Python 3.10 or higher, MCP Python SDK 2.x. NVIDIA GPU with drivers installed on the host machine.

## Links

- PyPI: https://pypi.org/project/gpu-server/
- MCP Registry entry: https://registry.modelcontextprotocol.io/v0/servers?search=io.github.moezdil/mcp-gpu-server
- Model Context Protocol: https://modelcontextprotocol.io
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- uv: https://docs.astral.sh/uv/
- NVML (nvidia-ml-py): https://pypi.org/project/nvidia-ml-py/
