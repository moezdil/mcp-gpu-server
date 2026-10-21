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

Queries NVML (pynvml) directly when available. Falls back to nvidia-smi subprocess if NVML is not accessible. Returns clean JSON in both cases.

## Install

    pip install gpu-server

The package moved from `mcp-gpu-server` to `gpu-server` on PyPI in 0.2.1. The old name is frozen at 0.1.3, which does not start with mcp 2.x. The command is still `mcp-gpu-server`.

## Connect to your MCP client

Add this to your MCP client config file:

    {
      "mcpServers": {
        "gpu": {
          "command": "mcp-gpu-server"
        }
      }
    }


## Run tests

    uv run pytest tests/test_server.py   # no GPU needed
    uv run python tests/test_gpu.py      # on a GPU host

## Requirements

Python 3.10 or higher, MCP Python SDK 2.x. NVIDIA GPU with drivers installed on the host machine.
