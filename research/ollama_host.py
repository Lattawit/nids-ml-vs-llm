"""Resolve a reachable Ollama host for the llm_classification_*.py scripts.

On this project's dev machine, Ollama runs as a native Windows install, not
inside WSL, so `localhost:11434` (the ollama-python default) is only reachable
from WSL when WSL mirrored networking is on. Otherwise WSL must reach it via
the WSL2 gateway IP (`ip route`'s `default via`), which is assigned by Hyper-V
per boot and is NOT stable across reboots, so it's detected at runtime instead
of hardcoded. This is unrelated to deploy/docker-compose, which runs Ollama as
a container reached at OLLAMA_HOST=http://ollama:11434 and needs no change.
"""

import os
import socket
import subprocess

import ollama


def _reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wsl_gateway_ip() -> str | None:
    try:
        result = subprocess.run(
            ["ip", "route"], capture_output=True, text=True, timeout=2, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        parts = line.split()
        if line.startswith("default") and "via" in parts:
            return parts[parts.index("via") + 1]
    return None


def resolve_ollama_host(port: int = 11434) -> str:
    """Return a reachable http://host:port base URL for Ollama.

    Order: OLLAMA_HOST env var (explicit override) > localhost (mirrored
    networking / native Linux Ollama) > auto-detected WSL2 gateway IP
    (native Windows Ollama reached from WSL).
    """
    env_host = os.getenv("OLLAMA_HOST")
    if env_host:
        return env_host

    if _reachable("localhost", port):
        return f"http://localhost:{port}"

    gateway = _wsl_gateway_ip()
    if gateway and _reachable(gateway, port):
        return f"http://{gateway}:{port}"

    raise RuntimeError(
        f"Could not reach Ollama at localhost:{port}"
        + (f" or the WSL gateway {gateway}:{port}" if gateway else "")
        + ". Is the Ollama daemon running? Set OLLAMA_HOST to override."
    )


def get_ollama_client() -> ollama.Client:
    """An ollama.Client pinned to a host resolved once at call time."""
    return ollama.Client(host=resolve_ollama_host())
