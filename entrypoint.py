"""Start one ComfyUI instance, then one-at-a-time Runpod processing."""
from __future__ import annotations
import subprocess, sys, time
import requests
from handler import handle

def ready(url="http://127.0.0.1:8188", timeout=180):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if requests.get(url + "/system_stats", timeout=2).ok: return
        except requests.RequestException: pass
        time.sleep(1)
    raise RuntimeError("ComfyUI readiness failed")

def main():
    process = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--disable-auto-launch", "--fp8_e4m3fn-unet", "--cache-none"], cwd="/opt/ComfyUI")
    try:
        ready()
        import runpod
        runpod.serverless.start({"handler": handle, "return_aggregate_stream": False})
    finally:
        process.terminate()
        try: process.wait(timeout=20)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)

if __name__ == "__main__": main()
