"""Strict Runpod handler and private-volume LoRA discovery."""
from __future__ import annotations

import base64
from io import BytesIO
import logging
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import threading
import time
import uuid

from PIL import Image, UnidentifiedImageError
import requests
from workflow import build_workflow, model_config

VOLUME = Path("/runpod-volume/wan22")
COMFY_INPUT = Path("/opt/ComfyUI/input")
MAX_VIDEO = 7 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_PROMPT = 4096
SAVE_NODE = "94"
ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
JOB_LOCK = threading.Lock()
LOG = logging.getLogger(__name__)


class ComfyStuckError(RuntimeError):
    """Raised when ComfyUI cannot confirm that an interrupted prompt stopped."""


def terminate_worker() -> None:
    """Fail closed: Runpod replaces the container instead of overlapping jobs."""
    os.kill(os.getpid(), signal.SIGTERM)


def fail():
    raise ValueError("invalid request")


def decode_png(encoded: str) -> tuple[bytes, tuple[int, int]]:
    try:
        data = base64.b64decode(encoded, validate=True)
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) > MAX_IMAGE_BYTES:
            fail()
        with Image.open(BytesIO(data)) as image:
            if image.width * image.height > MAX_PIXELS:
                fail()
            image.verify()
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                fail()
            image.load()
            if image.width * image.height > MAX_PIXELS:
                fail()
            return data, (image.width, image.height)
    except (ValueError, TypeError, SyntaxError, base64.binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError):
        fail()


def validate(payload: object) -> dict:
    allowed = {"image", "prompt", "negative_prompt", "duration", "seed", "loras"}
    required = {"image", "prompt", "negative_prompt", "duration"}
    if not isinstance(payload, dict) or set(payload) - allowed or not required <= set(payload): fail()
    if not isinstance(payload["image"], str) or len(payload["image"]) > (MAX_IMAGE_BYTES * 4 // 3 + 4): fail()
    if not all(isinstance(payload[key], str) and len(payload[key]) <= MAX_PROMPT for key in ("prompt", "negative_prompt")): fail()
    if not isinstance(payload["duration"], int) or isinstance(payload["duration"], bool) or not 1 <= payload["duration"] <= 10: fail()
    if "seed" in payload and (not isinstance(payload["seed"], int) or isinstance(payload["seed"], bool) or not 0 <= payload["seed"] <= 0xffffffff): fail()
    loras = payload.get("loras", [])
    if not isinstance(loras, list) or len(loras) > 4: fail()
    names = set()
    for item in loras:
        if not isinstance(item, dict) or set(item) != {"name", "high_strength", "low_strength"} or not isinstance(item["name"], str) or not ALIAS.fullmatch(item["name"]) or item["name"] in names: fail()
        names.add(item["name"])
        for key in ("high_strength", "low_strength"):
            value = item[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 2: fail()
    image, dimensions = decode_png(payload["image"])
    return {**payload, "image_bytes": image, "dimensions": dimensions, "seed": payload.get("seed", int.from_bytes(os.urandom(4), "big")), "loras": loras}


def lora_marker(path: Path) -> str | None:
    tokens = re.split(r"[^a-z0-9]+", path.stem.lower())
    stem = path.stem
    high_prefix = stem[:4].lower() == "high" and len(stem) > 4 and stem[4] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    low_prefix = stem[:3].lower() == "low" and len(stem) > 3 and stem[3] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    is_high = "highnoise" in tokens or "high" in tokens or "h" in tokens or any(re.search(r"\dhi$", token) for token in tokens) or high_prefix
    is_low = "lownoise" in tokens or "low" in tokens or "l" in tokens or any(re.search(r"\dlo$", token) for token in tokens) or low_prefix
    if is_high and is_low: fail()
    if is_high: return "high"
    if is_low: return "low"
    return None


def discover_lora(alias: str, root: Path = VOLUME / "loras") -> dict[str, str]:
    if not ALIAS.fullmatch(alias): fail()
    try: root = root.resolve(strict=True)
    except OSError: fail()
    folder = root / alias
    if folder.is_symlink() or not folder.is_dir() or folder.resolve().parent != root: fail()
    high, low, shared = [], [], []
    for path in folder.iterdir():
        if path.suffix.lower() != ".safetensors": continue
        if path.is_symlink() or not path.is_file(): fail()
        resolved = path.resolve(strict=True)
        try: resolved.relative_to(folder.resolve())
        except ValueError: fail()
        marker = lora_marker(path)
        (high if marker == "high" else low if marker == "low" else shared).append(path)
    if len(shared) == 1 and not high and not low:
        name = f"{alias}/{shared[0].name}"; return {"high": name, "low": name}
    if len(high) != 1 or len(low) != 1 or shared: fail()
    return {"high": f"{alias}/{high[0].name}", "low": f"{alias}/{low[0].name}"}


def valid_video(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size <= MAX_VIDEO and path.read_bytes()[:8][4:8] == b"ftyp"
    except OSError:
        return False


class ComfyClient:
    def __init__(self, url="http://127.0.0.1:8188", output=Path("/opt/ComfyUI/output")):
        self.url, self.output = url, output.resolve()

    def run(self, graph: dict, prefix: str, timeout=900) -> Path:
        ident = None
        try:
            try:
                response = requests.post(self.url + "/prompt", json={"prompt": graph}, timeout=15)
            except requests.RequestException as error:
                raise ComfyStuckError("ambiguous ComfyUI submission") from error
            response.raise_for_status()
            try:
                ident = response.json()["prompt_id"]
            except (KeyError, TypeError, ValueError) as error:
                raise ComfyStuckError("ambiguous ComfyUI submission") from error
            if not isinstance(ident, str) or not ident:
                raise ComfyStuckError("ambiguous ComfyUI submission")
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                response = requests.get(self.url + "/history/" + ident, timeout=15)
                response.raise_for_status(); history = response.json().get(ident, {})
                status = history.get("status", {}).get("status_str")
                if status in {"error", "failed", "cancelled"}: raise RuntimeError("ComfyUI generation failed")
                if status == "success": return self._output(history, prefix)
                time.sleep(1)
            raise RuntimeError("ComfyUI generation timed out")
        except Exception as error:
            if ident is not None and not self.cancel(ident):
                raise ComfyStuckError("ComfyUI prompt did not stop") from error
            raise

    def cancel(self, ident: str, timeout: float = 30) -> bool:
        """Interrupt a running prompt, delete it if queued, and confirm absence."""
        for endpoint, body in (("/interrupt", {}), ("/queue", {"delete": [ident]})):
            try:
                response = requests.post(self.url + endpoint, json=body, timeout=10)
                response.raise_for_status()
            except requests.RequestException:
                LOG.warning("ComfyUI cancellation request failed", exc_info=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = requests.get(self.url + "/queue", timeout=10)
                response.raise_for_status()
                queue = response.json()
                entries = queue.get("queue_running", []) + queue.get("queue_pending", [])
                if not any(isinstance(item, list) and len(item) > 1 and item[1] == ident for item in entries):
                    return True
            except (requests.RequestException, TypeError, ValueError):
                LOG.warning("Unable to confirm ComfyUI queue state", exc_info=True)
            time.sleep(.5)
        return False

    def cleanup_prefix(self, prefix: str) -> None:
        """Remove all direct output artifacts for this unguessable job prefix."""
        for path in self.output.glob(prefix + "*"):
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)
            except OSError:
                LOG.warning("Unable to remove ComfyUI output", exc_info=True)

    def _output(self, history: dict, prefix: str) -> Path:
        # ComfyUI's PreviewVideo intentionally uses the legacy "images" UI key.
        videos = history.get("outputs", {}).get(SAVE_NODE, {}).get("images", [])
        if not isinstance(videos, list) or len(videos) != 1: raise RuntimeError("unexpected output")
        item = videos[0]
        if not isinstance(item, dict) or set(item) - {"filename", "subfolder", "type"} or item.get("type") != "output": raise RuntimeError("unexpected output")
        filename, subfolder = item.get("filename"), item.get("subfolder", "")
        if not isinstance(filename, str) or not isinstance(subfolder, str) or Path(filename).name != filename or Path(subfolder).is_absolute() or ".." in Path(subfolder).parts or not filename.startswith(prefix) or Path(filename).suffix.lower() != ".mp4": raise RuntimeError("unexpected output")
        candidate = (self.output / subfolder / filename).resolve()
        try: candidate.relative_to(self.output)
        except ValueError: raise RuntimeError("unexpected output")
        if not candidate.is_file(): raise RuntimeError("unexpected output")
        return candidate


def encode_video(source: Path, target: Path, duration: int) -> None:
    bitrate = max(256, int((MAX_VIDEO - 256 * 1024) * 8 / duration / 1000))
    for _ in range(3):
        subprocess.run(["ffmpeg", "-y", "-i", str(source), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-b:v", f"{bitrate}k", "-maxrate", f"{bitrate}k", "-bufsize", f"{bitrate * 2}k", str(target)], check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        if valid_video(target): return
        bitrate = max(128, int(bitrate * .65))
    raise RuntimeError("video too large")


def handle(job: dict, client: ComfyClient | None = None) -> dict:
    with JOB_LOCK:
        work = Path("/tmp/worker") / uuid.uuid4().hex; work.mkdir(parents=True)
        comfy_input = source = prefix = None
        comfy = client or ComfyClient()
        fatal = False
        try:
            request = validate(job.get("input")); width, height = request["dimensions"]
            (work / "input.png").write_bytes(request["image_bytes"])
            comfy_input = COMFY_INPUT / (work.name + ".png"); shutil.copyfile(work / "input.png", comfy_input)
            resolved = [{**item, **discover_lora(item["name"])} for item in request["loras"]]
            prefix = "runpod_" + work.name
            graph = build_workflow(input_name=comfy_input.name, output_prefix=prefix, prompt=request["prompt"], negative_prompt=request["negative_prompt"], seed=request["seed"], duration=request["duration"], width=width, height=height, loras=resolved, config=model_config())
            source = comfy.run(graph, prefix); target = work / "result.mp4"; encode_video(source, target, request["duration"])
            return {"video": base64.b64encode(target.read_bytes()).decode("ascii")}
        except ComfyStuckError:
            fatal = True
            LOG.exception("worker stopping after unconfirmed ComfyUI cancellation")
        except Exception:
            LOG.exception("worker generation failure")
        finally:
            try:
                if source is not None: source.unlink(missing_ok=True)
                if comfy_input is not None: comfy_input.unlink(missing_ok=True)
            except OSError: pass
            if prefix is not None and hasattr(comfy, "cleanup_prefix"):
                comfy.cleanup_prefix(prefix)
            shutil.rmtree(work, ignore_errors=True)
        if fatal:
            terminate_worker()
        return {"error": "generation failed"}
