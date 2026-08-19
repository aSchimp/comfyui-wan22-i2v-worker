import base64
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest
import handler
from workflow import build_workflow, dimensions, model_config

CONFIG = {"HIGH_MODEL": "high.safetensors", "LOW_MODEL": "low.safetensors", "TEXT_ENCODER": "text.safetensors", "VAE_MODEL": "vae.safetensors"}

def png(width=32, height=24):
    result = BytesIO(); Image.new("RGB", (width, height)).save(result, "PNG")
    return result.getvalue()

def payload(**changes):
    data = {"image": base64.b64encode(png()).decode(), "prompt": "a scene", "negative_prompt": "none", "duration": 2}
    data.update(changes); return data

@pytest.mark.parametrize("bad", [{}, payload(duration=11), payload(extra=1), payload(seed=-1), payload(image="bad"), payload(prompt="x" * 4097), payload(loras=[{"name": "one", "high_strength": 1, "low_strength": 1}, {"name": "one", "high_strength": 1, "low_strength": 1}])])
def test_validation_rejects_bad_contract(bad):
    with pytest.raises(ValueError): handler.validate(bad)

@pytest.mark.parametrize("data", [b"\x89PNG\r\n\x1a\n", png()[:-8]])
def test_validation_rejects_malformed_png(data):
    with pytest.raises(ValueError): handler.validate(payload(image=base64.b64encode(data).decode()))

def test_png_decode_and_pixel_limit(monkeypatch):
    assert handler.validate(payload())["dimensions"] == (32, 24)
    monkeypatch.setattr(handler, "MAX_PIXELS", 10)
    with pytest.raises(ValueError): handler.validate(payload())

def test_model_config_and_extreme_dimensions_are_bounded():
    assert model_config(CONFIG) == CONFIG
    with pytest.raises(ValueError): model_config({**CONFIG, "HIGH_MODEL": "../escape.safetensors"})
    width, height = dimensions(16_000_000, 1)
    assert 16 <= width <= 1280 and 16 <= height <= 1280

def test_pair_shared_markers_and_symlinks(tmp_path):
    root = tmp_path / "loras"; pair = root / "pair"; pair.mkdir(parents=True)
    (pair / "part_HIGH_NOISE.safetensors").touch(); (pair / "part-low.safetensors").touch()
    assert handler.discover_lora("pair", root)["high"].startswith("pair/")
    shared = root / "shared"; shared.mkdir(); (shared / "asset.safetensors").touch()
    assert handler.discover_lora("shared", root)["high"] == "shared/asset.safetensors"
    false = root / "false"; false.mkdir(); (false / "flower.safetensors").touch()
    assert handler.discover_lora("false", root)["low"] == "false/flower.safetensors"
    escaped = root / "escaped"; escaped.mkdir(); outside = tmp_path / "outside.safetensors"; outside.touch(); (escaped / "part-high.safetensors").symlink_to(outside)
    with pytest.raises(ValueError): handler.discover_lora("escaped", root)

def test_exact_fixed_graph_schema_and_lora_order():
    graph = build_workflow(input_name="input.png", output_prefix="out", prompt="p", negative_prompt="n", seed=1, duration=2, width=320, height=240, config=CONFIG, loras=[{"high": "x/a.safetensors", "low": "x/b.safetensors", "high_strength": 1, "low_strength": 0}])
    kinds = {key: value["class_type"] for key, value in graph.items()}
    lora = next(key for key, value in kinds.items() if value == "LoraLoaderModelOnly")
    assert graph["9"]["inputs"]["model"] == [lora, 0]
    assert set(graph["90"]["inputs"]) == {"model", "add_noise", "noise_seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "latent_image", "start_at_step", "end_at_step", "return_with_leftover_noise"}
    assert set(graph["93"]["inputs"]) == {"images", "fps"}
    assert graph["90"]["inputs"]["model"] == ["9", 0]
    assert graph["91"]["inputs"]["model"] == ["10", 0]
    assert set(graph["94"]["inputs"]) == {"video", "filename_prefix", "format", "codec", "codec.encoding", "codec.encoding.crf"}
    assert graph["8"]["inputs"]["length"] == 33
    w, h = dimensions(320, 240); assert w % 16 == h % 16 == 0 and w > h

def test_history_output_path_checks(tmp_path):
    output = tmp_path / "output"; output.mkdir(); video = output / "runpod_id_00001_.mp4"; video.write_bytes(b"\0\0\0\x18ftypisom")
    client = handler.ComfyClient(output=output)
    history = {"outputs": {"94": {"videos": [{"filename": video.name, "subfolder": "", "type": "output"}]}}}
    assert client._output(history, "runpod_id") == video
    history["outputs"]["94"]["videos"][0]["filename"] = "other.mp4"
    with pytest.raises(RuntimeError): client._output(history, "runpod_id")

def test_cancel_interrupts_dequeues_and_confirms(monkeypatch, tmp_path):
    calls = []
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"queue_running": [], "queue_pending": []}
    monkeypatch.setattr(handler.requests, "post", lambda url, **kwargs: calls.append((url, kwargs["json"])) or Response())
    monkeypatch.setattr(handler.requests, "get", lambda *args, **kwargs: Response())
    client = handler.ComfyClient(output=tmp_path)
    assert client.cancel("prompt-id", timeout=.1)
    assert calls == [(client.url + "/interrupt", {}), (client.url + "/queue", {"delete": ["prompt-id"]})]

def test_run_raises_stuck_when_cancellation_is_unconfirmed(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"prompt_id": "prompt-id"}
    monkeypatch.setattr(handler.requests, "post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(handler.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(handler.requests.ConnectionError()))
    client = handler.ComfyClient(output=tmp_path)
    monkeypatch.setattr(client, "cancel", lambda ident: False)
    with pytest.raises(handler.ComfyStuckError): client.run({}, "prefix", timeout=0)

def test_ambiguous_submission_failure_is_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(handler.requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(handler.requests.ConnectionError()))
    with pytest.raises(handler.ComfyStuckError):
        handler.ComfyClient(output=tmp_path).run({}, "prefix")

def test_cleanup_prefix_removes_only_owned_outputs(tmp_path):
    owned = tmp_path / "job_1.mp4"; owned.write_bytes(b"x")
    other = tmp_path / "other.mp4"; other.write_bytes(b"x")
    handler.ComfyClient(output=tmp_path).cleanup_prefix("job_")
    assert not owned.exists() and other.exists()

def test_iterative_encode_and_handle_source_cleanup(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"; source.write_bytes(b"\0\0\0\x18ftypisom")
    target = tmp_path / "target.mp4"; calls = []
    def fake_run(*args, **kwargs):
        calls.append(args); target.write_bytes(b"\0\0\0\x18ftypisom" if len(calls) == 2 else b"x" * (handler.MAX_VIDEO + 1))
    monkeypatch.setattr(handler.subprocess, "run", fake_run)
    handler.encode_video(source, target, 10); assert len(calls) == 2
    input_dir = tmp_path / "input"; input_dir.mkdir(); monkeypatch.setattr(handler, "COMFY_INPUT", input_dir); monkeypatch.setattr(handler, "model_config", lambda: CONFIG); monkeypatch.setattr(handler, "discover_lora", lambda _: {})
    class Client:
        def run(self, graph, prefix): return source
        def cleanup_prefix(self, prefix): pass
    monkeypatch.setattr(handler, "encode_video", lambda src, dst, duration: dst.write_bytes(b"\0\0\0\x18ftypisom"))
    assert "video" in handler.handle({"input": payload()}, Client())
    assert not source.exists() and not list(input_dir.iterdir())
