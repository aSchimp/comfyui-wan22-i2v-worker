# ComfyUI Wan2.2 I2V worker

An auditable, fixed-workflow Runpod Serverless worker for image-to-video on a
24 GB GPU. It starts one local ComfyUI process and accepts one job at a time.
The image contains code only; all runtime assets stay on the private volume.

## API

The handler accepts `{"image":"<base64 PNG>","prompt":"...",
"negative_prompt":"...","duration":1..10,"seed":<optional uint32>,
"loras":[{"name":"alias","high_strength":0..2,"low_strength":0..2}]}`.
There may be at most four aliases. Unknown keys, malformed image data, URLs,
paths, non-finite weights, and invalid bounds are rejected. Success returns
`{"video":"<base64 MP4>"}`; output is H.264, has an MP4 `ftyp` header, and is
at most 7 MiB. Failures deliberately return only `{"error":"generation failed"}`.

## Volume layout and configuration

Mount `/runpod-volume` and provide basename-only `HIGH_MODEL`, `LOW_MODEL`,
`TEXT_ENCODER`, and `VAE_MODEL` variables. `extra_model_paths.yaml` maps the
model categories beneath `/runpod-volume/wan22/models`; aliases live in
`/runpod-volume/wan22/loras/<alias>/`. A folder contains exactly one high and
one low marked artifact, or exactly one shared artifact. Markers are matched
case-insensitively. Runtime runs offline and never downloads assets.

The fixed graph uses native core nodes, paired FP8 branches, six Euler/Simple
steps split at three, CFG 1, shift 7, and 16 fps. Frame count is
`duration * 16 + 1`; dimensions preserve the source aspect near a 640×640
pixel budget and are multiples of 16.

## Security and operations

Callers cannot provide workflows, filenames, network locations, or filesystem
paths. LoRA aliases are validated before private-volume discovery, output lookup
is prefix-bound, subprocess arguments are not shell-interpolated, and temporary
input/output files are removed in `finally`. The image pins its base and the
ComfyUI v0.33.1 commit. The release workflows are tag-only and publish to the
private registry and public GHCR mirror respectively.

Future providers or model families should add a separate fixed builder and
explicit contract rather than accepting arbitrary graphs in this worker.
