# flux-server

OpenAI Images API compatible server for `black-forest-labs/FLUX.2-klein-4B`
(text-to-image and multi-reference image editing) built on FastAPI and diffusers'
`Flux2KleinPipeline`, with a Gradio debug UI mounted at `/ui`.

Device selection is automatic (cuda → mps → cpu). Developed on Apple Silicon (MPS);
the CUDA path is untested on this machine.

## Quickstart

```sh
make sync       # create .venv and install CPU PyTorch
make sync-cuda  # install CUDA PyTorch (NVIDIA GPU + driver required)
make download   # fetch model weights into the Hugging Face cache (once)
make run        # start the server on http://127.0.0.1:8000
```

Then open <http://127.0.0.1:8000/ui> for the debug UI.

Weights must already be present locally (`make download`); the server loads with
`FLUX_LOCAL_FILES_ONLY=1` by default and does not fetch from the network.

## API examples

Text-to-image:

```sh
curl -s http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox in the snow, film photo", "size": "1024x1024", "n": 1}' \
  | python3 -c 'import sys,json,base64; d=json.load(sys.stdin); open("out.png","wb").write(base64.b64decode(d["data"][0]["b64_json"]))'
```

Image editing (one or more reference images):

```sh
curl -s http://127.0.0.1:8000/v1/images/edits \
  -F image=@ref.jpg \
  -F prompt='make it a watercolor painting' \
  | python3 -c 'import sys,json,base64; d=json.load(sys.stdin); open("edit.png","wb").write(base64.b64decode(d["data"][0]["b64_json"]))'
```

Responses use `response_format=b64_json` only. Extra parameters `seed` and `steps`
are accepted; the seed used is echoed back in each `data[i].seed`.

### Inpainting (`mask`) and partial redraw (`strength`)

By default `/v1/images/edits` uses the model's native multi-reference editing: every
uploaded image conditions the generation and the prompt describes the result. Adding a
`mask` and/or `strength` switches to the inpainting pipeline (`Flux2KleinInpaintPipeline`,
same weights), where the **first** image is the canvas:

```sh
curl -s http://127.0.0.1:8000/v1/images/edits \
  -F image=@photo.png -F mask=@mask.png -F prompt='a red sports car' \
  -F strength=0.8 -F steps=8
```

- `mask`: transparent pixels (alpha = 0) mark the area to repaint, like the OpenAI API; a
  mask without an alpha channel is read as grayscale with white = repaint. Pixels outside
  the mask are copied back from the original image. Any size is accepted (resized to the
  image).
- `strength` (extension, `0 < strength <= 1`, default `1`): how far to move away from the
  first image; without a mask the whole image is redrawn from a noised version of itself
  (img2img). Only `floor(steps * strength)` denoising steps run, so with the default 4 steps
  the useful values are `0.25 / 0.5 / 0.75 / 1`; raise `steps` for finer control.
- A second `image[]` entry becomes the reference for the repainted area (at most one extra
  image in this mode). `size` must stay `auto` — the output follows the first image
  (downscaled to ≤ 1 MP, multiples of 16).
- In the debug UI the mask can be painted directly on the first reference image (Image to
  Image → **Inpainting** accordion) instead of uploading a file.

### Cancelling a generation

Every request gets an id, echoed in the `X-Request-Id` response header. Send your own
`X-Request-Id` (1-128 characters from `[A-Za-z0-9._-]`) to be able to cancel while the
request is still running:

```sh
curl -s http://127.0.0.1:8000/v1/images/generations -H 'X-Request-Id: job-42' \
  -H 'Content-Type: application/json' -d '{"prompt": "...", "n": 4}' &
curl -s -X POST http://127.0.0.1:8000/v1/images/job-42/cancel   # -> {"request_id": "job-42", "cancelled": true}
```

The generation stops at the next denoising step (or before the next image when `n > 1`)
and the original request fails with `409` / `code: "cancelled"`; an unknown or finished
id gives `404`. Requests whose client disconnects (closed tab, `Ctrl-C`, SDK timeout) are
cancelled the same way automatically, so a queued job never runs for nobody. With the
`openai` SDK: `client.images.generate(..., extra_headers={"X-Request-Id": "job-42"})`.
`/health` reports the number of `in_flight` requests.

## Debug UI

Open <http://127.0.0.1:8000/ui> (`/` redirects there). The header shows a live status line
from `/health` (model, device, dtype, offload, default steps, in-flight requests; refreshed
when the API base changes). Tabs: **Text to Image**, **Image to Image** (multi-reference
editing), **Status** (`/health` and `/v1/models`). The page runs in the server process but
talks to the `/v1` API through the official `openai` Python SDK, so it doubles as an
end-to-end API check. Each tab is two columns: the left has the prompt (**Shift+Enter**
submits), the **Generate / Edit** and **Cancel** buttons, the basic parameters (`size`,
`quality`, `output_format`) and a collapsed **Advanced** accordion (`steps`, `seed`, `n`,
custom width/height for `size=custom`, `output_compression` for jpeg/webp); the Text to
Image tab also offers a few clickable example prompts. The right column is the result
gallery with a collapsed **Request / response details** accordion below it: the request
parameters and the equivalent `curl` command update live as you edit the form (no need to
wait for a generation), and the response metadata (seeds, timings) fills in as the
generation progresses. With `n > 1` the page sends one `n=1` request per image, chaining
seeds `s, s+1, …` exactly as the server does internally for a single `n` request (so the
images are identical), and shows each image as soon as it finishes. Result images are
returned to the browser inline (base64 data URLs, byte-exact API output in the chosen
`output_format`) — nothing is written to disk or kept on the server, and the gallery's
download button saves them as `seed-<n>.<ext>`. A status line under the buttons tracks the
run (`Generating · 1/4 done`, `Done · 4 images · server 57.3s / total 58.0s`, …);
**Cancel** cancels the run in progress through the cancel endpoint — the current image
stops at the end of its current denoising step (up to one step of latency, e.g. ~15 s at
1024² on an M1 Max) and no further images are started; the line then reads
`Cancelled · k/n done`. Reference images for Image to Image are dropped into a gallery (its
order is the `image[]` order); **Send to Image to Image** under the Text to Image results
puts the selected image there and switches tabs, and **Continue editing with selected
result** on the Image to Image tab feeds the selected result back as the sole reference.
The collapsed **Inpainting** accordion under the reference gallery holds a paint canvas
(`gr.ImageEditor`) that loads the first reference automatically: brush over the area to
repaint and it is sent as the `mask` (painted area = repaint); a mask file can still be
uploaded there instead, but a non-empty painting always wins over the file. The **API base**
and **model** fields in the collapsed **Connection** accordion apply to every request, so
the page can drive a remote server or send `model=gpt-image-1` to test client
compatibility. `FLUX_UI=0` disables the page; `FLUX_UI_API_BASE` sets the default API base.

`scripts/smoke_openai_sdk.py` runs the same generate + edit flow from the command line
against a running server (`--base-url`, `--image`, `--skip-edit`; images land in `--out-dir`).

`scripts/tune_parameters.py` benchmarks size/steps combinations against a running server
without changing any defaults. It prints one JSON record per request and a fastest successful
combination for each tested size:

```sh
uv run python scripts/tune_parameters.py --warmup
uv run python scripts/tune_parameters.py --sizes 512x512,768x768,1024x1024 --steps 2,4 --repeats 2
```

The server configuration being tested is reported by `/health`, including device, dtype, and
offload. Set `FLUX_DTYPE` and `FLUX_OFFLOAD` before starting the server to compare those modes.

## Configuration

All settings are read from environment variables with the `FLUX_` prefix
(a `.env` file is also loaded). See [.env.example](.env.example) for the full,
documented list.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLUX_MODEL` | `black-forest-labs/FLUX.2-klein-4B` | Model id to serve |
| `FLUX_MODEL_PATH` | *(empty)* | Optional local snapshot directory |
| `FLUX_MODEL_ID` | `flux.2-klein-4b` | Id reported by `/v1/models` (request `model` is not validated) |
| `FLUX_DEVICE` | `auto` | `auto` \| `cuda` \| `mps` \| `cpu` |
| `FLUX_DTYPE` | `auto` | `auto` \| `bfloat16` \| `float16` \| `float32` |
| `FLUX_OFFLOAD` | `auto` | `auto` \| `none` \| `model` \| `sequential` |
| `FLUX_WEIGHTS` | `auto` | `auto` \| `diffusers` \| `single_file` |
| `FLUX_HOST` / `FLUX_PORT` | `127.0.0.1` / `8000` | Bind address |
| `FLUX_DEFAULT_STEPS` | `4` | Default inference steps |
| `FLUX_QUALITY_STEPS` | `{"low":2,"medium":4,"high":8,"auto":4,"standard":4,"hd":8}` | JSON quality→steps map used when `steps` is absent (replaces the whole map; unknown values → default steps) |
| `FLUX_MAX_N` | `10` | Max images per request |
| `FLUX_MAX_PIXELS` | `4194304` | Max output pixels per image |
| `FLUX_MAX_REF_IMAGES` | `16` | Max reference images for edits |
| `FLUX_MAX_UPLOAD_MB` | `32` | Max size per uploaded file (image/mask) |
| `FLUX_MAX_IN_FLIGHT` | `8` | Max concurrent requests; excess requests get 503 `code: "overloaded"` |
| `FLUX_UI` | `1` | Mount the Gradio UI at `/ui` |
| `FLUX_UI_API_BASE` | `http://127.0.0.1:{PORT}/v1` | API base used by the UI |
| `FLUX_WARMUP` | `0` | Run a warmup generation at startup |
| `FLUX_LOCAL_FILES_ONLY` | `1` | Never download weights at runtime |

## Development notes

- This checkout lives inside iCloud Drive, which re-applies the macOS hidden flag to
  dot-prefixed paths and makes CPython skip hidden `.pth` files in a `.venv`.
- `make test`, `make lint`, `make fmt` run pytest and ruff inside that environment.
