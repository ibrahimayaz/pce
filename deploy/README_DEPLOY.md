# GUI deployment

## Online GUI (Hugging Face Space, Gradio SDK)

The public GUI runs at https://huggingface.co/spaces/ibrahimayaz/pce-predictor. Its source is
[`hf_space/`](hf_space/):

| File | Content |
|---|---|
| `app.py` | Gradio interface; uses the same final pipelines as the Flask GUI |
| `models_v2/` | Copy of `../../models_v2/` (final CatBoost pipelines and metadata) |
| `rev/` | `features.py` from `revision/rev/`, needed to load the pipelines |
| `admixtures.json` | Descriptors X13–X18 of the 22 admixtures (preset list) |
| `requirements.txt`, `README.md` | Package versions and Space configuration (`sdk: gradio`) |

To update the Space, upload the changed files (Files → Add file → Upload files) or use the CLI:

```bash
pip install -U huggingface_hub
hf auth login
hf upload ibrahimayaz/pce-predictor deploy/hf_space . --repo-type=space
```

If `models_v2/` or `revision/rev/features.py` change, copy them into `hf_space/` first.

Notes: the Space runs on ZeroGPU hardware, which requires one `@spaces.GPU` function at start-up; `app.py`
defines an unused placeholder and all predictions run on CPU. Server-side rendering is disabled
(`ssr_mode=False`) because it stopped the Space at start-up.

## Flask GUI in a container (alternative hosts)

`Dockerfile` serves the Flask GUI (`app.py`) with gunicorn on port 7860 (e.g. Render, Railway or a Docker
Space):

```bash
docker build -f deploy/Dockerfile -t pce-gui .
docker run -p 7860:7860 pce-gui      # http://localhost:7860
```
