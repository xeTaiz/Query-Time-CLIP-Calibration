# Query-time CLIP Calibration

Query-time calibration of lifted CLIP features.

## Environment

Install [uv](https://docs.astral.sh/uv/) and a compatible NVIDIA driver.

```sh
uv sync --locked
```

Image reproduction also needs a CUDA 13.0 toolkit. Set `CUDA_HOME_IMAGE` to its installation path.

## Usage

Run calibration on your own data:

```sh
uv run --locked python calibration.py \
  --gaussians my_data/gaussians.pt \
  --features my_data/features.pt \
  --queries my_data/queries.pt \
  --output scores.pt
```

Save floating-point tensors with `torch.save`. `N` is the number of Gaussians and `Q` is the number of queries.

- `gaussians.pt`: a dictionary with exactly `means` `[N, 3]`, `scales` `[N, 3]`, `quats` `[N, 4]` and `opacity` `[N]`. Use positive scales, WXYZ unit quaternions and opacity in `[0, 1]`, not log-scales or opacity logits.
- `features.pt`: a tensor `[N, 512]` with one CLIP feature per Gaussian in the same row order.
- `queries.pt`: a tensor `[Q, 512]` of text embeddings in the same CLIP space. The bundled reference uses ViT-B-16 with `laion2b_s34b_b88k`.

The default multiclass run needs at least two queries and eight eligible Gaussians. `scores.pt` contains `scores` `[N, Q]`, an `eligible` mask `[N]` and metadata.

See `uv run --locked python calibration.py --help` for options.

## Reproduction

Place the curated benchmark data in `data/`. It is not included in a normal checkout.

```sh
uv run --locked python -m eval.reproduce --output results/reproduction
```

This runs all four datasets and both appendix vocabularies. Results and comparisons are saved in the output folder.

The verified run matched 696 of 756 values at the paper's two-decimal precision, with no missing results or evaluation failures.

| Dataset / vocabulary | Matching values | Largest difference (percentage points) |
| --- | ---: | ---: |
| ScanNet / 19 | 103 / 126 | 0.052499 |
| ScanNet / 10 | 117 / 126 | 0.097696 |
| ScanNet++ / 100 | 104 / 126 | 0.171453 |
| ScanNet++ / 20 | 120 / 126 | 0.040350 |
| LERF-OVS | 126 / 126 | 0.004948 |
| 3D-OVS | 126 / 126 | 0.004922 |

Both image datasets match at printed precision. Point results are not fully exact. Parameters were not retuned to remove the differences.
