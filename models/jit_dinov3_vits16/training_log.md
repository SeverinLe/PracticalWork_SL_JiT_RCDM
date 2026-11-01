# `jit_dinov3` — training log

Source data for §4.5 (hardware and compute) of the report.  Filled from W&B runs
`c48aue7x` (DinoV3) and `6h9y3bt6` (RETFound CFP) on 2026-09-27; the rubric's top
band requires precise hardware and compute time, not estimates.

`step0055000.pt` is not a separate run — it is the checkpoint written at the
first session boundary of `c48aue7x`, kept as a training-length ablation.

## Runs

| Field | `final.pt` | `step0055000.pt` |
|---|---|---|
| W&B run id | `c48aue7x` | `c48aue7x` (same run) |
| Git commit | `a27a5e7` | `a27a5e7` |
| Steps | 100 000 | 55 000 |
| Encoder | DinoV3 ViT-S/16, frozen, `h_dim` 384 | same |
| Preset | `JiT_S_16` — hidden 384, depth 12, heads 6, patch 16, `cond_dim` 128 | same |
| Parameters | 25.7 M | 25.7 M |
| Image size | 224 × 224 | 224 × 224 |
| Objective | flow matching, x-prediction, `t ~ sigmoid(N(-0.8, 0.8))` | same |
| EMA | decay 0.9999, stored in checkpoint | same |
| Optimiser | AdamW, lr 1e-4, β = (0.9, 0.95), weight decay 0.0 | same |
| LR schedule | linear warmup 2000 steps, then constant | same |
| Batch size | 128 (`--grad_accum 1`, effective 128) | same |
| CFG dropout | 0.1 | same |
| Dataset | Messidor-2 train, 972 images (packed cache `train_packed.pt`) | same |
| Hardware | Google Colab — 1 × NVIDIA A100-SXM4-40GB, 12 vCPU, 83 GiB RAM, Python 3.12.13 | same |
| Wall-clock | 12 h 23 m elapsed (incl. a 2 h 33 m gap between sessions) | — |
| GPU-hours | **9.83** | 5.35 (first session) |
| Final train loss | 4.68 × 10⁻⁴ | — |
| Checkpoint sha256 | see `final.cfg.json` | see `step0055000.cfg.json` |

Executed on Colab, so there is no local `wandb/run-*-c48aue7x` directory; the
figures above come from the W&B API (`run.config`, `wandb-metadata.json`, and the
`step` / `_timestamp` history).

### Session structure

`c48aue7x` covers steps 0 → 100 000 **from scratch** in two Colab sessions:

| Session | Steps | GPU-h | Throughput | Started (UTC) |
|---|---|---|---|---|
| 1 | 100 → 55 100 | 5.35 | 10 275 steps/h | 2026-05-16 21:06 |
| 2 | 55 100 → 100 000 | 4.48 | 10 023 steps/h | 2026-05-17 05:00 |

The matching throughput across both sessions is the evidence that batch size did
not change mid-run — the `wandb-metadata.json` args are only captured for the
session that wrote them last, so the config alone cannot rule that out.

## Comparison run — `jit_retfound_cfp/final.pt`

Recorded here so §4.5 can state the two are matched at equal compute.

| | DinoV3 `c48aue7x` | RETFound CFP `6h9y3bt6` |
|---|---|---|
| Git commit | `a27a5e7` | `9e6694e` |
| `h_dim` | 384 | 1024 |
| Parameters | 25.7 M | 25.8 M |
| Batch / lr / warmup / steps | 128 / 1e-4 / 2000 / 100 k | 128 / 1e-4 / 2000 / 100 k |
| `cfg_dropout` / EMA / `cond_dim` | 0.1 / 0.9999 / 128 | 0.1 / 0.9999 / 128 |
| Sessions | 2 | 3 |
| GPU-hours | 9.83 | 10.13 |
| Final train loss | 4.68 × 10⁻⁴ | 4.38 × 10⁻⁴ |

Every hyper-parameter except `h_dim` is identical, and both runs cover steps
0 → 100 000. RETFound's extra 0.3 GPU-hours are two resume rollbacks that
recomputed ≈ 2 200 steps (12 100 → 10 100 and 95 300 → 95 100); the step range it
ultimately covers is still exactly 0 → 100 000, so the budgets are comparable.

Worth stating in the report: RETFound reached a *slightly lower* final training
loss than DinoV3 while failing E1 at chance level (MRR 0.046 vs chance 0.036).
The inversion failure is therefore not underfitting — see
`experiments/results/E7_conditioning_bottleneck/` for the 128/1024 projector
bottleneck as the more likely mechanism.

## Provenance note

Both files were previously named `jit_rcdm_retfound_step0100000.pt` and
`jit_rcdm_step0055000.pt`.  The first name was wrong: the stored `model_cfg`
reads `h_dim: 384`, which is DinoV3 (RETFound would be 1024).  Renamed during
the 2026-08-03 restructure; `scripts/train.py` now writes `encoder` into
`model_cfg` so the ambiguity cannot recur.

## Related runs (checkpoints deleted, metrics still in W&B)

| Run | What it was | Why it is not here |
|---|---|---|
| `oavlyyc5` | 50 000 steps with an incorrect `h`, batch size 8 (≈ 2 000 steps/h) | Known-bad; kept as a documented negative result, not as weights. **Not an ancestor of `c48aue7x`** — that run trained from scratch, so no bad-`h` weights leak into the released model |
| `nmf6l3rz` | hidden 768 / `cond_dim` 768 | Capacity ablation |
| `rctoipbq` | `cond_dim` 64 | Bottleneck-width ablation |
| `6gecr1sw` | "S32" (stored config says patch 16) | Ablation; name and config disagree |
| `sq7b3ske`, `34elcigr` | t-sampler ablation, μ = −0.8 | Crashed at 19 k / 9.2 k steps |

See `archive/MANIFEST.md` for sizes and full configurations.
