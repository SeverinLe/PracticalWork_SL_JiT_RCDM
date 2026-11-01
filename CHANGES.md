# JiT-RCDM: changes relative to RCDM and JiT

Final-version change log. It lists what this repository takes from each
upstream, what it changes, and what it adds. Upstream behaviour was checked
against the `main` branches of both repositories.

- **RCDM**: [facebookresearch/RCDM](https://github.com/facebookresearch/RCDM)
  (Bordes et al. 2022). A guided-diffusion UNet conditioned on a frozen SSL
  representation.
- **JiT**: [LTH14/JiT](https://github.com/LTH14/JiT). A plain ViT denoiser,
  x-prediction, flow matching, class-conditional.

In one sentence: **JiT's denoiser and flow-matching recipe, with JiT's class
label replaced by RCDM's frozen SSL representation `h`.**

`[fix-N]` tags refer to code comments in `rcdm/jit.py`, `rcdm/conditioning.py`
and `scripts/train.py`.

---

## At a glance

| | RCDM | JiT | **This repo** |
|---|---|---|---|
| Conditioning signal | SSL representation (ResNet-50 SSL models) | class label `y` | frozen SSL representation `h` (DINOv3 / RETFound, 384 or 1024-dim CLS) |
| Denoiser | guided-diffusion UNet | ViT (JiT-B/L/H) | ViT, **JiT-S preset** (new) |
| Injection | conditional BatchNorm | adaLN-Zero + 32 in-context class tokens | **adaLN-Zero only** |
| Conditioning width | 512 | `hidden_dim` | **`cond_dim = 128`** (new bottleneck) |
| Objective | DDPM, ε-prediction, 1000 steps | flow matching, x-pred, **v-loss** | flow matching, x-pred, **x-loss** |
| Sampler | DDPM, 100 respaced steps | Heun, 50 steps | Heun, 50 steps |
| CFG | none | label dropout 0.1, extra null-class row | `h` dropout 0.1, **learnable `null_h`** |
| Resolution | 128 px | 256 px | **224 px** |
| Encoder at train time | run online | n/a | **precomputed once, cached** |

---

## 1. From RCDM

**Kept**
- Conditioning a generator on a frozen SSL representation instead of a class
  label. The encoder stays frozen (`eval()`, `requires_grad=False`).
- Two normalisations that are never mixed: ImageNet mean/std for the encoder
  input, `[-1, 1]` for the diffusion target.
- `ConditioningProjector` = `Linear(h_dim, cond_dim) + SiLU`
  (`rcdm/conditioning.py`). Both widths are now configurable.
- `RepresentationDataset` concept: `(image, h)` pairs (`rcdm/dataset.py`).

**Changed**

| RCDM | This repo | Why |
|---|---|---|
| guided-diffusion UNet + DDPM (ε-pred, 1000 steps) | JiT ViT + flow matching (§2) | ViT backbone matching the encoders; 50-step ODE sampling |
| conditional BatchNorm on `(B, C, H, W)` maps | adaLN-Zero on `(B, N, D)` tokens | cBN has no meaning for a token sequence. `ConditionalBatchNorm2d` is still in `conditioning.py` but nothing uses it |
| ResNet-50 SSL encoders (vissl) | registry of DINOv3 ViT-S/16, DINOv3 ViT-L/16, RETFound-MAE, RETFound-DINOv3 (`rcdm/encoders/`) | retinal vs web domain, MAE vs DINO objective |
| projector output 512 | `cond_dim = 128` | regularising bottleneck for 972 training images |
| encoder run every training step | representations cached once (`data/scripts/precompute_reps.py`) | training, sampling and probes see byte-identical `h` |
| 128 px | 224 px | the encoders' native resolution; their 14 × 14 patch grid is fixed |
| no CFG | CFG with learnable `null_h` (§2) | |

---

## 2. From JiT

**Kept**
- ViT denoiser: Conv patch embed → 12 × block → final layer → unpatchify.
- adaLN-Zero with a 6-way modulation (shift/scale/gate for attention and FFN),
  zero-initialised so every block starts as the identity `[fix-5a]`.
- RMSNorm `[fix-5a]`, 2D RoPE on Q/K `[fix-5b]`, per-head QK-RMSNorm
  `[fix-5c]`, bias-free SwiGLU `[fix-5d]`. The SwiGLU inner width is
  `4·h·2/3`, rounded up to a multiple of 256; this equals JiT's value (1024)
  at `h = 384`.
- Final layer: RMSNorm + adaLN shift/scale (no gate), zero-initialised.
- x-prediction on the linear path `z_t = t·x + (1−t)·ε`, `noise_scale = 1`.
- Logit-normal t-sampler `t = σ(N(−0.8, 0.8))` `[fix-4b]`.
- Heun ODE, 50 steps, uniform grid 0 → 1, plain Euler on the last step.
- AdamW with β = (0.9, 0.95) `[fix-4a]`, weight decay 0.
- EMA of the weights, decay 0.9999 `[fix-1]`.

**Changed**

| JiT | This repo | Note |
|---|---|---|
| `c = t_emb + class_emb(y)` | `c = t_emb + cond_proj(h)` | continuous `h` replaces the label table |
| conditioning width = `hidden_dim` | `cond_dim = 128` | adaLN MLPs become `128 → 6·hidden_dim` |
| 32 in-context class tokens prepended mid-network | removed | `h` enters only through adaLN |
| bottleneck patch embed (`p×p` conv → 128 → 1×1 → hidden) | single `p×p` conv → hidden | |
| fixed sin-cos `pos_embed` **and** RoPE | RoPE only `[fix-5b]` | |
| timestep MLP: freq 256 → hidden → hidden | sinusoidal(`cond_dim`) → 4·`cond_dim` → `cond_dim` | |
| null class = extra embedding row | learnable `null_h ∈ ℝ^{h_dim}`, before the projector, in the EMA `[fix-3]` | dropout probability 0.1 in both |
| **v-loss** `‖v − v̂‖²`, `(1−t)` clamped at 0.05 | **x-loss** `‖x − x̂‖²` by default; JiT's v-loss via `--loss_space v` `[fix-6]` | x-loss is JiT's loss reweighted by `(1−t)²`, which down-weights the clean end of the path. All reported generators use x-loss |
| CFG blended on velocity, optional t-interval | CFG blended on `x̂`, no interval | identical at a given `(z, t)`, since `v` is affine in `x̂`; JiT's default interval [0, 1] is also the full range |
| two EMAs (0.9999, 0.9996) | one EMA (0.9999), always applied on load | |
| lr = 5e-5 · batch / 256, constant | lr 1e-4, linear warmup 2000 steps, then constant `[fix-4c]`; grad-norm clip 1.0 | |
| presets JiT-B/L/H (/16, /32), 256 px | `JiT_S_16` / `JiT_S_32`: hidden 384, depth 12, 6 heads, ≈ 25.7 M params, 224 px | all reported generators use `JiT_S_16` |
| — | gradient accumulation `--grad_accum` `[fix-4d]` | |
| — | `t_mu`, `t_sigma` exposed as flags and stored in `model_cfg` | |

---

## 3. Added (in neither upstream)

- **Encoder registry** (`rcdm/encoders/`): one module per encoder exposing
  `H_DIM` and `load()`, plus a token-sequence output for attentive probing.
  `transforms.py` holds the 224 px transform and the nuisance transforms that
  E3 and E5 share.
- **Representation caches** record `encoder`, `h_dim`, `paths` and `labels`.
  A token-cache variant and a packed variant with uint8 images embedded (for
  Colab, via `pack_dataset.py`) are also supported.
- **Checkpoint contract.** Checkpoints carry `model_cfg` (architecture,
  `encoder`, `t_mu`, `t_sigma`) and the EMA state `[fix-1]`. The architecture
  is rebuilt from `model_cfg`, never from CLI flags, and EMA weights are
  applied on load.
- **Guards that raise instead of producing plausible nonsense:**
  - `train.py` refuses a cache whose `encoder` or `h_dim` contradicts
    `--encoder`.
  - Probes refuse a generator/encoder pair whose `h_dim` doesn't match.
  - Encoder input is always 224 px, whatever the generator's resolution
    `[fix-2]`.
- **Provenance.** Every experiment run writes `run_config.json` (arguments,
  seed, git commit, checkpoint sha256, library versions) before any compute
  starts. Every checkpoint has a `*.cfg.json` sidecar with its sha256 and W&B
  run id.

---

## 4. Final training configuration (all four generators)

`JiT_S_16`, 224 px, `cond_dim` 128, batch 128, lr 1e-4, warmup 2000, 100 000
steps (≈ 13 000 epochs over 972 images), `cfg_dropout` 0.1, EMA 0.9999,
`t ~ σ(N(−0.8, 0.8))`, AdamW (0.9, 0.95), weight decay 0, grad clip 1.0, one
A100. Only the encoder, and therefore `h_dim`, differs between runs.

The `jit_dinov3_vits16` and `jit_retfound_cfp` checkpoints predate the `t_mu`
and `t_sigma` flags, so their `model_cfg` does not record them. They were
trained with the then-hard-coded (−0.8, 0.8). The two ViT-L/16 checkpoints
record the same values.
