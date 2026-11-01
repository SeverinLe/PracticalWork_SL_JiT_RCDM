# Prediction mechanism: JiT vs JiT-RCDM

How the denoiser is trained and sampled in upstream
[JiT](https://github.com/LTH14/JiT) (`denoiser.py`, `main_jit.py`, branch
`main`) and in this repository (`rcdm/jit.py` `FlowMatching`,
`scripts/train.py`). Architecture differences are in `changes_pr26.md`.

Notation: `x` is the clean image in `[-1, 1]`, `ε ~ N(0, I)`, and `t ∈ [0, 1]`,
where `t = 0` is pure noise and `t = 1` is data. The network outputs `x̂`.

---
## 1. Side by side

| Step | JiT | JiT-RCDM (this repo) |
|---|---|---|
| Noisy input | `z_t = t·x + (1−t)·ε`, `noise_scale = 1` | same |
| Time sampling | `t = σ(u)`, `u ~ N(−0.8, 0.8)` | same (`--t_mu`, `--t_sigma`, defaults −0.8 / 0.8) |
| Conditioning | `c = t_emb + class_emb(y)`, plus 32 in-context label tokens | `c = t_emb + cond_proj(h)`, `h` = frozen encoder CLS, `cond_dim` 128 |
| Network output | `x̂ = net(z_t, t, y)` (x-prediction) | `x̂ = net(z_t, t, h)` (x-prediction) |
| Loss space | **velocity**: `v̂ = (x̂ − z_t) / max(1−t, 0.05)` | **image**: no conversion |
| Loss | `‖v − v̂‖²` | `‖x − x̂‖²` |
| CFG dropout | p = 0.1, label → extra null-class index | p = 0.1, `h` → learnable vector `null_h` |
| Sampler | Heun, 50 steps, grid `linspace(0, 1, 51)`, Euler on the last step | same |
| Velocity in sampler | `v = (x̂ − z) / (1 − t)` | same, `(1 − t)` floored at 1e-8 |
| CFG at sampling | on velocity: `v_u + s·(v_c − v_u)`, applied for `t` in `[interval_min, interval_max]` (default `[0, 1]`) | on `x̂`: `x̂_u + s·(x̂_c − x̂_u)`, all `t` |
| Default guidance | `cfg = 1.0` | `1.0` on every probe; `3.0` only in `sample_grid.py` (qualitative) |
| EMA | two copies, 0.9999 and 0.9996 | one copy, 0.9999, always applied on load |

---

## 2. The two differences in mechanism

### 2.1 Loss space: v-loss (JiT) vs x-loss (ours)

Both networks predict `x`. They differ in where the error is measured. Because
`v − v̂ = (x − x̂) / (1 − t)`:

```
L_JiT(t)  = ‖x − x̂‖² / max(1 − t, 0.05)²
L_ours(t) = ‖x − x̂‖²
```

So the two objectives differ only by a per-sample weight `w(t)` on the same
image-space error:

| `t` | 0.1 | 0.31 (median) | 0.5 | 0.7 | 0.9 | ≥ 0.95 |
|---|---|---|---|---|---|---|
| JiT `w(t)` | 1.2 | 2.1 | 4 | 11 | 100 | 400 (cap) |
| ours `w(t)` | 1 | 1 | 1 | 1 | 1 | 1 |

Relative to JiT, our objective down-weights the clean end of the path
(`t → 1`) by up to 400×. The minimiser of either loss at a fixed `(z_t, t)`
is the same conditional mean `E[x | z_t, h]`. The weighting changes which `t`
the limited capacity is spent on, not what the optimum is.

This was a choice I regret and cannot remember why I changed it: "x-prediction" with "x-loss" - it is untagged and applies identically to all
four trained generators. 

*Open question, not a result:* together with `t_mu = −0.8`, x-loss puts most
of the training signal at noisy `t`, where the MSE-optimal `x̂ ≈ E[x | h]`
ignores `z_t`. That is one candidate cause of the generators being
deterministic in `h`. Testing it requires retraining with a v-loss which will be done and reported on in the upcoming master thesis.

### 2.2 CFG blend: on `x̂` (ours) vs on `v` (JiT): equivalent

At a fixed `(z, t)` the velocity is affine in `x̂`:

```
v_u + s·(v_c − v_u)  =  ( [x̂_u + s·(x̂_c − x̂_u)] − z ) / (1 − t)
```

Blending `x̂` and then converting gives exactly JiT's velocity blend. JiT's
default interval `[0, 1]` covers the whole path, so with defaults the two
samplers are the same. With `s = 1.0`, which every probe uses, the blend is
the identity, and our sampler skips the unconditional pass entirely.

### 2.3 What is otherwise the same

The interpolation path, the noise scale, the t-sampler, the Heun integrator
with a final Euler step, the 50-step grid, and x-prediction itself are
identical. Given the same `x̂` function, both samplers produce the same
trajectory.

---

## 3. Changes that were essential to make it work

### A. Required by replacing the class label with `h`

Without these the model cannot be built or conditioned.

1. **`class_emb(y)` → `cond_proj(h)`.** `Linear(h_dim, cond_dim) + SiLU`
   maps a continuous 384- or 1024-dim representation into the conditioning
   vector. There is no label table.
2. **Learnable `null_h` for CFG** `[fix-3]`. A continuous input has no "extra
   index" to reserve, so the unconditional input is a learned vector in
   `h`-space. It is trained by the 10% dropout and included in the EMA.
3. **In-context label tokens removed.** In JiT they are copies of the label
   embedding. With no label they have nothing to carry, so `h` enters only
   through adaLN-Zero.
4. **`h_dim` per generator.** The projector width is tied to the encoder, so
   each encoder gets its own generator. `train.py` and `load_generator()`
   refuse mismatched pairs.

### B. Required for correct results

Each of these failed silently: the model ran, but the numbers were wrong.

5. **Encoder input fixed at 224 px** `[fix-2]`. A /16 ViT has a fixed 14 × 14
   position grid. Feeding it the generator's resolution corrupts `h` without
   an error.
6. **DINOv3 `use_gated_mlp: false`.** Otherwise transformers re-initialises a
   missing projection randomly on every load, and `h` differs between runs.
7. **EMA saved in the checkpoint and applied at inference** `[fix-1]`.
   Previously sampling used the raw weights, and resuming discarded the EMA
   history.
8. **Representations cached once, aligned by index** (`reps[i]` ↔
   `paths[i]`). Training, sampling and every probe see byte-identical `h`.
   Caches record their `encoder`, because three encoders share `h_dim = 1024`.
9. **Architecture read from the checkpoint's `model_cfg`, never from CLI
   flags.** A checkpoint can't be loaded into the wrong configuration.

### C. Restoring the JiT recipe

10. t-sampler `σ(N(−0.8, 0.8))` instead of `σ(N(0, 1))` `[fix-4b]`.
11. AdamW β₂ = 0.95 `[fix-4a]`. Linear LR warmup `[fix-4c]`.
12. RMSNorm, 2D RoPE, QK-norm and SwiGLU in place of LayerNorm, learned
    `pos_embed`, `nn.MultiheadAttention` and a GELU FFN `[fix-5a–d]`.

### D. Design choices (not required, but they shape the results)

- **`cond_dim = 128` bottleneck.** This is from neither upstream. It sets how
  much of `h` the generator can use: 33% of the width at `h_dim` 384, 12.5% at
  1024.
- **`JiT_S_16` preset** (hidden 384, depth 12, ≈ 25.7 M params) and **224 px**
  instead of JiT-B and 256 px, sized for 972 training images and one A100.
- **Single EMA, no CFG interval.** Simplifications that are equivalent at
  JiT's defaults.
