# Frozen encoders

The pretrained SSL encoders under study — the *inputs* of the project, as
opposed to `models/`, which holds the generators trained on top of them.

All are frozen (`requires_grad=False`, `eval()`), consume 224 × 224
ImageNet-normalised input, and expose the same interface through
`rcdm/encoders/`:

```python
from rcdm.encoders import get_encoder, build_transform
encoder = get_encoder("dinov3", device="cuda")
h = encoder(build_transform()(img).unsqueeze(0).to("cuda"))   # (1, encoder.h_dim)
```

| Registry name | Model | SSL family | `h_dim` | Weights | Tracked |
|---|---|---|---|---|---|
| `dinov3` | DinoV3 ViT-S/16 | joint-embedding | 384 | `dinov3_vits16/` | no (86 MB) |
| `dinov3_vitl16` | DinoV3 ViT-L/16 (web) | joint-embedding | 1024 | `dinov3_vitl16/` | no (1.1 GB) |
| `retfound_cfp` | RETFound MAE ViT-L/16, colour-fundus weights | reconstructive (MAE) | 1024 | `retfound_cfp/` | no (3.7 GB) |
| `retfound_dinov3` | RETFound-DINOv3 ViT-L/16, colour-fundus weights | joint-embedding (DINOv3) | 1024 | `dinov3_vitl16_retfound_med/` | no (1.4 GB) |
| `resnet50` | ImageNet-supervised ResNet-50 | supervised (control) | 2048 | torchvision zoo | n/a |

The three-way split — joint-embedding vs. reconstructive vs. supervised — is the
scientific spine of the study: it is what lets conclusions be about *SSL
representation spaces* rather than about one checkpoint.

## The comparison these entries are for

The study's claim is about *SSL families*, so every other variable has to be held
constant — and until `retfound_dinov3` existed, none of them were:

| Entry | Objective | Pre-training data | Arch | `h_dim` |
|---|---|---|---|---|
| `retfound_cfp` | MAE | CFP, MEH | ViT-L/16 | 1024 |
| `retfound_dinov3` | DINOv3 | CFP, MEH | ViT-L/16 | 1024 |
| `dinov3_vitl16` | DINOv3 | LVD-1689M web | ViT-L/16 | 1024 |
| `dinov3` | DINOv3 | LVD-1689M web | ViT-S/16 | 384 |

Read as pairs, each isolating one variable:

- **`retfound_cfp` vs `retfound_dinov3`** — *objective* (MAE vs DINOv3). Data,
  architecture and width identical. **This is the pair the RQ1 claim rests on.**
- **`retfound_dinov3` vs `dinov3_vitl16`** — *domain* (retinal vs web). Objective,
  architecture and width identical.
- **`dinov3_vitl16` vs `dinov3`** — *scale* (ViT-L vs ViT-S), which also changes the
  conditioning bottleneck from 12.5 % to 33 %.

`retfound_cfp` vs `dinov3` — the comparison the study actually ran — varies all
three at once and cannot attribute anything on its own. Say which pair supports
which claim.

## Why there are three ViT-L/16 entries

They are controls, not additional models of interest. `cond_dim` is fixed at 128
for every generator, so representation width directly sets how much of `h` the
generator can see: 33 % at 384, 12.5 % at 1024. `E7_conditioning_bottleneck`
measures the consequence — variance retained 0.504 for `dinov3` against 0.214 for
`retfound_cfp`. Because RETFound *also* fails E1 at chance level, "MAE inverts
poorly" and "1024-dim through a 128-dim map inverts poorly" were not separable.

**`retfound_dinov3` is the one that resolves it** (train with
`notebooks/colab_train_retfound_dinov3.ipynb`). It is DINOv3 on the same CFP corpus,
same architecture, same width, same author as `retfound_cfp`, so the objective is
the only thing that differs.

**`dinov3_vitl16` is registered but optional.** It would isolate *domain* against
`retfound_dinov3`. Worth training only if the objective pair turns out interesting;
otherwise it is future work.

### Equal width is not equal loss — measure the ceiling

A caveat that changes how E7 should be read. What a 128-dim map *can* retain
depends on the spectrum of `h`, and the objectives produce very different spectra
(PCA over the 972 training representations):

| encoder | participation ratio | dims for 90 % var | top-128 PCA ceiling |
|---|---|---|---|
| `dinov3` | 12.6 | 70 | 0.959 |
| `retfound_cfp` | 5.5 | 13 | **0.990** |
| `retfound_dinov3` | 100.7 | 267 | **0.738** |

RETFound MAE's features are extremely anisotropic — 13 dimensions carry 90 % of the
variance — so a well-aligned 128-dim projection could retain 99 %. The learned one
retains 21 %. **The width is therefore not what fails for RETFound; the projection's
alignment is.** Conversely `retfound_dinov3` is near-isotropic and faces a genuinely
harder compression problem (ceiling 0.738), so if it inverts better anyway, the
objective explanation is strengthened rather than confounded.

Report `retained_learned` against `retained_ceiling`, not `retained_learned` alone:
the gap is what separates "the bottleneck is too narrow" from "the projection failed
to use it".

## DinoV3 ViT-S/16

Expected layout:

```
encoders/dinov3_vits16/
├── config.json
└── model.safetensors
```

Loaded with `local_files_only=True`, so no network request is ever made and the
weights are pinned for reproducibility.  `download_dinov3_vits16.sh` fetches
them.

### The `use_gated_mlp` trap — read before running anything

The checkpoint was trained with a standard 2-projection FFN, but the shipped
`config.json` sets `"use_gated_mlp": true`, which makes transformers expect a
3-projection gated FFN.  The missing `gate_proj` is then **randomly initialised
on every load**, so `h` differs between runs and every downstream number becomes
irreproducible — silently.

Fix it once, in `config.json`:

```json
"use_gated_mlp": false
```

`rcdm/encoders/dinov3.py` refuses to load rather than letting the corruption
through. The check no longer hard-codes "must be false": it reads the `gate_proj`
keys actually present in `model.safetensors` and raises when the flag disagrees
with them. That matters because the flag is *legitimately* true for the H+ and
7B variants, whose gated FFN is real — a blanket rule would block them.

## DinoV3 ViT-L/16 (the width-matched control)

**Two weight formats exist and both load.** `load_vitl16` detects which is
present — a directory takes the transformers path, a file takes the timm path —
and prefers the `.pth` when both are there.

```
encoders/dinov3_vitl16/
├── config.json                       ── A. HuggingFace directory
└── model.safetensors                    (download_dinov3_vitl16.sh)
        …or…
└── dinov3_vitl16_pretrain_lvd.pth    ── B. official facebookresearch/dinov3
```

**A — HuggingFace directory.** Fetch with `download_dinov3_vitl16.sh` (gated —
accept the licence at
<https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m> first, then
`export HF_TOKEN=hf_...`). Loaded via `AutoModel.from_pretrained`, and the script
validates before you train on it: `use_gated_mlp` against the `gate_proj` keys
actually in the weight file, and `hidden_size == 1024`.

**B — official `.pth`.** A flat backbone state dict (`cls_token`,
`storage_tokens`, `blocks.*`, `rope_embed.periods`), not a transformers
checkpoint, so `AutoModel` cannot read it. It goes through timm
(`vit_large_patch16_dinov3`) using the key conversion shared with
`retfound_dinov3.py`, and the load is asserted exact — 0 missing, 0 unexpected.
The download script does **not** fetch this form; take it from the dinov3 repo or
the Meta access e-mail.

Its qkv biases are all exactly zero (distilled release), so the bias-free timm
variant is faithful — but the loader *checks* that before dropping them rather
than assuming it, because `retfound_dinov3`'s key bias is **not** zero and
dropping it there costs ~4 % relative L2. Same converter, opposite correct answer.

Same LVD-1689M corpus and objective as the ViT-S/16 entry; only the student width
differs. 24 layers, 16 heads, 1024-dim, 4 register tokens (so `forward_tokens`
returns 201 tokens: CLS + 4 registers + 196 patches — against 197 for both
RETFound variants, which have none).

## RETFound-DINOv3 ViT-L/16 (`retfound_dinov3`)

DINOv3 self-distillation pre-trained on colour fundus photographs — the
joint-embedding counterpart to `retfound_cfp`'s MAE, from the same author and
corpus.

**Sources**

| | |
|---|---|
| Weights | <https://huggingface.co/YukunZhou/retfound_dinov3> (gated) |
| Method / code | <https://github.com/HORIZONHealthcare/Med_DINOv3> |
| Model card | `dinov3_vitl16_retfound_med/README.md` (RETFound_DINOv3, CC BY-NC 4.0) |
| Compute | 4 × A100 80GB, 10 days (per the model card) |

Expected layout:

```
encoders/dinov3_vitl16_retfound_med/
├── config.json              hand-written model card, NOT a transformers config
├── README.md                upstream model card
└── retfound_dinov3.pth      1.4 GB
```

`config.json` records `"SSL strategy": "DINOV3"`, `"image modality": "CFP"`,
`"dataset": "MEH"`, ViT-Large, 24 layers, 16 heads, 1024-dim, patch 16.

### Loading — why it goes through timm, not transformers

The `.pth` is a raw DINOv3 *pre-training* checkpoint in the upstream
facebookresearch/dinov3 layout: `{"teacher": {"backbone.*", "dino_head.*",
"ibot_head.*"}}`. The two heads are the DINO and iBOT projections (65 536
prototypes each) used only by the self-distillation loss and are dropped. The
directory's `config.json` is a model card in the same hand-written style as
RETFound's, **not** a transformers `DINOv3ViTConfig`, so `AutoModel.from_pretrained`
cannot read it — `rcdm/encoders/retfound_dinov3.py` converts the weights into
timm's `vit_large_patch16_dinov3_qkvb` instead, and verifies the load leaves **zero**
missing or unexpected keys.

Three conversion details, each of which would silently corrupt `h`:

1. **LayerScale renames.** DINOv3's `ls1.gamma`/`ls2.gamma` are timm's
   `gamma_1`/`gamma_2`, mirroring `timm.models.eva.checkpoint_filter_fn`.
2. **No register tokens.** Official DINOv3 ViT-L/16 has 4 storage tokens; *this*
   checkpoint has none. timm defaults to 4, so the model is built with
   `num_reg_tokens=0` or four randomly-initialised tokens are prepended to every
   sequence. The sequence is 197 tokens (CLS + 196 patches) — the same as
   RETFound's, not the 201 of the `dinov3` entry.
3. **The key bias must be kept.** The fused `attn.qkv.bias` has a non-zero key
   component (max 10.33). timm's stock converter discards it, on the grounds that
   the k bias "is supposed to be fixed at 0" — true for the released DINOv3 models,
   where weight decay drove it there. Without RoPE a k-bias *is* a no-op: it adds a
   per-query constant to every logit and softmax is shift-invariant. With RoPE the
   bias is rotated per key position, the term stops being constant across keys, and
   the invariance fails. Measured: dropping it gives cosine 0.9991 but ~4 % relative
   L2 error — invisible to E3's cosines, not invisible to `E1_exact_embedding`'s
   offset ratios. The loader writes it into the `k_bias` buffer timm already
   concatenates, so nothing is lost.

RoPE periods are deliberately *not* loaded: timm recomputes them in float32, the
checkpoint stores the same values in bfloat16, and
`timm_periods.to(bfloat16) == checkpoint_periods` exactly. The recomputed buffer is
the same embedding at higher precision.

## RETFound MAE ViT-Large/16

Expected layout:

```
encoders/retfound_cfp/
├── RETFound_mae_natureCFP.pth
├── config.json
└── README.md
```

Access-gated: request via <https://github.com/rmaphoh/RETFound_MAE> or the
HuggingFace model card, accept the CC BY-NC 4.0 licence, then download.
**Non-commercial use only.**

Two weight sets exist and are *not* interchangeable:

| File | Pre-training corpus | Use with |
|---|---|---|
| `RETFound_mae_natureCFP.pth` | colour fundus photography | Messidor-2 ✅ (what this project uses) |
| `RETFound_mae_natureOCT.pth` | optical coherence tomography | OCT datasets |

Both are ViT-L/16 with a 1024-dim CLS token; only the corpus differs.  Any
report must state which one produced each result.

The `.pth` is a raw MAE pre-training checkpoint containing both the encoder and
the MAE decoder.  `rcdm/encoders/retfound.py` drops the `decoder_*` and
`mask_token` keys before loading and then **raises** if any encoder key is still
missing or unexpected — a partial load would mean silently random weights.

## ResNet-50

No local weights: `torchvision.models.resnet50(weights=DEFAULT)` downloads to
`~/.cache/torch` on first use.  The classification head is replaced with
`Identity`, giving the 2048-dim avgpool trunk output — the same representation
RCDM used originally.

## Adding an encoder

Create `rcdm/encoders/<name>.py` exposing `H_DIM` and `load(device, **kwargs)`
returning a frozen module with `.name`, `.h_dim` and
`forward(x) -> (B, h_dim)`, then add one line to `ENCODERS` in
`rcdm/encoders/__init__.py`.  Nothing else changes: the dataset, denoiser,
trainer and every experiment are parametrised by `h_dim` throughout.
