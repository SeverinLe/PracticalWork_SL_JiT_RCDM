# JiT-RCDM: probing SSL representations of retinal images through generation

Master's practical work (JKU, 2026). **Research question: what does a frozen
self-supervised encoder actually keep about a retinal fundus scan, and what does
it throw away?**

A linear probe only tells you whether information can be *decoded* from a
representation, not whether it is *present*. Following Bordes et al. (2022),
this project inverts the encoder instead. A diffusion generator is conditioned
on a frozen representation `h`. Sampling images from that `h` shows what the
encoding fixes and what it leaves open.

The implementation merges two upstream frameworks:

- **RCDM** ([facebookresearch/RCDM](https://github.com/facebookresearch/RCDM))
  conditions a diffusion model on an SSL representation `h` instead of a class
  label.
- **JiT** ([LTH14/JiT](https://github.com/LTH14/JiT)) is a plain ViT denoiser
  trained with flow matching.

Every deviation from the two upstream repositories is listed in `CHANGES.md`.

This repository contains exactly what is needed to rebuild the representation
caches, train the four generators, and run the experiments the report is built
on: E1, E1c, E2, E3, E4 and E5 (attentive probing).

---

## Dataset: Messidor-2

| | |
|---|---|
| Dataset | **Messidor-2**, colour fundus photographs for diabetic-retinopathy (DR) screening |
| Source | <https://www.adcis.net/en/third-party/messidor2/> (registration required, may not be redistributed) |
| Version used | the **preprocessed release**, already arranged into train/val/test and five severity-grade folders |
| Labels | 5-grade DR severity, read from the parent folder name of each image |
| Resolution | variable at source; resized in the pipeline (no offline preprocessing) |
| Cite | Decencière et al. (2014); Abràmoff et al. (2013) |

| Split | Total | `anodr` (no DR) | `bmilddr` | `cmoderatedr` | `dseveredr` | `eproliferativedr` |
|---|---|---|---|---|---|---|
| train | 972 | 568 | 151 | 193 | 41 | 19 |
| val | 246 | 143 | 38 | 49 | 11 | 5 |
| test | 526 | 306 | 81 | 105 | 23 | 11 |

The folder names sort into severity order (`a` to `e`). Place the images as

```
data/raw/messidor2/{train,val,test}/<grade>/*.png
```

Report these three properties next to any result:

- **Class imbalance.** 58% of the training split is no-DR, and the two most
  advanced grades have 41 and 19 training images. Predicting the majority class
  gives 0.58 plain accuracy, so every downstream metric here is balanced
  accuracy or macro-AUC.
- **Image-level splits.** The splits come from the preprocessed release.
  Messidor-2 contains both eyes of each patient, so patient-level leakage
  between splits cannot be ruled out.
- **Test count.** Older notes give the test split as 527 images. The extra
  file was a stray non-retinal `test/cat.png`, since removed. The test split
  is 526.

`data/README.md` has the cache format and further details.

---

## File structure

```
.
├── README_PR26.md                this file
├── requirements.txt              pinned dependencies (Python 3.13)
├── CHANGES.md                    every deviation from RCDM and JiT, [fix-N] tags
├── .gitignore                    keeps data, weights and results out of git
│
├── rcdm/                         library code
│   ├── jit.py                    JiT ViT denoiser, FlowMatching loss, 50-step Heun sampler + CFG
│   ├── conditioning.py           ConditioningProjector (h → 128), AdaLNZero, RMSNorm
│   ├── dataset.py                (image, h) pairs from a representation cache
│   └── encoders/                 the frozen encoders under study
│       ├── __init__.py           registry: name → (loader, h_dim, description)
│       ├── transforms.py         224 px ImageNet transform + E3 nuisance transforms
│       ├── dinov3.py             DINOv3 ViT-S/16 and ViT-L/16
│       ├── retfound.py           RETFound-MAE ViT-L/16
│       ├── retfound_dinov3.py    RETFound-DINOv3 ViT-L/16
│       └── resnet50.py           supervised control (imported by the registry)
│
├── data/
│   ├── README.md                 dataset, splits, cache format
│   ├── scripts/
│   │   ├── precompute_reps.py    frozen encoder → <split>_reps.pt / <split>_tokens.pt
│   │   └── pack_dataset.py       embed images into the cache, for Colab training
│   ├── raw/messidor2/            images (not tracked)
│   └── processed/messidor2/      <encoder>/{train,val,test}_{reps,tokens}.pt (not tracked)
│
├── encoders/                     frozen pretrained SSL weights: study INPUTS
│   ├── README.md                 sources, gated download, the use_gated_mlp fix
│   ├── download_dinov3_vits16.sh
│   ├── download_dinov3_vitl16.sh
│   ├── dinov3_vits16/config.json             (+ model.safetensors, not tracked)
│   ├── dinov3_vitl16/                        (dinov3_vitl16_pretrain_lvd.pth, not tracked)
│   ├── retfound_cfp/{config.json,README.md}  (+ RETFound_mae_natureCFP.pth, not tracked)
│   └── dinov3_vitl16_retfound_med/{config.json,README.md}  (+ retfound_dinov3.pth, not tracked)
│
├── models/                       trained JiT-RCDM generators: study OUTPUTS
│   ├── README.md                 inventory, W&B run ids
│   ├── jit_dinov3_vits16/                     final.cfg.json, training_log.md   (+ final.pt)
│   ├── jit_retfound_cfp/                      final.cfg.json, README.md        (+ final.pt)
│   ├── jit_dinov3_vitl16/                     final.cfg.json                   (+ jit_rcdm_final.pt)
│   └── jit_dinov3_vitl16_retfound_med/        jit_rcdm_dinov3_retfound_final.cfg.json (+ .pt)
│
├── scripts/
│   └── train.py                  train one generator on a cached representation file
├── notebooks/
│   ├── colab_train_dinov3_vitl16.ipynb       Colab/A100 training, DINOv3 ViT-L/16 generator
│   └── colab_train_retfound_dinov3.ipynb     Colab/A100 training, RETFound-DINOv3 generator
│
├── experiments/
│   ├── README.md                 what each probe measures; "Reading a result honestly"
│   ├── common.py                 run dirs, provenance, seeding, generator/encoder loading
│   ├── E1_fidelity.py            RQ1: does the generation re-encode to its own h? (MRR, FID)
│   ├── E1c_PCA.py                RQ1: do generations land on the real test manifold? (standalone)
│   ├── E2_sample_variability.py  RQ2: what varies across samples from one h?
│   ├── E3_invariance.py          RQ3: cosine and retrieval MRR of h under nuisance transforms
│   ├── E4_dimension_structure.py RQ4: what do individual dimensions of h control?
│   ├── E5_downstream_probe.py    linear-probe helpers used by E5_attention_probing
│   ├── E5_attention_probing.py   RQ5: does a DR grade survive h → image → h′?
│   └── results/                  run outputs (not tracked; directory skeleton only)
│
└── tests/
    └── test_pipeline.py          shape/contract tests, no weights, seconds on CPU
```

`encoders/` and `models/` are kept apart on purpose. The frozen encoder is the
independent variable of the study and the generator trained on it is the
dependent one. Each encoder gets its own generator, because the conditioning
path is sized for one `h_dim`.

---

## Setup

```bash
uv python install 3.13
uv venv .venv --python 3.13 && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_pipeline.py        # sanity check, no weights needed (or: pytest tests/ -v)
```

Always call `.venv/bin/python`. If the venv is broken or not activated, Python
silently falls back to another interpreter, and the errors look like missing
packages.

Devices: `mps` (Apple Silicon) for caching and experiments, `cuda` (Colab A100)
for training, `cpu` fallback. Every script takes `--device`.

### Encoder weights (gated downloads)

| Registry name | Model | SSL objective | Pre-training data | `h_dim` | Weights |
|---|---|---|---|---|---|
| `dinov3` | DINOv3 ViT-S/16 | joint-embedding | LVD-1689M (web) | 384 | <https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m> |
| `dinov3_vitl16` | DINOv3 ViT-L/16 | joint-embedding | LVD-1689M (web) | 1024 | <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m> |
| `retfound_cfp` | RETFound ViT-L/16, CFP weights | MAE (reconstructive) | colour fundus, MEH | 1024 | <https://github.com/rmaphoh/RETFound_MAE> |
| `retfound_dinov3` | RETFound-DINOv3 ViT-L/16 | joint-embedding | colour fundus, MEH | 1024 | <https://huggingface.co/YukunZhou/retfound_dinov3> |

`encoders/README.md` has the download procedure and the expected file names.
It also documents the DINOv3 **`use_gated_mlp: false`** fix: without it,
transformers randomly initialises a missing projection on every load, and `h`
changes from run to run.

Each encoder pair varies one thing:

- `retfound_cfp` vs `retfound_dinov3`: the SSL objective. Data, architecture
  and width are the same.
- `retfound_dinov3` vs `dinov3_vitl16`: the pre-training domain.
- `dinov3_vitl16` vs `dinov3`: the scale.

Every comparison has to say which of these variables it isolates.

### Trained generators

| Directory / file | Encoder | `h_dim` | Steps | W&B run | sha256 (prefix) |
|---|---|---|---|---|---|
| `models/jit_dinov3_vits16/final.pt` | `dinov3` | 384 | 100k | `c48aue7x` | `f5f20832` |
| `models/jit_retfound_cfp/final.pt` | `retfound_cfp` | 1024 | 100k | `6h9y3bt6` | `0bc02564` |
| `models/jit_dinov3_vitl16/jit_rcdm_final.pt` | `dinov3_vitl16` | 1024 | 100k | `dfjj1g7h` | `c13fd666` |
| `models/jit_dinov3_vitl16_retfound_med/jit_rcdm_dinov3_retfound_final.pt` | `retfound_dinov3` | 1024 | 100k | `qp2b6wu1` | `498f3900` |

The weights (about 410 MB each) are not in git. Each one has a committed
`*.cfg.json` sidecar with its architecture (`model_cfg`), training step, W&B
run id and full sha256, so a received file can be verified. All four use the
`JiT_S_16` preset: hidden 384, depth 12, 6 heads, patch 16, `cond_dim` 128,
about 25.7 M parameters. Training settings: batch 128, lr 1e-4, 2000 warmup
steps, `cfg_dropout` 0.1, EMA 0.9999, one A100. Only `h_dim` differs.

---

## Pipeline

```
raw images ─► data/scripts/precompute_reps.py ─► data/processed/messidor2/<encoder>/<split>_reps.pt
           ─► scripts/train.py                ─► models/<name>/*.pt
           ─► experiments/E*.py               ─► experiments/results/<experiment>/<encoder>/<tag>/
```

The encoder never runs inside the training loop. Representations are cached
once, so training, sampling and every probe see byte-identical `h` for the same
image.

Below, `ENC` is one of the four registry names, and `CKPT` is the matching
generator from the table above.

### 1. Cache representations (once per encoder × split)

```bash
for split in train val test; do
    python data/scripts/precompute_reps.py --encoder $ENC \
        --data_dir data/raw/messidor2/$split \
        --out_file data/processed/messidor2/$ENC/${split}_reps.pt --device mps
    # full token sequences, needed by E5_attention_probing only
    python data/scripts/precompute_reps.py --encoder $ENC --tokens \
        --data_dir data/raw/messidor2/$split \
        --out_file data/processed/messidor2/$ENC/${split}_tokens.pt --device mps
done
```

### 2. Train a generator

Locally or on any CUDA machine:

```bash
python scripts/train.py --encoder $ENC \
    --reps_file data/processed/messidor2/$ENC/train_reps.pt \
    --save_dir  models/<name> \
    --model S16 --batch_size 128 --lr 1e-4 --warmup_steps 2000 \
    --total_steps 100000 --cfg_dropout 0.1 --ema_decay 0.9999 --device cuda
```

On Colab, use the two notebooks in `notebooks/`. They pack the cache with
`data/scripts/pack_dataset.py`, so the images travel inside the `.pt` file, and
they resume from the last checkpoint. `train.py` refuses to start if
`--encoder` disagrees with the encoder recorded in the cache.

### 3. Run the experiments

Add `--device mps` (or `cuda`) to every command.

| RQ | Experiment | Command |
|---|---|---|
| RQ1 fidelity | `E1_fidelity` | `python experiments/E1_fidelity.py --checkpoint $CKPT --encoder $ENC --n_per_class 40 --seed 0` |
| RQ1 manifold | `E1c_PCA` | `python experiments/E1c_PCA.py --checkpoint $CKPT --encoder $ENC --seed 0` |
| RQ2 variability | `E2_sample_variability` | `python experiments/E2_sample_variability.py --checkpoint $CKPT --encoder $ENC --n_per_class 6 --n_samples 8 --seed 0` |
| RQ3 invariance | `E3_invariance` | `python experiments/E3_invariance.py --checkpoint $CKPT --encoder $ENC --n_per_class 30 --n_visual 1 --seed 0` (add `--skip_generation` for the encoder-only part, no checkpoint needed) |
| RQ4 dimensions | `E4_dimension_structure` | `python experiments/E4_dimension_structure.py --checkpoint $CKPT --encoder $ENC --reps_file data/processed/messidor2/$ENC/train_reps.pt --n_queries 10 --k_neighbours 16 --n_dims 64 --seed 0` |
| RQ5 attentive probe | `E5_attention_probing` | `python experiments/E5_attention_probing.py --encoder $ENC --checkpoint $CKPT --reps_dir data/processed/messidor2/$ENC --n_seeds 3` |

Notes:

- **E3** reports both cosine similarity and retrieval MRR, i.e. whether
  `h(T(x))` still retrieves `h(x)` among all sampled test images. Per-image
  reciprocal ranks are written to `cosine_similarity.csv` (long form) and
  `mrr_per_image.csv` (one row per image, one column per transform). The
  per-transform MRR, CI, top-1 rate and chance MRR go to `summary.json`.
- **E5** requires the token caches from step 1. Always pass `--checkpoint` and
  `--reps_dir` explicitly, because the script's defaults point at a
  `models/jit_dinov3/` path that no longer exists. Class imbalance is
  corrected with inverse-frequency weights in both read-outs:
  `CrossEntropyLoss(weight=…)` for the attentive probe, and
  `class_weight="balanced"` for the linear probe.
- `--cfg_scale` defaults to **1.0** on every probe, because classifier-free
  guidance raises fidelity and suppresses diversity, which are exactly the
  quantities being measured. Report any other value as a sensitivity analysis.

---

## Outputs and provenance

Every run writes `experiments/results/<experiment>/<encoder>/<tag>/` containing:

- `run_config.json`: all arguments, seed, git commit (with `-dirty`),
  checkpoint sha256, `model_cfg`, library versions. It is written before any
  compute starts, so an interrupted run still leaves a record.
- `metrics.csv` and/or experiment-specific CSVs
- `summary.json`
- `figures/`

`--tag` names the run directory; without it, a UTC timestamp is used. The
report's numbers come from runs tagged `full_seed0`.

---

## Read before interpreting a result

`experiments/README.md` § "Reading a result honestly" is required reading. In
short:

- **The generators are deterministic in `h`** on their own training
  distribution: `k` samples from one `h` are (near-)identical. This is a
  property of this generator at this dataset size (972 training images). It
  voids E2 as a statement about the encoders. E3, E4 and the generator-free
  parts of E5 are unaffected.
- **A low self-read-back MRR in E1 does not by itself mean the generator
  failed.** E1 re-encodes generations with the same encoder that produced `h`.
  An encoder whose embedding space is angularly compressed can fail to read
  generated images back even when the images carry identity. Read E1 for each
  encoder together with E1c.
- Metrics are balanced accuracy and macro-AUC, never plain accuracy.

---

## Citation

```
Bordes, Balestriero, Vincent (2022). High Fidelity Visualization of What Your
  Self-Supervised Representation Knows About. arXiv:2112.09164
Li, Katabi, He (2024). Return of Unconditional Generation: A Self-supervised
  Representation Generation Method. arXiv:2312.03701
Siméoni et al. (2025). DINOv3. arXiv:2508.10104
Zhou et al. (2023). A foundation model for generalizable disease detection from
  retinal images. Nature 622, 156-163.
Decencière et al. (2014). Feedback on a publicly distributed image database:
  the Messidor database. Image Analysis & Stereology 33(3).
Abràmoff et al. (2013). Automated analysis of retinal images for detection of
  referable diabetic retinopathy. JAMA Ophthalmology 131(3).
```
