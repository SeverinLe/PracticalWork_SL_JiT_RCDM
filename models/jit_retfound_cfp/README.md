# `jit_retfound_cfp` — JiT-RCDM generator conditioned on RETFound

**RETFound MAE ViT-L/16 (colour fundus weights)**, `h_dim = 1024`.
Trained to 100 000 steps; W&B run `6h9y3bt6`.  See `final.cfg.json` for the
architecture, sha256 and EMA flag, and `config.yaml` / `output.log` for the
training environment.

This is the second encoder family in the study, and it is what lets the report
say anything about *SSL representation spaces* rather than about one DinoV3
checkpoint.

| Encoder | SSL family | `h_dim` | Generator |
|---|---|---|---|
| DinoV3 ViT-S/16 | joint-embedding | 384 | `models/jit_dinov3/` ✅ |
| RETFound MAE ViT-L/16 | reconstructive (MAE), retinal domain | 1024 | this directory ✅ |
| ResNet-50 | supervised (control) | 2048 | not planned |

`h_dim` 384 → 1024 enlarges only the conditioning projector
(`Linear(h_dim, 128)`), not the denoiser, so the two runs cost essentially the
same and are comparable at equal compute.  Encoding is slower — ViT-L/16 against
ViT-S/16 — but that cost is paid once, during precompute.

## A naming caveat, now resolved

Before the 2026-08-03 restructure this directory was a placeholder, because every
checkpoint that existed then reported `h_dim: 384` — i.e. DinoV3 — including one
file *named* `jit_rcdm_retfound_step0100000.pt`.  That file was misnamed; see
`archive/MANIFEST.md`.  The checkpoint now in this directory is a different run:
`h_dim: 1024`, W&B run `6h9y3bt6` against DinoV3's `c48aue7x`.

## What the probes found

Run with `--tag full_seed0`; results under
`experiments/results/<experiment>/retfound_cfp/full_seed0/`.

**RQ1 fails, and it is the gate.** The inversion is statistically
indistinguishable from chance: MRR 0.046 with a 95% CI of [0.028, 0.068] against
a chance MRR of 0.036, and mean rank 75.5 [68.3, 82.5] against chance 77.5.  Both
intervals contain chance.  Given `h`, this generator produces *a* retina, not
*the* retina `h` came from.

Note what it is not: FID is 72.4, **better** than the DinoV3 generator's 87.0.
The samples look more like fundus photographs while carrying no identity
information at all, which is a clean demonstration that distributional realism
and invertibility are orthogonal — and a warning against reading FID as evidence
that a conditional generator is conditioning on anything.

Everything downstream that reads *pixels* therefore cannot be interpreted as a
statement about the representation:

| Probe | Reads | Status for this model |
|---|---|---|
| E2 sample variability / heatmap | generated pixels | **not interpretable** — E1 failed |
| E3 (a) cosine statistics | encoder only | valid |
| E3 (b) qualitative grids | generated pixels | **not interpretable** — E1 failed |
| E4 dimension structure | generated pixels | **not interpretable** — E1 failed |
| E5 linear probe | representations only | valid |

**RQ2 is doubly unanswerable here.** Beyond the E1 failure, the generator is a
deterministic map on its own training distribution: the `k` samples per `h` sit
at or below the 8-bit quantisation floor for 22 of 30 conditioning images.  The
mean per-pixel std (0.0040) clears the floor (0.0039) by 0.75%, so E2's binary
guard does not fire — read `per_image_collapse` in the summary, not the verdict
alone.

**RQ3 is where this model earns its place**, and it survives the E1 failure
because the cosine statistics never touch the generator.  The invariance profile
is close to the opposite of DinoV3's: near-perfect under flips, rotation and
photometric change (mirror_horizontal 0.9996, grayscale 0.9942) and catastrophic
under spatial rescaling (zoom_x2 0.351, crop_center **−0.325**).  An MAE trained
without augmentation invariance keeps the spatial layout it was asked to
reconstruct, and discards very little colour.  DinoV3, trained with photometric
augmentation, does the reverse.

**RQ5 confirms the consequence.** A linear probe on frozen RETFound reaches
balanced accuracy 0.424 (macro-AUC 0.785) against DinoV3's 0.360 (0.671) — better,
as a retinal-domain model should be.  But under `crop_center` and `zoom_x2` it
collapses to predicting the majority class for all 526 test images: balanced
accuracy exactly 0.200, macro-AUC 0.500.

## Reproducing

```bash
for split in train val test; do
    python data/scripts/precompute_reps.py --encoder retfound_cfp \
        --data_dir data/raw/messidor2/$split \
        --out_file data/processed/messidor2/retfound_cfp/${split}_reps.pt \
        --batch_size 16 --device cuda
done

python scripts/train.py --encoder retfound_cfp \
    --reps_file data/processed/messidor2/retfound_cfp/train_reps.pt \
    --save_dir  models/jit_retfound_cfp \
    --model S16 --batch_size 128 --lr 1e-4 --warmup_steps 2000 \
    --total_steps 100000 --device cuda --wandb_project jit-rcdm
```
