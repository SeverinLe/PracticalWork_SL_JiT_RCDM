# Experiments

One script per research question.  Each writes a self-describing run directory:

```
experiments/results/<experiment>/<encoder>/<tag>/
├── run_config.json     every CLI argument, RNG seed, git commit,
│                       checkpoint sha256, library versions, timestamp
├── metrics.csv         per-image or per-condition raw measurements
├── summary.json        aggregates, confidence intervals, test statistics
└── figures/            publication-ready PNGs at 300 dpi
```

`run_config.json` is written *before* any compute starts, so even an interrupted
run leaves a record of what was attempted.  Nothing under `results/` is tracked
in git; figures selected for the write-up are copied into `report/figures/`.

## The probes

| Script | RQ | Question | Primary output |
|---|---|---|---|
| `E1_fidelity.py` | RQ1 | Is `h` invertible — can the generator reconstruct the scan it came from? | MRR + mean rank with bootstrap CI, FID |
| `E1_exact_embedding.py` | RQ1 | Does the re-encoded generation land *on* `h`, or only nearest to it? | offset ÷ nearest-real-neighbour distance ± CI, PCA scatter |
| `E2_sample_variability.py` | RQ2 | What does `h` encode vs. discard? | within-h vs between-h std/SSIM/LPIPS ratios |
| `E2_variance_heatmap.py` | RQ2 | *Where* in the retina does `h` leave things free? | shared-scale variance heatmaps over the fundus |
| `E3_invariance.py` | RQ3 | Which transformations is `h` invariant to? | cosine similarity per transform ± CI, Wilcoxon + Holm |
| `E4_dimension_structure.py` | RQ4 | Do individual dimensions carry separable factors? | masking/substitution grids, `dimensions.json` |
| `E5_downstream_probe.py` | RQ5 | Do the generative findings predict classifier behaviour? | balanced accuracy ± 95% CI, McNemar |
| `E5_attention_probing.py` | RQ5 | Does a grade survive `h → image → h′`, and does attentive pooling read more than `h`? | real-vs-generated agreement ± CI, balanced accuracy for both probes |
| `E5_weighted_ap.py` | RQ5 | Same, with the class-imbalance correction as a controlled variable | the above, per `--class_weighting` scheme, plus per-class loss shares |
| `E6_grade_transfer.py` | RQ6 | Can a DR grade be edited into `h`, and does it survive generation? | dose-response P(target) ± CI, labelled ladder grids |
| `sample_grid.py` | — | Qualitative figures | `[real \| k samples]` grids |

`common.py` holds the shared plumbing: run-directory creation, provenance
capture, seeding, generator/encoder loading, and the matplotlib style.

## How the probes fit together

E1 establishes that the inversion is faithful enough to be read at all — without
it, nothing downstream is interpretable.  `E1_exact_embedding.py` sharpens the
same question from a rank into a distance: MRR says the matching generation is
the *nearest* one, which is a much weaker claim than saying `h′` reproduces `h`,
and everything E4 and E6 do with edited representations depends on which of the
two holds.  E2 and E3 then read the representation
in two directions: what varies across samples from one `h` (discarded), and what
moves `h` itself (non-invariance).  E4 asks whether the structure is coordinate-
aligned enough to edit.  E5 is the external check: it takes E4's dimension
indices and E3's non-invariant transforms and asks whether they predict what a
linear probe can and cannot do.

That last link matters more than it looks.  E1–E4 measure what a *generator* can
render from `h`; a linear probe measures what a *downstream model* can decode.
These come apart in both directions, and RQ5 is the test of whether the
generative probe earns its keep as a diagnostic.

`E5_attention_probing.py` attacks RQ5 from two directions the linear probe
cannot.  It fits one probe on real training images and evaluates it twice: on
real test images, and on images the generator drew from those same test images'
`h` and the encoder then re-read.  The reported quantity is **agreement** — how
often the probe returns the same grade from the reconstruction as from the
original.  That is stricter than E1's rank: E1 asks whether `h′` is *nearest* to
`h` among the generated set, which a reconstruction can win while having lost
every feature that decides a grade.  Here the arbiter is a classifier trained on
the task, so it answers whether the *grade-relevant* content survived the round
trip.  Agreement is measured against the probe's call on the real image, not
against the ground-truth label: an image the probe misgrades identically on both
sides is a success for the reconstruction and a failure for the probe, and
conflating the two would hide both.

It also runs two read-outs side by side.  `linear_cls` is logistic regression on
`h` — exactly what the generator is conditioned on, so it is the like-for-like
comparison.  `attentive` is the AIM/DINOv2 protocol: a learned query attends over
the encoder's whole final-layer token sequence and pools it before a linear layer
classifies.  The gap between them measures how much grade signal lives *outside*
the vector E1–E4 and E6 study, which bounds the inversion confound from the side
E1 cannot reach — a feature missing from the generations may be absent from `h`
while sitting plainly in the patch tokens, and only this comparison separates
those.  Expect the gap to be larger for RETFound: MAE pre-training has no CLS
objective, so nothing forces its CLS token to summarise the image, whereas
DinoV3's joint-embedding loss trains that token directly.

E6 closes the loop the other way round.  E4 and E5 both stop at the
representation: E4 edits `h` and looks at pictures, E5 feeds edited `h` to a
probe.  E6 is the only experiment that goes `h → edit → image → encoder → h′ →
probe`, so the probe never sees the edited representation.  That distinction is
the experiment: probing an edited `h` directly is close to tautological, because
setting the grade-discriminative coordinates to a class's values is most of what
the probe reads.  Both numbers are reported side by side and the gap between them
is the result.  E6 also does not inherit E4's dimension selection — the k-NN
neighbourhood is not grade-structured on this dataset (purity 0.443 at k=16,
below the 0.584 majority-class rate), so E6 selects on a labelled class contrast
instead.  See its module docstring for the measurements behind that choice.

## Reading a result honestly

**Both generators are deterministic in `h`, which voids RQ2.** This is the first
thing to know about every number below.  Given an `h` from the training
distribution, the denoiser ignores its noise argument: tracing the sampler,
`std(x_pred)` across noise draws is ~0 at every `t`, so the noise contribution
decays as `(1−t)·σ₀` and is annihilated by `t=1`.  The `k` samples from one `h`
come out byte-identical — DinoV3 has 0/30 conditioning images above the 8-bit
floor, RETFound 8/30.  Feed an *off-manifold* `h` (rescaled, dimension-shuffled,
random) and diversity returns immediately, so this is memorisation of the
representation manifold, not a broken sampler.

With 972 training images and a representation that near-uniquely keys each one,
`p(x|h)` is effectively a point mass, and an MSE objective is minimised by
learning the deterministic map.  So E2 cannot report what `h` discards: there is
nothing for the model to vary.  That is a finding about *this generator at this
dataset size*, and it must not be restated as "`h` determines the retina" — the
two are indistinguishable from E2's numbers alone, which is exactly why the
normalised ratios are voided rather than reported as ~0.

E3, E4 and E5 do not depend on sampling spread — they turn on how `h` itself
moves — and remain readable.

**"Nearest" is not "exact", and E1's rank cannot tell them apart.**
`E1_exact_embedding.py` exists because a top-1 rank only says the matching
generation beat 153 competitors; it says nothing about how far away it landed.
The reportable quantity is `offset_over_nn_distance` — the distance from `h` to
the centroid of its re-encoded generations, divided by the distance from `h` to
the *nearest other real image*.  Below 1 the generation still identifies its
source; only near 0 is the embedding reproduced.  Report that ratio alongside the
cosine, not instead of it: cosine is blind to norm, so it cannot see the
shrinkage the same run measures.  The within-`h` spread in that run is a
descriptive scale for the offset, not a claim about the sampler — what the `k`
generated *images* do is E2's question.

**E4's selection rule is not the paper's selection rule.** The reference
implementation (`facebookresearch/RCDM`, `scripts/image_sample_manipulation.py`)
picks dimensions by counting how many of the k=20 neighbours are **non-zero** in
each dimension and keeping those above a threshold — a rule that only
discriminates because their SimCLR ResNet-50 avgpool feature is post-ReLU and so
non-negative and sparse.  Both ViT CLS caches here are dense and signed (0.0%
exact zeros, ~50% negative entries), so that rule would select all 384/1024
dimensions and silently turn the masked row into an unconditional sample.
`--selection common_nonzero` is the faithful port and refuses to run on a dense
cache; the default `common_active` is a dense-feature stand-in that fixes the
count instead of thresholding.  The masking and substitution steps themselves are
identical to the reference.  Say which rule produced a figure when reporting it —
they are not the same experiment.

**The inversion confound.** A feature missing from the samples can mean the
information is absent from `h`, *or* that the generator cannot render it.  With
972 training images the second explanation is always live.  E1's fidelity metric
bounds it (a faithful inversion makes "the generator cannot render it" less
plausible) and E5 bounds it from the other side, but the bound is partial and the
report says so.

**Balanced accuracy and McNemar disagree, and only one of them answers RQ5.**
E5 reports balanced accuracy because Messidor-2 is 58% no-DR.  A degraded
representation does not make the probe err randomly — it makes it predict the
majority class for everything, which *raises* raw accuracy to 0.582 while
balanced accuracy falls to chance (0.200) and macro-AUC to 0.500.  McNemar tests
raw correctness and therefore calls that total collapse non-significant
(p = 0.35 for RETFound under `crop_center`).  Read `delta_vs_A_full`, the paired
bootstrap on the balanced-accuracy difference, and check
`prediction_degeneracy.collapsed_to_single_class`.  McNemar is still recorded,
but it is a statement about plain accuracy and is labelled as one.

**FID on retinal images.** The Inception backbone is ImageNet-trained and poorly
calibrated for this domain, and the estimator is biased at a few hundred images.
Treat FID as a relative signal between configurations, never as an absolute
quality claim.

**Error bars.** E1, E3 and E5 compute their own confidence intervals
(bootstrap over images for E1/E3, over seeds for E5).  E2's variability metrics
and E1's FID need repeated `--seed` runs to get a spread — a single FID number
carries no error bar.

**Never read the round-trip agreement rate raw — read kappa.** This is the same
trap as balanced accuracy vs plain accuracy above, in a new place.  Two
prediction vectors agree by accident at a rate set entirely by how concentrated
the predictions are, so a probe that has partly collapsed onto one class scores
a *high* raw agreement for the wrong reason.  On DinoV3 the two probes look
indistinguishable raw — attentive 0.445, linear 0.452 — and are not close once
corrected:

| probe | raw | chance | kappa |
|---|---|---|---|
| attentive | 0.445 | 0.231 | **+0.278** |
| linear on `h` | 0.452 | 0.386 | **+0.109** |

The linear probe's chance baseline is 0.386 because it predicts one class for
52.5% of real images and 59.3% of generated ones; the attentive probe's most
common prediction covers 25.9% and 48.1%.  Corrected, the attentive probe
carries roughly 2.5x more grade signal through the round trip, and the raw rates
would have said the opposite of nothing.  `chance_agreement` and `cohen_kappa`
are both in `summary.json`, `prediction_degeneracy` records the concentration
that drives them, and `predictions.csv` holds the per-image calls so the whole
thing can be re-cut without refitting.

**The attentive probe needs its learning rate checked, not assumed.** Attentive
probing is normally reported at lr 1e-3, a value tuned for ImageNet-scale probe
sets.  On 972 training images that rate drives train loss 1.60 → 0.52 while
validation balanced accuracy *falls* 0.38 → 0.27, so the selected epoch is 1 and
the probe is never actually fitted — it reports its initialisation, which is
mean pooling.  The default here is 3e-5, where the validation optimum lands
mid-schedule.  `training_curves.png` is written on every run precisely so this
is visible rather than assumed; read it before quoting a number.  Epoch
selection is on validation, never test, and the 246-image validation split has
only 5 images in the rarest grade, so the curve is noisy — which is also why the
seed spread is reported alongside the test bootstrap.

## Conventions

- `--seed` on every script; recorded in `run_config.json`.
- `--cfg_scale 1.0` on every probe (E1–E4).  Classifier-free guidance
  extrapolates the conditioning signal, which raises fidelity and suppresses
  diversity — precisely the quantities E1 and E2 measure — so a tuned CFG value
  would let the sampler decide the result.  At 1.0 the sampler skips the
  `null_h` branch entirely and the generation is a pure function of `h`, which
  is also what makes the MRR comparable to Bordes et al.  `sample_grid.py` keeps
  3.0 because its output is qualitative figures where legibility is the point;
  the value belongs in the caption.  Varying it elsewhere is a sensitivity
  analysis and must be reported as one.
- `--tag` names a run directory; omitted, it defaults to a UTC timestamp.
- `--device cpu|cuda|mps`, falling back to CPU with a warning if unavailable.
- `--encoder` is checked against the checkpoint's `h_dim`; a mismatch raises
  rather than producing plausible nonsense.
- EMA weights are applied automatically when present, so every number comes from
  the same weights.

## Dependency between runs

`E5_downstream_probe.py` consumes `dimensions.json` from an E4 run and, for
condition C, a representation cache built from transformed images:

```bash
python experiments/E4_dimension_structure.py --checkpoint models/jit_dinov3/final.pt \
    --encoder dinov3 --n_queries 10 --n_dims 64 --seed 0

python experiments/E5_downstream_probe.py --encoder dinov3 \
    --dimensions_file experiments/results/E4_dimension_structure/dinov3/<tag>/dimensions.json \
    --n_seeds 5
```

E5 also needs `val`/`test` representation caches, and condition C needs the test
split re-encoded under an E3 transform.  Build the latter with

```bash
for v in crop_center zoom_x2 grayscale brightness_up; do
    python data/scripts/precompute_reps.py --encoder dinov3 \
        --data_dir data/raw/messidor2/test \
        --out_file data/processed/messidor2/dinov3/transformed/test_reps_${v}.pt \
        --variant $v --seed 0
done
```

`--variant` applies the transform at the source resolution, from the same
`rcdm/encoders/transforms.py` definitions E3 measures, so the E3 cosine and the
E5 accuracy drop refer to the identical transform.  `--transformed_reps` accepts
several caches at once; they are then compared against one shared baseline fit,
which is what makes the drops comparable to each other.

Pick the variants deliberately: at least one E3 found the encoder is *not*
invariant to, and at least one it is.  The invariant one is the control — if it
also drops, the drop is not about invariance.

`E5_weighted_ap.py` is `E5_attention_probing.py` with one thing changed: the
inverse-frequency class weighting both read-outs apply is hard-coded in the
former and exposed as `--class_weighting` in the latter.  Messidor-2 runs 58%
no-DR down to 19 proliferative training images, so that correction sets every
number the probe reports, and the default (`balanced`) reproduces
`E5_attention_probing.py` bit-for-bit — the point of the script is to be able
to *report* the choice as a sensitivity analysis instead of assuming it:

```bash
for w in none sqrt_inv balanced effective_number; do
    python experiments/E5_weighted_ap.py --encoder dinov3 \
        --class_weighting $w --tag "w_$w" --skip_generated --device mps
done
```

Schemes are renormalised to a sample-mean weight of 1, so switching one does not
also change the effective learning rate and confound the comparison; the weights
and each class's resulting share of the training loss are written to
`class_weights.csv` and `figures/class_weighting.png`.  Read the schemes against
balanced accuracy only — plain accuracy rises under `none` precisely because the
probe is collapsing onto no-DR.

`E5_attention_probing.py` needs **token** caches instead, which the pooled
`*_reps.pt` files do not contain — the attentive probe pools over a sequence, so
there has to be a sequence:

```bash
for split in train val test; do
    python data/scripts/precompute_reps.py --encoder dinov3 --tokens \
        --data_dir data/raw/messidor2/$split \
        --out_file data/processed/messidor2/dinov3/${split}_tokens.pt
done

python experiments/E5_attention_probing.py --encoder dinov3 \
    --checkpoint models/jit_dinov3/final.pt --device mps --n_seeds 3
```

These are large — 150 MB for dinov3's train split, ~374 MB for RETFound's — so
they are written as float16 to a separate file rather than folded into the
pooled cache the rest of the pipeline reads.  `tokens[:, 0]` is the pooled
vector, and the script *checks* that on load against `<split>_reps.pt`, refusing
to run if the path lists differ or the vectors disagree beyond float16 rounding
(~3e-4 relative).  Without that gate, a cache built from a different ordering or
a different checkpoint would yield entirely plausible numbers describing
different images from E1–E4 and E6.

The generation step (`h → image → h′`) runs inside the script and is the slow
part: ~31 min for Messidor-2's 526 test images at 50 ODE steps on an M-series
GPU.  It is cached to `<reps_dir>/generated/test_tokens_from_h.pt`, keyed on the
checkpoint sha256, step count, `cfg_scale`, seed and a hash of the image list, so
a cache built under different settings is regenerated rather than silently
reused.  `--n_test_images` takes a class-stratified subsample and subsets both
conditions together, so the round-trip pairing survives — that is what makes a
trial run cheap.
