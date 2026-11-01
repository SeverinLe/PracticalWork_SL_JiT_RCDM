"""
experiments/E3_invariance.py — RQ3: which transformations is h invariant to?

Merges the previous invariance_images.py + invariance_probe.py into one script
and scales the measurement from one image per class to a stratified sample, so
the cosine similarities become distributions with confidence intervals rather
than anecdotes.

Two levels of measurement
-------------------------
  (a) **Backbone invariance** — cosine similarity between h(T(x)) and h(x) for
      each transform T.  A perfectly invariant encoder returns 1.0.
  (b) **Generator-visible invariance** (optional, --n_visual > 0) — generate
      from h(T(x)) reusing the *same* starting noise across all variants, so any
      visual difference between rows is attributable to h alone rather than to
      the noise draw.  This is the controlled version; drawing fresh noise per
      variant instead mixes the two factors and answers a different question.

Why this matters clinically
---------------------------
Nuisance factors that survive into h are factors a downstream classifier can
silently latch onto.  Greyscale and intensity/contrast shifts stand in for
device and acquisition-protocol differences; if h moves under them, a grading
model trained on one site's images carries a hidden dependence on that site.

Statistics
----------
Per transform: mean cosine ± bootstrap 95% CI over images.  Transforms are
compared against the least-disruptive one with a paired Wilcoxon signed-rank
test (same images under both conditions), Holm-corrected across the family.

Per transform also: MRR ± bootstrap 95% CI and top-1 rate of retrieving the
true original among all sampled untransformed images, against the chance MRR
of a random ranking.  The per-image reciprocal ranks land in two places:
``cosine_similarity.csv`` (long form, column ``rr``) and ``mrr_per_image.csv``
(one row per image, one column per transform).  The latter is what the former
standalone ``E3_mrr.py`` derived from the CSV; it is written here so that a
run directory is complete without a second script.

Usage:
    python experiments/E3_invariance.py \
        --checkpoint models/jit_dinov3/final.pt \
        --encoder    dinov3 \
        --test_dir   data/raw/messidor2/test \
        --n_per_class 30 --n_visual 1 --seed 0 --device mps

Set --skip_generation to measure the encoder alone (no checkpoint needed).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image

from common import (
    create_run_dir,
    load_generator,
    load_probe_encoder,
    resolve_device,
    save_metrics,
    set_plot_style,
    set_seed,
    sha256_file,
    stratified_sample,
    unnorm,
)
from rcdm.encoders.transforms import VARIANT_ORDER, make_variants  # noqa: E402

EXPERIMENT = "E3_invariance"

# VARIANT_ORDER and make_variants live in rcdm/encoders/transforms.py: E5's
# condition C re-encodes the same transformed images through
# precompute_reps.py --variant, and the two must be the same transform for the
# E3 cosine and the E5 accuracy drop to be about the same thing.


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple:
    """Percentile bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = [values[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def holm_correction(pvalues: dict) -> dict:
    """Holm-Bonferroni step-down correction over a family of tests."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    corrected, running_max = {}, 0.0
    for i, (key, p) in enumerate(ordered):
        adjusted = min(1.0, (m - i) * p)
        running_max = max(running_max, adjusted)  # enforce monotonicity
        corrected[key] = running_max
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 — representation invariance")
    parser.add_argument("--checkpoint", default=None,
                        help="Required unless --skip_generation")
    parser.add_argument("--encoder", default="dinov3")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--test_dir", default="data/raw/messidor2/test")
    parser.add_argument("--n_per_class", type=int, default=30,
                        help="Images per class for the cosine-similarity statistics")
    parser.add_argument("--n_visual", type=int, default=1,
                        help="Images per class that additionally get a generated "
                             "comparison grid (0 = statistics only)")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Generations per variant in the qualitative grids")
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="1.0 = no guidance, and the default for every probe. "
                             "Only the visual panels use the generator here, but "
                             "they must come from the same sampler setting as the "
                             "rest of the study.")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Measure the encoder only; no generator needed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    if not args.skip_generation and not args.checkpoint:
        parser.error("--checkpoint is required unless --skip_generation is set")

    device = resolve_device(args.device)
    set_seed(args.seed)
    set_plot_style()
    rng = torch.Generator().manual_seed(args.seed)

    cfg, model, flow = None, None, None
    if not args.skip_generation:
        model, flow, cfg = load_generator(args.checkpoint, device)
    encoder, enc_transform = load_probe_encoder(args.encoder, device, cfg, args.encoder_ckpt)

    pairs = stratified_sample(args.test_dir, args.n_per_class, seed=args.seed)
    print(f"  [data] {len(pairs)} images over {len(set(c for c, _ in pairs))} classes")

    extra = {"n_images": len(pairs), "variants": VARIANT_ORDER}
    if args.checkpoint:
        extra["checkpoint_sha256"] = sha256_file(args.checkpoint)
        extra["model_cfg"] = cfg
    run_dir = create_run_dir(EXPERIMENT, args.encoder, args.tag, args, extra=extra)

    # ---- (a) backbone invariance -------------------------------------------
    rows, per_variant = [], {name: [] for name in VARIANT_ORDER}
    embeddings = {name: [] for name in VARIANT_ORDER}        # h(T(x)) per image, for MRR

    for class_name, path in pairs:
        img = Image.open(path).convert("RGB")
        variants = make_variants(img, args.crop_size, rng)

        with torch.no_grad():
            x = torch.stack([enc_transform(v) for v in variants.values()]).to(device)
            h = encoder(x)                                   # (V, h_dim)
        h_original = h[0:1]
        for name, vec in zip(variants, h.cpu()):
            embeddings[name].append(vec)
        cosines = F.cosine_similarity(h, h_original.expand_as(h), dim=1).cpu().numpy()

        for name, cosine in zip(variants, cosines):
            rows.append({"image": path.stem, "class": class_name,
                         "variant": name, "cosine": float(cosine)})
            per_variant[name].append(float(cosine))

    # ---- retrieval (MRR) ----------------------------------------------------
    # Cosine says how far h moves; MRR says whether it moves *past other
    # images*.  Query h(T(x_i)) against the gallery of untransformed h(x_j) over
    # all sampled images; rr = 1/rank of the true original x_i.  Strict ">" counts
    # only images that are better than the true one, so exact ties are resolved
    # optimistically -- irrelevant for continuous 1024-d vectors but stated.
    gallery = F.normalize(torch.stack(embeddings["original"]), dim=1)
    per_variant_rr = {}
    for name in VARIANT_ORDER:
        query = F.normalize(torch.stack(embeddings[name]), dim=1)
        sims = query @ gallery.T                                  # (N, N)
        true = sims.diag().unsqueeze(1)
        ranks = 1 + (sims > true).sum(dim=1).numpy()
        per_variant_rr[name] = 1.0 / ranks
    cursor = {name: 0 for name in VARIANT_ORDER}
    for row in rows:
        name = row["variant"]
        row["rr"] = float(per_variant_rr[name][cursor[name]])
        cursor[name] += 1
    torch.save({"image": [p.stem for _, p in pairs], "class": [c for c, _ in pairs],
                "h": {n: torch.stack(v) for n, v in embeddings.items()}},
               run_dir / "embeddings.pt")

    save_metrics(run_dir, "cosine_similarity", rows,
                 ["image", "class", "variant", "cosine", "rr"])

    # Wide form of the same reciprocal ranks: one row per image, so an image that
    # is retrieved under one transform and lost under another reads off one line.
    # Index i of every per_variant_rr array is pairs[i], by construction above.
    wide_rows = [{"image": path.stem, "class": class_name,
                  **{name: float(per_variant_rr[name][i]) for name in VARIANT_ORDER}}
                 for i, (class_name, path) in enumerate(pairs)]
    save_metrics(run_dir, "mrr_per_image", wide_rows, ["image", "class", *VARIANT_ORDER])

    # ---- statistics ---------------------------------------------------------
    summary = {}
    for name in VARIANT_ORDER:
        values = np.array(per_variant[name])
        if values.size == 0:
            continue
        mean, lo, hi = bootstrap_ci(values, args.n_boot, seed=args.seed)
        rr = per_variant_rr[name]
        rr_mean, rr_lo, rr_hi = bootstrap_ci(rr, args.n_boot, seed=args.seed)
        summary[name] = {"mean": mean, "ci95": [lo, hi], "std": float(values.std(ddof=1)),
                         "n": int(values.size),
                         "mrr": rr_mean, "mrr_ci95": [rr_lo, rr_hi],
                         "top1": float((rr == 1.0).mean()),
                         "chance_mrr": float(np.mean(1.0 / np.arange(1, len(rr) + 1)))}

    # Paired Wilcoxon against the transform that disturbs h least (excluding the
    # original, whose cosine is 1.0 by construction).
    comparisons = [n for n in summary if n != "original"]
    reference = max(comparisons, key=lambda n: summary[n]["mean"]) if comparisons else None
    if reference:
        try:
            from scipy.stats import wilcoxon

            raw_p = {}
            for name in comparisons:
                if name == reference:
                    continue
                stat, p = wilcoxon(per_variant[reference], per_variant[name])
                raw_p[name] = float(p)
            adjusted = holm_correction(raw_p)
            for name in raw_p:
                summary[name]["wilcoxon_vs_reference"] = {
                    "reference": reference,
                    "p_raw": raw_p[name],
                    "p_holm": adjusted[name],
                }
        except ImportError:
            print("  [stats] scipy not installed — skipping Wilcoxon tests")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure -------------------------------------------------------------
    import matplotlib.pyplot as plt

    names = [n for n in VARIANT_ORDER if n in summary and n != "original"]
    means = [summary[n]["mean"] for n in names]
    errors = np.array([[summary[n]["mean"] - summary[n]["ci95"][0] for n in names],
                       [summary[n]["ci95"][1] - summary[n]["mean"] for n in names]])

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(names, means, yerr=errors, capsize=3, color="#4C72B0")
    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.set_ylabel(r"cosine$(h(T(x)),\ h(x))$")
    ax.set_ylim(min(means) - 0.1, 1.02)
    ax.set_title(f"E3 — {args.encoder} invariance, n={summary[names[0]]['n']} images/"
                 f"transform\nerror bars: bootstrap 95% CI")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.savefig(run_dir / "figures" / "invariance_cosine.png")

    # ---- (b) generator-visible invariance ----------------------------------
    if not args.skip_generation and args.n_visual > 0:
        visual_pairs = stratified_sample(args.test_dir, args.n_visual, seed=args.seed)
        for class_name, path in visual_pairs:
            img = Image.open(path).convert("RGB")
            variants = make_variants(img, args.crop_size, rng)

            # One noise draw, reused for every variant: differences between rows
            # are then attributable to h alone.
            noise = torch.randn(args.n_samples, 3, cfg["image_size"], cfg["image_size"],
                                device=device, generator=torch.Generator(device=device)
                                .manual_seed(args.seed))

            grid_rows = []
            for name, variant in variants.items():
                with torch.no_grad():
                    h = encoder(enc_transform(variant).unsqueeze(0).to(device))
                    generated = flow.sample(model, noise.clone(),
                                            h.repeat(args.n_samples, 1),
                                            num_steps=args.num_steps,
                                            cfg_scale=args.cfg_scale)
                thumb = TF.to_tensor(variant.resize((cfg["image_size"],) * 2))
                grid_rows.append(torch.cat([thumb.unsqueeze(0).to(device),
                                            unnorm(generated)], dim=0))

            out = run_dir / "figures" / f"variants_{class_name}_{path.stem}.png"
            vutils.save_image(torch.cat(grid_rows), out,
                              nrow=args.n_samples + 1, padding=2)
            print(f"  {out.name}  (rows: {', '.join(variants)})")

    print("\n  cosine by transform:")
    for name in names:
        s = summary[name]
        print(f"    {name:20s} {s['mean']:.4f}  [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]")
    # "original" is kept here, unlike the cosine table: its MRR is the retrieval
    # ceiling (1.0 unless two images share an embedding), the row the others
    # are read against.
    chance = next(iter(summary.values()))["chance_mrr"]
    print(f"\n  retrieval of the true original (chance MRR {chance:.3f}):")
    print(f"    {'':20s} {'MRR':>6s}  {'95% CI':>16s}  {'top1':>5s}")
    for name in (n for n in VARIANT_ORDER if n in summary):
        s = summary[name]
        print(f"    {name:20s} {s['mrr']:.3f}  [{s['mrr_ci95'][0]:.3f}, {s['mrr_ci95'][1]:.3f}]"
              f"  {s['top1']:.3f}")
    print(f"\nDone — {run_dir}")


if __name__ == "__main__":
    main()
