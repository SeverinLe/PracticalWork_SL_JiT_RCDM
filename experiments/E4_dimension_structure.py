"""
experiments/E4_dimension_structure.py — RQ4: is the representation structured?

Bordes et al. (2022) found that individual dimensions of an SSL representation
carry separable, editable factors: take a query image, find its nearest
neighbours in representation space, identify the dimensions those neighbours
share, and that subset behaves like a handle on whatever the neighbourhood has
in common (in their ImageNet case, the dog-and-background signal, leaving the
clothing the dog was wearing untouched).

The three conditions below are a direct port of the reference implementation
(facebookresearch/RCDM, scripts/image_sample_manipulation.py), which builds
exactly three conditioning vectors and concatenates them into one batch:

    feat_zero[:, dims] = 0
    feat_transformation[:, dims] = feat_target[:, dims]

  1. **Select** the dimensions shared across the query's k-NN neighbourhood.
  2. **Mask** them (set to zero) and generate — what disappears was carried by
     those dimensions.
  3. **Substitute** them with the values from a donor image and generate — if
     the factor transplants, the representation is compositional in that
     direction.

Where this deviates from the reference, and why
-----------------------------------------------
*Dimension selection.*  The reference counts, over the k=20 neighbours, how many
have a **non-zero** value in each dimension and keeps those exceeding a
threshold (``n_common``, default 14 of 20).  That heuristic is only meaningful
for a representation with genuine exact zeros — theirs is a SimCLR ResNet-50
avgpool feature, i.e. post-ReLU and therefore non-negative and sparse.  The
ViT CLS features under study here are dense and signed (measured on
messidor2/train: 0.0% exact zeros, ~50% negative entries for both dinov3 and
retfound_cfp), so the reference rule selects *every* dimension and degenerates.
``--selection common_nonzero`` is the faithful port and is available for the
resnet50 control; it refuses to run on a dense cache rather than silently
returning all D dimensions.  ``common_active`` and ``low_variance`` are the
dense-feature stand-ins, and unlike the reference they select a *fixed* number
of dimensions (``--n_dims``) rather than however many clear a threshold.

*Donor image.*  The reference uses a hand-picked target image supplied by the
user (their clothed-dog / unclothed-dog pair).  ``--donor`` reproduces that with
``image``; the default ``least_similar`` instead transplants from the least
similar representation in the cache, which is the harder test but is not what
the paper shows.

*Noise.*  The reference draws fresh noise for the whole concatenated batch, so
its three rows differ through both h and the noise draw.  Here the same noise is
reused across conditions, so a difference between rows is attributable to the
edit alone.

Interpretation is deliberately qualitative here; the quantitative counterpart is
E5, which feeds the same masked representations to a linear probe and asks
whether the visually-removed information was also the decodable information.
The selected dimension indices are written to dimensions.json for exactly that.

Caveat to carry into the report: this is a heuristic, not a discovery
procedure — the dimensions it finds depend on k, on the selection rule, and on
the encoder, and there is no guarantee that a "factor" aligns with coordinate
axes at all.

Usage:
    python experiments/E4_dimension_structure.py \
        --checkpoint models/jit_dinov3/final.pt \
        --encoder    dinov3 \
        --reps_file  data/processed/messidor2/dinov3/train_reps.pt \
        --n_queries  10 --k_neighbours 16 --n_dims 64 --seed 0 --device mps
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from common import (
    create_run_dir,
    load_generator,
    load_probe_encoder,
    resolve_device,
    save_metrics,
    set_plot_style,
    set_seed,
    sha256_file,
    unnorm,
)

EXPERIMENT = "E4_dimension_structure"


def select_dimensions(neighbourhood: torch.Tensor, n_dims: int, selection: str,
                      n_common: int = 14) -> torch.Tensor:
    """
    Pick the dimensions that a k-NN neighbourhood has in common.

    Args:
        neighbourhood: (k, D) representations of the query and its neighbours.
        n_dims: how many dimensions to return (fixed-size rules only).
        selection: "common_nonzero" (the reference rule: dimensions non-zero in
            more than ``n_common`` of the k neighbours — variable size, requires
            a sparse representation), "common_active" (high mean magnitude
            relative to spread) or "low_variance" (smallest variance across the
            neighbourhood).
        n_common: threshold for "common_nonzero", counted in neighbours.

    Returns:
        1-D long tensor of dimension indices, ascending.
    """
    if selection == "common_nonzero":
        # Port of scripts/image_sample_manipulation.py: count in how many
        # neighbours each dimension is non-zero, keep those above the threshold.
        counts = (neighbourhood != 0).sum(dim=0)
        return (counts > n_common).nonzero(as_tuple=True)[0].sort().values
    if selection == "low_variance":
        score = -neighbourhood.var(dim=0)                    # higher = more agreement
    elif selection == "common_active":
        mean = neighbourhood.mean(dim=0).abs()
        spread = neighbourhood.std(dim=0) + 1e-8
        score = mean / spread                                # consistently large
    else:
        raise ValueError(f"unknown selection rule {selection!r}")
    return torch.topk(score, k=min(n_dims, score.numel())).indices.sort().values


def generate_from(model, flow, h: torch.Tensor, noise: torch.Tensor,
                  num_steps: int, cfg_scale: float) -> torch.Tensor:
    """Generate with a fixed noise draw so rows differ only through h."""
    with torch.no_grad():
        return flow.sample(model, noise.clone(), h.repeat(noise.shape[0], 1),
                           num_steps=num_steps, cfg_scale=cfg_scale)


def save_labelled_grid(out_path, rows: list, subtitle: str) -> None:
    """
    Render the three-condition grid with every panel labelled.

    A bare image grid is unreadable six months later: which row was masked, what
    the leftmost column is, and which image the substitution came from all have
    to be recoverable from the figure itself for it to go in the report.

    Args:
        out_path: destination .png.
        rows: one dict per condition, with keys ``label`` (left-hand row title),
            ``reference`` (C,H,W tensor in [0,1] for the leftmost column),
            ``reference_label`` (caption under it) and ``samples`` (N,C,H,W).
        subtitle: provenance line placed under the title.
    """
    import matplotlib.pyplot as plt

    n_samples = rows[0]["samples"].shape[0]
    n_cols = n_samples + 1
    fig, axes = plt.subplots(
        len(rows), n_cols, figsize=(1.7 * n_cols + 1.1, 1.7 * len(rows) + 1.0),
        gridspec_kw={"width_ratios": [1.12] + [1.0] * n_samples},
    )
    axes = np.atleast_2d(axes)

    def show(ax, image) -> None:
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    for r, row in enumerate(rows):
        # Leftmost column: the real image the row refers to, boxed to separate it
        # from the generations beside it.
        ax = axes[r][0]
        show(ax, row["reference"])
        for spine in ax.spines.values():
            spine.set(visible=True, linewidth=1.6, edgecolor="#444444")
        ax.set_ylabel(row["label"], fontsize=8, labelpad=8)
        ax.set_xlabel(row["reference_label"], fontsize=7, color="#444444", labelpad=3)

        for c in range(n_samples):
            ax = axes[r][c + 1]
            show(ax, row["samples"][c])
            for spine in ax.spines.values():
                spine.set_visible(False)

    axes[0][0].set_title("conditioned on", fontsize=8, color="#444444")
    for c in range(n_samples):
        axes[0][c + 1].set_title(f"sample {c + 1}", fontsize=8, color="#444444")

    fig.suptitle("E4 — masking and substituting shared dimensions", fontsize=10, y=0.995)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=7, color="#444444")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="E4 — dimension-level structure")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="dinov3")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--reps_file", default="data/processed/messidor2/dinov3/train_reps.pt",
                        help="Representation cache searched for nearest neighbours")
    parser.add_argument("--n_queries", type=int, default=10,
                        help="Query images. Bordes et al. showed one; ten keeps "
                             "the finding from being anecdotal.")
    parser.add_argument("--k_neighbours", type=int, default=16)
    parser.add_argument("--n_dims", type=int, default=64,
                        help="Dimensions to mask/substitute (ignored by "
                             "common_nonzero, which is threshold-based)")
    parser.add_argument("--selection", default="common_active",
                        choices=["common_active", "low_variance", "common_nonzero"],
                        help="common_nonzero is the reference rule and needs a "
                             "sparse (post-ReLU) representation such as resnet50")
    parser.add_argument("--n_common", type=int, default=14,
                        help="common_nonzero only: keep dimensions non-zero in "
                             "more than this many neighbours (reference default 14 of 20)")
    parser.add_argument("--donor", default="least_similar",
                        choices=["least_similar", "random", "image"],
                        help="Source of the substituted values. 'image' reproduces "
                             "the reference's hand-picked target and needs --donor_image.")
    parser.add_argument("--donor_image", default=None,
                        help="Donor image path, used when --donor image")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Generations per condition")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="1.0 = no guidance, and the default for every probe. "
                             "CFG would amplify the effect of an edited h and "
                             "overstate how much each dimension controls.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)
    set_plot_style()

    model, flow, cfg = load_generator(args.checkpoint, device)
    encoder, enc_transform = load_probe_encoder(args.encoder, device, cfg, args.encoder_ckpt)
    image_size = cfg["image_size"]

    cache = torch.load(args.reps_file, map_location="cpu", weights_only=False)
    reps, paths = cache["reps"].float(), cache["paths"]
    if cache.get("encoder") and cache["encoder"] != args.encoder:
        raise ValueError(f"{args.reps_file} was built with encoder "
                         f"'{cache['encoder']}', not '{args.encoder}'")
    if reps.shape[1] != cfg["h_dim"]:
        raise ValueError(f"cache has h_dim {reps.shape[1]}, generator expects {cfg['h_dim']}")
    sparsity = float((reps == 0).float().mean())
    print(f"  [cache] {reps.shape[0]} representations, D={reps.shape[1]}, "
          f"{sparsity:.1%} exact zeros")

    # The reference rule counts non-zero dimensions, which only discriminates if
    # the representation actually has zeros. On a dense cache it would select all
    # D dimensions and quietly turn the "masked" row into an unconditional sample.
    if args.selection == "common_nonzero" and sparsity < 0.01:
        raise ValueError(
            f"--selection common_nonzero needs a sparse representation, but "
            f"{args.reps_file} is {sparsity:.2%} zeros — the rule would select all "
            f"{reps.shape[1]} dimensions. It is the reference (post-ReLU ResNet-50) "
            f"heuristic; for a dense ViT CLS feature use common_active or low_variance."
        )
    if args.donor == "image" and not args.donor_image:
        parser.error("--donor image requires --donor_image")

    run_dir = create_run_dir(
        EXPERIMENT, args.encoder, args.tag, args,
        extra={"checkpoint_sha256": sha256_file(args.checkpoint), "model_cfg": cfg,
               "n_reps": int(reps.shape[0])},
    )

    display_transform = transforms.Compose([
        transforms.Resize(image_size), transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    normalised = F.normalize(reps, dim=1)
    rng = np.random.default_rng(args.seed)
    query_indices = rng.choice(len(paths), size=min(args.n_queries, len(paths)),
                               replace=False)

    # A --donor image is encoded once and reused for every query, matching the
    # reference's single hand-picked target.
    fixed_donor_h, fixed_donor_path = None, None
    if args.donor == "image":
        fixed_donor_path = Path(args.donor_image)
        with torch.no_grad():
            fixed_donor_h = encoder(
                enc_transform(Image.open(fixed_donor_path).convert("RGB"))
                .unsqueeze(0).to(device)
            ).float().cpu()
        print(f"  [donor] fixed target {fixed_donor_path.name}")

    rows, selected_dims = [], {}
    for query_idx in query_indices:
        query_idx = int(query_idx)
        query_path = Path(paths[query_idx])

        # Nearest neighbours by cosine similarity, query included.
        similarity = normalised @ normalised[query_idx]
        neighbour_idx = torch.topk(similarity, k=args.k_neighbours + 1).indices
        neighbourhood = reps[neighbour_idx]

        dims = select_dimensions(neighbourhood, args.n_dims, args.selection, args.n_common)
        selected_dims[query_path.stem] = dims.tolist()
        if dims.numel() == 0:
            print(f"  [skip] {query_path.stem}: selection rule matched no dimensions")
            continue

        if fixed_donor_h is not None:
            donor_path, donor_h = fixed_donor_path, fixed_donor_h
        else:
            # least_similar transplants from a genuinely different image rather
            # than a near-duplicate; random is the neutral control.
            donor_idx = (int(torch.argmin(similarity).item()) if args.donor == "least_similar"
                         else int(rng.choice(len(paths))))
            donor_path, donor_h = Path(paths[donor_idx]), reps[donor_idx: donor_idx + 1]

        h_original = reps[query_idx: query_idx + 1].to(device)
        h_masked = h_original.clone()
        h_masked[:, dims] = 0.0
        h_substituted = h_original.clone()
        h_substituted[:, dims] = donor_h[:, dims].to(device)

        noise = torch.randn(args.n_samples, 3, image_size, image_size, device=device)
        conditions = {
            "original": h_original,
            "masked": h_masked,
            "substituted": h_substituted,
        }

        grid_rows, cosines = [], {}
        for name, h in conditions.items():
            generated = generate_from(model, flow, h, noise, args.num_steps, args.cfg_scale)
            grid_rows.append(unnorm(generated).cpu())
            # How far the edit moved the representation, for the record.
            cosines[name] = float(
                F.cosine_similarity(h.cpu(), h_original.cpu(), dim=1).item()
            )
            rows.append({
                "query": query_path.stem,
                "condition": name,
                "donor": donor_path.stem if name == "substituted" else "",
                "n_dims_selected": int(dims.numel()),
                "cosine_to_original": cosines[name],
                "l2_to_original": float((h.cpu() - h_original.cpu()).norm().item()),
            })

        query_thumb = display_transform(Image.open(query_path).convert("RGB"))
        donor_thumb = display_transform(Image.open(donor_path).convert("RGB"))

        n_sel = int(dims.numel())
        out = run_dir / "figures" / f"dims_{query_path.stem}.png"
        save_labelled_grid(
            out,
            rows=[
                {"label": "original\nh unchanged",
                 "reference": query_thumb, "reference_label": f"query\n{query_path.stem}",
                 "samples": grid_rows[0]},
                {"label": f"masked\n{n_sel} dims -> 0\ncos={cosines['masked']:.3f}",
                 "reference": query_thumb, "reference_label": f"query\n{query_path.stem}",
                 "samples": grid_rows[1]},
                {"label": f"substituted\n{n_sel} dims <- donor\ncos={cosines['substituted']:.3f}",
                 "reference": donor_thumb, "reference_label": f"donor\n{donor_path.stem}",
                 "samples": grid_rows[2]},
            ],
            subtitle=(f"{args.encoder}  |  selection={args.selection}  "
                      f"|  k={args.k_neighbours}  |  {n_sel}/{reps.shape[1]} dims  "
                      f"|  cfg={args.cfg_scale}  |  seed={args.seed}"),
        )
        print(f"  {out.name}  ({n_sel} dims; rows: original, masked, "
              f"substituted<-{donor_path.stem})")

    save_metrics(run_dir, "metrics", rows,
                 ["query", "condition", "donor", "n_dims_selected",
                  "cosine_to_original", "l2_to_original"])

    # E5 reads this file to test whether the visually-removed dimensions are also
    # the ones a linear probe depends on.
    counts = [len(dims) for dims in selected_dims.values()]
    (run_dir / "dimensions.json").write_text(json.dumps({
        "selection": args.selection,
        "n_dims_requested": args.n_dims,
        "n_common": args.n_common if args.selection == "common_nonzero" else None,
        "n_dims_selected": {"min": min(counts, default=0), "max": max(counts, default=0)},
        "k_neighbours": args.k_neighbours,
        "donor": args.donor,
        "encoder": args.encoder,
        "h_dim": int(reps.shape[1]),
        "per_query": selected_dims,
        "union": sorted({d for dims in selected_dims.values() for d in dims}),
    }, indent=2))

    print(f"\n  dimensions.json written ({min(counts, default=0)}-{max(counts, default=0)} "
          f"dims per query) — feed it to E5")
    print(f"Done — {run_dir}")


if __name__ == "__main__":
    main()
