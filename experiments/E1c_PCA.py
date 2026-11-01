"""
experiments/E1c_PCA.py — do generations land on the real test manifold?

Self-contained by design: this script imports only ``rcdm``, never the other
experiment modules, so it can be copied to a server together with the model and
the data and run there.  Everything it needs from experiments/common.py is
duplicated below, which is the one place in this repository where duplication is
the right call.

Two questions, one pass over the test split.  Every test image is encoded, one
image is generated from each resulting h, and the generation is re-encoded:

  1. **Where do the generations sit?**  A 2-D PCA fitted on the real h, with the
     generated h' projected in out-of-sample, so "outside the real cloud" is a
     statement the figure can actually make.

  2. **Are they distinguishable at all?**  A classifier two-sample test: label
     real 0 and generated 1, fit logistic regression, read held-out accuracy.
     0.5 means a linear boundary cannot separate them; 1.0 means the generator
     has a systematic offset.  Three details make this honest —

       * the design is balanced by construction (one generation per real image),
         so 0.5 really is the baseline and balanced accuracy is reported anyway;
       * folds are grouped by source image, because a generation and the image it
         came from are highly correlated and splitting a pair across a fold
         boundary leaks the answer;
       * it is run in *both* 2-D and full representation space.  In full-d, a few
         hundred points per class in a 384- or 1024-dimensional space are
         linearly separable almost by construction, so only the cross-validated
         number means anything, and the 2-D score is the one the figure shows.

The grade probe from the earlier version is kept as a secondary line, now
printed next to its own majority-class baseline: the real and generated sets have
different class balances, and a single shared baseline makes the real score
unreadable.

Run:

    python experiments/E1c_PCA.py \
        --checkpoint models/jit_dinov3/final.pt --encoder dinov3 --device mps
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

sys.path.insert(0, str(REPO_ROOT))

from rcdm.encoders import build_transform, get_encoder, get_h_dim  # noqa: E402
from rcdm.jit import FlowMatching, create_jit_model                # noqa: E402

EXPERIMENT = "E1c_PCA"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Ordinal severity, so the grades get a sequential ramp (light -> dark) rather
# than a categorical rainbow: adjacent grades *should* look adjacent.
GRADE_ORDER = ["anodr", "bmilddr", "cmoderatedr", "dseveredr", "eproliferativedr"]
GRADE_COLOURS = {
    "anodr": "#A9D5EE",
    "bmilddr": "#66B3E0",
    "cmoderatedr": "#2E86C1",
    "dseveredr": "#1B5A8E",
    "eproliferativedr": "#0B2E4F",
}
REAL_COLOUR = "#9AA7B1"
GEN_COLOUR = "#D1603D"
REAL_CLOUD = "#D5D5D5"


# ---------------------------------------------------------------------------
# Plumbing — duplicated from experiments/common.py to keep this file standalone
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed torch (CPU + all accelerators), numpy and random."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Honour the requested device, falling back to CPU with a printed warning."""
    if requested == "mps" and not torch.backends.mps.is_available():
        print("  [device] mps unavailable -> cpu")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("  [device] cuda unavailable -> cpu")
        return torch.device("cpu")
    return torch.device(requested)


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 — checkpoints are too large to read into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """Current commit hash, with a -dirty suffix if the tree has changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _versions() -> dict:
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "platform": platform.platform()}
    for name in ("torchvision", "timm", "transformers", "numpy", "sklearn"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 — a missing optional dep is not an error
            versions[name] = None
    return versions


def create_run_dir(encoder: str, tag: str | None, args, extra: dict) -> Path:
    """results/<experiment>/<encoder>/<tag>/ with its manifest written up front."""
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / EXPERIMENT / encoder / tag
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": EXPERIMENT, "encoder": encoder, "tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "command": " ".join(sys.argv),
        "args": vars(args), "versions": _versions(), **extra,
    }
    (run_dir / "run_config.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"  [run] {run_dir}")
    return run_dir


def save_csv(run_dir: Path, name: str, rows: list, fieldnames: list) -> None:
    """Write a list of dicts to <run_dir>/<name>.csv."""
    out = run_dir / f"{name}.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [csv] {out.name} ({len(rows)} rows)")


def set_plot_style() -> None:
    """One matplotlib style for every figure, so the report looks like one document."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
    })


def unnorm(x: torch.Tensor) -> torch.Tensor:
    """Diffusion space [-1, 1] -> display space [0, 1]."""
    return (x.clamp(-1, 1) + 1.0) / 2.0


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """(3, H, W) in [-1, 1] -> PIL RGB, quantisation included."""
    array = (unnorm(x).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def list_split(split_dir, n_per_class: int | None = None, seed: int = 0) -> list:
    """
    Every image in an ImageFolder split, as (class_name, Path), sorted by class.

    The whole split is the point: one generation per real image makes the
    two-sample test balanced by construction.  ``n_per_class`` caps the count for
    quick smoke runs and should be left unset for a real run.
    """
    import random

    rng = random.Random(seed)
    picked = []
    for class_dir in sorted(p for p in Path(split_dir).iterdir() if p.is_dir()):
        images = sorted(p for p in class_dir.iterdir()
                        if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            print(f"  [warn] {class_dir.name}: no images, skipping")
            continue
        if n_per_class is not None and len(images) > n_per_class:
            images = sorted(rng.sample(images, n_per_class))
        picked.extend((class_dir.name, p) for p in images)
    return picked


def load_generator(checkpoint_path, device: torch.device, use_ema: bool = True):
    """
    Restore a trained JiT-RCDM generator.

    Architecture comes from the checkpoint's own ``model_cfg``, never from CLI
    flags.  EMA weights are applied by default, so every reported number comes
    from the same weights.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_cfg" not in state:
        raise ValueError(f"{checkpoint_path} has no 'model_cfg' — it predates the "
                         "JiT trainer and cannot be loaded by this pipeline.")
    cfg = state["model_cfg"]

    model = create_jit_model(
        image_size=cfg["image_size"], patch_size=cfg.get("patch_size", 16),
        hidden_dim=cfg["hidden_dim"], depth=cfg["depth"],
        num_heads=cfg["num_heads"], h_dim=cfg["h_dim"], cond_dim=cfg.get("cond_dim"),
    )
    model.load_state_dict(state["model"])

    ema_state = state.get("ema")
    if use_ema and ema_state:
        shadow = ema_state.get("shadow", {})
        applied = 0
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name].to(param.device))
                applied += 1
        print(f"  [model] EMA weights applied ({applied} tensors)")
    elif use_ema:
        print("  [model] WARNING: no EMA in checkpoint — using raw weights")

    model.eval().to(device)
    print(f"  [model] step {state.get('step')} | h_dim {cfg['h_dim']} "
          f"| hidden {cfg['hidden_dim']} | patch {cfg.get('patch_size')}")
    return model, FlowMatching(), cfg


def load_probe_encoder(encoder_name: str, device: torch.device, cfg: dict,
                       checkpoint_path=None):
    """
    Load the frozen encoder and verify it matches the generator it is paired with.

    A generator's conditioning projector is built for one representation width;
    pairing it with a different encoder produces plausible-looking but meaningless
    samples, so the h_dim mismatch is an error, not a warning.
    """
    if cfg.get("h_dim") != get_h_dim(encoder_name):
        raise ValueError(
            f"Encoder '{encoder_name}' has h_dim {get_h_dim(encoder_name)}, but the "
            f"generator expects {cfg['h_dim']} (trained with "
            f"'{cfg.get('encoder', 'unrecorded')}').")
    kwargs = {"checkpoint_path": checkpoint_path} if checkpoint_path else {}
    return get_encoder(encoder_name, device=str(device), **kwargs), build_transform()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def fit_pca(fit_vectors: np.ndarray, n_components: int = 2):
    """
    PCA on raw centred representations, with deterministic component signs.

    Not standardised: dividing each dimension by its own std would rescale the
    space and destroy the norm geometry, and would break comparability with the
    cosine metric the other probes use.  sklearn's component signs are arbitrary,
    so each is flipped to make its largest-magnitude loading positive — otherwise
    the figure mirrors itself between runs for no reason.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(n_components, *fit_vectors.shape))
    pca.fit(fit_vectors)
    dominant = np.abs(pca.components_).argmax(axis=1)
    signs = np.sign(pca.components_[np.arange(len(dominant)), dominant])
    signs[signs == 0] = 1.0
    pca.components_ *= signs[:, None]
    return pca


def c2st(real: np.ndarray, generated: np.ndarray, seed: int, n_splits: int = 5) -> dict:
    """
    Classifier two-sample test: can a linear boundary tell real from generated?

    Balanced by construction (one generation per real image), so chance is 0.5 —
    balanced accuracy is reported as well, because it stays 0.5 under any future
    imbalance.  Folds are grouped by source image: a generation and the image it
    was conditioned on are highly correlated, and letting a pair straddle a fold
    boundary would inflate the score.

    Returns:
        accuracy, balanced accuracy, and the per-fold spread.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = len(real)
    X = np.vstack([real, generated])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    groups = np.concatenate([np.arange(n), np.arange(n)])

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000, random_state=seed))
    folds = GroupKFold(n_splits=min(n_splits, n))
    predicted = cross_val_predict(model, X, y, cv=folds, groups=groups)

    per_fold = []
    for _, test_idx in folds.split(X, y, groups):
        per_fold.append(float(accuracy_score(y[test_idx], predicted[test_idx])))

    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "fold_accuracy_min": float(np.min(per_fold)),
        "fold_accuracy_max": float(np.max(per_fold)),
        "chance": 0.5,
        "n_per_class": int(n),
    }


def grade_probe(points: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    """
    5-fold CV accuracy of a logistic regression predicting DR grade, with the
    majority-class rate it has to beat.

    The two are only meaningful together: on a 58%-no-DR split an accuracy of
    0.58 is the probe predicting one class for everything.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _, counts = np.unique(labels, return_counts=True)
    majority = float(counts.max() / counts.sum())
    if len(counts) < 2 or counts.min() < 5:
        return {"cv_accuracy": float("nan"), "majority_rate": majority}

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=seed))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(model, points, labels, cv=folds)
    return {"cv_accuracy": float(scores.mean()), "majority_rate": majority}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_real_vs_generated(real_2d, gen_2d, explained, c2st_2d, c2st_full,
                           encoder, out_path) -> None:
    """The two-sample view: one colour per population, the test result in the title."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(real_2d[:, 0], real_2d[:, 1], s=12, c=REAL_COLOUR, linewidths=0,
               alpha=0.75, label=f"real ({len(real_2d)})")
    ax.scatter(gen_2d[:, 0], gen_2d[:, 1], s=12, c=GEN_COLOUR, linewidths=0,
               alpha=0.75, label=f"generated ({len(gen_2d)})")
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
    ax.set_title(f"E1c — real vs generated ({encoder})\n"
                 f"two-sample test: 2-D {c2st_2d['accuracy']:.3f} | "
                 f"full-d {c2st_full['accuracy']:.3f} | chance 0.500")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def plot_by_grade(real_2d, gen_2d, classes, explained, encoder, out_path) -> None:
    """The original E1c view: generations coloured by grade over the real cloud."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(real_2d[:, 0], real_2d[:, 1], s=10, c=REAL_CLOUD, linewidths=0,
               label=f"real ({len(real_2d)})", zorder=1)
    for grade in GRADE_ORDER:
        mask = classes == grade
        if not mask.any():
            continue
        ax.scatter(gen_2d[mask, 0], gen_2d[mask, 1], s=16, marker="X",
                   c=GRADE_COLOURS[grade], linewidths=0, zorder=3,
                   label=f"generated · {grade} ({int(mask.sum())})")
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
    ax.set_title(f"E1c — generations by grade ({encoder})")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E1c — 2-D PCA of the test split and its generations, plus a "
                    "classifier two-sample test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="dinov3")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--test_dir", default="data/raw/messidor2/test")
    parser.add_argument("--n_per_class", type=int, default=None,
                        help="Cap images per grade. Unset (the default) uses the "
                             "whole split — one generation per real image, which "
                             "is what balances the two-sample test.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="1.0 = no guidance. CFG extrapolates the very "
                             "conditioning signal this figure is inspecting.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)
    set_plot_style()

    model, flow, cfg = load_generator(args.checkpoint, device)
    encoder, enc_transform = load_probe_encoder(args.encoder, device, cfg,
                                                args.encoder_ckpt)
    image_size = cfg["image_size"]

    pairs = list_split(args.test_dir, args.n_per_class, seed=args.seed)
    n_images = len(pairs)
    print(f"  [data] {n_images} real images, one generation each "
          f"({2 * n_images} points total)")

    run_dir = create_run_dir(args.encoder, args.tag, args,
                             extra={"checkpoint_sha256": sha256_file(args.checkpoint),
                                    "model_cfg": cfg, "n_images": n_images})

    # ---- encode, generate one sample each, re-encode ------------------------
    h_real, h_gen, classes, stems = [], [], [], []

    for start in range(0, n_images, args.batch_size):
        chunk = pairs[start:start + args.batch_size]
        images = [Image.open(p).convert("RGB") for _, p in chunk]
        x_real = torch.stack([enc_transform(im) for im in images]).to(device)

        # Noise drawn on CPU from a batch-independent seed, so the result does
        # not depend on --batch_size or on the device.
        generator = torch.Generator(device="cpu").manual_seed(args.seed * 100_003 + start)
        noise = torch.randn(len(chunk), 3, image_size, image_size,
                            generator=generator).to(device)

        with torch.no_grad():
            h = encoder(x_real)
            generated = flow.sample(model, noise, h, num_steps=args.num_steps,
                                    cfg_scale=args.cfg_scale)
            # The same PIL round trip, quantisation included, that E1 uses, so
            # these points are comparable with E1's numbers.
            x_gen = torch.stack([enc_transform(tensor_to_pil(g))
                                 for g in generated]).to(device)
            h_real.append(h.float().cpu())
            h_gen.append(encoder(x_gen).float().cpu())

        classes.extend(name for name, _ in chunk)
        stems.extend(p.stem for _, p in chunk)
        print(f"  generated {min(start + len(chunk), n_images)}/{n_images}")

    h_real = torch.cat(h_real).numpy()
    h_gen = torch.cat(h_gen).numpy()
    classes = np.asarray(classes)

    # ---- PCA fitted on the real manifold, generations projected in ----------
    pca = fit_pca(h_real, n_components=2)
    real_2d = pca.transform(h_real)
    gen_2d = pca.transform(h_gen)
    explained = pca.explained_variance_ratio_

    # ---- the two-sample test, in 2-D and in full representation space -------
    c2st_2d = c2st(real_2d, gen_2d, args.seed)
    c2st_full = c2st(h_real, h_gen, args.seed)

    # ---- secondary: is grade linearly readable in these two components? -----
    grade_real = grade_probe(real_2d, classes, args.seed)
    grade_gen = grade_probe(gen_2d, classes, args.seed)

    # ---- outputs ------------------------------------------------------------
    plot_real_vs_generated(real_2d, gen_2d, explained, c2st_2d, c2st_full,
                           args.encoder,
                           run_dir / "figures" / "pca_real_vs_generated.png")
    plot_by_grade(real_2d, gen_2d, classes, explained, args.encoder,
                  run_dir / "figures" / "pca_generations_by_grade.png")

    # Long format — one row per point, with a source column, which is the shape
    # a scatterplot wants and the shape the two-sample test was run on.
    rows = []
    for i in range(n_images):
        rows.append({"image": stems[i], "class": classes[i], "source": "real",
                     "pc1": float(real_2d[i, 0]), "pc2": float(real_2d[i, 1])})
        rows.append({"image": stems[i], "class": classes[i], "source": "generated",
                     "pc1": float(gen_2d[i, 0]), "pc2": float(gen_2d[i, 1])})
    save_csv(run_dir, "points", rows, ["image", "class", "source", "pc1", "pc2"])

    summary = {
        "n_images": n_images,
        "explained_variance_ratio": [float(v) for v in explained],
        "c2st_2d": c2st_2d,
        "c2st_full_dim": c2st_full,
        "grade_probe_real_2d": grade_real,
        "grade_probe_generated_2d": grade_gen,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  PC1+PC2 explain {explained.sum():.1%} of the real variance")
    print(f"  two-sample test (chance 0.500)")
    print(f"    2-D    acc {c2st_2d['accuracy']:.3f} | balanced "
          f"{c2st_2d['balanced_accuracy']:.3f} | folds "
          f"[{c2st_2d['fold_accuracy_min']:.3f}, {c2st_2d['fold_accuracy_max']:.3f}]")
    print(f"    full-d acc {c2st_full['accuracy']:.3f} | balanced "
          f"{c2st_full['balanced_accuracy']:.3f} | folds "
          f"[{c2st_full['fold_accuracy_min']:.3f}, {c2st_full['fold_accuracy_max']:.3f}]")
    print(f"  grade probe in 2-D (accuracy vs its own majority rate)")
    print(f"    real      {grade_real['cv_accuracy']:.3f} vs "
          f"{grade_real['majority_rate']:.3f}")
    print(f"    generated {grade_gen['cv_accuracy']:.3f} vs "
          f"{grade_gen['majority_rate']:.3f}")
    print(f"  [done] {run_dir}")


if __name__ == "__main__":
    main()
