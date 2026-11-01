"""
experiments/E5_attention_probing.py — RQ5: do the generative findings predict
downstream classifier behaviour?

One probe, two evaluations, and the whole experiment is the comparison between
them:

  A_real        the probe is fitted on real training images and evaluated on
                real test images.  The base case — what a downstream classifier
                on this frozen encoder is worth.

  B_generated   the *same fitted probe*, evaluated on images the generator drew
                from those test images' own h, re-encoded through the same frozen
                encoder.  Nothing is refitted; the only thing that changes is
                whether the classifier is looking at the scan or at the
                generator's reconstruction of it.

    real image ──encoder──> h ──generator──> generated image ──encoder──> h'
         │                                                                │
         └────────────> probe ──> class          probe <─────────────────┘
                          A_real                  B_generated

The headline number is **agreement**: on how many images does the probe return
the same class from the reconstruction as from the original?  That is a
different and stricter question than E1's.  E1 asks whether h' is nearest to h
among the generated set — a ranking, which a reconstruction can win while still
having lost every feature that decides a grade.  Here the arbiter is a
classifier trained on the actual task, so what it measures is whether the
*grade-relevant* content survived the h → image → h' round trip.

Two probes, because the read-out is itself a variable
-----------------------------------------------------
  linear_cls   logistic regression on h alone.  h is exactly what the generator
               is conditioned on, so this is the like-for-like comparison.
  attentive    a learned query attends over the encoder's whole final-layer
               token sequence and pools it, then a linear layer classifies.
               This is the protocol AIM and DINOv2 report frozen features under,
               and it reads information h never carried.

The gap between them on A_real says how much grade signal lives outside the
vector the generator sees.  The gap between them on B_generated says whether
the round trip preserves that extra signal or only what h pinned down.

Note the generator is deterministic in h on this dataset (see experiments/
README.md), so one generation per test image is not a subsample of a
distribution — it is the whole conditional.

Requirements
------------
Token caches for all three splits, which the pooled *_reps.pt files do not hold:

    for split in train val test; do
        python data/scripts/precompute_reps.py --encoder dinov3 --tokens \
            --data_dir data/raw/messidor2/$split \
            --out_file data/processed/messidor2/dinov3/${split}_tokens.pt
    done

The generation step needs the generator checkpoint and runs once, caching its
re-encoded output so later runs skip it.

Usage:
    python experiments/E5_attention_probing.py \
        --encoder    dinov3 \
        --checkpoint models/jit_dinov3/final.pt \
        --reps_dir   data/processed/messidor2/dinov3 \
        --device mps --n_seeds 3
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    REPO_ROOT,
    create_run_dir,
    load_generator,
    load_probe_encoder,
    resolve_device,
    save_metrics,
    set_plot_style,
    set_seed,
    sha256_file,
)
# One definition of the metrics, the statistics, the linear probe and the
# generation-to-PIL conversion, imported rather than restated: the E5 scripts
# must not be able to drift into reporting differently-computed numbers under
# the same research question, and a generation must become an image here in
# exactly the way it does in E1.
from E1_fidelity import tensor_to_pil, to_training_view
from E5_downstream_probe import (
    bootstrap_test_ci,
    build_probe,
    ci95,
    mcnemar,
    paired_bootstrap_delta,
    prediction_degeneracy,
    score,
)

EXPERIMENT = "E5_attention_probing"

# Every ViT in the registry uses 64-dim attention heads (ViT-S 384/6,
# ViT-L 1024/16), so deriving the head count from the width reproduces the
# encoder's own geometry instead of imposing an unrelated constant.
DEFAULT_HEAD_DIM = 64


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

class AttentionPoolingClassifier(nn.Module):
    """
    Attentive-pooling read-out over a frozen token sequence.

    A learned query (`cls_token`, not the encoder's) attends over the sequence
    to produce one pooled vector, which a linear layer classifies.  Unlike mean
    pooling the weights are input-dependent, so the probe can concentrate on the
    few patches carrying a lesion — which is the whole reason to prefer it over
    a linear probe on a pooled vector for a task where the evidence is local.

    Both the query and the key projection are initialised at std 0.02, so the
    attention logits start near zero and the softmax starts near uniform: the
    probe begins life as mean pooling and learns to sharpen away from it.

    The BatchNorm is the counterpart of the linear probe's StandardScaler: token
    channels have very different scales and the classifier layer's weight decay
    would otherwise regularise them unequally.  It is affine-free, so it
    standardises without adding fitted degrees of freedom.
    """

    def __init__(
        self,
        dim: int,
        out_features: int,
        num_heads: int = 12,
        num_queries: int = 1,
        use_batch_norm: bool = True,
        qkv_bias: bool = False,
        linear_bias: bool = True,
        average_pool: bool = True,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"num_heads {num_heads} does not divide dim {dim}")
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.average_pool = average_pool

        # No query projection: a linear map applied to a learned constant is
        # just another learned constant, so it would be redundant parameters.
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.linear = nn.Linear(dim, out_features, bias=linear_bias)
        self.bn = (
            nn.BatchNorm1d(dim, affine=False, eps=1e-6)
            if use_batch_norm
            else nn.Identity()
        )

        # apply() only visits sub-*modules*, so the bare Parameter above needs
        # its own initialisation.
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) frozen token sequence.
        Returns:
            logits: (B, out_features).
        """
        B, N, C = x.shape
        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # softmax over the token axis -> a convex combination of value vectors,
        # weighted per image.  This is the pooling a linear probe cannot do.
        x_cls = F.scaled_dot_product_attention(q, k, v)
        x_cls = x_cls.transpose(1, 2).reshape(B, self.num_queries, C)
        # Averages over the queries; with a single query this just drops the axis.
        x_cls = x_cls.mean(dim=1) if self.average_pool else x_cls

        return self.linear(x_cls)


# ---------------------------------------------------------------------------
# Token caches
# ---------------------------------------------------------------------------

def load_token_cache(path: Path, encoder: str) -> dict:
    """
    Load one token cache, checking it belongs to the encoder under study.

    Tokens stay float16 on CPU because the caches are large (RETFound's training
    split is 374 MB) and batches are cast to float32 on the device as drawn.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. This experiment needs token caches, which the "
            "pooled *_reps.pt files do not contain — see the module docstring "
            "for the precompute_reps.py --tokens loop that builds them."
        )
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if "tokens" not in cache:
        raise ValueError(
            f"{path} holds pooled representations, not token sequences. "
            "Rebuild it with precompute_reps.py --tokens."
        )
    if cache.get("encoder") and cache["encoder"] != encoder:
        raise ValueError(f"{path} was built with encoder '{cache['encoder']}', not '{encoder}'")
    if cache.get("labels") is None:
        cache["labels"] = [Path(p).parent.name for p in cache["paths"]]
    return cache


def load_token_split(reps_dir: Path, split: str, encoder: str) -> dict:
    """Load `<reps_dir>/<split>_tokens.pt`."""
    return load_token_cache(Path(reps_dir) / f"{split}_tokens.pt", encoder)


def resolve_image_path(path) -> Path:
    """
    Locate a source image recorded in a cache.

    precompute_reps.py stores whatever ``--data_dir`` it was given, and the
    documented commands are run from the repo root, so the stored paths are
    relative to REPO_ROOT — while the experiment scripts are usually run from
    experiments/.  Try the path as recorded first so an absolute or
    already-correct relative path is left alone.
    """
    path = Path(path)
    if path.exists():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"{path} not found, and neither is {candidate}. The token cache records "
        "source image paths relative to the directory precompute_reps.py ran in."
    )


def check_cls_alignment(cache: dict, reps_dir: Path, split: str, tol: float = 2e-2) -> dict:
    """
    Verify `tokens[:, 0]` reproduces the pooled cache every other experiment uses.

    This is the one check that ties this script to the rest of the study.  If the
    token cache were built from a different split ordering, a different transform
    or a different checkpoint, every number here would still look plausible while
    describing different images from E1-E4 and E6.  The tolerance is loose
    because the tokens are stored in float16 against the pooled cache's float32,
    which is a relative error of ~3e-4 — anything near that is rounding, and
    anything far above it is a mismatch.

    Returns a dict recorded in run_config.json; raises if the caches disagree.
    """
    pooled_path = Path(reps_dir) / f"{split}_reps.pt"
    if not pooled_path.exists():
        print(f"  [check] {split}: no pooled cache to cross-check against")
        return {"checked": False, "reason": f"{pooled_path.name} absent"}

    pooled = torch.load(pooled_path, map_location="cpu", weights_only=False)
    if list(pooled["paths"]) != list(cache["paths"]):
        raise ValueError(
            f"{pooled_path} and the token cache describe different images or a "
            "different ordering — every downstream comparison would be mispaired."
        )
    cls = cache["tokens"][:, 0].float()
    deviation = (cls - pooled["reps"].float()).abs().max().item()
    scale = pooled["reps"].float().abs().max().item()
    if deviation > tol * max(scale, 1.0):
        raise ValueError(
            f"tokens[:, 0] disagrees with {pooled_path.name} by {deviation:.4g} "
            f"(scale {scale:.4g}). The two caches were not built from the same "
            "encoder state; rebuild both before reading any number from this run."
        )
    print(f"  [check] {split}: tokens[:, 0] matches {pooled_path.name} "
          f"(max |diff| {deviation:.4g})")
    return {"checked": True, "max_abs_diff": float(deviation), "scale": float(scale)}


# ---------------------------------------------------------------------------
# The round trip:  h -> generated image -> h'
# ---------------------------------------------------------------------------

def generate_and_reencode(h: torch.Tensor, paths, args, device, run_dir: Path,
                          n_examples: int = 8) -> dict:
    """
    Draw one image per conditioning vector and re-encode it through the same
    frozen encoder.

    The generation is converted to a uint8 PIL image and read back before
    encoding, exactly as E1 does: that quantisation is what a deployed pipeline
    would apply, and skipping it would measure a round trip nobody performs.

    cfg_scale defaults to 1.0 for the reason the repo applies to every probe —
    guidance extrapolates the conditioning signal and would let the sampler
    decide how faithful the reconstruction looks.

    The first ``n_examples`` originals and reconstructions are kept as a figure:
    the agreement rate says how often the probe was unmoved, and the pictures say
    what the probe was looking at when it was.

    Returns:
        {"tokens": (N, T, D) float16, "provenance": {...}}
    """
    from PIL import Image

    model, flow, cfg = load_generator(args.checkpoint, device)
    encoder, enc_transform = load_probe_encoder(
        args.encoder, device, cfg, args.encoder_ckpt)
    if cfg["h_dim"] != h.shape[1]:
        raise ValueError(f"generator expects h_dim {cfg['h_dim']}, "
                         f"representations are {h.shape[1]}")

    set_seed(args.seed)
    all_tokens, examples = [], []
    for start in range(0, len(h), args.sample_batch_size):
        h_batch = h[start: start + args.sample_batch_size].to(device)
        with torch.no_grad():
            noise = torch.randn(len(h_batch), 3, cfg["image_size"], cfg["image_size"],
                                device=device)
            generated = flow.sample(model, noise, h_batch,
                                    num_steps=args.num_steps, cfg_scale=args.cfg_scale)
            pils = [tensor_to_pil(g) for g in generated]
            x_gen = torch.stack([enc_transform(im) for im in pils]).to(device)
            all_tokens.append(encoder.forward_tokens(x_gen).to(torch.float16).cpu())

        for offset, generated_pil in enumerate(pils):
            index = start + offset
            if index < n_examples:
                real = Image.open(resolve_image_path(paths[index])).convert("RGB")
                examples.append((to_training_view(real, cfg["image_size"]),
                                 generated_pil, Path(paths[index]).parent.name))
        print(f"  generated {min(start + args.sample_batch_size, len(h))}/{len(h)}")

    save_reconstruction_figure(examples, run_dir)

    return {
        "tokens": torch.cat(all_tokens),
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "encoder": args.encoder,
            "num_steps": args.num_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "model_cfg": cfg,
        },
    }


def save_reconstruction_figure(examples, run_dir: Path) -> None:
    """A `[original | reconstruction]` strip, so the agreement rate has pictures."""
    if not examples:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(examples), figsize=(1.5 * len(examples), 3.4))
    axes = np.atleast_2d(axes)
    for column, (real, generated, class_name) in enumerate(examples):
        for row, (image, label) in enumerate(((real, "original"), (generated, "generated"))):
            ax = axes[row, column]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if column == 0:
                ax.set_ylabel(label, fontsize=8)
        axes[0, column].set_title(class_name, fontsize=6)
    fig.suptitle("E5 — what the probe sees on each side of the round trip", fontsize=9)
    fig.savefig(run_dir / "figures" / "reconstructions.png")
    plt.close(fig)


def reconstruction_tokens(h: torch.Tensor, paths, args, device,
                          run_dir: Path) -> torch.Tensor:
    """
    Re-encoded generations, from cache when the cache was built the same way.

    Generation is the expensive step and it is deterministic given
    (checkpoint, num_steps, cfg_scale, seed), so the result is cached.  Every one
    of those is recorded and re-checked on load: a cache built at a different
    guidance scale describes a different experiment and must not be reused
    silently.
    """
    import hashlib

    cache_path = Path(args.generated_cache) if args.generated_cache else (
        Path(args.reps_dir) / "generated" / "test_tokens_from_h.pt")
    # The image list is hashed into the key, not just its length: two different
    # --n_test_images subsamples of the same size would otherwise collide and one
    # run would silently score another's reconstructions.
    wanted = {"checkpoint_sha256": sha256_file(args.checkpoint),
              "encoder": args.encoder, "num_steps": args.num_steps,
              "cfg_scale": args.cfg_scale, "seed": args.seed,
              "images_sha256": hashlib.sha256(
                  "\n".join(str(p) for p in paths).encode()).hexdigest()}

    if cache_path.exists() and not args.regenerate:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        recorded = {k: cache["provenance"].get(k) for k in wanted}
        if recorded == wanted and len(cache["tokens"]) == len(h):
            print(f"  [generate] reusing {cache_path} ({len(cache['tokens'])} images)")
            return cache["tokens"]
        print(f"  [generate] {cache_path} was built differently — regenerating\n"
              f"      cached: {recorded}\n      wanted: {wanted}")

    print(f"  [generate] {len(h)} images, {args.num_steps} steps, "
          f"cfg {args.cfg_scale} — this is the slow step")
    result = generate_and_reencode(h, paths, args, device, run_dir)
    result["provenance"].update(images_sha256=wanted["images_sha256"],
                                paths=[str(p) for p in paths])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)
    print(f"  [generate] cached -> {cache_path} "
          f"({cache_path.stat().st_size / 1e6:.1f} MB)")
    return result["tokens"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def class_weights(labels: np.ndarray, classes: list) -> torch.Tensor:
    """
    Inverse-frequency weights, matching sklearn's ``class_weight="balanced"``.

    Weighting rather than resampling, for the reason the linear probe gives: the
    tail grades have as few as 19 training images and resampling would duplicate
    them heavily.  Using the same rule as the linear probe is what keeps the two
    balanced accuracies comparable.
    """
    counts = np.array([(labels == c).sum() for c in classes], dtype=float)
    return torch.tensor(len(labels) / (len(classes) * counts), dtype=torch.float32)


def _batches(n: int, batch_size: int, generator: torch.Generator, shuffle: bool):
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start: start + batch_size]


@torch.no_grad()
def predict(model, tokens, device, batch_size: int) -> np.ndarray:
    """Class probabilities for a whole split, in the model's class order."""
    model.eval()
    probabilities = []
    generator = torch.Generator().manual_seed(0)
    for index in _batches(len(tokens), batch_size, generator, shuffle=False):
        logits = model(tokens[index].to(device).float())
        probabilities.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(probabilities).numpy()


def train_attentive_probe(train_tokens, y_train_idx, val_tokens, y_val_idx, *,
                          classes, weights, device, seed, args):
    """
    Fit one attentive probe, selecting the epoch on validation balanced accuracy.

    Returns:
        (model, history, best) — model restored to its best-validation state.
    """
    from sklearn.metrics import balanced_accuracy_score

    set_seed(seed)
    dim = train_tokens.shape[-1]
    num_heads = args.num_heads or max(1, dim // DEFAULT_HEAD_DIM)
    model = AttentionPoolingClassifier(
        dim=dim,
        out_features=len(classes),
        num_heads=num_heads,
        num_queries=args.num_queries,
        use_batch_norm=not args.no_batch_norm,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    y_train_t = torch.as_tensor(y_train_idx, dtype=torch.long)
    y_val_np = np.asarray(y_val_idx)
    generator = torch.Generator().manual_seed(seed)

    steps_per_epoch = math.ceil(len(train_tokens) / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = steps_per_epoch * args.warmup_epochs
    step, lr = 0, args.lr

    best = {"epoch": -1, "val_balanced_accuracy": -1.0}
    best_state, history, since_improved = None, [], 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for index in _batches(len(train_tokens), args.batch_size, generator, shuffle=True):
            # Linear warmup then cosine decay — the schedule attentive probing is
            # normally reported under.  The warmup matters here because the query
            # is randomly initialised against a frozen, unnormalised input.
            if step < warmup_steps:
                lr = args.lr * (step + 1) / max(1, warmup_steps)
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            for group in optimizer.param_groups:
                group["lr"] = lr

            logits = model(train_tokens[index].to(device).float())
            loss = criterion(logits, y_train_t[index].to(device))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * len(index)
            n_seen += len(index)
            step += 1

        val_probabilities = predict(model, val_tokens, device, args.batch_size)
        val_accuracy = balanced_accuracy_score(y_val_np, val_probabilities.argmax(1))
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(1, n_seen),
                        "val_balanced_accuracy": float(val_accuracy), "lr": lr})

        if val_accuracy > best["val_balanced_accuracy"]:
            best = {"epoch": epoch, "val_balanced_accuracy": float(val_accuracy)}
            best_state = copy.deepcopy(model.state_dict())
            since_improved = 0
        else:
            since_improved += 1
            if args.patience and since_improved >= args.patience:
                print(f"      early stop at epoch {epoch} "
                      f"({args.patience} epochs without improvement)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best


# ---------------------------------------------------------------------------
# The agreement measurement
# ---------------------------------------------------------------------------

def prediction_agreement(pred_real, pred_generated, y_true, classes,
                         n_boot: int = 10_000, seed: int = 0) -> dict:
    """
    How often the probe returns the same class from the reconstruction as from
    the original — the quantity this experiment exists to report.

    Deliberately measured against the probe's own call on the real image, not
    against the ground-truth label.  The question is whether the round trip
    preserved what the classifier reads, which is separable from whether the
    classifier was right: an image the probe misgrades identically before and
    after is a *success* for the reconstruction and a failure for the probe, and
    conflating the two would hide both.

    ``agreement_when_correct`` restricts to images the probe grades correctly
    from the original, which is the subset a deployment would care about.

    A raw agreement rate is not readable on its own: two prediction vectors that
    are statistically independent still agree sometimes, and how often depends
    entirely on how concentrated the predictions are.  So ``chance_agreement``
    (the rate expected if the two were independent, given their own marginals)
    and Cohen's kappa (the same correction as a coefficient, 0 = chance,
    1 = perfect) are reported alongside.  Read kappa, not the raw rate.

    Returns overall and per-class rates with a bootstrap CI over images.
    """
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(seed)
    pred_real = np.asarray(pred_real)
    pred_generated = np.asarray(pred_generated)
    y_true = np.asarray(y_true)
    agree = pred_real == pred_generated

    draws = [agree[rng.integers(0, len(agree), len(agree))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])

    # Expected agreement if the probe's call on the generation were independent
    # of its call on the original, holding both marginal distributions fixed.
    chance = float(sum(
        (pred_real == name).mean() * (pred_generated == name).mean()
        for name in classes
    ))

    correct = pred_real == y_true
    result = {
        "agreement": float(agree.mean()),
        "ci95": [float(lo), float(hi)],
        "chance_agreement": chance,
        "cohen_kappa": float(cohen_kappa_score(pred_real, pred_generated,
                                               labels=list(classes))),
        "n_images": int(len(agree)),
        "n_disagree": int((~agree).sum()),
        "agreement_when_correct": (float(agree[correct].mean())
                                   if correct.any() else float("nan")),
        "n_correct_on_real": int(correct.sum()),
        "ci_source": "bootstrap over images",
    }
    result["per_class"] = {
        name: {"agreement": float(agree[y_true == name].mean()),
               "n": int((y_true == name).sum())}
        for name in classes if (y_true == name).any()
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E5 — attentive and linear probes, real vs generated")
    parser.add_argument("--encoder", default="dinov3")
    parser.add_argument("--checkpoint", default="models/jit_dinov3/final.pt",
                        help="Generator checkpoint used for the round trip")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--reps_dir", default="data/processed/messidor2/dinov3",
                        help="Directory holding {train,val,test}_tokens.pt")

    gen_group = parser.add_argument_group("generation")
    gen_group.add_argument("--num_steps", type=int, default=50,
                           help="Heun ODE steps, as E1 uses")
    gen_group.add_argument("--cfg_scale", type=float, default=1.0,
                           help="1.0 on every probe in this repo: guidance "
                                "extrapolates the conditioning signal and would "
                                "let the sampler decide the fidelity being measured")
    gen_group.add_argument("--sample_batch_size", type=int, default=16)
    gen_group.add_argument("--generated_cache", default=None,
                           help="Default: <reps_dir>/generated/test_tokens_from_h.pt")
    gen_group.add_argument("--regenerate", action="store_true",
                           help="Ignore an existing generation cache")
    gen_group.add_argument("--skip_generated", action="store_true",
                           help="Fit and report A_real only — no generator needed")
    gen_group.add_argument("--n_test_images", type=int, default=None,
                           help="Evaluate on a class-stratified subsample of the "
                                "test split instead of all of it.  Subsets both "
                                "conditions together so the round trip stays "
                                "paired; generation is the slow step, so this is "
                                "what makes a trial run cheap.")

    probe_group = parser.add_argument_group("probe")
    probe_group.add_argument("--num_heads", type=int, default=None,
                             help="Default: h_dim // 64, the encoder's own head width")
    probe_group.add_argument("--num_queries", type=int, default=1)
    probe_group.add_argument("--no_batch_norm", action="store_true",
                             help="Drop the affine-free BatchNorm that stands in "
                                  "for the linear probe's StandardScaler")

    fit_group = parser.add_argument_group("fitting")
    fit_group.add_argument("--epochs", type=int, default=100)
    fit_group.add_argument("--patience", type=int, default=25,
                           help="Epochs without a validation improvement before "
                                "stopping; 0 disables early stopping")
    fit_group.add_argument("--batch_size", type=int, default=64)
    fit_group.add_argument("--lr", type=float, default=3e-5,
                           help="Well below the 1e-3 attentive probing is usually "
                                "reported at, because that rate is tuned for "
                                "ImageNet-scale probe sets.  On 972 images 1e-3 "
                                "drives train loss from 1.60 to 0.52 while "
                                "validation balanced accuracy *falls* 0.38 -> 0.27, "
                                "so the selected epoch is 1 and the probe is never "
                                "actually fitted.  At 3e-5 the validation optimum "
                                "lands mid-schedule.  Check training_curves.png "
                                "before trusting any run.")
    fit_group.add_argument("--weight_decay", type=float, default=0.05)
    fit_group.add_argument("--warmup_epochs", type=int, default=5)
    fit_group.add_argument("--n_seeds", type=int, default=3,
                           help="Independent refits. Unlike the linear probe this "
                                "one is genuinely stochastic, so the spread over "
                                "seeds is a real error bar — but each seed is a "
                                "full training run.")

    parser.add_argument("--n_boot", type=int, default=10_000,
                        help="Bootstrap resamples of the test set for the CI")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="CPU is workable for dinov3 but slow for RETFound's "
                             "1024-wide tokens and for sampling; pass mps or cuda")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_plot_style()

    # ---- data ---------------------------------------------------------------
    reps_dir = Path(args.reps_dir)
    caches = {split: load_token_split(reps_dir, split, args.encoder)
              for split in ("train", "val", "test")}
    alignment = {split: check_cls_alignment(caches[split], reps_dir, split)
                 for split in caches}

    train_tokens = caches["train"]["tokens"]
    val_tokens = caches["val"]["tokens"]
    y_train = np.asarray(caches["train"]["labels"])
    y_val = np.asarray(caches["val"]["labels"])

    test_index = np.arange(len(caches["test"]["labels"]))
    if args.n_test_images and args.n_test_images < len(test_index):
        # Stratified, because Messidor-2 is 58% no-DR and a head-of-list slice of
        # a sorted path list would be one class.  Both conditions are built from
        # this index, so the round-trip pairing is preserved by construction.
        labels_all = np.asarray(caches["test"]["labels"])
        rng = np.random.default_rng(args.seed)
        per_class = max(1, args.n_test_images // len(set(labels_all)))
        test_index = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(labels_all == name),
                       size=min(per_class, int((labels_all == name).sum())),
                       replace=False)
            for name in sorted(set(labels_all))
        ]))
        print(f"  [data] test subsampled to {len(test_index)} images "
              f"(~{per_class}/class, --n_test_images)")

    test_tokens = caches["test"]["tokens"][test_index]
    test_paths = [caches["test"]["paths"][i] for i in test_index]
    y_test = np.asarray(caches["test"]["labels"])[test_index]

    classes = sorted(set(y_train))
    class_index = {name: i for i, name in enumerate(classes)}
    y_train_idx = np.array([class_index[c] for c in y_train])
    y_val_idx = np.array([class_index[c] for c in y_val])
    weights = class_weights(y_train, classes)
    layout = caches["train"].get("token_layout", {})
    h_dim = train_tokens.shape[2]

    print(f"  [data] train {tuple(train_tokens.shape)}, val {tuple(val_tokens.shape)}, "
          f"test {tuple(test_tokens.shape)}")
    print(f"  [data] {train_tokens.shape[1]} tokens x {h_dim} channels "
          f"({layout.get('n_register_tokens', '?')} register, "
          f"{layout.get('n_patch_tokens', '?')} patch), {len(classes)} classes")
    for name in classes:
        print(f"    {name:20s} train {int((y_train == name).sum()):4d}  "
              f"val {int((y_val == name).sum()):4d}  test {int((y_test == name).sum()):4d}")

    run_dir = create_run_dir(
        EXPERIMENT, args.encoder, args.tag, args,
        extra={"classes": classes, "n_train": int(len(y_train)),
               "n_val": int(len(y_val)), "n_test": int(len(y_test)),
               "token_layout": layout, "h_dim": int(h_dim),
               "num_heads": args.num_heads or max(1, h_dim // DEFAULT_HEAD_DIM),
               "cls_alignment_check": alignment,
               "device": str(device)},
    )

    # ---- the round trip -----------------------------------------------------
    conditions = {"A_real": (test_tokens, y_test)}
    if not args.skip_generated:
        # Condition on the float32 pooled cache rather than the float16 tokens:
        # check_cls_alignment has just proven they are the same vector, and the
        # generator should see h at the precision it was trained with.
        pooled_test = reps_dir / "test_reps.pt"
        if pooled_test.exists():
            h_test = torch.load(pooled_test, map_location="cpu",
                                weights_only=False)["reps"].float()[test_index]
        else:
            h_test = test_tokens[:, 0].float()
        conditions["B_generated"] = (
            reconstruction_tokens(h_test, test_paths, args, device, run_dir), y_test)
    else:
        print("  [generate] --skip_generated — A_real only")

    # ---- fit ----------------------------------------------------------------
    # One fit, on real training images.  Both conditions are evaluations of that
    # same probe; nothing is refitted on generations.
    rows, curve_rows = [], []
    per_seed: dict = {name: {} for name in conditions}

    for seed in range(args.n_seeds):
        print(f"  [fit] attentive seed {seed}")
        model, history, best = train_attentive_probe(
            train_tokens, y_train_idx, val_tokens, y_val_idx,
            classes=classes, weights=weights, device=device, seed=seed, args=args)
        print(f"      best epoch {best['epoch']}  "
              f"val bal_acc {best['val_balanced_accuracy']:.4f}")
        for entry in history:
            curve_rows.append({"seed": seed, **entry})

        for name, (tokens, labels) in conditions.items():
            y_proba = predict(model, tokens, device, args.batch_size)
            y_pred = np.array([classes[i] for i in y_proba.argmax(1)])
            metrics = score(labels, y_pred, y_proba, classes)
            rows.append({"condition": name, "probe": "attentive", "seed": seed,
                         "val_balanced_accuracy": best["val_balanced_accuracy"],
                         "best_epoch": best["epoch"], **metrics})
            per_seed[name][seed] = (labels, y_pred, y_proba,
                                    best["val_balanced_accuracy"])
            print(f"      {name:14s} bal_acc {metrics['balanced_accuracy']:.4f}  "
                  f"auc {metrics['macro_auc']:.4f}")

    # ---- the linear-on-h probe ----------------------------------------------
    # Fitted on tokens[:, 0], which check_cls_alignment has proven identical to
    # the pooled cache E5_downstream_probe reads.  Deriving both probes' inputs
    # from one tensor means they cannot be looking at different images.
    linear_predictions = {}
    probe = build_probe(seed=args.seed)
    probe.fit(train_tokens[:, 0].float().numpy(), y_train)
    for name, (tokens, labels) in conditions.items():
        x_test = tokens[:, 0].float().numpy()
        y_pred = probe.predict(x_test)
        y_proba = probe.predict_proba(x_test)
        metrics = score(labels, y_pred, y_proba, list(probe.classes_))
        rows.append({"condition": name, "probe": "linear_cls", "seed": args.seed,
                     "val_balanced_accuracy": float("nan"), "best_epoch": -1,
                     **metrics})
        linear_predictions[name] = (labels, y_pred, y_proba, list(probe.classes_))
        print(f"  [linear] {name:14s} bal_acc {metrics['balanced_accuracy']:.4f}  "
              f"auc {metrics['macro_auc']:.4f}")

    save_metrics(run_dir, "metrics", rows,
                 ["condition", "probe", "seed", "balanced_accuracy", "macro_auc",
                  "val_balanced_accuracy", "best_epoch"])
    save_metrics(run_dir, "training_curves", curve_rows,
                 ["seed", "epoch", "train_loss", "val_balanced_accuracy", "lr"])

    # Per-image calls on both sides of the round trip.  Agreement is a
    # descriptive statistic over these, and writing them out means it can be
    # re-cut — by class, by confidence, by image — without refitting anything.
    selected = max(per_seed["A_real"], key=lambda s: per_seed["A_real"][s][3])
    prediction_rows = [
        {"path": path, "true": true_label,
         "attentive_real": per_seed["A_real"][selected][1][i],
         "attentive_generated": (per_seed["B_generated"][selected][1][i]
                                 if "B_generated" in conditions else ""),
         "linear_real": linear_predictions["A_real"][1][i],
         "linear_generated": (linear_predictions["B_generated"][1][i]
                              if "B_generated" in conditions else "")}
        for i, (path, true_label) in enumerate(zip(test_paths, y_test))
    ]
    save_metrics(run_dir, "predictions", prediction_rows,
                 ["path", "true", "attentive_real", "attentive_generated",
                  "linear_real", "linear_generated"])

    # ---- aggregate ----------------------------------------------------------
    summary = {}
    for name in conditions:
        seeds = per_seed[name]
        accuracies = np.array([r["balanced_accuracy"] for r in rows
                               if r["condition"] == name and r["probe"] == "attentive"])
        seed_mean, seed_lo, seed_hi = ci95(accuracies)

        # The reported predictions come from the seed with the best *validation*
        # balanced accuracy — model selection as it would actually be done, and
        # never on the test set.  The seed spread below records what that choice
        # was worth, so a reader can check it did not drive the result.
        chosen = max(seeds, key=lambda s: seeds[s][3])
        labels, y_pred, y_proba, val_accuracy = seeds[chosen]

        summary[name] = {"attentive": bootstrap_test_ci(
            labels, y_pred, y_proba, classes, args.n_boot, seed=0)}
        summary[name]["attentive"]["seed_spread"] = {
            "balanced_accuracy_mean": seed_mean, "ci95": [seed_lo, seed_hi],
            "n_seeds": int(len(accuracies)),
            "note": "a real interval here — the attentive probe is stochastic",
        }
        summary[name]["attentive"]["selected_seed"] = {
            "seed": int(chosen), "val_balanced_accuracy": float(val_accuracy),
            "criterion": "highest validation balanced accuracy",
        }
        summary[name]["attentive"]["prediction_degeneracy"] = prediction_degeneracy(y_pred)

        lin_labels, lin_pred, lin_proba, lin_classes = linear_predictions[name]
        summary[name]["linear_cls"] = bootstrap_test_ci(
            lin_labels, lin_pred, lin_proba, lin_classes, args.n_boot, seed=0)
        summary[name]["linear_cls"]["prediction_degeneracy"] = prediction_degeneracy(lin_pred)
        # How much the token sequence carries that h does not.
        summary[name]["attentive_minus_linear"] = paired_bootstrap_delta(
            labels, lin_pred, y_pred, classes, args.n_boot, seed=0)

    # ---- the round-trip comparison ------------------------------------------
    if "B_generated" in conditions:
        chosen = summary["A_real"]["attentive"]["selected_seed"]["seed"]
        _, attn_real, _, _ = per_seed["A_real"][chosen]
        # Same seed on both sides: comparing predictions from two different
        # probes would confound the round trip with the fit.
        _, attn_generated, _, _ = per_seed["B_generated"][chosen]
        _, lin_real, _, _ = linear_predictions["A_real"]
        _, lin_generated, _, _ = linear_predictions["B_generated"]

        summary["round_trip"] = {
            "attentive": {
                **prediction_agreement(attn_real, attn_generated, y_test, classes,
                                       args.n_boot, seed=0),
                "delta_balanced_accuracy": paired_bootstrap_delta(
                    y_test, attn_real, attn_generated, classes, args.n_boot, seed=0),
                "mcnemar": mcnemar(attn_real == y_test, attn_generated == y_test),
                "seed": int(chosen),
            },
            "linear_cls": {
                **prediction_agreement(lin_real, lin_generated, y_test, classes,
                                       args.n_boot, seed=0),
                "delta_balanced_accuracy": paired_bootstrap_delta(
                    y_test, lin_real, lin_generated, classes, args.n_boot, seed=0),
                "mcnemar": mcnemar(lin_real == y_test, lin_generated == y_test),
            },
        }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figures ------------------------------------------------------------
    import matplotlib.pyplot as plt

    names = list(conditions)
    positions = np.arange(len(names))
    width = 0.38

    def _series(key):
        means = np.array([summary[n][key]["balanced_accuracy"]["mean"] for n in names])
        lows = np.array([summary[n][key]["balanced_accuracy"]["ci95"][0] for n in names])
        highs = np.array([summary[n][key]["balanced_accuracy"]["ci95"][1] for n in names])
        return means, np.vstack([means - lows, highs - means])

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for offset, key, colour, label in (
        (-width / 2, "attentive", "#4C72B0", "attentive (all tokens)"),
        (+width / 2, "linear_cls", "#DD8452", "linear (h only)"),
    ):
        means, errors = _series(key)
        ax.bar(positions + offset, means, width, yerr=errors, capsize=3,
               color=colour, label=label)
    ax.axhline(1.0 / len(classes), ls="--", lw=0.8, color="grey",
               label=f"chance ({1 / len(classes):.2f})")
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylabel("balanced accuracy")
    title = f"E5 — probes on frozen {args.encoder} features"
    if "round_trip" in summary:
        # Kappa, not the raw agreement rate: the two probes have very different
        # prediction concentrations, so their chance-agreement baselines differ
        # and the raw rates are not comparable to each other.
        title += (f"\nround-trip agreement (Cohen's kappa): attentive "
                  f"{summary['round_trip']['attentive']['cohen_kappa']:+.3f}, "
                  f"linear {summary['round_trip']['linear_cls']['cohen_kappa']:+.3f}")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.savefig(run_dir / "figures" / "real_vs_generated.png")
    plt.close(fig)

    # The training curves belong in the output: with 972 images the probe
    # overfits, and the val curve is the evidence that epoch selection caught it.
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for seed in range(args.n_seeds):
        curve = [r for r in curve_rows if r["seed"] == seed]
        if curve:
            ax.plot([r["epoch"] for r in curve],
                    [r["val_balanced_accuracy"] for r in curve],
                    lw=1.0, alpha=0.85, label=f"seed {seed}")
    ax.axhline(1.0 / len(classes), ls="--", lw=0.8, color="grey")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation balanced accuracy")
    ax.set_title(f"E5 — attentive probe fitting ({args.encoder})")
    ax.legend(fontsize=7)
    fig.savefig(run_dir / "figures" / "training_curves.png")
    plt.close(fig)

    headline = {
        name: {"attentive": round(summary[name]["attentive"]["balanced_accuracy"]["mean"], 4),
               "linear_cls": round(summary[name]["linear_cls"]["balanced_accuracy"]["mean"], 4)}
        for name in names
    }
    if "round_trip" in summary:
        headline["agreement_real_vs_generated"] = {
            probe: {"raw": round(summary["round_trip"][probe]["agreement"], 4),
                    "chance": round(summary["round_trip"][probe]["chance_agreement"], 4),
                    "kappa": round(summary["round_trip"][probe]["cohen_kappa"], 4)}
            for probe in ("attentive", "linear_cls")
        }
    print("\n  " + json.dumps(headline, indent=2))
    print(f"Done — {run_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
