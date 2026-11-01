"""
data/scripts/precompute_reps.py

Run a frozen encoder over an image split once and cache the representations.

The encoder is a fixed feature extractor in this project, so running it inside
the training loop (as upstream RCDM did) recomputes identical vectors every
epoch.  Caching once also guarantees that training, sampling and every probe in
experiments/ see byte-identical h for the same image.

Output (.pt):
    {
        "paths"   : list[str]    N image paths, sorted
        "labels"  : list[str]    parent directory name of each path (the class)
        "reps"    : Tensor(N, D) float32
        "encoder" : str          registry name, e.g. "dinov3"
        "h_dim"   : int          D
    }

reps[i] belongs to paths[i] — the whole pipeline depends on that alignment.

With ``--tokens`` the cache holds the encoder's full final-layer sequence
instead, for E5's attentive probe:

    {
        "paths", "labels", "encoder", "h_dim", "variant", "seed"  as above
        "tokens"       : Tensor(N, T, D) float16
        "token_layout" : dict     which index is CLS / register / patch
    }

Written to a separate ``*_tokens.pt`` file rather than added to the pooled
cache: these are ~100x larger (RETFound's train split is 374 MB against 4 MB),
and every other consumer in the pipeline only ever wants the pooled vector.
``tokens[:, 0]`` is the pooled representation, so the two caches can always be
cross-checked against each other — E5 does exactly that on load.

Usage
-----
    # DinoV3 (384-dim) over the training split
    python data/scripts/precompute_reps.py \
        --encoder  dinov3 \
        --data_dir data/raw/messidor2/train \
        --out_file data/processed/messidor2/dinov3/train_reps.pt

    # RETFound (1024-dim), all three splits
    for split in train val test; do
        python data/scripts/precompute_reps.py \
            --encoder  retfound_cfp \
            --data_dir data/raw/messidor2/$split \
            --out_file data/processed/messidor2/retfound_cfp/${split}_reps.pt
    done

    # Token sequences for E5_attention_probing.py, all three splits
    for split in train val test; do
        python data/scripts/precompute_reps.py \
            --encoder  dinov3 --tokens \
            --data_dir data/raw/messidor2/$split \
            --out_file data/processed/messidor2/dinov3/${split}_tokens.pt
    done

Both E5 probes need val and test as well as train, so run all three splits for
any encoder you intend to probe.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rcdm.encoders import (  # noqa: E402
    ENCODER_NAMES,
    VARIANT_ORDER,
    build_transform,
    describe,
    encode_batch,
    encode_tokens_batch,
    get_encoder,
    make_variant,
)

VALID_EXTENSIONS = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def collect_image_paths(data_dir: Path) -> list:
    """
    Recursively collect image paths under data_dir, sorted for determinism.

    Handles the ImageFolder layout used by both datasets:
        <split>/<class>/<image>.png
    """
    paths = sorted(
        str(p) for p in Path(data_dir).rglob("*")
        if p.suffix.lower() in VALID_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No images found under {data_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache frozen-encoder representations for a dataset split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Encoders:\n  " + "\n  ".join(f"{n:14s} {describe(n)}" for n in ENCODER_NAMES),
    )
    parser.add_argument("--encoder", required=True, choices=ENCODER_NAMES)
    parser.add_argument("--data_dir", required=True,
                        help="Split root, e.g. data/raw/messidor2/train")
    parser.add_argument("--out_file", required=True,
                        help="Destination .pt, e.g. data/processed/messidor2/dinov3/train_reps.pt")
    parser.add_argument("--encoder_ckpt", default=None,
                        help="Override the encoder weights path (defaults per encoder)")
    parser.add_argument("--variant", default=None, choices=VARIANT_ORDER[1:],
                        help="Apply one of E3's nuisance transforms at the source "
                             "resolution before encoding.  This builds the cache "
                             "E5's condition C consumes, without writing a "
                             "transformed copy of the split to disk.")
    parser.add_argument("--tokens", action="store_true",
                        help="Cache the encoder's full final-layer token "
                             "sequence (N, T, D) float16 instead of the pooled "
                             "(N, D) vector.  This is what E5's attentive probe "
                             "reads; nothing else in the pipeline uses it.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds the only stochastic variant ('translate')")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    paths = collect_image_paths(Path(args.data_dir))
    labels = [Path(p).parent.name for p in paths]
    print(f"Found {len(paths)} images in {args.data_dir}")
    print(f"  classes: {sorted(set(labels))}")

    pre_transform = None
    if args.variant:
        rng = torch.Generator().manual_seed(args.seed)
        pre_transform = lambda img: make_variant(img, args.variant, rng=rng)  # noqa: E731
        print(f"  variant: {args.variant} (seed {args.seed})")

    print(f"Loading encoder '{args.encoder}' — {describe(args.encoder)}")
    kwargs = {"checkpoint_path": args.encoder_ckpt} if args.encoder_ckpt else {}
    encoder = get_encoder(args.encoder, device=args.device, **kwargs)

    transform = build_transform()  # always 224 px, ImageNet normalisation

    cache = {
        "paths": paths,
        "labels": labels,
        "encoder": args.encoder,
        "h_dim": encoder.h_dim,
        # None for a clean cache; the variant name identifies a condition-C
        # cache so it can never be mistaken for the untransformed split.
        "variant": args.variant,
        "seed": args.seed,
    }

    if args.tokens:
        layout = encoder.token_layout()
        print(f"  tokens: {layout['n_tokens']} per image "
              f"(1 CLS + {layout['n_register_tokens']} register + "
              f"{layout['n_patch_tokens']} patch), {encoder.h_dim} channels")
        out = encode_tokens_batch(paths, encoder, transform,
                                  device=args.device, batch_size=args.batch_size,
                                  pre_transform=pre_transform)
        if out.shape[1] != layout["n_tokens"]:
            raise RuntimeError(f"Expected {layout['n_tokens']} tokens, got {out.shape[1]}")
        cache["tokens"] = out
        cache["token_layout"] = layout
    else:
        out = encode_batch(paths, encoder, transform,
                           device=args.device, batch_size=args.batch_size,
                           pre_transform=pre_transform)
        cache["reps"] = out

    if out.shape[0] != len(paths):
        raise RuntimeError(f"Alignment broken: {out.shape[0]} rows for {len(paths)} paths")
    if out.shape[-1] != encoder.h_dim:
        raise RuntimeError(f"Expected h_dim {encoder.h_dim}, got {out.shape[-1]}")

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_file)
    size_mb = out_file.stat().st_size / 1e6
    print(f"Saved {tuple(out.shape)} "
          f"{'token sequences' if args.tokens else 'representations'} "
          f"-> {out_file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
