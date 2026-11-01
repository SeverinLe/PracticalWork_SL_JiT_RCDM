"""
rcdm.encoders — the frozen encoders under study.

Every experiment in this project treats the encoder as the independent variable,
so all four are loaded through one registry with one interface:

    from rcdm.encoders import get_encoder, build_transform

    encoder = get_encoder("dinov3", device="cuda")   # frozen, eval mode
    h = encoder(transform(img).unsqueeze(0).to("cuda"))   # (1, encoder.h_dim)

Contract every encoder satisfies:

    encoder.name              str   registry key, written into result manifests
    encoder.h_dim             int   width of h (384 / 1024 / 1024 / 2048)
    encoder(x)                (B, 3, 224, 224) -> (B, h_dim), no grad, eval mode
    encoder.forward_tokens(x) (B, 3, 224, 224) -> (B, T, h_dim)
    encoder.token_layout()    dict  which index in that sequence is what

``forward_tokens`` exists for E5's attentive probe, which pools over the whole
sequence instead of reading the single pooled vector.  Everything else in the
pipeline — training, sampling, E1-E4, E6 — conditions on ``forward(x)`` alone,
because that is the only thing the generator was trained on.

Two invariants hold across every encoder and E5 checks both rather than
trusting them:

    tokens[:, 0] == forward(x)          token 0 is always the pooled vector
    tokens.shape[-1] == encoder.h_dim   channel j means the same axis in every
                                        token, which is what lets E5 apply E4's
                                        CLS-space dimension indices sequence-wide

Adding an encoder means adding a module with ``H_DIM`` and ``load(device, ...)``
and one line in ENCODERS below — no changes anywhere else in the pipeline, which
is parametrised by ``h_dim`` throughout.

``dinov3`` and ``dinov3_vitl16`` share one module and differ only in student
width.  The pair exists so that RETFound ViT-L/16 has a width-matched comparison:
against ``dinov3`` (384) the RETFound contrast varies SSL family, pre-training
domain *and* representation width at once, and the 128-dim conditioning
bottleneck bites four times harder at 1024 than at 384 — see
experiments/E7_conditioning_bottleneck.py.
"""

from typing import Callable, Dict

import torch
import torch.nn as nn
from PIL import Image

from . import dinov3, resnet50, retfound, retfound_dinov3
from .transforms import (  # noqa: F401  (public re-exports)
    ACQUISITION_VARIANTS,
    ENCODER_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VARIANT_ORDER,
    build_transform,
    make_variant,
    make_variants,
)

# name -> (loader, h_dim, one-line description used in --help and manifests)
ENCODERS: Dict[str, tuple] = {
    "dinov3": (
        dinov3.load,
        dinov3.H_DIM,
        "DinoV3 ViT-S/16, joint-embedding SSL, 384-dim CLS",
    ),
    "dinov3_vitl16": (
        dinov3.load_vitl16,
        dinov3.H_DIM_VITL16,
        "DinoV3 ViT-L/16, joint-embedding SSL, 1024-dim CLS "
        "(width-matched control for retfound_cfp)",
    ),
    "retfound_cfp": (
        retfound.load,
        retfound.H_DIM,
        "RETFound MAE ViT-L/16 (colour fundus weights), reconstructive SSL, 1024-dim CLS",
    ),
    "retfound_dinov3": (
        retfound_dinov3.load,
        retfound_dinov3.H_DIM,
        "RETFound-DINOv3 ViT-L/16 (colour fundus), joint-embedding SSL, 1024-dim CLS "
        "(same data and architecture as retfound_cfp — isolates the SSL objective)",
    ),
    "resnet50": (
        resnet50.load,
        resnet50.H_DIM,
        "ImageNet-supervised ResNet-50, 2048-dim avgpool (control)",
    ),
}

ENCODER_NAMES = tuple(ENCODERS)


def get_encoder(name: str, device: str = "cpu", **kwargs) -> nn.Module:
    """
    Load a frozen encoder by registry name.

    Args:
        name: one of ENCODER_NAMES.
        device: "cpu", "cuda" or "mps".
        **kwargs: forwarded to the encoder's loader, e.g. checkpoint_path.

    Returns:
        A frozen nn.Module in eval mode with ``.name`` and ``.h_dim`` set.
    """
    if name not in ENCODERS:
        raise ValueError(
            f"Unknown encoder {name!r}. Available: {', '.join(ENCODER_NAMES)}"
        )
    loader: Callable = ENCODERS[name][0]
    return loader(device=device, **kwargs)


def get_h_dim(name: str) -> int:
    """Representation width for an encoder, without loading its weights."""
    if name not in ENCODERS:
        raise ValueError(
            f"Unknown encoder {name!r}. Available: {', '.join(ENCODER_NAMES)}"
        )
    return ENCODERS[name][1]


def describe(name: str) -> str:
    """One-line human-readable description, for --help text and manifests."""
    return ENCODERS[name][2]


@torch.no_grad()
def encode_image(image_path, encoder: nn.Module, transform, device: str = "cpu") -> torch.Tensor:
    """
    Encode a single image file.

    Returns:
        h: (1, encoder.h_dim)
    """
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    return encoder(x)


@torch.no_grad()
def encode_batch(image_paths, encoder: nn.Module, transform,
                 device: str = "cpu", batch_size: int = 64,
                 verbose: bool = True, pre_transform=None) -> torch.Tensor:
    """
    Encode a list of image paths in batches.

    Used by data/scripts/precompute_reps.py to cache every representation once,
    so the encoder never runs inside the training loop.

    Args:
        pre_transform: optional PIL -> PIL callable applied at the source
            resolution, before ``transform`` resizes to 224 px.  This is how E5's
            condition C caches representations of transformed images without
            writing a transformed copy of the split to disk.

    Returns:
        reps: (N, encoder.h_dim) float32 on CPU, with reps[i] belonging to
        image_paths[i] — index alignment is what the whole pipeline relies on.
    """
    all_reps = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            if pre_transform is not None:
                img = pre_transform(img)
            imgs.append(transform(img))
        x = torch.stack(imgs).to(device)
        all_reps.append(encoder(x).float().cpu())
        if verbose and (i // batch_size) % 10 == 0:
            print(f"  encoded {min(i + batch_size, len(image_paths))}/{len(image_paths)}")
    return torch.cat(all_reps, dim=0)


@torch.no_grad()
def encode_tokens_batch(image_paths, encoder: nn.Module, transform,
                        device: str = "cpu", batch_size: int = 32,
                        verbose: bool = True, pre_transform=None,
                        dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    Encode a list of image paths into full token sequences.

    Same contract as ``encode_batch`` — reps[i] belongs to image_paths[i] — but
    returns the whole sequence rather than the pooled vector.

    Args:
        dtype: storage precision.  Defaults to float16 because these caches are
            two orders of magnitude larger than the pooled ones (RETFound's
            training split is 972 x 197 x 1024) and the probe standardises its
            input anyway, so the low 13 bits of mantissa carry nothing it reads.

    Returns:
        tokens: (N, T, h_dim) on CPU in ``dtype``.
    """
    all_tokens = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            if pre_transform is not None:
                img = pre_transform(img)
            imgs.append(transform(img))
        x = torch.stack(imgs).to(device)
        all_tokens.append(encoder.forward_tokens(x).to(dtype).cpu())
        if verbose and (i // batch_size) % 10 == 0:
            print(f"  encoded {min(i + batch_size, len(image_paths))}/{len(image_paths)}")
    return torch.cat(all_tokens, dim=0)
