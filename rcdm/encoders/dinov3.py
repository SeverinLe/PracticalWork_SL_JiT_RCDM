"""
rcdm/encoders/dinov3.py

Frozen DinoV3 encoders — joint-embedding (DINO-style) SSL.  Two registry entries
share this module:

    dinov3          ViT-S/16,  384-dim CLS   encoders/dinov3_vits16/
    dinov3_vitl16   ViT-L/16, 1024-dim CLS   encoders/dinov3_vitl16/

Both are the LVD-1689M web-image checkpoints distilled from the ViT-7B teacher;
only the student width differs.  ``dinov3_vitl16`` exists to remove the
architecture confound from the RETFound comparison: RETFound is only released as
ViT-L/16 (1024-dim), so pairing it with DinoV3 ViT-S/16 varies SSL family, domain
*and* representation width at once.  At h_dim 1024 the conditioning bottleneck is
identical to RETFound's (128/1024), leaving SSL family + pre-training domain as
the only differences.  See experiments/E7_conditioning_bottleneck.py for why that
matters.

Origin: entirely new relative to the upstream repos.  RCDM loaded a torchvision
ResNet-50 inline in train.py and ran it on every training step; JiT had no image
encoder at all (it was class-conditional).

Why this encoder is in the study
--------------------------------
It represents the *joint-embedding* branch of SSL, as opposed to RETFound's
*reconstructive* (MAE) branch.  DINO-style training pulls together augmented
views, so nearby vectors correspond to perceptually similar images — the
property representation-conditioned generation relies on.  At 384 dimensions it
also keeps the conditioning path light.

The use_gated_mlp trap
----------------------
The local checkpoint was trained with a standard 2-projection FFN (up_proj +
down_proj), but the config.json shipped with `use_gated_mlp: true`, which makes
transformers expect a 3-projection gated FFN.  The missing `gate_proj` is then
randomly initialised *on every load*, producing different h vectors run to run.
`assert_config_sane()` below fails loudly rather than letting that corrupt an
experiment silently.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .transforms import ENCODER_INPUT_SIZE, build_transform  # noqa: F401  (re-exported)

# CLS-token width per variant.  Architecture constant: ViT-S = 384, ViT-B = 768,
# ViT-L = 1024.  H_DIM keeps the ViT-S value so the registry contract documented
# in encoders/__init__.py ("a module with H_DIM and load()") still reads.
H_DIM = 384
H_DIM_VITL16 = 1024

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = _REPO_ROOT / "encoders" / "dinov3_vits16"
# The ViT-L/16 weights come in two shapes depending on where they were fetched:
# the HuggingFace directory (config.json + model.safetensors, what
# download_dinov3_vitl16.sh produces) or the official facebookresearch/dinov3
# .pth state dict.  load_vitl16() detects which and routes accordingly.
_VITL16_DIR = _REPO_ROOT / "encoders" / "dinov3_vitl16"
DEFAULT_CHECKPOINT_VITL16 = _VITL16_DIR / "dinov3_vitl16_pretrain_lvd.pth"
if not DEFAULT_CHECKPOINT_VITL16.exists():
    DEFAULT_CHECKPOINT_VITL16 = _VITL16_DIR

# timm architecture for the official .pth.  The released ViT-L is distilled and
# its qkv biases are all exactly zero, so the bias-free variant is the faithful
# target; the "_qkvb" variant would load the same weights with 24 zero biases.
TIMM_MODEL_VITL16 = "vit_large_patch16_dinov3"


class DinoV3Encoder(nn.Module):
    """
    Frozen DinoV3 ViT-S/16 exposing a uniform ``forward(x) -> (B, 384)``.

    The wrapper exists so that every encoder in this package has the same
    interface despite different underlying APIs (transformers here, timm for
    RETFound, torchvision for ResNet-50).
    """

    def __init__(self, backbone: nn.Module, name: str = "dinov3"):
        super().__init__()
        self.backbone = backbone
        # Instance attributes, not class constants: this class serves both the
        # 384-dim ViT-S/16 and the 1024-dim ViT-L/16 entry, and a stale class-level
        # h_dim would pass every downstream check while describing the wrong space.
        self.name = name
        self.h_dim = int(backbone.config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) ImageNet-normalised tensor.
        Returns:
            h: (B, 384) CLS token from the final layer.
        """
        out = self.backbone(pixel_values=x)
        return out.last_hidden_state[:, 0, :]

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        The full final-layer token sequence, for E5's attentive probe.

        Args:
            x: (B, 3, 224, 224) ImageNet-normalised tensor.
        Returns:
            tokens: (B, 201, 384) — [CLS, 4 registers, 196 patches].
            ``tokens[:, 0]`` is exactly what ``forward()`` returns.
        """
        return self.backbone(pixel_values=x).last_hidden_state

    def token_layout(self) -> dict:
        """Which index in the sequence is what — see encoders/__init__.py."""
        config = self.backbone.config
        n_register = int(getattr(config, "num_register_tokens", 0))
        side = int(config.image_size) // int(config.patch_size)
        return {
            "cls_index": 0,
            "n_register_tokens": n_register,
            "n_patch_tokens": side * side,
            "patch_grid": [side, side],
            "n_tokens": 1 + n_register + side * side,
        }


class DinoV3TimmEncoder(nn.Module):
    """
    Frozen DinoV3 backed by timm rather than transformers.

    The official facebookresearch/dinov3 releases are flat .pth state dicts, which
    only timm's EVA implementation can host; the HuggingFace directory form is
    served by DinoV3Encoder above.  Both satisfy the same registry contract, so
    nothing downstream needs to know which one it was handed.
    """

    def __init__(self, vit: nn.Module, name: str):
        super().__init__()
        self.vit = vit
        self.name = name
        self.h_dim = int(vit.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, h_dim) CLS token."""
        return self.vit.forward_features(x)[:, 0]

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, T, h_dim); T = 201 for the 4-register ViT-L/16."""
        return self.vit.forward_features(x)

    def token_layout(self) -> dict:
        """Which index in the sequence is what — see encoders/__init__.py."""
        rows, cols = self.vit.patch_embed.grid_size
        n_prefix = int(getattr(self.vit, "num_prefix_tokens", 1))
        return {
            "cls_index": 0,
            "n_register_tokens": n_prefix - 1,
            "n_patch_tokens": int(rows * cols),
            "patch_grid": [int(rows), int(cols)],
            "n_tokens": n_prefix + int(rows * cols),
        }


def assert_config_sane(checkpoint_path: Path, expected_h_dim: int | None = None) -> None:
    """
    Raise if config.json disagrees with the weights actually in the checkpoint.

    Two failure modes, both silent without this check:

    1. ``use_gated_mlp`` disagreeing with the weight file.  The shipped ViT-S/16
       config sets it true while the checkpoint has a 2-projection FFN, so
       transformers randomly initialises the missing ``gate_proj`` *on every load*
       and h differs run to run.  Rather than hard-coding "must be false" — which
       would be wrong for the H+ and 7B variants, where the gated FFN is real —
       this reads the safetensors key list and compares.
    2. A width mismatch between the directory and the registry entry pointed at
       it, e.g. the ViT-L/16 folder holding S/16 weights.  That would build a
       conditioning projector for the wrong space.

    Args:
        checkpoint_path: local HuggingFace-format directory.
        expected_h_dim: registry width for this entry; skipped when None.
    """
    import json

    checkpoint_path = Path(checkpoint_path)
    config_file = checkpoint_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"{config_file} not found. See encoders/README.md for how to obtain "
            "the DinoV3 checkpoints."
        )
    config = json.loads(config_file.read_text())
    flag = bool(config.get("use_gated_mlp", False))

    has_gate = None
    weight_file = checkpoint_path / "model.safetensors"
    if weight_file.exists():
        try:
            from safetensors import safe_open

            with safe_open(str(weight_file), framework="pt") as handle:
                has_gate = any("gate_proj" in k for k in handle.keys())
        except Exception:  # noqa: BLE001 — an unreadable index is not this check's business
            has_gate = None

    if has_gate is None:
        # Cannot inspect the weights; fall back to the one case known to be broken.
        if flag:
            raise ValueError(
                f"{config_file} has use_gated_mlp: true and the weight file could "
                "not be inspected to confirm that gate_proj exists. If this is the "
                "ViT-S/16 checkpoint the flag is wrong — set it to false, or h will "
                "differ between runs."
            )
    elif flag != has_gate:
        raise ValueError(
            f"{config_file} has use_gated_mlp: {str(flag).lower()} but "
            f"{weight_file.name} {'contains' if has_gate else 'has no'} gate_proj "
            f"weights. transformers would {'ignore real weights' if has_gate else 'randomly initialise the missing projection on every load'}, "
            f"so h would be {'incomplete' if has_gate else 'different run to run'}. "
            f"Set use_gated_mlp to {str(has_gate).lower()}."
        )

    if expected_h_dim is not None and int(config.get("hidden_size", -1)) != expected_h_dim:
        raise ValueError(
            f"{checkpoint_path.name} has hidden_size {config.get('hidden_size')}, "
            f"but this registry entry expects {expected_h_dim}. The wrong variant "
            "is in this directory."
        )


def _load(checkpoint_path, device: str, name: str, expected_h_dim: int) -> DinoV3Encoder:
    """Shared loading path for every DinoV3 variant."""
    from transformers import AutoModel

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found. DinoV3 weights are gated downloads; see "
            "encoders/README.md for the procedure and the expected layout."
        )
    assert_config_sane(checkpoint_path, expected_h_dim=expected_h_dim)

    backbone = AutoModel.from_pretrained(str(checkpoint_path), local_files_only=True)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    encoder = DinoV3Encoder(backbone, name=name)
    if encoder.h_dim != expected_h_dim:
        raise RuntimeError(
            f"{name}: loaded backbone reports h_dim {encoder.h_dim}, expected "
            f"{expected_h_dim}."
        )
    return encoder.eval().to(device)


def load(device: str = "cpu", checkpoint_path=DEFAULT_CHECKPOINT, **_) -> DinoV3Encoder:
    """
    Load the frozen DinoV3 ViT-S/16 backbone (384-dim CLS).

    Args:
        device: "cpu", "cuda" or "mps".
        checkpoint_path: local HuggingFace-format directory containing
            config.json and model.safetensors.  Loaded with
            local_files_only=True so no network request is ever made — the
            weights are pinned for reproducibility.

    Returns:
        DinoV3Encoder in eval mode with all parameters frozen.
    """
    return _load(checkpoint_path, device, "dinov3", H_DIM)


def load_vitl16(device: str = "cpu", checkpoint_path=DEFAULT_CHECKPOINT_VITL16,
                **_) -> DinoV3Encoder:
    """
    Load the frozen DinoV3 ViT-L/16 backbone (1024-dim CLS).

    Same pre-training corpus and objective as the ViT-S/16 entry — only the
    student width differs — which is what makes it the width-matched control for
    RETFound ViT-L/16.

    Args:
        device: "cpu", "cuda" or "mps".
        checkpoint_path: local HuggingFace-format directory; defaults to
            encoders/dinov3_vitl16/.

    Returns:
        DinoV3Encoder in eval mode with all parameters frozen.
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        return _load(checkpoint_path, device, "dinov3_vitl16", H_DIM_VITL16)
    return _load_official_pth(checkpoint_path, device, "dinov3_vitl16", H_DIM_VITL16)


def _load_official_pth(checkpoint_path: Path, device: str, name: str,
                       expected_h_dim: int) -> "DinoV3Encoder":
    """
    Load an official facebookresearch/dinov3 .pth into timm.

    The released file is a flat backbone state dict — ``cls_token``,
    ``storage_tokens``, ``blocks.*``, ``rope_embed.periods`` — not a HuggingFace
    directory, so ``AutoModel.from_pretrained`` cannot read it.  The key mapping
    is the same one ``rcdm/encoders/retfound_dinov3.py`` documents (LayerScale
    renamed to timm's ``gamma_1``/``gamma_2``, pre-training-only tensors dropped);
    the converter is shared rather than duplicated.

    Two differences from the RETFound-DINOv3 checkpoint:

    * this one **has** 4 ``storage_tokens``, so the model is built with
      ``num_reg_tokens=4`` and the sequence is 201 = CLS + 4 + 196, against 197
      there.  ``token_layout()`` reports it, and E5's attentive probe pools the
      whole sequence, so the length is not interchangeable across encoders.
    * its qkv biases are all exactly zero (distilled release), so the bias-free
      timm variant is used and the split q/v biases are dropped — verified zero
      before dropping rather than assumed.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required for the DinoV3 ViT-L/16 .pth: pip install timm") from exc

    from .retfound_dinov3 import convert_state_dict

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state, k_biases = convert_state_dict(raw)

    # The bias-free variant is only faithful if the biases really are zero.
    stray = {k: v for k, v in state.items() if k.endswith(("q_bias", "v_bias"))}
    nonzero = [k for k, v in {**stray, **k_biases}.items() if float(v.abs().max()) != 0.0]
    if nonzero:
        raise RuntimeError(
            f"{checkpoint_path.name} has non-zero qkv biases ({len(nonzero)} tensors, "
            f"e.g. {nonzero[0]}), which the bias-free timm variant cannot represent. "
            "Load it through the _qkvb variant as rcdm/encoders/retfound_dinov3.py does."
        )
    state = {k: v for k, v in state.items() if k not in stray}

    model = timm.create_model(
        TIMM_MODEL_VITL16, pretrained=False, num_classes=0,
        img_size=ENCODER_INPUT_SIZE, num_reg_tokens=4,
    )
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "DinoV3 ViT-L/16 weights did not load cleanly — some parameters would "
            f"be randomly initialised.\n  missing: {result.missing_keys}\n"
            f"  unexpected: {result.unexpected_keys}"
        )

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    encoder = DinoV3TimmEncoder(model, name=name)
    if encoder.h_dim != expected_h_dim:
        raise RuntimeError(f"{name}: got h_dim {encoder.h_dim}, expected {expected_h_dim}")
    return encoder.eval().to(device)
