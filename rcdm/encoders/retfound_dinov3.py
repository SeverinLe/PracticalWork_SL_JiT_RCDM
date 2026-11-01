"""
rcdm/encoders/retfound_dinov3.py

Frozen RETFound-DINOv3 ViT-L/16 encoder — joint-embedding (DINOv3) SSL on colour
fundus photographs.

Why this encoder is the one the study needed
-------------------------------------------
Until this checkpoint existed, the RETFound comparison varied three things at
once.  ``retfound_cfp`` is MAE on retinal CFP at ViT-L/16, and the only
joint-embedding encoder available was DinoV3 ViT-S/16 — web images, 384-dim.  So
"MAE inverts poorly" could not be separated from "web pre-training inverts
poorly" or from "1024-dim squeezed through a 128-dim conditioning projector
inverts poorly".  See experiments/E7_conditioning_bottleneck.py for the last one,
which is the most dangerous: cond_dim is fixed at 128, so a 1024-dim encoder gets
a 12.5% bottleneck against DinoV3 ViT-S/16's 33%.

This checkpoint holds everything constant except the SSL objective:

    retfound_cfp       MAE (reconstructive)   CFP, MEH   ViT-L/16   1024-dim
    retfound_dinov3    DINOv3 (joint-emb.)    CFP, MEH   ViT-L/16   1024-dim

Same modality, same hospital source, same architecture, same width, same author
(Yukun Zhou).  The conditioning bottleneck is therefore *identical* to
RETFound's, and a difference in E1 fidelity between the two is attributable to
the SSL family alone.  That is the single-variable contrast the report's §4.2
claim needs.

Sources
-------
    https://huggingface.co/YukunZhou/retfound_dinov3
    https://github.com/HORIZONHealthcare/Med_DINOv3

Why the checkpoint needs converting
-----------------------------------
The .pth is a raw DINOv3 *pre-training* checkpoint in the upstream
facebookresearch/dinov3 layout, not a HuggingFace or timm state dict::

    {"teacher": {"backbone.*": ..., "dino_head.*": ..., "ibot_head.*": ...}}

``backbone.*`` is the ViT-L/16 encoder (24 blocks, 1024-dim, RoPE, LayerScale,
non-gated MLP).  ``dino_head`` and ``ibot_head`` are the DINO and iBOT projection
heads (65 536 prototypes each) used only by the self-distillation loss; they have
no counterpart in an encoder-only model and are dropped.

The directory's ``config.json`` is a hand-written model card in the same style as
RETFound's — ``{"architectures": ["ViT-Large"], "SSL strategy": "DINOV3", ...}``
— **not** a transformers ``DINOv3ViTConfig``.  ``AutoModel.from_pretrained``
cannot read it, which is why this module goes through timm rather than reusing
the ``dinov3`` entry's HuggingFace path.

Three conversion details that would each silently corrupt h
-----------------------------------------------------------
1. **Key renames.**  timm's EVA implementation names LayerScale ``gamma_1`` /
   ``gamma_2`` where DINOv3 uses ``ls1.gamma`` / ``ls2.gamma``.  Mirrors
   ``timm.models.eva.checkpoint_filter_fn``.

2. **No register tokens.**  Official DINOv3 ViT-L/16 carries 4 register
   ("storage") tokens; this checkpoint has *none* — only ``cls_token``.  timm
   defaults to 4, so the model must be built with ``num_reg_tokens=0`` or 4
   randomly-initialised tokens are prepended to every sequence.  The sequence is
   therefore 197 tokens (CLS + 196 patches), the same as RETFound's, rather than
   the 201 of the ``dinov3`` entry.

3. **The k-bias must be kept.**  The checkpoint's fused ``attn.qkv.bias`` has a
   non-zero key component (max |k bias| = 10.33).  timm's stock converter drops
   it — its comment says the k bias "is supposed to be fixed at 0", which holds
   for the released DINOv3 models where weight decay drove it there.  Without
   RoPE a k-bias *is* a no-op, because it adds a per-query constant to every
   logit and softmax is shift-invariant.  With RoPE the bias is rotated per key
   position, so the term stops being constant across keys and the invariance
   fails.  Measured here: dropping it gives cosine 0.9991 against the faithful
   output but ~4% relative L2 error — invisible to E3's cosines, not invisible to
   E1_exact_embedding's offset ratios.  ``load()`` writes it into the non-persistent
   ``k_bias`` buffer that timm already concatenates into the qkv bias, so the load
   is exact and nothing is dropped.

RoPE periods are deliberately *not* loaded from the checkpoint.  timm recomputes
them in float32 from ``rope_theta``; the checkpoint stores the same values in
bfloat16, and ``timm_periods.to(bfloat16) == checkpoint_periods`` exactly.  The
recomputed buffer is the same embedding at higher precision.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .transforms import build_transform  # noqa: F401  (re-exported for callers)

# CLS-token width of ViT-Large/16.
H_DIM = 1024

# timm architecture.  The "_qkvb" variant is required: this checkpoint has q and v
# biases (the plain variant is built with qkv_bias=False and would drop them).
TIMM_MODEL = "vit_large_patch16_dinov3_qkvb"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    _REPO_ROOT / "encoders" / "dinov3_vitl16_retfound_med" / "retfound_dinov3.pth"
)

# Self-distillation heads: used by the pre-training loss, absent from an encoder.
_HEAD_PREFIXES = ("dino_head", "ibot_head")


class RETFoundDINOv3Encoder(nn.Module):
    """Frozen RETFound-DINOv3 ViT-L/16 exposing a uniform ``forward(x) -> (B, 1024)``."""

    h_dim = H_DIM
    name = "retfound_dinov3"

    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) ImageNet-normalised tensor.
        Returns:
            h: (B, 1024) CLS token.
        """
        # forward_features returns the full sequence (B, 197, 1024); token 0 is CLS.
        return self.vit.forward_features(x)[:, 0]

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        The full final-layer token sequence, for E5's attentive probe.

        Args:
            x: (B, 3, 224, 224) ImageNet-normalised tensor.
        Returns:
            tokens: (B, 197, 1024) — [CLS, 196 patches].  This checkpoint was
            trained without register tokens, so there are none to skip.
        """
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


def convert_state_dict(raw: dict) -> tuple:
    """
    Convert an upstream DINOv3 pre-training checkpoint to timm's EVA layout.

    Args:
        raw: the object saved in the .pth — either the full file
            ``{"teacher": {...}}`` or an already-unwrapped state dict.

    Returns:
        (state_dict, k_biases) — ``state_dict`` loads into TIMM_MODEL with
        ``strict=True``; ``k_biases`` maps ``<block>.attn.k_bias`` to the key-bias
        tensor timm keeps as a zero buffer.  See the module docstring for why it
        is returned separately rather than discarded.
    """
    state = raw.get("teacher", raw)
    state = state.get("model", state)

    backbone = {
        key[len("backbone."):]: value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone:
        # Some exports drop the prefix; accept those as long as the heads are gone.
        backbone = {
            key: value for key, value in state.items()
            if not key.startswith(_HEAD_PREFIXES)
        }
    if not backbone:
        raise ValueError(
            "No backbone weights found in the checkpoint. Expected a DINOv3 "
            "pre-training file with a 'teacher' entry holding 'backbone.*' keys."
        )

    out: dict = {}
    k_biases: dict = {}
    for key, value in backbone.items():
        if key.startswith(_HEAD_PREFIXES):
            continue
        # mask_token is pre-training only; RoPE periods are recomputed by timm.
        if key == "mask_token" or key.endswith(".periods") or key.endswith(".bias_mask"):
            continue
        if key.startswith("local_cls_norm"):
            continue

        value = value.float()   # the upstream file mixes float32 and bfloat16

        if key.endswith("attn.qkv.bias"):
            q_bias, k_bias, v_bias = value.chunk(3, dim=-1)
            out[key.replace("qkv.bias", "q_bias")] = q_bias
            out[key.replace("qkv.bias", "v_bias")] = v_bias
            # "blocks.N.attn.qkv.bias" -> "blocks.N.attn.k_bias"
            k_biases[key.replace(".qkv.bias", ".k_bias")] = k_bias
            continue

        key = key.replace("ls1.gamma", "gamma_1").replace("ls2.gamma", "gamma_2")
        key = key.replace("storage_tokens", "reg_token")
        out[key] = value

    return out, k_biases


def load(device: str = "cpu", checkpoint_path=DEFAULT_CHECKPOINT,
         strict_keys: bool = True, **_) -> RETFoundDINOv3Encoder:
    """
    Load the frozen RETFound-DINOv3 ViT-L/16 encoder.

    Args:
        device: "cpu", "cuda" or "mps".
        checkpoint_path: path to retfound_dinov3.pth.
        strict_keys: if True (default), raise when any encoder parameter is
            missing or unexpected after conversion. A silent mismatch here means
            part of the backbone is randomly initialised, which would make every
            downstream number irreproducible without failing visibly.

    Returns:
        RETFoundDINOv3Encoder in eval mode with all parameters frozen.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for the RETFound-DINOv3 encoder: pip install timm"
        ) from exc

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found. RETFound-DINOv3 weights are access-gated; "
            "see encoders/README.md."
        )

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state, k_biases = convert_state_dict(raw)

    # num_reg_tokens=0: this checkpoint has no storage tokens, and timm's default of
    # 4 would prepend randomly-initialised ones to every sequence.
    vit = timm.create_model(
        TIMM_MODEL,
        pretrained=False,
        num_classes=0,
        img_size=224,
        num_reg_tokens=0,
    )

    result = vit.load_state_dict(state, strict=False)
    if strict_keys and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(
            "RETFound-DINOv3 weights did not load cleanly — some encoder "
            f"parameters would be randomly initialised.\n  missing: {result.missing_keys}\n"
            f"  unexpected: {result.unexpected_keys}\n"
            "Pass strict_keys=False only if you understand the consequence."
        )

    # Restore the key bias that timm parks at zero — see the module docstring.
    applied = 0
    for name, module in vit.named_modules():
        key = f"{name}.k_bias"
        if key in k_biases and hasattr(module, "k_bias"):
            module.k_bias.copy_(k_biases[key].to(module.k_bias.dtype))
            applied += 1
    if strict_keys and applied != len(k_biases):
        raise RuntimeError(
            f"Applied {applied} of {len(k_biases)} key biases. The attention "
            "implementation changed; the conversion needs revisiting."
        )

    for param in vit.parameters():
        param.requires_grad = False
    vit.eval()

    encoder = RETFoundDINOv3Encoder(vit)
    return encoder.eval().to(device)
