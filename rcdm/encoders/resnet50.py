"""
rcdm/encoders/resnet50.py

Frozen ImageNet-supervised ResNet-50 — the control condition.

This is RCDM's original encoder (Bordes et al. 2022), kept so the study has a
*supervised, out-of-domain* baseline against the two self-supervised retinal
encoders.  Bordes et al. found supervised representations invert far less
faithfully than SSL ones (MRR 0.69 vs 0.97-0.99 on ImageNet); reproducing that
contrast on retinal data is what makes the SSL-vs-supervised claim testable
here rather than assumed.

The representation is the 2048-dim global-average-pooled trunk output, obtained
by replacing the classification head with Identity.
"""

import torch
import torch.nn as nn

from .transforms import build_transform  # noqa: F401  (re-exported for callers)

# Width of the ResNet-50 avgpool output.
H_DIM = 2048


class ResNet50Encoder(nn.Module):
    """Frozen ResNet-50 trunk exposing a uniform ``forward(x) -> (B, 2048)``."""

    h_dim = H_DIM
    name = "resnet50"

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) ImageNet-normalised tensor.
        Returns:
            h: (B, 2048) avgpool trunk representation.
        """
        return self.backbone(x)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        A token sequence for E5's attentive probe.

        A ResNet has no token sequence, so the closest analogue is the final
        conv feature map flattened into its 49 spatial positions.  The avgpool
        vector is prepended as token 0 so that ``tokens[:, 0] == forward(x)``
        holds for every encoder in the registry — E5's cache check relies on
        that invariant to prove tokens and CLS caches describe the same images.

        Args:
            x: (B, 3, 224, 224) ImageNet-normalised tensor.
        Returns:
            tokens: (B, 50, 2048) — [avgpool, 49 spatial positions].
        """
        net = self.backbone
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        feature_map = net.layer4(net.layer3(net.layer2(net.layer1(x))))  # (B, 2048, 7, 7)
        spatial = feature_map.flatten(2).transpose(1, 2)                 # (B, 49, 2048)
        pooled = feature_map.mean(dim=(2, 3), keepdim=False).unsqueeze(1)  # (B, 1, 2048)
        return torch.cat([pooled, spatial], dim=1)

    def token_layout(self) -> dict:
        """Which index in the sequence is what — see encoders/__init__.py."""
        return {
            "cls_index": 0,          # the avgpool vector standing in for a CLS token
            "n_register_tokens": 0,
            "n_patch_tokens": 49,
            "patch_grid": [7, 7],
            "n_tokens": 50,
        }


def load(device: str = "cpu", **_) -> ResNet50Encoder:
    """
    Load the ImageNet-supervised ResNet-50 trunk from torchvision.

    Unlike the other two encoders this downloads its weights from torchvision's
    model zoo on first use, then caches them in ~/.cache/torch.

    Returns:
        ResNet50Encoder in eval mode with all parameters frozen.
    """
    from torchvision import models

    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()  # drop the 1000-way head -> (B, 2048)

    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    encoder = ResNet50Encoder(backbone)
    return encoder.eval().to(device)
