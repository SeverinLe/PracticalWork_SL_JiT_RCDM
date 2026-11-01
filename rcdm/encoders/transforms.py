"""
rcdm/encoders/transforms.py

Preprocessing shared by every frozen encoder in this package.

All three encoders (DinoV3 ViT-S/16, RETFound MAE ViT-L/16, ResNet-50) consume
ImageNet-normalised input.  RETFound's official pipeline
(RETFound_MAE/util/datasets.py) uses timm's IMAGENET_DEFAULT_MEAN/STD, which
are the same constants; DinoV3 and torchvision's ResNet-50 use them too.

Note the deliberate separation of two normalisations in this project:

    encoder input   ImageNet mean/std      (this module)
    generator input [-1, 1]                (rcdm/dataset.py)

They are never mixed.  h is computed in encoder space; the diffusion target x
lives in [-1, 1].
"""

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Every supported encoder is a /16 ViT (or a ResNet trained at this resolution)
# with a fixed 14x14 positional-embedding grid: 224 / 16 = 14 patches per side.
ENCODER_INPUT_SIZE = 224


def build_transform(image_size: int = ENCODER_INPUT_SIZE) -> transforms.Compose:
    """
    PIL image -> (3, image_size, image_size) ImageNet-normalised tensor.

    Args:
        image_size: edge length in pixels.  Leave at the default 224 unless you
            know what you are doing — a ViT's positional embeddings are tied to
            the 14x14 grid, and any other resolution silently interpolates or
            corrupts them.  The *generator's* resolution is a separate setting
            and does not constrain this one.

    Returns:
        A torchvision Compose: Resize -> CenterCrop -> ToTensor -> Normalize.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Nuisance transforms — the variant set E3 measures invariance over
# ---------------------------------------------------------------------------
#
# These live here rather than in experiments/E3_invariance.py because E5's
# condition C encodes the *same* transformed images through
# data/scripts/precompute_reps.py.  Two definitions of "grayscale" that drifted
# apart would silently make E3's cosine and E5's accuracy drop refer to
# different transforms, and the whole point of RQ5 is to compare them.
#
# Transforms are applied at the source resolution; build_transform() resizes to
# 224 px afterwards, so they only need to be semantically meaningful.

# Display order; also the order of rows in E3's qualitative grids.
VARIANT_ORDER = [
    "original",
    "crop_center",
    "zoom_x2",
    "rot90",
    "mirror_horizontal",
    "mirror_vertical",
    "translate",
    "grayscale",
    "contrast_low",
    "brightness_up",
]

# The variants that model acquisition/device variation rather than framing.
# A downstream classifier that moves under these carries a site dependence.
ACQUISITION_VARIANTS = ("grayscale", "contrast_low", "brightness_up")

DEFAULT_CROP_SIZE = 128


def make_variant(img: Image.Image, name: str, crop_size: int = DEFAULT_CROP_SIZE,
                 rng: torch.Generator | None = None) -> Image.Image:
    """
    Apply one named transform from VARIANT_ORDER to a PIL image.

    Args:
        img: source image at its native resolution.
        name: a key of VARIANT_ORDER.
        crop_size: edge length for "crop_center".
        rng: generator for "translate", the only stochastic variant.  Pass a
            seeded one whenever the result is cached, or the representations
            will not be reproducible.

    Returns:
        The transformed PIL image (a copy; the input is never mutated).
    """
    width, height = img.size

    if name == "original":
        return img.copy()
    if name == "crop_center":
        return TF.center_crop(img, [crop_size, crop_size])
    if name == "zoom_x2":
        # True 2x zoom: crop the central half, which the encoder then upsamples.
        return TF.center_crop(img, [height // 2, width // 2])
    if name == "rot90":
        # TF.rotate takes counter-clockwise angles, so clockwise 90 is -90.
        return TF.rotate(img, angle=-90, expand=True)
    if name == "mirror_horizontal":
        return TF.hflip(img)
    if name == "mirror_vertical":
        return TF.vflip(img)
    if name == "translate":
        # Shift by up to +/-20% per axis, black fill on the revealed border.
        max_dx, max_dy = int(0.20 * width), int(0.20 * height)
        dx = int(torch.randint(-max_dx, max_dx + 1, (1,), generator=rng).item())
        dy = int(torch.randint(-max_dy, max_dy + 1, (1,), generator=rng).item())
        return TF.affine(img, angle=0.0, translate=[dx, dy], scale=1.0,
                         shear=[0.0, 0.0], fill=0)
    if name == "grayscale":
        # Kept as 3 identical channels so the RGB encoder still applies.
        return TF.to_grayscale(img, num_output_channels=3)
    if name == "contrast_low":
        return TF.adjust_contrast(img, contrast_factor=0.6)
    if name == "brightness_up":
        return TF.adjust_brightness(img, brightness_factor=1.3)

    raise ValueError(f"Unknown variant {name!r}. Available: {', '.join(VARIANT_ORDER)}")


def make_variants(img: Image.Image, crop_size: int = DEFAULT_CROP_SIZE,
                  rng: torch.Generator | None = None) -> dict:
    """
    Ordered dict {variant_name: PIL image} of the original plus every transform.

    Returns them in VARIANT_ORDER, which is also the row order of E3's grids.
    """
    return {name: make_variant(img, name, crop_size, rng) for name in VARIANT_ORDER}
