#!/usr/bin/env bash
# =============================================================================
# Download DINOv3 ViT-L/16 (LVD-1689M) into encoders/dinov3_vitl16/.
#
# This is the width-matched control for RETFound ViT-L/16: same 1024-dim CLS, so
# the 128-dim conditioning bottleneck is identical and the remaining differences
# against retfound_cfp are SSL family (self-distillation vs MAE) and pre-training
# domain (web vs retinal).  See rcdm/encoders/dinov3.py.
#
# TWO WEIGHT FORMATS EXIST, AND BOTH WORK — this script fetches the first.
#
#   A. HuggingFace directory   encoders/dinov3_vitl16/
#                                config.json + model.safetensors
#      What this script downloads (from facebook/dinov3-vitl16-pretrain-lvd1689m).
#      Loaded with transformers AutoModel.from_pretrained(local_files_only=True).
#
#   B. Official .pth           encoders/dinov3_vitl16/dinov3_vitl16_pretrain_lvd.pth
#      The facebookresearch/dinov3 release: a flat backbone state dict
#      (cls_token, storage_tokens, blocks.*, rope_embed.periods), NOT a
#      transformers checkpoint — AutoModel cannot read it. Loaded through timm
#      (vit_large_patch16_dinov3) after the key conversion shared with
#      rcdm/encoders/retfound_dinov3.py. Download it yourself from the dinov3
#      repo or the Meta access e-mail; this script does not fetch it.
#
# rcdm/encoders/dinov3.py:load_vitl16 detects which is present — a directory
# takes path A, a file takes path B — and DEFAULT_CHECKPOINT_VITL16 prefers the
# .pth when both exist. Either yields the same encoder: 1024-dim CLS, 4 register
# tokens, so forward_tokens returns 201 = CLS + 4 + 196. The .pth path also
# asserts the qkv biases are zero before dropping them, which holds for this
# distilled release but not for every DINOv3 checkpoint.
#
# So: run this script if you have neither, and skip it if you already placed the
# .pth in encoders/dinov3_vitl16/.
#
# Prerequisites — the repo is gated:
#   1. Open https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
#      while logged in, accept the licence, wait for access to show as granted.
#   2. Token must allow gated repos:
#        • Classic token: "Read" is enough.
#        • Fine-grained token: enable "Access to public gated repositories".
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash encoders/download_dinov3_vitl16.sh
# or:
#   bash encoders/download_dinov3_vitl16.sh --token hf_...
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTDIR="${REPO_ROOT}/encoders/dinov3_vitl16"
HF_REPO="facebook/dinov3-vitl16-pretrain-lvd1689m"
TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --token) [[ -n "${2:-}" ]] || { echo "Error: --token needs a value"; exit 1; }
             TOKEN="$2"; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1"; echo "Try --help"; exit 1 ;;
  esac
done

# `hf` is the current entrypoint (huggingface_hub >= 0.34); keep the old name as
# a fallback so this works on an older pinned environment too.
if command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli download)
else
  echo "Error: neither 'hf' nor 'huggingface-cli' found."
  echo "  pip install -r requirements.txt   (huggingface_hub is pinned there)"
  exit 1
fi

mkdir -p "${OUTDIR}"
echo "Downloading ${HF_REPO}"
echo "  -> ${OUTDIR}"
echo ""

TOKEN_ARGS=()
[[ -n "${TOKEN}" ]] && TOKEN_ARGS=(--token "${TOKEN}")

# Only the two files the loader needs; --local-dir keeps them out of the HF cache
# so the checkpoint is pinned inside the repo, as encoders/README.md requires.
if ! "${HF_CLI[@]}" "${HF_REPO}" config.json model.safetensors \
      --local-dir "${OUTDIR}" "${TOKEN_ARGS[@]}"; then
  echo ""
  echo "──────────────── Hugging Face download failed ────────────────"
  echo "On 403 Forbidden / 'public gated repositories':"
  echo "  • Accept the licence at https://huggingface.co/${HF_REPO} while logged in."
  echo "  • Fine-grained token: enable read access to *public gated* repos,"
  echo "    or use a classic read token instead."
  echo "  • Pass it explicitly:  HF_TOKEN=hf_... $0"
  echo "──────────────────────────────────────────────────────────────"
  exit 1
fi

echo ""
echo "Downloaded:"
ls -lh "${OUTDIR}"/config.json "${OUTDIR}"/model.safetensors

# Validate before anything trains on it.  assert_config_sane compares the
# config's use_gated_mlp flag against the gate_proj keys actually present in the
# weight file and checks hidden_size == 1024 — the two ways a DINOv3 directory
# can silently produce a wrong or irreproducible h.
echo ""
echo "Validating checkpoint ..."
PYBIN="${REPO_ROOT}/.venv/bin/python"
[[ -x "${PYBIN}" ]] || PYBIN="$(command -v python3)"
"${PYBIN}" -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from rcdm.encoders.dinov3 import assert_config_sane, H_DIM_VITL16
assert_config_sane('${OUTDIR}', expected_h_dim=H_DIM_VITL16)
import json
cfg = json.load(open('${OUTDIR}/config.json'))
print('  hidden_size       :', cfg['hidden_size'])
print('  layers / heads    :', cfg['num_hidden_layers'], '/', cfg['num_attention_heads'])
print('  patch_size        :', cfg['patch_size'])
print('  register tokens   :', cfg.get('num_register_tokens', 0))
print('  use_gated_mlp     :', cfg.get('use_gated_mlp', False), '(checked against the weight file)')
print()
print('OK — validated. Next:')
print('  python data/scripts/precompute_reps.py --encoder dinov3_vitl16 \\\\')
print('      --data_dir data/raw/messidor2/train \\\\')
print('      --out_file data/processed/messidor2/dinov3_vitl16/train_reps.pt --device mps')
"
