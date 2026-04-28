#!/bin/bash
set -euo pipefail

# =============================================================================
# VBench Evaluation Models - One-Click Download Script
#
# Downloads all pretrained models required by VBench's 16 evaluation dimensions.
# Uses hf-mirror.com as primary source for HuggingFace URLs, with fallback
# to alternative mirrors for models whose original sources are inaccessible.
#
# Usage: bash scripts/download_vbench_models.sh
# =============================================================================

CACHE_DIR="$HOME/.cache/vbench"

# HuggingFace mirror (change if you have another mirror or proxy)
HF_MIRROR="https://hf-mirror.com"

# GitHub mirror for DINO repo
GITHUB_MIRROR="https://ghfast.top/https://github.com"

# RAFT source: original Dropbox is blocked, use HF mirror instead
RAFT_URL="${RAFT_URL:-${HF_MIRROR}/ddrfan/RAFT/resolve/main/raft-things.pth}"

echo "============================================"
echo " VBench Model Downloader"
echo " Cache: $CACHE_DIR"
echo "============================================"

mkdir -p "$CACHE_DIR"

# ── Helper: download a file, skip if already exists ──────────────────────────
download() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [ -f "$dest" ]; then
        # Quick integrity check: must be > 1KB or explicitly allowed
        if [ "$(stat -c%s "$dest")" -gt 1024 ] || [ "${4:-}" = "allow_small" ]; then
            echo "[OK] $desc — already exists ($(du -sh "$dest" | cut -f1))"
            return 0
        else
            echo "[RE-DOWNLOAD] $desc — file too small ($(stat -c%s "$dest") bytes), re-downloading..."
            rm -f "$dest"
        fi
    fi

    echo "[DOWNLOADING] $desc"
    echo "  URL: $url"
    echo "  Dest: $dest"

    local retries=3
    local delay=5
    for i in $(seq 1 $retries); do
        if wget --no-check-certificate -q -O "$dest.tmp" "$url" 2>&1; then
            mv "$dest.tmp" "$dest"
            echo "  -> OK ($(du -sh "$dest" | cut -f1))"
            return 0
        else
            echo "  -> Attempt $i/$retries failed${delay:+, retrying in ${delay}s...}"
            sleep "$delay"
        fi
    done

    # Clean up failed download
    rm -f "$dest.tmp"
    echo "[FAILED] $desc — all retries exhausted"
    return 1
}

# ── Helper: clone a git repo, skip if directory exists ──────────────────────
clone_repo() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [ -d "$dest" ] && [ -d "$dest/.git" ]; then
        echo "[OK] $desc — already cloned"
        return 0
    fi

    echo "[CLONING] $desc"
    local retries=3
    for i in $(seq 1 $retries); do
        if git clone --depth 1 "$url" "$dest" 2>&1; then
            echo "  -> OK"
            return 0
        else
            echo "  -> Attempt $i/$retries failed"
            rm -rf "$dest"
            sleep 5
        fi
    done

    echo "[FAILED] $desc — all retries exhausted"
    return 1
}

echo ""
echo "──── 1/11 CLIP ViT-B-32 (background_consistency, appearance_style) ────"
download \
    "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" \
    "$CACHE_DIR/clip_model/ViT-B-32.pt" \
    "CLIP ViT-B-32"

echo ""
echo "──── 2/11 CLIP ViT-L-14 (aesthetic_quality) ────────────────────────────"
download \
    "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt" \
    "$CACHE_DIR/clip_model/ViT-L-14.pt" \
    "CLIP ViT-L-14"

echo ""
echo "──── 3/11 DINO ViT-B/16 (subject_consistency) ────────────────────────"
clone_repo \
    "${GITHUB_MIRROR}/facebookresearch/dino" \
    "$CACHE_DIR/dino_model/facebookresearch_dino_main" \
    "DINO repo (github mirror)"

download \
    "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth" \
    "$CACHE_DIR/dino_model/dino_vitbase16_pretrain.pth" \
    "DINO ViT-B/16 weights"

echo ""
echo "──── 4/11 GRiT (object_class, multiple_objects, color, spatial_relationship) ──"
# Original Azure URL is 409 (public access revoked). Use HF mirror instead.
download \
    "${HF_MIRROR}/trimble/GRiT/resolve/main/models/grit_b_densecap_objectdet.pth" \
    "$CACHE_DIR/grit_model/grit_b_densecap_objectdet.pth" \
    "GRiT densecap model (hf-mirror)"

echo ""
echo "──── 5/11 AMT-S (motion_smoothness) ────────────────────────────────────"
download \
    "${HF_MIRROR}/lalala125/AMT/resolve/main/amt-s.pth" \
    "$CACHE_DIR/amt_model/amt-s.pth" \
    "AMT-S optical flow model (hf-mirror)"

echo ""
echo "──── 6/11 MUSIQ SPAQ (imaging_quality) ────────────────────────────────"
download \
    "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth" \
    "$CACHE_DIR/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth" \
    "MUSIQ image quality model"

echo ""
echo "──── 7/11 Tag2Text Swin-B (scene) ─────────────────────────────────────"
download \
    "${HF_MIRROR}/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth" \
    "$CACHE_DIR/caption_model/tag2text_swin_14m.pth" \
    "Tag2Text Swin-B 14M (hf-mirror)"

echo ""
echo "──── 8/11 Aesthetic predictor (aesthetic_quality) ─────────────────────"
download \
    "${HF_MIRROR}/LAION-AI/aesthetic-predictor/resolve/main/sa_0_4_vit_l_14_linear.pth" \
    "$CACHE_DIR/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth" \
    "LAION aesthetic predictor (hf-mirror)" \
    "allow_small"

echo ""
echo "──── 9/11 UMT L16 (human_action) ──────────────────────────────────────"
download \
    "${HF_MIRROR}/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth" \
    "$CACHE_DIR/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth" \
    "UMT video understanding model (hf-mirror)"

echo ""
echo "──── 10/11 ViCLIP (temporal_style, overall_consistency) ───────────────"
download \
    "${HF_MIRROR}/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth" \
    "$CACHE_DIR/ViCLIP/ViClip-InternVid-10M-FLT.pth" \
    "ViCLIP video-language model (hf-mirror)"

echo ""
echo "──── 11/11 RAFT (dynamic_degree) ──────────────────────────────────────"
download \
    "$RAFT_URL" \
    "$CACHE_DIR/raft_model/models/raft-things.pth" \
    "RAFT optical flow model (hf-mirror)"

echo ""
echo "============================================"
echo " Done! Checking all models..."
echo "============================================"
echo ""

ERRORS=0

check() {
    local f="$1"
    local desc="$2"
    if [ -f "$f" ]; then
        echo "  [OK] $desc"
    else
        echo "  [MISSING] $desc — $f"
        ERRORS=$((ERRORS + 1))
    fi
}

check "$CACHE_DIR/clip_model/ViT-B-32.pt"              "CLIP ViT-B-32"
check "$CACHE_DIR/clip_model/ViT-L-14.pt"              "CLIP ViT-L-14"
check "$CACHE_DIR/dino_model/dino_vitbase16_pretrain.pth" "DINO ViT-B/16 weights"
check "$CACHE_DIR/dino_model/facebookresearch_dino_main/hubconf.py" "DINO repo"
check "$CACHE_DIR/grit_model/grit_b_densecap_objectdet.pth" "GRiT densecap"
check "$CACHE_DIR/amt_model/amt-s.pth"                 "AMT-S"
check "$CACHE_DIR/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth" "MUSIQ SPAQ"
check "$CACHE_DIR/caption_model/tag2text_swin_14m.pth"  "Tag2Text"
check "$CACHE_DIR/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth" "LAION aesthetic"
check "$CACHE_DIR/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth" "UMT L16"
check "$CACHE_DIR/ViCLIP/ViClip-InternVid-10M-FLT.pth" "ViCLIP"
check "$CACHE_DIR/raft_model/models/raft-things.pth"    "RAFT optical flow"

echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo "All models ready!"
else
    echo "WARNING: $ERRORS model(s) missing. See above for details."
    echo "For models behind HuggingFace, try setting a different mirror:"
    echo "  HF_MIRROR=https://your-mirror.com bash scripts/download_vbench_models.sh"
    exit 1
fi
