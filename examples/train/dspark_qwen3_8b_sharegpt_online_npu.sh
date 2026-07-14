#!/bin/bash
# Online DSpark Training Script — Ascend NPU (910B/910C) variant
#
# Same pipeline as dspark_qwen3_0_6b_sharegpt_online.sh scaled to Qwen3-8B,
# adapted for Ascend:
#   - ASCEND_RT_VISIBLE_DEVICES instead of CUDA_VISIBLE_DEVICES (always assign
#     explicitly: an empty string is NOT the same as unset)
#   - --draft-attn-impl sdpa (flex attention needs inductor, unavailable on
#     NPU; train.py also auto-falls-back, this just makes it explicit). If
#     sdpa hits operator issues on your CANN version, try eager.
#   - multi-card training defaults to DDP (fp32 replicated master weights);
#     the DSpark draft fits per card, so no --fsdp-shard needed
#   - the vLLM server must run on vllm-ascend >= v0.20.2rc1 (first version
#     with extract_hidden_states support); see my_docs/910b_环境与训练上手指南.md
#
# DSpark extends DFlash with a Markov head (intra-block token dependency) and
# a confidence head (per-position acceptance prediction).
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dspark_qwen3_8b_sharegpt_online_npu.sh

### Example E2E run for DSpark Qwen3-8B on 5k samples from ShareGPT ###

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="sharegpt"                # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="./output/dspark_qwen3_8b_sharegpt_npu"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=5
LR=3e-4

# DSpark-specific parameters
SPECULATOR_TYPE="dspark"
BLOCK_SIZE=8
MAX_ANCHORS=3072
NUM_LAYERS=5
# Full verifier vocab (GLM-5.2 / Red Hat recipe): omit --draft-vocab-size.
# To prune to the top-K frequent tokens instead (smaller logits/loss, needs
# d2t/t2d mapping at serve time), set e.g. DRAFT_VOCAB_SIZE=32000 and add
# --draft-vocab-size "$DRAFT_VOCAB_SIZE" back to the train command. NOTE:
# once a pruned run cached d2t.npy/t2d.npy in OUTPUT_DIR, delete them (or use
# a fresh OUTPUT_DIR) to return to full vocab.
TARGET_LAYER_IDS="2 18 33"  # Must match vLLM's eagle_aux_hidden_state_layer_ids

# Markov + confidence head settings
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"   # vanilla | gated | rnn
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# NPU assignments (online training needs separate NPUs for vLLM and training)
VLLM_NPUS="0,1"
NUM_VLLM_NPUS=2     # must match the card count in VLLM_NPUS
TRAIN_NPUS="2,3"
NUM_TRAIN_NPUS=2
VLLM_TP=1           # vLLM tensor parallel; raise to 2/4 when one card can't
                    # hold the target (e.g. Qwen3-32B bf16 ~65GB > 64GB card)
VLLM_DP=$(( NUM_VLLM_NPUS / VLLM_TP ))   # auto: DP*TP must equal NUM_VLLM_NPUS

# Optimizer: default is muon; if Muon misbehaves on your torch_npu build,
# uncomment the adamw fallback below (added to the train command).
# OPTIMIZER_ARGS="--optimizer adamw"
OPTIMIZER_ARGS=""
# =======================================

# Step 1: Prepare data (CPU-only, no device flags needed)
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM (vllm-ascend) server in the background
echo "=== Step 2: Launching vLLM server ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --tensor-parallel-size "$VLLM_TP" --data-parallel-size "$VLLM_DP" \
       --port "$VLLM_PORT" &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
# --noproxy: intranet http_proxy/https_proxy settings would otherwise hijack
# the localhost request and this loop would never succeed; 127.0.0.1 avoids
# IPv6 localhost resolution mismatches.
WAITED=0
until curl -sf --noproxy '*' "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server process exited before becoming healthy;" \
             "check its output above." >&2
        exit 1
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $((WAITED % 60)) -eq 0 ]; then
        echo "still waiting for http://127.0.0.1:${VLLM_PORT}/health (${WAITED}s elapsed)..."
    fi
done
echo "vLLM server ready."

# Step 3: Train DSpark against the live vLLM server
echo "=== Step 3: Training ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --draft-attn-impl sdpa \
    $OPTIMIZER_ARGS \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
echo "Recommended: python scripts/check_norm_canary.py $OUTPUT_DIR/checkpoints"
