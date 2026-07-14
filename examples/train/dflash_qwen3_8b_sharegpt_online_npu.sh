#!/bin/bash
# Online DFlash Training Script — Ascend NPU (910B/910C) variant
#
# Same pipeline as dflash_qwen3_8b_sharegpt_online_5k.sh, adapted for Ascend:
#   - ASCEND_RT_VISIBLE_DEVICES instead of CUDA_VISIBLE_DEVICES
#   - --draft-attn-impl sdpa (flex attention needs inductor, unavailable on NPU;
#     train.py also auto-falls-back, this just makes it explicit). If sdpa hits
#     operator issues on your CANN version, try --draft-attn-impl eager.
#   - the vLLM server must run on vllm-ascend with support for
#     speculative_config method "extract_hidden_states" and the
#     ExampleHiddenStatesConnector KV connector (verify before a long run)
#
# Prerequisites on the NPU host: torch_npu matching your torch version, CANN
# toolkit, and vllm + vllm-ascend in the serving environment.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dflash_qwen3_8b_sharegpt_online_npu.sh

### Example E2E run for DFlash Qwen3-8B on 5k samples from ShareGPT ###

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="sharegpt"                # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="./output/dflash_qwen3_8b_sharegpt_npu"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=5
LR=3e-4

# DFlash-specific parameters
SPECULATOR_TYPE="dflash"          # dflash or dspark
BLOCK_SIZE=8
MAX_ANCHORS=3072
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="2 18 33"  # Must match vLLM's eagle_aux_hidden_state_layer_ids

# NPU assignments (online training needs separate NPUs for vLLM and training)
VLLM_NPUS="0,1"
TRAIN_NPUS="2,3"
NUM_TRAIN_NPUS=2

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
    -- --data-parallel-size 2 --port "$VLLM_PORT" &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 3: Train against the live vLLM server
# Multi-card training defaults to DDP (fp32 replicated master weights); the
# draft model is small enough to fit per card. Add --fsdp-shard only if it
# does not.
echo "=== Step 3: Training ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --draft-attn-impl sdpa \
    $OPTIMIZER_ARGS \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
