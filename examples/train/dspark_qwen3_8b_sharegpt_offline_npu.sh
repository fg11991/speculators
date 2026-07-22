#!/bin/bash
# Offline DSpark Training Script — Ascend NPU (910B/910C) variant
#
# Runs the full offline pipeline: data preparation, vLLM server launch, hidden
# states generation to disk, server shutdown, then training from the
# pre-generated hidden states (NPUs are reused sequentially, so vLLM and
# training can share the same cards).
#
# Ascend notes:
#   - ASCEND_RT_VISIBLE_DEVICES instead of CUDA_VISIBLE_DEVICES (always assign
#     explicitly: an empty string is NOT the same as unset)
#   - --draft-attn-impl sdpa (flex attention needs inductor, unavailable on
#     NPU); if sdpa hits operator issues on your CANN version, try eager
#   - multi-card training defaults to DDP (fp32 replicated master weights);
#     the DSpark draft fits per card, so no --fsdp-shard needed
#   - vllm-ascend >= v0.20.2rc1 required (first version with
#     extract_hidden_states support); see my_docs/910b_环境与训练上手指南.md
#
# Disk usage: hidden states are ~ tokens x num_target_layers x hidden_size x
# 2 bytes per sample (Qwen3-8B, 3 layers: ~24 KB/token, up to ~200 MB per
# 8k-token sample). Budget roughly 100-300 GB for 5k ShareGPT samples and
# point HIDDEN_STATES_DIR at a large, fast disk.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dspark_qwen3_8b_sharegpt_offline_npu.sh

### Example E2E run for DSpark Qwen3-8B on 5k samples from ShareGPT ###

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="sharegpt"                # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="./output/dspark_qwen3_8b_sharegpt_offline_npu"
HIDDEN_STATES_DIR="$OUTPUT_DIR/hidden_states"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=5
LR=3e-4
CONCURRENCY=32                    # Parallel requests to vLLM during generation

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

# NPU assignments (offline reuses the same NPUs sequentially)
NPUS="0,1,2,3"
NUM_NPUS=4          # training: torchrun nproc_per_node
VLLM_TP=1           # vLLM tensor parallel; raise to 2/4 when one card can't
                    # hold the target (e.g. Qwen3-32B bf16 ~65GB > 64GB card)
VLLM_DP=$(( NUM_NPUS / VLLM_TP ))   # auto: DP*TP must equal the card count

# Optimizer: default is muon; if Muon misbehaves on your torch_npu build,
# uncomment the adamw fallback below (added to the train command).
# OPTIMIZER_ARGS="--optimizer adamw"
OPTIMIZER_ARGS=""
# =======================================

# Steps 1-4 are skipped entirely when a previous run already generated the
# hidden states (marker written after successful generation). Data prep is
# skipped together with generation on purpose: hidden states are keyed to the
# packed samples, so re-running prepare_data against existing hidden states
# could silently mismatch them. Set FORCE_REGEN=1 to redo everything.
GENERATION_MARKER="$HIDDEN_STATES_DIR/.generation_complete"
if [ -f "$GENERATION_MARKER" ] && [ "${FORCE_REGEN:-0}" != "1" ]; then
    echo "=== Hidden states already generated ($GENERATION_MARKER exists) ==="
    echo "=== Skipping data prep, vLLM launch and generation (FORCE_REGEN=1 to redo) ==="
else

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
ASCEND_RT_VISIBLE_DEVICES="$NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --tensor-parallel-size "$VLLM_TP" --data-parallel-size "$VLLM_DP" \
       --port "$VLLM_PORT" &
VLLM_PID=$!

# If anything below fails (set -e), don't leave the vLLM server orphaned on
# the NPUs. A second kill after the normal Step 4 shutdown is harmless.
cleanup() {
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

# Step 3: Generate hidden states to disk
echo "=== Step 3: Generating hidden states ==="
python scripts/data_generation_offline.py \
    --preprocessed-data "$OUTPUT_DIR" \
    --endpoint "http://localhost:${VLLM_PORT}/v1" \
    --output "$HIDDEN_STATES_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --concurrency "$CONCURRENCY" \
    --validate-outputs

# Step 4: Stop vLLM server to free NPU memory for training
echo "=== Step 4: Stopping vLLM server ==="
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "vLLM server stopped. NPUs freed for training."

# Only reached on full success (set -e): mark generation as complete so
# reruns go straight to training.
touch "$GENERATION_MARKER"

fi  # end of generation block

# Step 5: Train DSpark from the pre-generated hidden states
echo "=== Step 5: Training ==="
ASCEND_RT_VISIBLE_DEVICES="$NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
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
    --on-missing raise

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
echo "Recommended: python scripts/check_norm_canary.py $OUTPUT_DIR/checkpoints"
