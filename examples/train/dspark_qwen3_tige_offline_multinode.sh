#!/bin/bash
# Multi-node offline DSpark training on the tige Ascend platform (speculators).
#
# ONE script, run the SAME command on every node, only NODE_RANK differs
# (tige contract, same as DeepSpec's example/tige scripts):
#
#   # single node (defaults)
#   bash examples/train/dspark_qwen3_tige_offline_multinode.sh
#
#   # N nodes: on node i (i = 0..N-1)
#   NNODES=8 NODE_RANK=<i> MASTER_ADDR=<node0-ip> \
#       bash examples/train/dspark_qwen3_tige_offline_multinode.sh
#
# Phases (all coordinated through markers on the SHARED disk — OUTPUT_DIR,
# HIDDEN_STATES_DIR and the checkpoint dir MUST be visible to every node):
#   1. prepare_data          node 0 only, others wait on a marker
#   2. hidden-states gen     EVERY node: local vLLM + its 1/NNODES sample
#                            shard (data_generation_offline --world-size/--rank),
#                            then barrier until all node shards are done
#   3. training              torchrun --nnodes x --nproc_per_node, DDP
#
# Multi-node notes for speculators (verified against the code):
#   - scripts/train.py uses standard torchrun env:// init + hccl; the batch
#     sampler shards data by global rank. Plain multi-node torchrun just works.
#   - DDP (default) is the RIGHT strategy across nodes for the small draft:
#     one gradient all-reduce per step crosses the fabric. Do NOT add
#     --fsdp-shard on >1 node unless the draft truly cannot fit: it is
#     ZeRO-3-style sharding across ALL ranks, so every layer's parameter
#     all-gather crosses the inter-node fabric (DeepSpec measured 29 s/it @
#     2 nodes -> 180 s/it @ 4 nodes for the same reason). speculators has NO
#     hybrid_shard/HSDP option.
#   - There is NO gradient accumulation / global-batch-size knob: effective
#     batch = NNODES x NPROC_PER_NODE packed sequences per step. Consider
#     scaling LR when you scale nodes.
#   - Online mode is NOT multi-node friendly (single --vllm-endpoint in
#     train/data.py); this script is offline by design.
#
# FORCE_REGEN=1 re-runs data prep and this node's generation shard.

set -euo pipefail

# ============ tige multi-node wiring ============
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29510}

# ============ Ascend runtime ============
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-2400}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1200}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
# Keep intranet proxies away from node-local vLLM traffic.
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"
export NO_PROXY="$no_proxy"

# ============ Configuration ============
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
DATASET=${DATASET:-sharegpt}
# Both of these MUST be on the shared data disk:
OUTPUT_DIR=${OUTPUT_DIR:-/data/speculators/dspark_qwen3_8b_offline}
HIDDEN_STATES_DIR=${HIDDEN_STATES_DIR:-$OUTPUT_DIR/hidden_states}
VLLM_PORT=${VLLM_PORT:-8000}
MAX_SAMPLES=${MAX_SAMPLES:-50000}
SEQ_LENGTH=${SEQ_LENGTH:-8192}
EPOCHS=${EPOCHS:-5}
LR=${LR:-3e-4}
CONCURRENCY=${CONCURRENCY:-32}

# vLLM parallelism during generation (per node). For Qwen3-8B one replica
# fits per card: TP1 x DP8. For Qwen3-32B (bf16 ~65 GB > 64 GB card) set
# VLLM_TP=2 and DP is recomputed to 4 automatically (DP*TP == cards/node).
NUM_VISIBLE_NPUS=$(awk -F, '{print NF}' <<< "$ASCEND_RT_VISIBLE_DEVICES")
VLLM_TP=${VLLM_TP:-1}
VLLM_DP=${VLLM_DP:-$(( NUM_VISIBLE_NPUS / VLLM_TP ))}   # DP*TP must equal cards/node

# Training processes per node (one per NPU).
NPROC_PER_NODE=${NPROC_PER_NODE:-$NUM_VISIBLE_NPUS}

# DSpark parameters (full verifier vocab; see the single-node script for the
# pruned-vocab variant). MAX_ANCHORS is the main activation-memory knob.
SPECULATOR_TYPE=${SPECULATOR_TYPE:-dspark}
BLOCK_SIZE=${BLOCK_SIZE:-8}
MAX_ANCHORS=${MAX_ANCHORS:-1024}
NUM_LAYERS=${NUM_LAYERS:-5}
TARGET_LAYER_IDS=${TARGET_LAYER_IDS:-"2 18 33"}  # Qwen3-8B (36 layers); adjust per model
MARKOV_RANK=${MARKOV_RANK:-256}
MARKOV_HEAD_TYPE=${MARKOV_HEAD_TYPE:-vanilla}
LOSS_FN=${LOSS_FN:-'{"ce": 0.1, "tv": 0.9}'}
CONFIDENCE_HEAD_ALPHA=${CONFIDENCE_HEAD_ALPHA:-1.0}
# OPTIMIZER_ARGS="--optimizer adamw"   # fallback if muon misbehaves
OPTIMIZER_ARGS=${OPTIMIZER_ARGS:-""}
# =======================================

PREPARE_MARKER="$OUTPUT_DIR/.prepare_complete"
GEN_MARKER_DIR="$HIDDEN_STATES_DIR/.gen_markers"
GEN_MARKER="$GEN_MARKER_DIR/rank${NODE_RANK}.of${NNODES}"
mkdir -p "$OUTPUT_DIR" "$GEN_MARKER_DIR"

wait_for_file() {  # wait_for_file <path> <what>
    local waited=0
    while [ ! -f "$1" ]; do
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 60)) -eq 0 ]; then
            echo "[node $NODE_RANK] waiting for $2 ($1, ${waited}s elapsed)..."
        fi
    done
}

echo "[node $NODE_RANK/$NNODES] master=$MASTER_ADDR:$MASTER_PORT model=$MODEL"

# ---------- Phase 1: data preparation (node 0 only) ----------
if [ "$NODE_RANK" -eq 0 ]; then
    if [ -f "$PREPARE_MARKER" ] && [ "${FORCE_REGEN:-0}" != "1" ]; then
        echo "=== Phase 1: data already prepared, skipping ==="
    else
        echo "=== Phase 1: preparing data (node 0) ==="
        rm -f "$PREPARE_MARKER"
        python scripts/prepare_data.py \
            --model "$MODEL" \
            --data "$DATASET" \
            --output "$OUTPUT_DIR" \
            --max-samples "$MAX_SAMPLES" \
            --seq-length "$SEQ_LENGTH"
        touch "$PREPARE_MARKER"
    fi
else
    echo "=== Phase 1: waiting for node 0 to prepare data ==="
    wait_for_file "$PREPARE_MARKER" "data preparation"
fi

# ---------- Phase 2: hidden-states generation (every node, own shard) ----------
if [ -f "$GEN_MARKER" ] && [ "${FORCE_REGEN:-0}" != "1" ]; then
    echo "=== Phase 2: this node's shard already generated, skipping ==="
else
    rm -f "$GEN_MARKER"
    echo "=== Phase 2: launching local vLLM (TP=$VLLM_TP x DP=$VLLM_DP) ==="
    python scripts/launch_vllm.py "$MODEL" \
        --hidden-states-path "$HIDDEN_STATES_DIR" \
        --target-layer-ids $TARGET_LAYER_IDS \
        -- --tensor-parallel-size "$VLLM_TP" --data-parallel-size "$VLLM_DP" \
           --port "$VLLM_PORT" &
    VLLM_PID=$!
    cleanup() {
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    }
    trap cleanup EXIT

    echo "Waiting for vLLM server to be ready..."
    WAITED=0
    until curl -sf --noproxy '*' "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM exited before becoming healthy; see output above." >&2
            exit 1
        fi
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $((WAITED % 60)) -eq 0 ]; then
            echo "still waiting for vLLM health (${WAITED}s elapsed)..."
        fi
    done

    echo "=== Phase 2: generating shard $NODE_RANK/$NNODES ==="
    python scripts/data_generation_offline.py \
        --preprocessed-data "$OUTPUT_DIR" \
        --endpoint "http://localhost:${VLLM_PORT}/v1" \
        --output "$HIDDEN_STATES_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --concurrency "$CONCURRENCY" \
        --world-size "$NNODES" \
        --rank "$NODE_RANK" \
        --validate-outputs

    echo "=== Phase 2: stopping vLLM ==="
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    trap - EXIT
    touch "$GEN_MARKER"
fi

echo "=== Phase 2: waiting for all $NNODES generation shards ==="
for ((i = 0; i < NNODES; i++)); do
    wait_for_file "$GEN_MARKER_DIR/rank${i}.of${NNODES}" "generation shard $i"
done

# ---------- Phase 3: multi-node training (DDP via torchrun) ----------
echo "=== Phase 3: training ($NNODES nodes x $NPROC_PER_NODE ranks) ==="
torchrun \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --nproc_per_node "$NPROC_PER_NODE" \
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

echo "[node $NODE_RANK] Done. Checkpoints: $OUTPUT_DIR/checkpoints/"
if [ "$NODE_RANK" -eq 0 ]; then
    echo "Recommended: python scripts/check_norm_canary.py $OUTPUT_DIR/checkpoints"
fi
