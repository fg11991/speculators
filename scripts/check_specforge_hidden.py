"""Validate a SpecForge DFlash hidden-states file before a bulk training run.

Two checks on ONE real SpecForge DFlash sample (data_i.ckpt, generated with
``--model-type dflash --target-model-backend hf``), before committing to
``--legacy-data --legacy-data-format specforge``:

1. SHAPE/KEY: the file goes through ``standardize_data_specforge`` and produces
   the fields the trainer expects; the fc-input width must be
   ``num_target_layers * hidden``.

2. NORM CONVENTION (the one thing source review can't prove): speculators applies
   ``verifier_lm_head(verifier_norm(verifier_last_hidden_states))`` to rebuild the
   target logits, so ``last_hidden_state`` must be the PRE-final-norm state. We
   rebuild logits from the verifier's own norm+lm_head and measure how often
   argmax predicts the next input token (teacher-forcing next-token accuracy).
   A good verifier scores high on natural text; a near-zero score means the
   stored hidden is post-norm (or mismatched) -- fix generation or skip
   verifier_norm for this data, do NOT bulk-train on it.

Usage:
    python scripts/check_specforge_hidden.py <sample.ckpt> \
        --verifier-name-or-path Qwen/Qwen3-8B --num-target-layers 3
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM

from speculators.train.data import standardize_data_specforge

# A weak sanity floor. Real natural-text teacher-forcing accuracy is usually far
# higher; this only needs to separate "sensible logits" from "garbage".
_MIN_NEXT_TOKEN_ACC = 0.3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sample", help="One SpecForge DFlash data_i.ckpt (uncompressed)")
    ap.add_argument("--verifier-name-or-path", required=True)
    ap.add_argument(
        "--num-target-layers",
        type=int,
        required=True,
        help="len(--target-layer-ids) used in training; fc input width = this * hidden",
    )
    args = ap.parse_args()

    raw = torch.load(args.sample, weights_only=True, map_location="cpu")
    print(f"raw keys: {sorted(raw)}")

    std = standardize_data_specforge(raw)
    hs = std["hidden_states"]
    last = std["verifier_last_hidden_states"]
    input_ids = std["input_ids"].long()
    seq, hidden = last.shape
    fc_width = hs.shape[-1]
    print(f"seq={seq} hidden={hidden} fc_width={fc_width}")

    # Check 1: shapes/keys
    ok_shape = (
        hs.shape[0] == seq
        and input_ids.shape[0] == seq
        and fc_width == args.num_target_layers * hidden
    )
    print(
        f"[1] shape/key: {'OK' if ok_shape else 'FAIL'} "
        f"(expected fc_width={args.num_target_layers * hidden})"
    )

    # Check 2: norm convention -- rebuild logits with the verifier's own head.
    model = AutoModelForCausalLM.from_pretrained(
        args.verifier_name_or_path, torch_dtype=torch.float32
    ).eval()
    norm = model.model.norm
    lm_head = model.lm_head
    with torch.no_grad():
        logits = lm_head(norm(last.float()))  # [seq, vocab]
        pred = logits[:-1].argmax(-1)
        nxt = input_ids[1:]
        mask = std["loss_mask"][1:].bool()
        acc = (pred[mask] == nxt[mask]).float().mean().item()
    ok_norm = acc >= _MIN_NEXT_TOKEN_ACC
    print(
        f"[2] norm convention: next-token acc={acc:.3f} -> "
        f"{'OK (pre-norm)' if ok_norm else 'FAIL (likely post-norm/mismatch)'}"
    )

    if ok_shape and ok_norm:
        print("PASS: safe to bulk-train with --legacy-data-format specforge")
        return 0
    print("DO NOT bulk-train yet -- fix the failing check above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
