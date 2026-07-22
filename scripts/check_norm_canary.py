"""Canary check for the FSDP2 mixed-precision defect on trained checkpoints.

Before the fp32-master-weights fix, root-owned parameters (norm layers, fc,
markov/confidence heads) were stored in bf16 and every optimizer update was
rounded away: near 1.0 the bf16 spacing (2^-9 ~= 2e-3) exceeds the typical
per-step update of norm weights (1e-7 ~ 2e-5), freezing them at their init
value of exactly 1.0.

This script scans every *norm* tensor in a checkpoint and reports
max |w - 1|. A value of exactly 0 on a *trained* checkpoint means the
checkpoint was produced by the defective path and should be retrained.
(A freshly initialized, untrained checkpoint also reads 0 -- that is not
a hit.)

Usage:
    python scripts/check_norm_canary.py <checkpoint_dir_or_safetensors> [...]

Exits non-zero if any scanned checkpoint hits the canary.
"""

import argparse
import sys
from pathlib import Path

from safetensors import safe_open


def iter_safetensors(path: Path):
    if path.is_file() and path.suffix == ".safetensors":
        yield path
    elif path.is_dir():
        yield from sorted(path.rglob("*.safetensors"))


def check_file(path: Path, pattern: str) -> bool:
    """Print per-tensor deviations; return True if the canary is hit."""
    frozen = []
    found = False
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open handle is not a dict
            if pattern not in key:
                continue
            found = True
            weight = f.get_tensor(key).float()
            deviation = (weight - 1.0).abs().max().item()
            marker = "  <-- FROZEN AT INIT" if deviation == 0.0 else ""
            print(f"  {key}: max|w-1| = {deviation:.6e}{marker}")
            if deviation == 0.0:
                frozen.append(key)

    if not found:
        print(f"  (no tensors matching '{pattern}')")
        return False
    if frozen:
        print(
            f"  HIT: {len(frozen)} norm tensor(s) exactly at init value 1.0. "
            "If this checkpoint was trained, it was produced by the defective "
            "mixed-precision path."
        )
    return bool(frozen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Checkpoint directories (searched recursively) or .safetensors files",
    )
    parser.add_argument(
        "--pattern",
        default="norm",
        help="Substring selecting the tensors to scan (default: 'norm')",
    )
    args = parser.parse_args()

    any_hit = False
    any_file = False
    for path in args.paths:
        for st_file in iter_safetensors(path):
            any_file = True
            print(f"{st_file}:")
            any_hit |= check_file(st_file, args.pattern)

    if not any_file:
        print("No .safetensors files found under the given paths.", file=sys.stderr)
        return 2
    return 1 if any_hit else 0


if __name__ == "__main__":
    sys.exit(main())
