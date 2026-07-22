"""Unit tests for Ulysses sequence-parallel primitives (Stage 1).

Runs a 2-rank gloo group on CPU via ``mp.spawn`` and checks:
  * ``gather_seq_scatter_heads`` produces the full sequence with sharded heads,
    with the exact expected content;
  * ``scatter_seq_gather_heads`` inverts it (round-trip identity);
  * the all-to-all is autograd-correct (gradients round-trip in shape/content);
  * ``sp_size == 1`` (group=None) is the identity, so default training is
    unchanged.
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from speculators.train.sequence_parallel import (
    all_to_all,
    gather_seq_scatter_heads,
    scatter_seq_gather_heads,
)

B, S_LOCAL, H, D = 1, 3, 4, 2  # sp_size=2 -> full seq 6, heads/rank 2


def _encode(rank: int) -> torch.Tensor:
    # value = rank*1000 + seq*100 + head*10 + d  (unique, decodable)
    x = torch.zeros(B, S_LOCAL, H, D)
    for s in range(S_LOCAL):
        for h in range(H):
            for d in range(D):
                x[0, s, h, d] = rank * 1000 + s * 100 + h * 10 + d
    return x


def _worker(rank: int, world_size: int):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29607",
        RANK=str(rank), WORLD_SIZE=str(world_size),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    try:
        x = _encode(rank).requires_grad_(True)

        # enter attention: full seq, sharded heads
        g = gather_seq_scatter_heads(x, seq_dim=1, head_dim=2, group=group)
        assert g.shape == (B, S_LOCAL * world_size, H // world_size, D), g.shape

        # content check: pos p -> origin_rank=p//S_LOCAL, local_seq=p%S_LOCAL;
        # this rank holds head shard [rank*(H//ws) : (rank+1)*(H//ws)]
        hs = H // world_size
        for p in range(S_LOCAL * world_size):
            orank, lseq = divmod(p, S_LOCAL)
            for hloc in range(hs):
                hglob = rank * hs + hloc
                for d in range(D):
                    exp = orank * 1000 + lseq * 100 + hglob * 10 + d
                    assert g[0, p, hloc, d].item() == exp, (p, hloc, d, g[0, p, hloc, d].item(), exp)

        # exit attention: must invert exactly
        back = scatter_seq_gather_heads(g, seq_dim=1, head_dim=2, group=group)
        assert torch.equal(back, x.detach()), "round-trip not identity"

        # autograd: grad of sum flows back to a full-ones grad on x
        back.sum().backward()
        assert x.grad is not None and torch.equal(x.grad, torch.ones_like(x)), "bad grad"

        if rank == 0:
            print("PASS: 2-rank gather/scatter round-trip + content + autograd")
    finally:
        dist.destroy_process_group()


def test_sp1_identity():
    """sp_size==1 (group=None) is the identity -- default path unchanged."""
    x = torch.randn(1, 5, 4, 2, requires_grad=True)
    assert torch.equal(gather_seq_scatter_heads(x, 1, 2, None), x)
    assert torch.equal(scatter_seq_gather_heads(x, 1, 2, None), x)
    assert torch.equal(all_to_all(x, 2, 1, None), x)


def test_two_rank_all_to_all():
    mp.spawn(_worker, args=(2,), nprocs=2, join=True)


if __name__ == "__main__":
    test_sp1_identity()
    print("PASS: sp_size=1 identity")
    test_two_rank_all_to_all()
