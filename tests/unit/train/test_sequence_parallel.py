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
    anchor_loss_scale,
    gather_seq_scatter_heads,
    scatter_seq_gather_heads,
    shard_anchors,
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


V_GLOBAL = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # 6 "anchor positions"
# Uneven split so anchor_loss_scale's count weighting is actually exercised.
_SPLIT = {0: slice(0, 4), 1: slice(4, 6)}


def _loss_worker(rank: int, world_size: int, result_path: str):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29608",
        RANK=str(rank), WORLD_SIZE=str(world_size),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    try:
        w = torch.tensor([3.0], requires_grad=True)
        v = V_GLOBAL[_SPLIT[rank]]
        per_pos = (w - v) ** 2
        local_count = torch.tensor(float(v.numel()))
        loss = per_pos.sum() / local_count  # mean over this rank's positions

        scale = anchor_loss_scale(local_count, group)
        (loss * scale).backward()

        # simulate DDP: average grads over the world
        g = w.grad.clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        g /= world_size

        if rank == 0:
            # single-rank reference over ALL positions
            wr = torch.tensor([3.0], requires_grad=True)
            (((wr - V_GLOBAL) ** 2).mean()).backward()
            ok = torch.allclose(g, wr.grad, atol=1e-6)
            torch.save({"ok": bool(ok), "sharded": g.item(), "single": wr.grad.item()}, result_path)
    finally:
        dist.destroy_process_group()


def _shard_worker(rank: int, world_size: int, result_path: str):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29609",
        RANK=str(rank), WORLD_SIZE=str(world_size),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    try:
        # each rank draws a DIFFERENT random set; shard_anchors must force agreement
        # on rank 0's draw, then stride -- so union == rank0's full set, disjoint.
        torch.manual_seed(rank)
        pos = torch.randperm(8)[:6].contiguous()
        valid = torch.ones(6, dtype=torch.bool)
        local_pos, _ = shard_anchors(pos.clone(), valid.clone(), group, rank)
        gathered = [torch.zeros_like(local_pos) for _ in range(world_size)]
        dist.all_gather(gathered, local_pos)
        if rank == 0:
            union = torch.cat(gathered).sort().values
            # rank0's own broadcast draw:
            torch.manual_seed(0)
            expect = torch.randperm(8)[:6].sort().values
            ok = torch.equal(union, expect) and len(torch.unique(torch.cat(gathered))) == 6
            torch.save({"ok": bool(ok)}, result_path)
    finally:
        dist.destroy_process_group()


def test_sp1_identity():
    """sp_size==1 (group=None) is the identity -- default path unchanged."""
    x = torch.randn(1, 5, 4, 2, requires_grad=True)
    assert torch.equal(gather_seq_scatter_heads(x, 1, 2, None), x)
    assert torch.equal(scatter_seq_gather_heads(x, 1, 2, None), x)
    assert torch.equal(all_to_all(x, 2, 1, None), x)
    assert anchor_loss_scale(torch.tensor(7.0), None) == 1.0


def test_two_rank_all_to_all():
    mp.spawn(_worker, args=(2,), nprocs=2, join=True)


def _spawn_and_check(fn, label):
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "r.pt")
    mp.spawn(fn, args=(2, p), nprocs=2, join=True)
    r = torch.load(p, weights_only=False)
    assert r["ok"], f"{label} FAILED: {r}"
    print(f"PASS: {label}", {k: v for k, v in r.items() if k != "ok"})


if __name__ == "__main__":
    test_sp1_identity()
    print("PASS: sp_size=1 identity (incl anchor_loss_scale==1)")
    test_two_rank_all_to_all()
    _spawn_and_check(_loss_worker, "anchor loss-scale grad == single-rank")
    _spawn_and_check(_shard_worker, "shard_anchors union == rank0 draw, disjoint")
