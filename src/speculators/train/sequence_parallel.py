"""Ulysses sequence-parallel (USP) primitives for DFlash/DSpark draft training.

Ulysses SP shards the *sequence* dimension across an SP group. Inside attention
it switches to *head* sharding via an all-to-all, so each rank attends over the
full (gathered) sequence with a subset of heads, then switches back. This lets a
long sequence be split across ``sp_size`` ranks, cutting the per-rank activation
memory (input hidden, verifier logits, anchors, ...) by ~``sp_size``.

See ``my_docs/2026-07-22_usp适配文档.md``. The all-to-all follows the standard
Ulysses pattern (DeepSpeed / verl ``utils/ulysses.py``, which specforge adapts).

Stage 1 (this module): the collective primitives + autograd wrapper only. The
attention integration, sequence-sharded dataloader, DSpark global-position
gather and SP loss reduction are separate stages. With ``sp_size == 1`` (the
default) every function here is the identity, so default training is unchanged.
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


def _group_size(group: ProcessGroup | None) -> int:
    if group is None:
        return 1
    return dist.get_world_size(group)


def _all_to_all_tensor(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: ProcessGroup | None,
) -> torch.Tensor:
    """Split ``x`` along ``scatter_dim`` into ``world_size`` equal chunks, all-to-all
    them across ``group``, and concatenate the received chunks along ``gather_dim``.

    Equal-sized chunks are required by ``dist.all_to_all``; the scattered dim must
    be divisible by the group size.
    """
    world_size = _group_size(group)
    if world_size == 1:
        return x
    if x.size(scatter_dim) % world_size != 0:
        raise ValueError(
            f"scatter_dim {scatter_dim} size ({x.size(scatter_dim)}) must be "
            f"divisible by sp world_size ({world_size})"
        )
    inputs = [t.contiguous() for t in x.chunk(world_size, dim=scatter_dim)]
    outputs = [torch.empty_like(inputs[0]) for _ in range(world_size)]
    dist.all_to_all(outputs, inputs, group=group)
    return torch.cat(outputs, dim=gather_dim).contiguous()


class _SeqAllToAll(torch.autograd.Function):
    """Autograd-aware all-to-all. Backward is the same all-to-all with the
    scatter/gather dims swapped."""

    @staticmethod
    def forward(ctx, group, x, scatter_dim, gather_dim):  # type: ignore[override]
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _all_to_all_tensor(x, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        grad_input = _all_to_all_tensor(
            grad_output, ctx.gather_dim, ctx.scatter_dim, ctx.group
        )
        return (None, grad_input, None, None)


def all_to_all(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: ProcessGroup | None,
) -> torch.Tensor:
    """Autograd-aware all-to-all; identity when the group has a single rank."""
    if _group_size(group) == 1:
        return x
    return _SeqAllToAll.apply(group, x, scatter_dim, gather_dim)


def gather_seq_scatter_heads(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    group: ProcessGroup | None,
) -> torch.Tensor:
    """Enter attention: ``[..., s_local (full heads), ...]`` -> ``[..., s_full
    (heads sharded), ...]``. Gathers the sequence, scatters the heads.

    ``x`` is typically ``[batch, seq_local, num_heads, head_dim]`` with
    ``seq_dim=1, head_dim=2``. Requires ``num_heads % sp_size == 0``.
    """
    return all_to_all(x, scatter_dim=head_dim, gather_dim=seq_dim, group=group)


def scatter_seq_gather_heads(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    group: ProcessGroup | None,
) -> torch.Tensor:
    """Exit attention (inverse of :func:`gather_seq_scatter_heads`): ``[..., s_full
    (heads sharded), ...]`` -> ``[..., s_local (full heads), ...]``. Scatters the
    sequence, gathers the heads."""
    return all_to_all(x, scatter_dim=seq_dim, gather_dim=head_dim, group=group)
