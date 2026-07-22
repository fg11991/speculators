"""Single source of truth for distributed training topology.

Stores local_rank, SP/DP sizes and ranks, and process groups.
All other modules should import getters from here rather than
maintaining their own distributed state.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

logger = logging.getLogger("speculators")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_local_rank: int = 0
_rank: int = 0
_world_size: int = 1
_is_distributed: bool = False

_sp_size: int = 1
_sp_rank: int = 0
_dp_size: int = 1
_dp_rank: int = 0

_sp_group: ProcessGroup | None = None
_dp_group: ProcessGroup | None = None


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------


def get_local_rank() -> int:
    return _local_rank


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _world_size


def is_distributed() -> bool:
    return _is_distributed


def get_sp_group() -> ProcessGroup | None:
    return _sp_group


def get_dp_group() -> ProcessGroup | None:
    return _dp_group


def get_sp_size() -> int:
    return _sp_size


def get_sp_rank() -> int:
    return _sp_rank


def get_dp_size() -> int:
    return _dp_size


def get_dp_rank() -> int:
    return _dp_rank


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _init_sp_process_groups(rank: int, world_size: int, sp_size: int) -> None:
    """Initialize sequence-parallel and data-parallel process groups.

    SP groups use contiguous ranks (e.g. sp_size=2, world_size=4: {0,1}, {2,3}).
    DP groups use strided ranks (e.g. sp_size=2, world_size=4: {0,2}, {1,3}).
    """
    global _sp_group, _dp_group, _sp_size, _sp_rank, _dp_size, _dp_rank  # noqa: PLW0603

    if sp_size <= 0:
        raise ValueError(f"sp_size must be positive, got {sp_size}")

    if world_size % sp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by sp_size ({sp_size})"
        )

    dp_size = world_size // sp_size

    sp_group = None
    for i in range(dp_size):
        sp_ranks = list(range(i * sp_size, (i + 1) * sp_size))
        pg = dist.new_group(sp_ranks)
        if rank in sp_ranks:
            sp_group = pg

    dp_group = None
    for i in range(sp_size):
        dp_ranks = list(range(i, world_size, sp_size))
        pg = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            dp_group = pg

    if sp_group is None or dp_group is None:
        raise RuntimeError("Failed to initialize SP/DP process groups")

    _sp_group = sp_group
    _dp_group = dp_group
    _sp_size = sp_size
    _sp_rank = rank % sp_size
    _dp_size = dp_size
    _dp_rank = rank // sp_size


def maybe_setup_distributed(sp_size: int = 1) -> None:
    """Set up distributed training if launched with ``torchrun``.

    Always populates the module-level topology state so that callers
    can use the getter functions regardless of whether SP is enabled.
    Process groups are always created when distributed — with
    ``sp_size == 1`` the DP group spans all ranks and each SP group
    contains a single rank.
    """
    global _local_rank, _rank, _is_distributed, _world_size  # noqa: PLW0603

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "LOCAL_RANK" in os.environ

    _local_rank = local_rank
    _is_distributed = distributed

    if not distributed:
        return

    torch.accelerator.set_device_index(local_rank)
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        raise ValueError("No accelerator found")
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend, device_id=local_rank)

    _rank = dist.get_rank()
    _world_size = dist.get_world_size()

    _init_sp_process_groups(_rank, _world_size, sp_size)

    logger.info(
        f"Started distributed with local_rank={local_rank}, "
        f"dp_size={_dp_size}, sp_size={_sp_size}",
        extra={"override_rank0_filter": True},
    )


def maybe_destroy_distributed() -> None:
    """Destroy the distributed process group if using distributed training."""
    global _is_distributed, _local_rank, _rank, _world_size  # noqa: PLW0603
    global _sp_size, _sp_rank, _dp_size, _dp_rank  # noqa: PLW0603
    global _sp_group, _dp_group  # noqa: PLW0603

    if not _is_distributed:
        return

    dist.destroy_process_group()
    logger.info(
        "Destroyed distributed process group",
        extra={"override_rank0_filter": True},
    )

    _is_distributed = False
    _local_rank = 0
    _rank = 0
    _world_size = 1
    _sp_size = 1
    _sp_rank = 0
    _dp_size = 1
    _dp_rank = 0
    _sp_group = None
    _dp_group = None


_MIN_SHARD_SIZE = 2  # shard_size 1 = fully replicated params, i.e. DDP


def resolve_hsdp_mesh_dims(world_size: int, shard_size: int) -> tuple[int, int]:
    """Validate an HSDP shard-group width and return (replicate, shard) dims.

    ``shard_size`` is the number of ranks each parameter is sharded across
    (typically the cards per node, so the per-layer all-gather stays on the
    fast intra-node fabric); the remaining ``world_size // shard_size`` groups
    hold replicas and only exchange gradients once per step.
    """
    if shard_size < _MIN_SHARD_SIZE:
        raise ValueError(
            f"fsdp shard_size must be >= {_MIN_SHARD_SIZE}, got {shard_size}. "
            "For fully replicated parameters use DDP (drop --fsdp-shard) instead."
        )
    if shard_size > world_size:
        raise ValueError(
            f"fsdp shard_size ({shard_size}) cannot exceed world_size ({world_size})"
        )
    if world_size % shard_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"fsdp shard_size ({shard_size})"
        )
    return world_size // shard_size, shard_size


def build_hsdp_mesh(shard_size: int) -> DeviceMesh:
    """Build the 2D (replicate, shard) device mesh for HSDP.

    Ranks are laid out row-major, so with torchrun's contiguous per-node
    ranks a ``shard_size`` equal to the cards per node keeps every shard
    group inside one node.
    """
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        raise ValueError("No accelerator found; HSDP requires a device backend")
    replicate, shard = resolve_hsdp_mesh_dims(dist.get_world_size(), shard_size)
    return init_device_mesh(
        acc.type,
        (replicate, shard),
        mesh_dim_names=("replicate", "shard"),
    )


def apply_fully_sharded(
    model: torch.nn.Module,
    param_dtype: torch.dtype = torch.bfloat16,
    hsdp_shard_size: int | None = None,
):
    """Applies torch FSDP fully_shard to the model, wrapping layers in FSDPModule.

    Assumes the model has a `layers` attribute containing the decoder layers.
    Model should be validated with SpeculatorModel.verify_training_compatible()
    before calling this function.

    With ``hsdp_shard_size`` set, parameters are sharded only within groups of
    that many ranks and replicated across groups (HSDP): the per-layer
    parameter all-gather never crosses the inter-node fabric, only the
    once-per-step gradient all-reduce does. Left as ``None``, parameters are
    sharded across all ranks (ZeRO-3).
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
    )
    mesh = build_hsdp_mesh(hsdp_shard_size) if hsdp_shard_size is not None else None

    for layer in model.layers:  # type: ignore[union-attr]
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)

    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    return model
