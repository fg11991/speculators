import pytest

from speculators.train.distributed import resolve_hsdp_mesh_dims


@pytest.mark.parametrize(
    ("world_size", "shard_size", "expected"),
    [
        (16, 8, (2, 8)),  # 2 nodes x 8 cards: shard within node
        (64, 8, (8, 8)),  # 8 nodes x 8 cards
        (64, 16, (4, 16)),  # widen shard group to 2 nodes (DeepSpec-style)
        (8, 8, (1, 8)),  # single node: degenerates to plain full shard
        (4, 2, (2, 2)),
    ],
)
def test_resolve_hsdp_mesh_dims(world_size, shard_size, expected):
    assert resolve_hsdp_mesh_dims(world_size, shard_size) == expected


def test_shard_size_must_divide_world_size():
    with pytest.raises(ValueError, match="divisible"):
        resolve_hsdp_mesh_dims(16, 6)


def test_shard_size_cannot_exceed_world_size():
    with pytest.raises(ValueError, match="cannot exceed"):
        resolve_hsdp_mesh_dims(8, 16)


@pytest.mark.parametrize("shard_size", [0, 1, -8])
def test_shard_size_below_two_rejected(shard_size):
    # shard_size 1 would mean fully replicated params, i.e. DDP -- reject and
    # point users at the DDP path instead.
    with pytest.raises(ValueError, match=">= 2"):
        resolve_hsdp_mesh_dims(8, shard_size)
