import torch

from speculators.train.data import (
    LEGACY_STANDARDIZE_FNS,
    standardize_data_specforge,
    standardize_data_v1,
)

SEQ, HID, NUM_LAYERS = 7, 4, 3


def _dflash_sample():
    # SpecForge DFlash prepare_hidden_states (--model-type dflash
    # --target-model-backend hf) DataPoint: hidden_state = K captured layers
    # concatenated [1, seq, K*hid]; last_hidden_state = verifier final [1, seq, hid];
    # input_ids/loss_mask 1D; aux_hidden_state None.
    return {
        "input_ids": torch.arange(SEQ),
        "loss_mask": torch.ones(SEQ, dtype=torch.bool),
        "hidden_state": torch.randn(1, SEQ, NUM_LAYERS * HID),
        "aux_hidden_state": None,
        "last_hidden_state": torch.randn(1, SEQ, HID),
    }


def test_specforge_dflash_maps_to_v1_final_dict():
    out = standardize_data_specforge(_dflash_sample())
    assert set(out) == {
        "hidden_states",
        "input_ids",
        "verifier_last_hidden_states",
        "loss_mask",
    }
    assert out["hidden_states"].shape == (SEQ, NUM_LAYERS * HID)  # fc input
    assert out["verifier_last_hidden_states"].shape == (SEQ, HID)
    assert out["input_ids"].shape == (SEQ,)
    assert out["loss_mask"].shape == (SEQ,)


def test_specforge_dflash_matches_v1_layout():
    # A v1 sample built from the same per-layer tensors must yield an identical
    # standardized dict, proving the two formats reconcile.
    aux_layers = [torch.randn(SEQ, HID) for _ in range(NUM_LAYERS)]
    last = torch.randn(SEQ, HID)
    input_ids = torch.arange(SEQ)
    loss_mask = torch.ones(SEQ, dtype=torch.bool)

    v1 = standardize_data_v1(
        {
            "hidden_states": [*aux_layers, last],
            "input_ids": input_ids,
            "loss_mask": loss_mask,
        }
    )
    sf = standardize_data_specforge(
        {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "hidden_state": torch.cat(aux_layers, dim=-1).unsqueeze(0),
            "aux_hidden_state": None,
            "last_hidden_state": last.unsqueeze(0),
        }
    )
    for key in v1:
        assert torch.equal(v1[key], sf[key]), key


def test_registry_exposes_both():
    assert LEGACY_STANDARDIZE_FNS["v1"] is standardize_data_v1
    assert LEGACY_STANDARDIZE_FNS["specforge"] is standardize_data_specforge
