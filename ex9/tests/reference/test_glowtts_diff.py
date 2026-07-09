"""Optional Glow-TTS reference-diff tests for speech flow components."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from nf_assignment.flows.coupling import SequenceAffineCoupling
from nf_assignment.flows.normalization import ActNorm
from nf_assignment.flows.permutation import InvConvNear
from nf_assignment.networks.wavenet import WaveNetConditioner

REPOS_ROOT = Path(__file__).resolve().parents[3]
GLOW_TTS_ROOT = REPOS_ROOT / "glow-tts"


pytestmark = pytest.mark.skipif(
    not (GLOW_TTS_ROOT / "modules.py").is_file(),
    reason="repos/glow-tts is not available",
)


def _import_glow_tts():
    sys.path.insert(0, str(GLOW_TTS_ROOT))
    import attentions  # noqa: PLC0415
    import modules  # noqa: PLC0415

    return attentions, modules


def test_actnorm_matches_glow_tts_fixed_parameters() -> None:
    _, modules = _import_glow_tts()
    reference = modules.ActNorm(4)
    ours = ActNorm(4)
    with torch.no_grad():
        reference.logs.copy_(torch.tensor([[[0.1], [-0.2], [0.0], [0.3]]]))
        reference.bias.copy_(torch.tensor([[[0.5], [-0.1], [0.2], [0.0]]]))
        ours.logs.copy_(reference.logs)
        ours.bias.copy_(reference.bias)
    x = torch.randn(3, 4, 5)
    mask = torch.ones(3, 1, 5)

    ref_z, ref_log_det = reference(x, mask, reverse=False)
    z, log_det = ours(x, mask=mask)
    ref_x, _ = reference(ref_z, mask, reverse=True)
    x_reconstructed, inverse_log_det = ours.inverse(z, mask=mask)

    assert torch.allclose(z, ref_z)
    assert torch.allclose(log_det, ref_log_det)
    assert torch.allclose(x_reconstructed, ref_x)
    assert torch.allclose(log_det + inverse_log_det, torch.zeros_like(log_det))


def test_inv_conv_near_matches_glow_tts_fixed_parameters() -> None:
    _, modules = _import_glow_tts()
    torch.manual_seed(4)
    reference = modules.InvConvNear(8, n_split=4)
    ours = InvConvNear(8, n_split=4)
    with torch.no_grad():
        ours.weight.copy_(reference.weight)
    x = torch.randn(2, 8, 6)
    mask = torch.ones(2, 1, 6)

    ref_z, ref_log_det = reference(x, mask, reverse=False)
    z, log_det = ours(x, mask=mask)
    ref_x, _ = reference(ref_z, mask, reverse=True)
    x_reconstructed, inverse_log_det = ours.inverse(z, mask=mask)

    assert torch.allclose(z, ref_z, atol=1e-6)
    assert torch.allclose(log_det, ref_log_det, atol=1e-6)
    assert torch.allclose(x_reconstructed, ref_x, atol=1e-6)
    assert torch.allclose(
        log_det + inverse_log_det, torch.zeros_like(log_det), atol=1e-6
    )


def test_sequence_coupling_matches_glow_tts_fixed_parameters() -> None:
    attentions, _ = _import_glow_tts()
    torch.manual_seed(5)
    reference = attentions.CouplingBlock(
        8,
        hidden_channels=4,
        kernel_size=3,
        dilation_rate=2,
        n_layers=2,
        gin_channels=6,
        p_dropout=0.0,
    )
    conditioner = WaveNetConditioner(
        4,
        8,
        hidden_channels=4,
        kernel_size=3,
        dilation_rate=2,
        num_layers=2,
        condition_channels=6,
        dropout=0.0,
        zero_init=False,
    )
    conditioner.load_state_dict(reference.state_dict())
    ours = SequenceAffineCoupling(8, conditioner)
    x = torch.randn(2, 8, 7)
    condition = torch.randn(2, 6, 7)
    mask = torch.ones(2, 1, 7)

    ref_z, ref_log_det = reference(x, mask, reverse=False, g=condition)
    z, log_det = ours(x, mask=mask, condition=condition)
    ref_x, _ = reference(ref_z, mask, reverse=True, g=condition)
    x_reconstructed, inverse_log_det = ours.inverse(z, mask=mask, condition=condition)

    assert torch.allclose(z, ref_z, atol=1e-6)
    assert torch.allclose(log_det, ref_log_det, atol=1e-6)
    assert torch.allclose(x_reconstructed, ref_x, atol=1e-6)
    assert torch.allclose(
        log_det + inverse_log_det, torch.zeros_like(log_det), atol=1e-6
    )
