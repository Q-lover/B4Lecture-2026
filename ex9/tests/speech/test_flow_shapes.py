"""Speech flow shape and reversibility tests."""

import torch
from nf_assignment.flows.coupling import AffineCouplingTransform, SequenceAffineCoupling
from nf_assignment.flows.normalization import ActNorm
from nf_assignment.flows.permutation import InvConvNear
from nf_assignment.networks.wavenet import WaveNetConditioner
from nf_assignment.speech.model import ConditionalSequenceFlow, build_speech_flow


def _sequence_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    steps = torch.arange(max_length).view(1, 1, max_length)
    return (steps < lengths.view(-1, 1, 1)).to(dtype=torch.float32)


def test_speech_flow_reconstructs_hubert_conditioned_features() -> None:
    torch.manual_seed(0)
    model = build_speech_flow(
        coded_sp_channels=48,
        condition_channels=256,
        hidden_channels=16,
        num_blocks=2,
        num_layers_per_block=2,
        kernel_size=3,
        dilation_rate=2,
    )
    x = torch.randn(3, 48, 13)
    condition = torch.randn(3, 256, 13)
    mask = _sequence_mask(torch.tensor([13, 11, 9]), 13)
    x = x * mask

    z, log_det = model(x, condition=condition, mask=mask)
    x_reconstructed, inverse_log_det = model.inverse(z, condition=condition, mask=mask)

    assert isinstance(model, ConditionalSequenceFlow)
    assert z.shape == x.shape
    assert log_det.shape == (3,)
    assert torch.isfinite(log_det).all()
    assert torch.allclose(x_reconstructed, x, atol=1e-5)
    assert torch.allclose(
        log_det + inverse_log_det, torch.zeros_like(log_det), atol=1e-5
    )
    assert torch.allclose(z * (1.0 - mask), torch.zeros_like(z))


def test_speech_flow_accepts_ppg_condition_channels() -> None:
    torch.manual_seed(1)
    model = build_speech_flow(
        coded_sp_channels=48,
        condition_channels=40,
        hidden_channels=16,
        num_blocks=1,
        num_layers_per_block=2,
        kernel_size=3,
        dilation_rate=2,
    )
    x = torch.randn(2, 48, 7)
    condition = torch.randn(2, 40, 7)

    z, log_det = model(x, condition=condition)
    x_reconstructed, inverse_log_det = model.inverse(z, condition=condition)

    assert z.shape == (2, 48, 7)
    assert log_det.shape == (2,)
    assert torch.allclose(x_reconstructed, x, atol=1e-5)
    assert torch.allclose(
        log_det + inverse_log_det, torch.zeros_like(log_det), atol=1e-5
    )


def test_actnorm_data_dependent_init_reconstructs_masked_input() -> None:
    torch.manual_seed(2)
    layer = ActNorm(4, data_dep_init=True)
    x = torch.randn(2, 4, 5)
    mask = _sequence_mask(torch.tensor([5, 3]), 5)
    x = x * mask

    z, log_det = layer(x, mask=mask)
    x_reconstructed, inverse_log_det = layer.inverse(z, mask=mask)

    assert layer.initialized
    assert torch.allclose(x_reconstructed, x, atol=1e-5)
    assert torch.allclose(
        log_det + inverse_log_det, torch.zeros_like(log_det), atol=1e-5
    )


def test_zero_initialized_sequence_coupling_starts_as_identity() -> None:
    torch.manual_seed(3)
    conditioner = WaveNetConditioner(
        4,
        8,
        hidden_channels=8,
        kernel_size=3,
        dilation_rate=2,
        num_layers=2,
        condition_channels=6,
        zero_init=True,
    )
    coupling = SequenceAffineCoupling(8, conditioner)
    x = torch.randn(2, 8, 6)
    condition = torch.randn(2, 6, 6)

    z, log_det = coupling(x, condition=condition)

    assert torch.allclose(z, x, atol=1e-6)
    assert torch.allclose(log_det, torch.zeros_like(log_det))
    assert isinstance(coupling.transform, AffineCouplingTransform)


def test_masked_sequence_transforms_do_not_depend_on_batch_max_length() -> None:
    torch.manual_seed(4)
    length = 5
    x_valid = torch.randn(1, 8, length)
    condition_valid = torch.randn(1, 6, length)
    transforms = [
        ActNorm(8, data_dep_init=False),
        InvConvNear(8, n_split=4),
        SequenceAffineCoupling(
            8,
            WaveNetConditioner(
                4,
                8,
                hidden_channels=8,
                kernel_size=3,
                dilation_rate=2,
                num_layers=3,
                condition_channels=6,
                dropout=0.0,
                zero_init=False,
            ),
        ),
    ]
    for transform in transforms:
        transform.eval()

    outputs_by_max_length = []
    log_dets_by_max_length = []
    for max_length, padding_scale in [(7, 10.0), (13, -7.0)]:
        x = torch.randn(1, 8, max_length) * padding_scale
        x[:, :, :length] = x_valid
        condition = torch.randn(1, 6, max_length) * -padding_scale
        condition[:, :, :length] = condition_valid
        mask = _sequence_mask(torch.tensor([length]), max_length)

        layer_outputs = []
        layer_log_dets = []
        hidden = x
        for transform in transforms:
            hidden, log_det = transform(hidden, mask=mask, condition=condition)
            layer_outputs.append(hidden[:, :, :length])
            layer_log_dets.append(log_det)
        outputs_by_max_length.append(layer_outputs)
        log_dets_by_max_length.append(layer_log_dets)

    for short_output, long_output in zip(
        outputs_by_max_length[0], outputs_by_max_length[1], strict=True
    ):
        assert torch.allclose(short_output, long_output, atol=1e-6)
    for short_log_det, long_log_det in zip(
        log_dets_by_max_length[0], log_dets_by_max_length[1], strict=True
    ):
        assert torch.allclose(short_log_det, long_log_det, atol=1e-6)
