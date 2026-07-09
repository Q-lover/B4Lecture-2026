import torch
from nf_assignment.toy.data import TwoMoons
from nf_assignment.toy.model import build_realnvp_2d
from nf_assignment.toy.sample import sample_model
from nf_assignment.toy.train import train_forward_kld


def test_toy_forward_kld_training_smoke() -> None:
    torch.manual_seed(0)
    target = TwoMoons()
    model = build_realnvp_2d(num_layers=2, hidden_dims=(8,), target=target)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = train_forward_kld(
        model,
        optimizer,
        target,
        batch_size=32,
        num_steps=3,
        log_every=1,
    )
    samples = sample_model(model, 16)

    assert len(history) == 3
    assert all(torch.isfinite(torch.tensor(item["loss"])) for item in history)
    assert samples.shape == (16, 2)
    assert torch.isfinite(samples).all()
