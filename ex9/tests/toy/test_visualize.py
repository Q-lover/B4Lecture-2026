import torch
from nf_assignment.toy.visualize import plot_density_heatmaps, plot_warped_base_grid


def test_plot_density_heatmaps_writes_png(tmp_path) -> None:
    def target_log_prob(points: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(points.pow(2), dim=1)

    def model_log_prob(points: torch.Tensor) -> torch.Tensor:
        shifted = points - torch.tensor([0.5, -0.25], device=points.device)
        return -0.5 * torch.sum(shifted.pow(2), dim=1)

    output_path = tmp_path / "density_heatmap.png"

    plot_density_heatmaps(
        target_log_prob,
        model_log_prob,
        output_path,
        grid_size=16,
        chunk_size=17,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_warped_base_grid_writes_png(tmp_path) -> None:
    def model_log_prob(points: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(points.pow(2), dim=1)

    output_path = tmp_path / "warped_grid.png"

    plot_warped_base_grid(
        model_log_prob,
        model_log_prob,
        lambda points: points,
        output_path,
        density_grid_size=16,
        base_grid_size=5,
        line_points=11,
        chunk_size=17,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
