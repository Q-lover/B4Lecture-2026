# -*- coding: utf-8 -*-
"""Diffusion Model の学習スクリプト。実装済み・変更不要。"""

import logging
import os

import diffusers
import hydra
import torch
import torch.nn as nn
from datasets import load_dataset
from diffusion_skeleton import DiffusionModel
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid


def train_one_epoch(model, loader, optimizer, device):
    """1エポック分の訓練を行い、平均損失を返す。"""
    model.train()
    total = 0.0
    for batch in loader:
        images = batch["images"].to(device)
        loss = model.training_step(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def visualize(model, num_timesteps, num_samples, image_size, writer, epoch):
    """生成画像を TensorBoard に記録する。"""
    model.eval()
    shape = (num_samples[0] * num_samples[1],) + tuple(image_size)
    with torch.no_grad():
        generated = model.generate(num_timesteps, shape)
    generated = (generated + 1) / 2  # [-1, 1] → [0, 1]
    grid = make_grid(generated, nrow=num_samples[1])
    writer.add_image("Generated Images", grid, epoch)
    model.train()


@hydra.main(config_path="conf", config_name="default.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train the diffusion model."""
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    train_dataset = load_dataset(
        "huggan/smithsonian_butterflies_subset", split="train", cache_dir=cfg.datadir
    )
    preprocess = transforms.Compose(
        [
            transforms.Resize(cfg.plot.image_size[-2:]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    train_dataset.set_transform(transform)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True
    )

    unet = diffusers.UNet2DModel(**cfg.model)
    criterion = nn.MSELoss()
    diffmodel = DiffusionModel(unet, criterion, **cfg.diffusion).to(device)

    optimizer = torch.optim.Adam(diffmodel.parameters(), **cfg.optimizer)

    writer = SummaryWriter(log_dir=os.path.join(outdir, "tensorboard"))

    for epoch in range(1, cfg.train.num_epochs + 1):
        loss = train_one_epoch(diffmodel, train_loader, optimizer, device)
        writer.add_scalar("train_loss", loss, epoch)
        print(f"[Epoch {epoch:4d}]  loss: {loss:.4f}")

        if epoch % cfg.plot.every_n_epochs == 0:
            logging.info(f"Epoch {epoch}: Generating images...")
            visualize(
                diffmodel,
                cfg.diffusion.num_timesteps,
                cfg.plot.num_samples,
                cfg.plot.image_size,
                writer,
                epoch,
            )

    writer.close()
    torch.save(diffmodel.state_dict(), os.path.join(outdir, "model.pth"))


if __name__ == "__main__":
    main()
