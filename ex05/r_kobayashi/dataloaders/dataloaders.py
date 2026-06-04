"""Data loading utilities for mel-spectrogram autoencoder training."""

import re
from pathlib import Path

import librosa
import torch
import torchaudio
import torchvision.transforms as T
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from utils.seed import make_seed_worker, make_torch_generator


class MelSpectrogramDataset(Dataset):
    """Dataset for mel spectrogram reconstruction."""

    def __init__(
        self,
        file_list: str | Path | list,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        target_frames: int,
        device: torch.device | None = None,
        is_train: bool = True,
    ) -> None:
        """dataset initialize

        Args:
            file_list (str | Path | list): A text file listing audio paths (one per line) or a list of audio paths.
            sample_rate (int): Target sampling rate used by librosa.
            n_fft (int): FFT size used for mel spectrogram.
            hop_length (int): Hop length between frames.
            n_mels (int): Number of mel bins.
            target_frames (int): Fixed number of frames after padding/trim.
            device (torch.device | None, optional): Device to use for tensors. Defaults to None.
            is_train (bool, optional): Whether the dataset is for training. Defaults to True.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.is_train = is_train
        if isinstance(file_list, list):
            self.files = [Path(f) for f in file_list]
        else:
            self.files = self._load_file_list(file_list)

        self.sample_rate = sample_rate
        self.target_frames = target_frames
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        ).to(self.device)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power").to(self.device)

        self.resize = T.Resize(
            (224, 224), antialias=True
        )  # Optional: resize for certain models

    def __len__(self) -> int:
        """Return the number of files in the dataset."""
        return len(self.files)

    def _load_file_list(self, file_list_path: str | Path) -> list[Path]:
        """Load a text file containing one audio path per line.

        Parameters
        ----------
        file_list_path : str | Path
            Path to the list file.

        Returns
        -------
        files : list[Path]
            Resolved audio paths (relative paths are resolved to the list file directory).
        """
        files: list[Path] = []
        list_path = Path(file_list_path)
        base_dir = list_path.parent
        with open(list_path, "r") as f:
            content = f.read()
            for line in content.splitlines():
                path = Path(line)
                files.append(path if path.is_absolute() else (base_dir / path))
        return files

    def _pad_or_trim_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Pad or trim mel spectrogram to a fixed frame count.

        Parameters
        ----------
        mel : torch.Tensor
            Input mel spectrogram with shape (1, n_mels, frames).

        Returns
        -------
        mel : torch.Tensor
            Mel spectrogram padded or trimmed to target_frames.
        """
        frames = mel.shape[-1]
        if frames > self.target_frames:
            return mel[..., : self.target_frames]
        if frames < self.target_frames:
            pad_amount = self.target_frames - frames
            return F.pad(mel, (0, pad_amount))
        return mel

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        """Load one audio file, convert to mel, and return metadata.

        Parameters
        ----------
        index : int
            Index into the file list.

        Returns
        -------
        sample : tuple[torch.Tensor, int, str]
            (mel, digit, speaker) where mel is (1, n_mels, frames).
        """
        file_path = self.files[index]
        filename = file_path.name

        # Load audio
        waveform_np, _ = librosa.load(file_path, sr=self.sample_rate)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(self.device)

        # Compute mel spectrogram
        mel = self.mel(waveform)
        mel = self.to_db(mel)
        mel = self._pad_or_trim_mel(mel)

        mel_resized = self.resize(mel)
        mel_rgb = mel_resized.repeat(3, 1, 1)

        match = re.search(r"(model_\d+)", filename)
        machine_id = match.group(1) if match else "unknown"

        if self.is_train:
            state = "normal" if "normal" in filename else "anomaly"
            binary_label = 0 if state == "normal" else 1
            return mel_rgb, binary_label, filename
        else:
            return mel_rgb, machine_id, filename


def create_dataloader(
    file_list: str | Path | list,
    batch_size: int = 32,
    shuffle: bool = True,
    sample_rate: int = 8000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 40,
    target_frames: int = 40,
    seed: int = 42,
    device: torch.device | None = None,
    is_train: bool = True,
    sampler=None,
) -> DataLoader:
    """Build a DataLoader for mel-spectrogram reconstruction.

    Parameters
    ----------
    file_list : str | Path | list
        Text file listing audio paths or a list of audio paths.
    batch_size : int, optional
        Batch size.
    shuffle : bool, optional
        Whether to shuffle samples each epoch.
    sample_rate : int, optional
        Target sampling rate used by librosa.
    n_fft : int, optional
        FFT size used for mel spectrogram.
    hop_length : int, optional
        Hop length between frames.
    n_mels : int, optional
        Number of mel bins.
    target_frames : int, optional
        Fixed number of frames after padding/trim.
    seed : int, optional
        Seed for deterministic shuffling and workers.
    is_train : bool, optional
        Whether the dataloader is for training.
    sampler : torch.utils.data.Sampler, optional
        Custom sampler for the dataloader.

    Returns
    -------
    dataloader : DataLoader
        Configured DataLoader instance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = make_torch_generator(seed)
    seed_worker = make_seed_worker(seed)
    dataset = MelSpectrogramDataset(
        file_list,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        target_frames=target_frames,
        device=device,
        is_train=is_train,
    )

    if sampler is not None:
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=seed_worker,
    )
