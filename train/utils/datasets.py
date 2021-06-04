import pickle as pkl
from pathlib import Path
from random import randint

import torch
from torch.utils.data import Dataset


class TESSDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        file_names: set[str],
        window_size: float,
        data_dim: int = 1,
        rnd_window: bool = False,
    ) -> None:
        super().__init__()

        # TODO dynamic classes encoding
        self.classes = [
            'angry',
            'disgust',
            'fear',
            'happy',
            'neutral',
            'ps',
            'sad',
        ]

        self.window_size = window_size
        self.rnd_window = rnd_window
        self.data_dim = data_dim

        files = [f for f in data_dir.glob('**/*.pkl') if f.stem in file_names]
        self.data = []
        for pth in files:
            with open(pth, 'rb') as f:
                d = pkl.load(f)
            emotion = pth.stem.split('_')[-1]
            c = self.classes.index(emotion)
            self.data.append((d, c))

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO handle too big windows
        features, file_class = self.data[index]

        if self.rnd_window:
            length = features.shape[1]
            margin = length - self.window_size
            start_idx = randint(0, margin)
        else:
            start_idx = 0
        end_idx = start_idx + self.window_size

        features_t = torch.tensor(features[:, start_idx:end_idx])
        if self.data_dim == 1:
            features_t = features_t.view(-1)
        elif self.data_dim == 2:
            features_t.unsqueeze_(0)

        return features_t, file_class

    def __len__(self) -> int:
        return len(self.data)
