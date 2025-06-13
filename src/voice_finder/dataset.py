import torch
from pathlib import Path
import zarr
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="zarr.codecs.vlen_utf8")


def collate_fn(group):
    batched = {}

    for key in group[0].keys():
        if isinstance(group[0][key], torch.Tensor):
            batched[key] = torch.concatenate([x[key] for x in group], axis=0).float()
        elif isinstance(group[0][key], list):
            batched[key] = [item for sublist in group for item in sublist[key]]
        else:
            batched[key] = [x[key] for x in group]

    return batched


class VoiceEmbedderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: Path,
        sequence_length: int,
        samples_per_item: int = 4,
        rng=None,
        xent_augmentations: list = [],
        number_xent_augmentations=0,
        triplet_augmentations: list = [],
        number_triplet_augmentations=0,
        all_samples_per_client: bool = False,
    ):
        super().__init__()
        self._rng = rng if rng else np.random.default_rng(seed=1337)

        self._dataset_path = dataset_path
        self._dataset = zarr.open(dataset_path)

        # sequence_length * <model-stride> / 1000 = seconds
        self._sequence_length = sequence_length
        self._samples_per_item = samples_per_item

        # Precount all the samples for each client
        self._client_samples = {}
        self.prepare_client_samples()

        self._xent_augmentations = xent_augmentations
        self._number_xent_augmentations = number_xent_augmentations

        self._triplet_augmentations = triplet_augmentations
        self._number_triplet_augmentations = number_triplet_augmentations

        self._all_samples_per_client = all_samples_per_client

    def prepare_client_samples(self):
        for client_idx in np.unique(self._dataset["client_index"][:]):
            self._client_samples[client_idx.item()] = np.where(
                self._dataset["client_index"][:] == client_idx
            )[0]
        self._client_indices = np.array(list(self._client_samples.keys()))

    def __len__(self):
        return len(self._client_indices)

    def __getitem__(self, idx: int):
        """
        Select a client, pull samples_per_item number of audio samples from that client.
        """

        client_idx = self._client_indices[idx]

        # Randomly sample samples_per_item number of samples from this client
        if self._all_samples_per_client:
            sample_indices = self._client_samples[client_idx]
        else:
            sample_indices = np.random.choice(
                self._client_samples[client_idx],
                replace=False,
                size=self._samples_per_item,
            )

        features = self._dataset["features"][sample_indices, :]
        attention_masks = self._dataset["attention_mask"][sample_indices, :]

        for _ in range(self._number_triplet_augmentations):
            aug = self._rng.choice(self._triplet_augmentations)

            features, _ = aug(features, attention_masks)

        features_augmented = features.copy()

        for _ in range(self._number_xent_augmentations):
            aug = self._rng.choice(self._xent_augmentations)

            features_augmented, _ = aug(features_augmented, attention_masks)
        client_idx = np.resize(client_idx, len(sample_indices))

        client_id = [
            x.decode().strip() for x in self._dataset["client_id"][sample_indices]
        ]

        path = [x.decode().strip() for x in self._dataset["path"][sample_indices]]

        sample = {
            "client_index": client_idx,
            "sample_index": sample_indices,
            "features": features,
            "attention_mask": attention_masks,
            "features_augmented": features_augmented,
            "client_id": client_id,
            "path": path,
        }

        sample = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in sample.items()
        }

        return sample
