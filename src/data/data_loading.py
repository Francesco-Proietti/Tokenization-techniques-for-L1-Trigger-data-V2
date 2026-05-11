from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset, get_worker_info
import lightning as pl


class L1TriggerDataset(IterableDataset):
    """
    IterableDataset for L1-trigger data from parquet files.

    Streams data lazily from parquet files instead of loading all into memory.
    Each event contains PUPPI particles with features: pT, eta, phi.

    Supports multi-process loading via PyTorch workers.
    """

    def __init__(
        self,
        parquet_dirs: List[str],
        max_particles: int = 128,
        features: List[str] = ["L1T_PUPPIPart_PT", "L1T_PUPPIPart_Eta", "L1T_PUPPIPart_Phi", "L1T_PUPPIPart_PuppiW"],
        puppiw_threshold: float = 0.05,
        preprocessing: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            parquet_dirs: List of directories containing parquet files.
            max_particles: Maximum number of particles per event.
            features: List of feature names to extract.
            puppiw_threshold: Minimum PUPPI weight for particles.
            preprocessing: Whether to apply preprocessing.
        """
        super().__init__()

        self.dataset = ds.dataset(parquet_dirs, format="parquet")
        self.max_particles = max_particles
        self.features = features
        self.coords = features[:-1] #Exclude PuppiW from coordinates
        self.puppiw_threshold = puppiw_threshold
        self.preprocessing = preprocessing

    def _process_event(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single event row into padded features and mask.

        Returns:
            features: [max_particles, n_coords] tensor
            mask: [max_particles] boolean tensor
        """
        n_coords = len(self.coords)
        feats = np.zeros((self.max_particles, n_coords), dtype=np.float32)
        mask = np.zeros(self.max_particles, dtype=bool)

        # Apply puppiw filter
        puppiw = row["L1T_PUPPIPart_PuppiW"]
        valid_mask = np.array(puppiw) >= self.puppiw_threshold

        #Preprocessing 
        if self.preprocessing:
            # Normalize pT and eta
            pt = np.array(row["L1T_PUPPIPart_PT"])
            eta = np.array(row["L1T_PUPPIPart_Eta"])
            phi = np.array(row["L1T_PUPPIPart_Phi"])
            pt = np.log(pt + 1e-8) - 1.8  
            eta = eta / 3
            phi = phi / np.pi
            row["L1T_PUPPIPart_PT"] = pt
            row["L1T_PUPPIPart_Eta"] = eta
            row["L1T_PUPPIPart_Phi"] = phi

        # Filter particles
        for feat_idx, feat_name in enumerate(self.coords):
            particles_feat = row[feat_name]
            particles_feat = np.array(particles_feat)[valid_mask]
            n_particles = min(len(particles_feat), self.max_particles)
            feats[:n_particles, feat_idx] = particles_feat[:n_particles]
            mask[:n_particles] = True
        
        return torch.FloatTensor(feats), torch.BoolTensor(mask)

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over all events using pyarrow.dataset scanner.

        Handles multi-process loading by splitting files across workers.
        """
        worker_info = get_worker_info()

        if worker_info is None:
            scanner = self.dataset.scanner(columns=self.features)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files = self.dataset.files
            files_per_worker = len(files) // num_workers
            extra = len(files) % num_workers

            start_idx = worker_id * files_per_worker + min(worker_id, extra)
            end_idx = start_idx + files_per_worker + (1 if worker_id < extra else 0)

            worker_files = files[start_idx:end_idx]
            scanner = self.dataset.scanner(files=worker_files, columns=self.features)

        batch_reader = scanner.to_reader()

        for batch in batch_reader:
            df = batch.to_pandas()

            for i in range(len(df)):
                row = df.iloc[i]
                features, mask = self._process_event(row)
                yield features, mask


class L1TriggerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for L1-trigger data.
    """

    def __init__(
        self,
        parquet_dirs_train: List[str],
        parquet_dirs_val: Optional[List[str]] = None,
        parquet_dirs_test: Optional[List[str]] = None,
        max_particles: int = 128,
        batch_size: int = 32,
        num_workers: int = 0,
        features: List[str] = ["L1T_PUPPIPart_PT", "L1T_PUPPIPart_Eta", "L1T_PUPPIPart_Phi", "L1T_PUPPIPart_PuppiW"],
        puppiw_threshold: float = 0.05,
    ):
        """
        Initialize the DataModule.

        Args:
            parquet_dirs_train: Directories containing training parquet files.
            parquet_dirs_val: Optional directories containing validation data.
            parquet_dirs_test: Optional directories containing test data.
            max_particles: Maximum particles per event.
            batch_size: Batch size for dataloaders.
            num_workers: Workers for dataloaders.
            features: Features to extract.
            puppiw_threshold: Minimum PUPPI weight.
        """
        super().__init__()

        self.train_dirs = parquet_dirs_train
        self.val_dirs = parquet_dirs_val or []
        self.test_dirs = parquet_dirs_test or []
        self.max_particles = max_particles
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.features = features
        self.puppiw_threshold = puppiw_threshold

    def train_dataloader(self):
        """Return training dataloader."""
        self.train_dataset = L1TriggerDataset(
            parquet_dirs=self.train_dirs,
            max_particles=self.max_particles,
            features=self.features,
            puppiw_threshold=self.puppiw_threshold,
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
            #drop_last=True,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        self.val_dataset = L1TriggerDataset(
            parquet_dirs=self.val_dirs,
            max_particles=self.max_particles,
            features=self.features,
            puppiw_threshold=self.puppiw_threshold,
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            #drop_last=True,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        self.test_dataset = L1TriggerDataset(
            parquet_dirs=self.test_dirs,
            max_particles=self.max_particles,
            features=self.features,
            puppiw_threshold=self.puppiw_threshold,
        )
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            #drop_last=True,
        )
