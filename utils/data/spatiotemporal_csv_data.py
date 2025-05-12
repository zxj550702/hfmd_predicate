import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import utils.data.functions
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_epoch = 500


class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,
        adj_path: str,
        batch_size: int,
        seq_len: int,
        pre_len: int,
        split_ratio: float = 0.8,
        normalize: bool = True,
        data: str = 'jphfmd',
        **kwargs
    ):
        super(SpatioTemporalCSVDataModule, self).__init__()
        self._feat_path = feat_path
        self._adj_path = adj_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self.dataName = data
        self._feat = utils.data.functions.load_features(self._feat_path)
        if data == 'jphfmd':
            self._feat_max_val = np.max(self._feat)
            self._feat_min_val = np.min(self._feat)
        else:
            self._feat_max_val = np.max(self._feat[:, -1])
            self._feat_min_val = np.min(self._feat[:, -1])
        self._adj = utils.data.functions.load_adjacency_matrix(self._adj_path)

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        args, _ = parent_parser.parse_known_args()
        batch_size = 32 if args.data == 'cnhfmd' else 8
        parser.add_argument("--batch_size", type=int, default=batch_size)
        parser.add_argument("--seq_len", type=int, default=4)
        parser.add_argument("--pre_len", type=int, default=4)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        parser.add_argument("--normalize", type=bool, default=True)
        global max_epoch
        max_epoch = 500
        parser.set_defaults(max_epochs=max_epoch)
        return parser

    def setup(self, stage: str = None):
        (
            self.train_dataset,
            self.val_dataset,
        ) = utils.data.functions.generate_torch_datasets(
            self._feat,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
            dataName=self.dataName
        )
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=len(self.val_dataset)
                          )

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def feat_min_val(self):
        return self._feat_min_val

    @property
    def adj(self):
        return self._adj
