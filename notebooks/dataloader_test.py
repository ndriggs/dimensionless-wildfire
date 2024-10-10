from torch.utils.data import DataLoader, IterableDataset
from tfrecord.torch.dataset import TFRecordDataset
import torch
import itertools
from sympy.physics import units
from data import nondim
import numpy as np


class MultiTFRecordDataset(IterableDataset):
    def __init__(self, file_patterns):
        self.datasets = [
            TFRecordDataset(file_pattern, index_path=None)
            for file_pattern in file_patterns
        ]

    def __iter__(self):
        return itertools.chain.from_iterable(self.datasets)

# train_files = [f'../data/modified_ndws/train_conus_west_ndws_0{i:02}.tfrecord' for i in range(39)]
# dataset = MultiTFRecordDataset(train_files)
# loader = DataLoader(dataset, batch_size=32)
#
# for batch in loader :
#
#     print(batch['elevation'][0,10])




class NondimFireDataset(IterableDataset):
    def __init__(self, file_patterns, units_dict):
        self.units_dict = units_dict
        self.nondims, self.cols, self.rows = nondim.dimensionless_vars(units_dict)
        self.nondims = np.array(self.nondims).astype(float)
        self.nondimensional = False

        self.datasets = [
            TFRecordDataset(file_pattern, index_path=None)
            for file_pattern in file_patterns
        ]

        self._nondimensionalize_data()
        self.nondimensional = True



    def __iter__(self):
        return itertools.chain.from_iterable(self.datasets)



    def _nondimensionalize_data(self):
        if self.nondimensional:
            raise ValueError("Dataset is already dimensionless.")

        batches = []
        for batch in self:
            return

