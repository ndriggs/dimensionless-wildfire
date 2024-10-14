from torch.utils.data import DataLoader, IterableDataset
from tfrecord.torch.dataset import TFRecordDataset
import torch
import itertools
from sympy.physics import units
from ..data import nondim
import numpy as np
import json
from ..data.data_preprocessing import impute_mean, impute_fire_mask


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



class Loader:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for x, y in zip(np.load(self.path +".npy", allow_pickle=True), np.load(self.path + "_target.npy", allow_pickle=True)):
            yield x.reshape((x.shape[0], 64, 64)), y.reshape((64, 64))


class NondimFireDataset(IterableDataset):
    def __init__(self, file_patterns, units_dict=dict(), constants=dict(), target="viirs_FireMask", positive=[], nondimensional=False):
        self.nondimensional = nondimensional
        if not self.nondimensional:
            self.constants = constants
            self.units_dict = units_dict
            self.nondims, self.cols, self.rows = nondim.dimensionless_vars(units_dict)
            self.nondims = np.array(self.nondims).astype(float).squeeze()

            # set all values in positive to be positive in the nondim variables
            for vec in self.nondims:
                for i, var in enumerate(self.cols):
                    if var in positive and vec[i] < 0:
                        vec[i] = -vec[i]

            # check that all positive values are in fact positive
            for vec in self.nondims:
                for i, var in enumerate(self.cols):
                    if var in positive:
                        assert vec[i] >= 0, f"Some vectors cannot have all requested positive entries. Failed for {self.cols}, {vec}."

            self.datasets = [
                TFRecordDataset(file_pattern, index_path=None)
                for file_pattern in file_patterns
            ]


            self._nondimensionalize_data(file_patterns, target)
            self.nondimensional = True




        else:  # nondimensional = True
            self.nondims = np.load(file_patterns + "_nondims.npy")
            self.cols = np.load(file_patterns + "_cols.npy")
            self.rows = np.load(file_patterns + "_rows.npy")
            self.constants = np.load(file_patterns + "_constants.npy")
            with open(file_patterns + "_units.json", "rb") as f:
                self.units_dict = json.load(f)

            self.datasets = [
                Loader(file_pattern) for file_pattern in file_patterns
            ]


    def __iter__(self):
        return itertools.chain.from_iterable(self.datasets)



    def _nondimensionalize_data(self, paths, target):
        if self.nondimensional:
            raise ValueError("Dataset is already dimensionless.")

        for i, path in enumerate(paths):
            non_dim = []
            targets = []
            for batch in TFRecordDataset(path, index_path=None):
                # make data into shape (1, vars, batch, dim) and nondims into shape (nondim, vars, 1, 1), then broadcast and product out the vars dimension.
                # FIXME: remove the string "elevation" from the below.
                # FIXME: do preprocessing elsewhere
                batch["tmmx"] = impute_mean(batch["tmmx"])
                batch["tmmn"] = impute_mean(batch["tmmn"])
                batch["viirs_FireMask"] = impute_fire_mask(batch["viirs_FireMask"])
                batch["viirs_PrevFireMask"] = impute_fire_mask(batch["viirs_PrevFireMask"])
                data = np.expand_dims(np.array([batch[key] if key in batch.keys() else np.ones_like(batch["elevation"]) * self.constants[key] for key in self.cols]), 0)
                targets.append(batch[target])
                non_dim.append((data ** np.expand_dims(self.nondims, 2)).prod(axis=1))
            np.save(path+f"_0{i:02}.npy", non_dim, allow_pickle=False)
            np.save(path + f"_0{i:02}_target.npy", targets, allow_pickle=False)

        self.datasets = [
            Loader(path+f"_0{i:02}") for i, path in enumerate(paths)
        ]


    def dimensionalize(self, item):
        return


    def save(self, path):
        num_batches = 0
        for i, batch in enumerate(self):
            np.save(path + f"_0{i:02}.npy", batch)
            num_batches += 1

        np.save(path + "_nondims.npy", self.nondims)
        np.save(path + "_cols.npy", self.cols)
        np.save(path + "_rows.npy", self.rows)
        np.save(path + "_constants.npy", self.constants)
        np.save(path + "_num_files.npy", [num_batches])
        with open(path + "_units.json", "w") as f:
            json.dump(f, self.units_dict)

    def __getitem__(self, item):
        for i, batch in enumerate(self):
            if i==item:
                return batch


