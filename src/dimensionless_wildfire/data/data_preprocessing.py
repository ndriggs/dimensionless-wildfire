# code for creating input tensor (dimensionless or unchanged features) from a batch 

from torch.utils.data import DataLoader, IterableDataset
from tfrecord.torch.dataset import TFRecordDataset
import torch
import itertools
import numpy as np

# max, mean, min, and variance for each feature as found in the training data
# used for scaling before inputting into network
DATA_STATS = {
    'NDVI': (0.999, 0.474, -0.979, 0.052),
    'bi': (151.679,	53.317,	-10.098, 444.672),
    'chili': (254.0, 181.843, 0.0, 1882.764),
    'elevation': (4176.0, 1317.760, -73.0, 611800.527),
    'erc': (113.304, 69.119, -0.782, 471.267),
    'fuel1': (4.487, -0.379, -7.182, 1.488),
    'fuel2': (14.405, -1.155, -11.433, 3.027),
    'fuel3': (6.188, -0.263, -4.503, 0.827),
    'impervious': (100.0, 1.210, 0.0, 37.884),
    'pdsi': (10.676, -1.970, -19.947, 5.784),
    'population': (18564.754, 27.870, 0.0, 64926.134),
    'pr': (32.764, 0.173, -0.288, 1.435),
    'sph': (0.015, 0.005, 0.0, 0.000005),
    'th': (895.525, 217.910, -872.872, 7222.749),
    'tmmn': (303.568, 282.886, 0.0, 48.983),
    'tmmx': (320.478, 298.714, 0.0, 60.725),
    'vs': (12.882, 3.314, -4.144, 1.593),
    'water': (100.0, 2.441, 0.0, 172.001)
}

def scale_and_concat_all_features(batch) : 
    scaled_features = []
    for key in DATA_STATS.keys() :
        feature = batch[key].unsqueeze(1) # (B, H*W) -> (B, C, H*W)
        scaled_feature = (feature - DATA_STATS[key][2]) / (DATA_STATS[key][0] - DATA_STATS[key][2]) 
        scaled_features.append(scaled_feature)
    scaled_features.append(impute_fire_mask(batch['viirs_PrevFireMask'].unsqueeze(1)))
    features = torch.cat(scaled_features, dim=1) # concat along the channels dimension
    features = features.view(-1,19,64,64) # reshape to (B, C, H, W)
    return features

def normalize_and_concat_all_features(batch) :
    normalized_features = []
    for key in DATA_STATS.keys() :
        feature = batch[key].unsqueeze(1) # (B, H*W) -> (B, C, H*W)
        normalized_feature = (feature - DATA_STATS[key][1]) / np.sqrt(DATA_STATS[key][3])
        normalized_features.append(normalized_feature)
    normalized_features.append(impute_fire_mask(batch['viirs_PrevFireMask'].unsqueeze(1)))
    features = torch.cat(normalized_features, dim=1) # concat along the channels dimension
    features = features.view(-1,19,64,64) # reshape to (B, C, H, W)
    return features

def reshape_fire_mask(batch) :
    fire_mask = batch['viirs_FireMask'].unsqueeze(1)
    return fire_mask.view(-1,1,64,64)


def impute_mean(variable) :
    variable[variable==0] = variable.mean()
    return variable


def impute_fire_mask(mask) :
    # for -1 (unknown) values, set to 1 if at least 1 neighbor is 1, and to 0 otherwise
    if (mask == -1).sum() > 0:
        mask = mask.view(-1, 1, 64, 64)
        for (batch, channel, i, j) in zip(np.where(mask == -1)):
            mask[batch, channel, i, j] = max(mask[batch, channel, i-1, j], \
                 mask[batch, channel, i+1, j], mask[batch, channel, i, j-1], \
                    mask[batch, channel, i, j+1], 0)
    return mask.view(-1, 1, 64*64)


class MultiTFRecordDataset(IterableDataset):
    def __init__(self, file_patterns):
        self.datasets = [
            TFRecordDataset(file_pattern, index_path=None)
            for file_pattern in file_patterns
        ]

    def __iter__(self):
        return itertools.chain.from_iterable(self.datasets)

# train_files = [f'modified_ndws/train_conus_west_ndws_0{i:02}.tfrecord' for i in range(39)]
# train_dataset = MultiTFRecordDataset(train_files)
# train_loader = DataLoader(train_dataset, batch_size=200)

# train_scaled = []
# train_normalized = []
# train_fire_masks = []
# for batch in train_loader :
#     train_scaled.append(scale_and_concat_all_features(batch))
#     train_normalized.append(normalize_and_concat_all_features(batch))
#     train_fire_masks.append(reshape_fire_mask(batch))

# scaled_training_data = torch.cat(train_scaled, dim=0)
# normalized_training_data = torch.cat(train_normalized, dim=0)
# train_fire_mask = torch.cat(train_fire_masks, dim=0)
# torch.save(scaled_training_data, 'modified_ndws/scaled_training_data.pt')
# torch.save(normalized_training_data, 'modified_ndws/normalized_training_data.pt')
# torch.save(train_fire_mask, 'modified_ndws/train_fire_masks.pt')

# test_files = [f'modified_ndws/test_conus_west_ndws_0{i:02}.tfrecord' for i in range(13)]
# test_dataset = MultiTFRecordDataset(test_files)
# test_loader = DataLoader(test_dataset, batch_size=200)

# test_scaled = []
# test_normalized = []
# test_fire_masks = []
# for batch in test_loader :
#     test_scaled.append(scale_and_concat_all_features(batch))
#     test_normalized.append(normalize_and_concat_all_features(batch))
#     test_fire_masks.append(reshape_fire_mask(batch))

# scaled_test_data = torch.cat(test_scaled, dim=0)
# normalized_test_data = torch.cat(test_normalized, dim=0)
# test_fire_mask = torch.cat(test_fire_masks, dim=0)
# torch.save(scaled_test_data, 'modified_ndws/scaled_test_data.pt')
# torch.save(normalized_test_data, 'modified_ndws/normalized_test_data.pt')
# torch.save(test_fire_mask, 'modified_ndws/test_fire_masks.pt')

# val_files = [f'modified_ndws/eval_conus_west_ndws_0{i:02}.tfrecord' for i in range(7)]
# val_dataset = MultiTFRecordDataset(val_files)
# val_loader = DataLoader(val_dataset, batch_size=200)

# val_scaled = []
# val_normalized = []
# val_fire_masks = []
# for batch in val_loader :
#     val_scaled.append(scale_and_concat_all_features(batch))
#     val_normalized.append(normalize_and_concat_all_features(batch))
#     val_fire_masks.append(reshape_fire_mask(batch))

# scaled_val_data = torch.cat(val_scaled, dim=0)
# normalized_val_data = torch.cat(val_normalized, dim=0)
# val_fire_mask = torch.cat(val_fire_masks, dim=0)
# torch.save(scaled_val_data, 'modified_ndws/scaled_val_data.pt')
# torch.save(normalized_val_data, 'modified_ndws/normalized_val_data.pt')
# torch.save(val_fire_mask, 'modified_ndws/val_fire_masks.pt')
