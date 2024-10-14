

from ..models.cnn_ae import CNNAutoEncoder
from ..notebooks import dataloader_test as dlt
from ..data import drop_prob_setup as setup


import torch

import pytorch_lightning as pl


# Define datasets
train_files = [f'../data/modified_ndws/train_conus_west_ndws_0{i:02}.tfrecord' for i in range(39)]
train_data = dlt.NondimFireDataset(train_files, setup.units_, positive=setup.positive, constants=setup.constants)
train_loader = dlt.DataLoader(train_data, batch_size=32)

test_files = [f'../data/modified_ndws/test_conus_west_ndws_0{i:02}.tfrecord' for i in range(13)]
test_data = dlt.NondimFireDataset(test_files, setup.units_, positive=setup.positive, constants=setup.constants)
test_loader = dlt.DataLoader(test_data, batch_size=32)


# Define model
model = CNNAutoEncoder(in_channels=(train_data.nondims), learning_rate=1e-3, max_epochs=100, power=0.9)
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

# Training setup
trainer = pl.Trainer(
    gpus=torch.cuda.device_count(),
    max_epochs=2,
    callbacks=[lr_monitor],
    logger=pl.loggers.TensorBoardLogger('logs/'),
    checkpoint_callback=pl.callbacks.ModelCheckpoint(dirpath='checkpoints/')
)


# the pytorch lightning docs call the test_loader val, which is probably standard practice. However, we are using
# files with test in the name as the validation set, so I used the name test for this data.
trainer.fit(model, train_loader, test_loader)
trainer.test(model, test_loader)
