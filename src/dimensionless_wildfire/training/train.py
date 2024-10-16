from torch.utils.data import DataLoader
import torch
import argparse
import torch
import lightning as pl
# from your_modules import Callbacks
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import sys
import os
from ..data.fire_dataset import FireDataset
from ..models.aspp_cnn import AsppCNN
from ..models.cnn_ae import CNNAutoEncoder
from ..notebooks.dataloader_test import NondimFireDataset
from ..data import drop_prob_setup
from ..data import keep_prob_setup
from ..data import learn_prob_setup
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='aspp_cnn')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr_schedule', type=str, default='poly')
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--max_lr', type=float, default=6e-3)
    parser.add_argument('--gamma', type=float, default=0.99994)
    parser.add_argument('--cycle_length', type=int, default=2000)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--nondim_setup', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.data == 'scaled' :
        train_data = torch.load('src/dimensionless_wildfire/data/modified_ndws/scaled_training_data.pt')
        val_data = torch.load('src/dimensionless_wildfire/data/modified_ndws/scaled_val_data.pt')
        test_data = torch.load('src/dimensionless_wildfire/data/modified_ndws/scaled_test_data.pt')

        train_targets = torch.load('src/dimensionless_wildfire/data/modified_ndws/train_fire_masks.pt')
        val_targets = torch.load('src/dimensionless_wildfire/data/modified_ndws/val_fire_masks.pt')
        test_targets = torch.load('src/dimensionless_wildfire/data/modified_ndws/test_fire_masks.pt')

        train_dataset = FireDataset(train_data, train_targets)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        val_dataset = FireDataset(val_data, val_targets)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        test_dataset = FireDataset(test_data, test_targets)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    elif args.data == 'normalized' :
        train_data = torch.load('src/dimensionless_wildfire/data/modified_ndws/normalized_training_data.pt')
        val_data = torch.load('src/dimensionless_wildfire/data/modified_ndws/normalized_val_data.pt')
        test_data = torch.load('src/dimensionless_wildfire/data/modified_ndws/normalized_test_data.pt')

        train_targets = torch.load('src/dimensionless_wildfire/data/modified_ndws/train_fire_masks.pt')
        val_targets = torch.load('src/dimensionless_wildfire/data/modified_ndws/val_fire_masks.pt')
        test_targets = torch.load('src/dimensionless_wildfire/data/modified_ndws/test_fire_masks.pt')

        train_dataset = FireDataset(train_data, train_targets)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        val_dataset = FireDataset(val_data, val_targets)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        test_dataset = FireDataset(test_data, test_targets)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    elif args.data == 'nondim':
        setup = args.nondim_setup
        if setup is None:
            raise ValueError('Must specify a non-dimensional setup')
        elif setup == "keep":
            setup = keep_prob_setup
        elif setup == "drop":
            setup = drop_prob_setup
        elif setup == "learn":
            setup = learn_prob_setup

        train_files = [f'src/dimensionless_wildfire/data/modified_ndws/train_conus_west_ndws_0{i:02}.tfrecord' for i in range(39)]
        train_dataset = NondimFireDataset(train_files, setup.units_, positive=setup.positive,
                                           constants=setup.constants)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

        val_files = [f'src/dimensionless_wildfire/data/modified_ndws/eval_conus_west_ndws_0{i:02}.tfrecord' for i in range(13)]
        val_dataset = NondimFireDataset(val_files, setup.units_, positive=setup.positive, constants=setup.constants)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

        test_files = [f'src/dimensionless_wildfire/data/modified_ndws/test_conus_west_ndws_0{i:02}.tfrecord' for i in range(13)]
        test_dataset = NondimFireDataset(test_files, setup.units_, positive=setup.positive, constants=setup.constants)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    else:
        raise ValueError('Unrecognized value for --data')

    if (args.data == 'scaled') or (args.data == 'normalized') :
        experiment_name = f'{args.data}_{args.model}_{args.lr_schedule}_{args.batch_size}_{np.random.randint(100000)}'
    elif args.data == 'nondim' :
        experiment_name = f'nondim_{args.nondim_setup}_{args.model}_{np.random.randint(100000)}'

    first_item, _ = train_dataset[0]
    in_channels = first_item.shape[0]
    if args.model == 'aspp_cnn' :
        model = AsppCNN(in_channels=in_channels, learning_rate=args.learning_rate, max_epochs=args.max_epochs,
                        power=args.power, lr_schedule=args.lr_schedule, min_lr=args.min_lr, 
                        max_lr=args.max_lr, gamma=args.gamma, cycle_length=args.cycle_length)
    elif args.model == 'cnn_ae' :
        model = CNNAutoEncoder(in_channels=in_channels, learning_rate=args.learning_rate, max_epochs=args.max_epochs,
                               power=args.power, lr_schedule=args.lr_schedule, min_lr=args.min_lr, 
                               max_lr=args.max_lr, gamma=args.gamma, cycle_length=args.cycle_length)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor="val_f1")

    # Training setup
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        # devices=torch.cuda.device_count(),
        max_epochs=args.max_epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        # fast_dev_run=2, # for when testing
        enable_checkpointing=True, # so it returns the best model
        logger=pl.pytorch.loggers.TensorBoardLogger('logs/', name=experiment_name) 
        # max_time = "00:12:00:00",
        # num_nodes = args.num_nodes,
    )
 
    best_model = trainer.fit(model, train_dataloader, test_dataloader)
    trainer.test(best_model, val_dataloader)

    print(vars(args))

if __name__ == '__main__':
    main()
