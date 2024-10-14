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
    parser.add_argument('--data', type=str)
    parser.add_argument('--accelerator', type=str, default='gpu')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.data == 'scaled' :
        train_data = torch.load('src/dimensionless_wildfire/data/modified_ndws/scaled_training_data.pt')
        val_data = torch.load('src/dimensionless_wildfire/data/modified_ndws/scaled_val_data.pt')
    elif args.data == 'normalized' :
        train_data = torch.load('src/dimensionless_wildfire/data/modified_ndws/normalized_training_data.pt')
        val_data = torch.load('src/dimensionless_wildfire/data/modified_ndws/normalized_val_data.pt')

    train_targets = torch.load('src/dimensionless_wildfire/data/modified_ndws/train_fire_masks.pt')
    val_targets = torch.load('src/dimensionless_wildfire/data/modified_ndws/val_fire_masks.pt')

    train_dataset = FireDataset(train_data, train_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = FireDataset(val_data, val_targets)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.model == 'aspp_cnn' :
        model = AsppCNN(in_channels=19, learning_rate=args.learning_rate, max_epochs=args.max_epochs, 
                        power=args.power, lr_schedule=args.lr_schedule, min_lr=args.min_lr, 
                        max_lr=args.max_lr, gamma=args.gamma, cycle_length=args.cycle_length)
    elif args.model == 'cnn_ae' :
        model = CNNAutoEncoder(in_channels=19, learning_rate=args.learning_rate, max_epochs=args.max_epochs, 
                               power=args.power, lr_schedule=args.lr_schedule, min_lr=args.min_lr, 
                               max_lr=args.max_lr, gamma=args.gamma, cycle_length=args.cycle_length)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    # Training setup
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        # devices=torch.cuda.device_count(),
        max_epochs=args.max_epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        # fast_dev_run=2, #### for when testing
        logger=pl.pytorch.loggers.TensorBoardLogger('logs/')
        # max_time = "00:12:00:00",
        # num_nodes = args.num_nodes,
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()
