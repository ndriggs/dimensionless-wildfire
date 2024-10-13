from torch.utils.data import DataLoader
import torch
import argparse
import torch
import pytorch_lightning as pl
# from your_modules import Callbacks
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
import sys
import os
from ..data.fire_dataset import FireDataset
from ..models.aspp_cnn import AsppCNN
from ..models.cnn_ae import CNNAutoEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='aspp_cnn')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr_schedule', type=str, default='poly')
    parser.add_argument('--data', type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    
    train_dataset = FireDataset(inputs, targets)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = FireDataset(inputs, targets)
    val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = AsppCNN(in_channels=19, learning_rate=args.learning_rate, 
                    max_epochs=args.max_epochs, power=args.power)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Training setup
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.max_epochs,
        callbacks=[lr_monitor],
        logger=pl.loggers.TensorBoardLogger('logs/'),
        checkpoint_callback=pl.callbacks.ModelCheckpoint(dirpath='checkpoints/')
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()
