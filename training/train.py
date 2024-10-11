from torch.utils.data import DataLoader
from ..data.fire_dataset import FireDataset
from ..models.aspp_cnn import AsppCNN
from ..models.cnn_ae import CNNAutoEncoder
import torch
import argparse
import torch
import pytorch_lightning as pl
# from your_modules import Callbacks
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='resnet18')
    # Add more arguments as needed
    return parser.parse_args()

def main():
    args = parse_args()
    
    train_dataset = FireDataset(inputs, targets)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    val_dataset = FireDataset(inputs, targets)
    val_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = AsppCNN(in_channels=3, learning_rate=1e-3, max_epochs=100, power=0.9)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Training setup
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=100,
        callbacks=[lr_monitor],
        logger=pl.loggers.TensorBoardLogger('logs/'),
        checkpoint_callback=pl.callbacks.ModelCheckpoint(dirpath='checkpoints/')
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()


# make models directory a package 
# 