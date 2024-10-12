import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from training.training_utils import tversky_loss
from torch.optim.lr_scheduler import PolynomialLR



class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class CNNAutoEncoder(nn.Module):
    def __init__(self, in_channels):
        super(CNNAutoEncoder, self).__init__()
        
        # Base model (MobileNetV2)
        self.base_model = models.mobilenet_v2(weights=None)
        self.base_model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Encoder (down_stack)
        self.down_stack = nn.ModuleList([
            self.base_model.features[:2],   # block_1_expand_relu
            self.base_model.features[2:4],  # block_3_expand_relu
            self.base_model.features[4:7],  # block_6_expand_relu
            self.base_model.features[7:14], # block_13_expand_relu
            self.base_model.features[14:17] # block_16_project
        ])
        
        # Decoder (up_stack)
        self.up_stack = nn.ModuleList([
            UpsampleBlock(160, 96),
            UpsampleBlock(96 + 96, 32),
            UpsampleBlock(32 + 32, 24),
            UpsampleBlock(24 + 24, 16)
        ])
        
        # Final layers
        self.final_upsample = nn.ConvTranspose2d(16+16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.Conv2d(1, 1, kernel_size=1, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Downsampling
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)
        
        skips = skips[::-1][1:]  # Reverse and remove last skip connection
        
        # Upsampling
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        
        # Final layers
        x = self.final_upsample(x)
        x = self.final_conv(x)
        return self.sigmoid(x)



class LightningCNNAutoEncoder(pl.LightningModule):
    def __init__(self, CNNAutoEncoder, learning_rate=1e-3, max_epochs=100, power=0.9):
        super(LightningCNNAutoEncoder, self).__init__()
        self.net = CNNAutoEncoder
        self.learning_rate = learning_rate
        self.power = power
        self.max_epochs = max_epochs


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = tversky_loss(y_hat, y)

        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        f1 = self.f1(y_hat, y)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = tversky_loss(y_hat, y)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        f1 = self.f1(y_hat, y)
        # self.log('test_loss', loss)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = PolynomialLR(optimizer, total_iters=self.max_epochs, power=self.power)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "polynomial_lr"
            }
        }



