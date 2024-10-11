import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import PolynomialLR
from torchmetrics import Precision, Recall, F1Score

class AsppCNN(pl.LightningModule):
    def __init__(self, in_channels, learning_rate=1e-3, max_epochs=100, power=0.9):
        super(AsppCNN, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.power = power

        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.aspp_conv1 = nn.Conv2d(128, 32, kernel_size=3, padding=1, dilation=1)
        self.aspp_conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=3, dilation=3)
        self.aspp_conv3 = nn.Conv2d(128, 32, kernel_size=3, padding=6, dilation=6)
        self.aspp_conv4 = nn.Conv2d(128, 32, kernel_size=3, padding=12, dilation=12)

        self.conv3 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.batchnorm = nn.BatchNorm2d(32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, x):
        # first 2 conv layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # aspp module
        x1 = self.relu(self.aspp_conv1(x))
        x2 = self.relu(self.aspp_conv2(x))
        x3 = self.relu(self.aspp_conv3(x))
        x4 = self.relu(self.aspp_conv4(x))
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # next 2 conv layers
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.batchnorm(x)

        # last layer
        x = self.final_conv(x)

        return F.sigmoid(x)

    def tversky_loss(y_pred, y_true, alpha=0.7, beta=0.3):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        true_pos = torch.sum(y_true * y_pred)
        false_neg = torch.sum(y_true * (1 - y_pred))
        false_pos = torch.sum((1 - y_true) * y_pred)
        tversky = true_pos / (true_pos + alpha * false_neg + beta * false_pos)
        return 1 - tversky

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = tversky_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = tversky_loss(y_hat, y)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        f1 = self.f1(y_hat, y)
        # self.log('val_loss', loss)
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