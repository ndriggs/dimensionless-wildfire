import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP_CNN(nn.Module) :
    """
    CNN with Atrous Spatial Pyramid Pooling (ASPP)
    Based on the architecture outlined in "Application of Explainable Artificial 
    Intelligence in Predicting Wildfire Spread: An ASPP-Enabled CNN Approach" by 
    Marjani et al. https://ieeexplore.ieee.org/document/10568207

    Used to predict fire spread at the next timestep at each pixel location
    """
    def __init__(self, in_channels) :
        super(ASPP_CNN, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.aspp_conv1 = nn.Conv2d(128, 32, kernel_size=3, padding=1, dilation=1)
        self.aspp_conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1, dilation=3)
        self.aspp_conv3 = nn.Conv2d(128, 32, kernel_size=3, padding=1, dilation=6)
        self.aspp_conv4 = nn.Conv2d(128, 32, kernel_size=3, padding=1, dilation=12)

        self.conv3 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.batchnorm = nn.BatchNorm2d(32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, x) :
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