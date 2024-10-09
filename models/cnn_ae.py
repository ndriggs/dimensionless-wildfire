import torch
import torch.nn as nn
import torchvision.models as models

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding='same', output_padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class CNNAutoEncoder(nn.Module):
    def __init__(self, in_channels):
        super(CNNAutoEncoder, self).__init__()
        
        # Base model (MobileNetV2)
        self.base_model = models.mobilenet_v2(weights=None)
        self.base_model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

        print('num layers:', len(self.base_model.features))
        
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
            UpsampleBlock(1280, 512),
            UpsampleBlock(512, 256),
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 64)
        ])
        
        # Final layers
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Downsampling
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        print(x.shape)
        
        skips = skips[::-1][1:]  # Reverse and remove last skip connection
        
        # Upsampling
        i = 1
        for up, skip in zip(self.up_stack, skips):
            print(i)
            i += 1
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        
        # Final layers
        x = self.final_upsample(x)
        x = self.final_conv(x)
        return self.sigmoid(x)
