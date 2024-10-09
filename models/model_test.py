import torch
from cnn_ae import CNNAutoEncoder
from aspp_cnn import AsppCNN
import aspp_cnn

cnn_ae = CNNAutoEncoder(18)
# aspp_cnn = AsppCNN(18)

x = torch.ones((32,18,64,64))

x1 = cnn_ae(x)
# x2 = aspp_cnn(x)

print(x1.shape)
# print(x2.shape)
