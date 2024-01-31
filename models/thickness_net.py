import numpy as np
from PIL import Image
import os
import math
import torch
import torch.nn as nn
import torch.optim as Optim
from .vq import VectorQuantizer
from .vq import VectorQuantizerEMA

class Encoder(nn.Module):
    def __init__(self, in_dim = 2):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
    
        self.conv1 = nn.Conv2d(self.in_dim, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)  # (batch_size, 32, 256, 256)
        x = torch.relu(x)
        x = self.conv2(x)  # (batch_size, 64, 128, 128)
        x = torch.relu(x)
        x = self.conv3(x)  # (batch_size, 128, 64, 64)
        x = torch.relu(x)
        x = self.conv4(x)  # (batch_size, 128, 64, 64)
        x = torch.relu(x)

        return x



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.deconv1(x)  # (batch_size, 128, 64, 64)
        x = torch.relu(x)
        x = self.deconv2(x)  # (batch_size, 32, 256, 256)
        x = torch.relu(x)
        x = self.deconv3(x)  # (batch_size, 1, 512, 512)

        return x


class thickness_Autoencoder(nn.Module):
    def __init__(self, in_dim = 2, decay=0.99, type=5):
        super(thickness_Autoencoder, self).__init__()

        self.commitment_cost = 0.25
        self.decay = decay
        self.in_dim = in_dim
        
        if decay > 0.0:
            self.vq = VectorQuantizerEMA(512, 64, self.commitment_cost, self.decay)
        else:
            self.vq = VectorQuantizer(512, 64, self.commitment_cost)
            
        self.encoder = Encoder(in_dim = self.in_dim)
        self.decoder = Decoder()


    def forward(self, x):
        encoded = self.encoder(x)  # (batch_size, 128, 64, 64)
        flattened = torch.flatten(encoded, start_dim=2) # (batch_size, 128, 4096)
        vq_loss, vq_feature, perplexity, encodings = self.vq(flattened) # (batch_size, 128, 4096)
        decoded = self.decoder(vq_feature.view(encoded.shape))  # (batch_size, 1, 512, 512)
        return decoded
