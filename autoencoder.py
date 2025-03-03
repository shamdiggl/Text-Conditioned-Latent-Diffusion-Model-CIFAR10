import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # First conv and norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        
        # Second conv and norm      
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        
        self.relu = nn.ReLU()
        
        # `in_channels` to `out_channels` mapping layer for residual connection
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.projection = nn.Identity()
            
            
    def forward(self, x: torch.Tensor):
        h = x

        # First normalization and convolution layer
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.relu(h)

        # Second normalization and convolution layer
        h = self.conv2(h)
        h = self.norm2(h)
      
        # Map and add residual
        return self.relu(self.projection(x) + h)
    
class ResNet_AutoEncoder(nn.Module):
    def __init__(self, channels: list, num_of_blocks: int, down_sample: list):
        super(ResNet_AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential()
                
        self.encoder.append(nn.Conv2d(in_channels=3, out_channels=channels[0], 
                            kernel_size=3, stride=1, padding=1))
        
        for i in range(len(channels)-1): #ResNet layers
            in_channel = channels[i]
            out_channel = channels[i+1]        
            for j in range(num_of_blocks):
                self.encoder.append(ResnetBlock(in_channel,out_channel))
                in_channel = out_channel
            if down_sample[i] == True:
                self.encoder.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                      kernel_size=3, stride=2, padding=1))
        
        self.encoder.append(nn.Conv2d(in_channels=channels[-1], out_channels=3, #Conv_out layer
                      kernel_size=3, stride=1, padding=1))
        
       
        
        self.decoder = nn.Sequential()
                
        self.decoder.append(nn.Conv2d(in_channels=3, out_channels=channels[-1], #Conv_in layer
                      kernel_size=3, stride=1, padding=1))
        
        for i in range(1,len(channels)): #ResNet layers
            in_channel = channels[-i]
            out_channel = channels[-i-1] 
            if down_sample[-i] == True:
                self.decoder.append(nn.ConvTranspose2d(in_channels=in_channel, out_channels=in_channel, 
                                    kernel_size=3, stride=2, padding=1, output_padding=1))
            for j in range(num_of_blocks):
                self.decoder.append(ResnetBlock(in_channel,out_channel))
                in_channel = out_channel
            
        
        self.decoder.append(nn.Conv2d(in_channels=channels[0], out_channels=3, #Conv_out layer
                      kernel_size=3, stride=1, padding=1))
        self.decoder.append(nn.Tanh())
        
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded