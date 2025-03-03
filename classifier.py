import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResnetBlock(32,64),
            nn.ReLU(),
            ResnetBlock(64,64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # Output: [batch_size, 64, 32, 32]
            nn.ReLU(),
            ResnetBlock(64,128),
            nn.ReLU(),
            ResnetBlock(128,128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 128, 16, 16]
            nn.ReLU(),
            ResnetBlock(128,128),
            nn.ReLU(),
            ResnetBlock(128,128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 128, 8, 8]

        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 512), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x