import torch
from torch import nn

class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)
    
class Discriminator(nn.Module):

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], 4, stride=2, padding_mode='reflect', padding=1),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
                
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature==features[-1] else 2)
            )
            in_channels = feature

        self.model = nn.Sequential(*layers)
        self.final = nn.Conv2d(in_channels, 1, 4, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        x = self.initial(input)
        x = self.model(x)
        
        return self.final(x)
    
    
def test():
    model = Discriminator()
    x = torch.rand(1, 3, 256, 256)
    y = torch.rand(1, 3, 256, 256)
    out = model(x, y)
    print(out.shape)
    

if __name__ == '__main__':
    test()

