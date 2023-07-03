import torch
from torch import nn

class Block(nn.Module):

    def __init__(self, in_Channels, out_channels, down=True, act='relu', use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_Channels, out_channels, 4, 2, 1, padding_mode='reflect')
            if down
            else nn.ConvTranspose2d(in_Channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=='relu' else nn.LeakyReLU(0.2)
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5) 
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        
        return self.dropout(x) if self.use_dropout else x
    

class Generator(nn.Module):

        def __init__(self, in_channels=3, features=64):
            super().__init__()
            self.down1 = nn.Sequential(
                nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode='reflect'),
                nn.LeakyReLU(0.2)
            )
            self.down2 = Block(features, features*2, down=True, act='leaky', use_dropout=False)
            self.down3 = Block(features*2, features*4, down=True, act='leaky', use_dropout=False)
            self.down4 = Block(features*4, features*8, down=True, act='leaky', use_dropout=False)
            self.down5 = Block(features*8, features*8, down=True, act='leaky', use_dropout=False)
            self.down6 = Block(features*8, features*8, down=True, act='leaky', use_dropout=False)
            self.down7 = Block(features*8, features*8, down=True, act='leaky', use_dropout=False)

            self.bottleneck = nn.Sequential(
                nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode='reflect'),
                nn.ReLU()
            )

            self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
            self.up2 = Block(features*8*2, features*8, down=False, act='relu', use_dropout=True)
            self.up3 = Block(features*8*2, features*8, down=False, act='relu', use_dropout=True)
            self.up4 = Block(features*8*2, features*8, down=False, act='relu', use_dropout=False)
            self.up5 = Block(features*8*2, features*4, down=False, act='relu', use_dropout=False)
            self.up6 = Block(features*4*2, features*2, down=False, act='relu', use_dropout=False)
            self.up7 = Block(features*2*2, features, down=False, act='relu', use_dropout=False)
            self.up8 = nn.Sequential(
                nn.ConvTranspose2d(features*2, in_channels, 4, 2, 1),
                nn.Tanh()
            )


        def forward(self, x):
            d1 = self.down1(x)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            d7 = self.down7(d6)
            
            neck = self.bottleneck(d7)

            up1 = self.up1(neck)
            up2 = self.up2(torch.cat([d7, up1], 1))
            up3 = self.up3(torch.cat([d6, up2], 1))
            up4 = self.up4(torch.cat([d5, up3], 1))
            up5 = self.up5(torch.cat([d4, up4], 1))
            up6 = self.up6(torch.cat([d3, up5], 1))
            up7 = self.up7(torch.cat([d2, up6], 1))
            up8 = self.up8(torch.cat([d1, up7], 1))

            return up8
        

def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)
    

if __name__ == "__main__":
    test()