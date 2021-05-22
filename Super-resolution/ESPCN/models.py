import torch
import torch.nn as nn
import math


class ESPCN(nn.Module):
    def __init__(self, scale_factor, input_num=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_num, 64, kernel_size=5, padding=5//2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=3//2),
            nn.Tanh()
        )
        self.up = nn.Sequential(
            nn.Conv2d(32, input_num*(scale_factor**2), kernel_size=3, padding=3//2),
            nn.PixelShuffle(scale_factor)
        )
        self._initialize_weight()

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0,
                                    std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return torch.sigmoid(x)


if __name__ == '__main__':
    e = ESPCN(4, 3)
    a = torch.rand([2, 3, 64, 64])
    b = e(a)
    print(b.shape)
















