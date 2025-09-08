import torch
from torch import nn
import torchvision
from IPython.core.debugger import set_trace


# Padded U-Net inspired from https://amaarora.github.io/2020/09/13/unet.html


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch
        self.pad = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.batchNorm2d = nn.BatchNorm2d(mid_ch)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        # return self.relu(self.conv2(self.pad(self.relu(self.conv1(self.pad(x))))))
        return self.batchNorm2d(self.relu(self.conv2(self.pad(self.batchNorm2d(self.relu(self.conv1(self.pad(x)))))))) # add BatchNorm


class Encoder(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, num_class, enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.out = nn.Softmax2d()

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        return out

"""
    Early Fusion
"""
class FIUNet(nn.Module):
    def __init__(self, num_class, enc_chs=(2, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head1 = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.head2 = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.out = nn.Softmax2d()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out1 = self.head1(out)
        out2 = self.head2(out)
        return out1, out2

"""
    Late Fusion
"""

class FOUNet(nn.Module):
    def __init__(self, num_class, enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder_fusion = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-2], num_class, 1)
        self.out = nn.Softmax2d()
        self.pad = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(dec_chs[-2], dec_chs[-2], 3)

    def forward(self, x1, x2):
        enc_ftrs1 = self.encoder(x1)
        out1 = self.decoder_fusion(enc_ftrs1[::-1][0], enc_ftrs1[::-1][1:])

        enc_ftrs2 = self.encoder(x2)
        out2 = self.decoder_fusion(enc_ftrs2[::-1][0], enc_ftrs2[::-1][1:])

        feat = torch.cat([out1, out2], dim=1)  # fusion output
        feat = self.conv2(self.pad(feat))  # add more conv. layer

        out1 = self.head(feat)
        out2 = self.head(feat)
        return out1, out2
