import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn = nn.GroupNorm(groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch)
        self.conv2 = ConvGNAct(out_ch, out_ch)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNIInstSeg(nn.Module):
    """
    UNI backbone + lightweight decoder + 3 heads (cell, boundary, center).
    """
    def __init__(self, backbone, decoder_ch=(512,256,128,64), heads=("cell","boundary","center")):
        super().__init__()
        self.backbone = backbone
        in_ch = backbone.embed_dim
        chs = [in_ch] + list(decoder_ch)
        ups = []
        for i in range(len(chs)-1):
            ups.append(UpsampleBlock(chs[i], chs[i+1]))
        self.decoder = nn.Sequential(*ups)
        head_modules = {}
        for name in heads:
            head_modules[name] = nn.Conv2d(chs[-1], 1, kernel_size=1)
        self.heads = nn.ModuleDict(head_modules)

    def forward(self, x):
        feats = self.backbone(x)["feat"]
        x = feats
        x = self.decoder(x)
        out = {name: self.heads[name](x) for name in self.heads}
        return out
