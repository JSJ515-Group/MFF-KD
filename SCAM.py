import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.attention import SCAM
class C2f_SC(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((5 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv3 = Conv(c1,(2+n)*self.c,1)
        self.attention = SCAM(c1)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out1 = torch.cat(y, 1)
        out2 = self.attention(x)
        out2 = self.cv3(out2)
        out = torch.cat((out1,out2),1)
        return self.cv2(out)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out1 = torch.cat(y, 1)
        out2 = self.attention(x)
        out2 = self.cv3(out2)
        out = torch.cat((out1, out2), 1)
        return self.cv2(out)


input = torch.rand(1,64,128,128)
sc = C2f_SC(64,64)
print(sc(input).shape)
