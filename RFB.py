import torch
import torch.nn as nn

class RFBconv(nn.Module):
    def __init__(self, c1):
        super(RFBconv, self).__init__()
        # 第一条分支
        c2 = c1
        self.branch1 = nn.Sequential(
            nn.Conv2d(c1,c2,1,1),
            nn.Conv2d(c2, c2, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=c2),
            nn.Conv2d(c2, c2, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=c2)
            # nn.Conv2d(c2, c2, kernel_size=3, dilation=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1),
            nn.Conv2d(c2, c2, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=c2),
            nn.Conv2d(c2, c2, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=c2),
            nn.Conv2d(c2, c2, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=c2,
                                        dilation=2),
            nn.Conv2d(c2, c2, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=c2,
                                        dilation=2),
        )

        # 第二条分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(c1, c2, 1,1),
            # nn.Conv2d(c2, c2, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=c2),
            # nn.Conv2d(c2, c2, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=c2),
            nn.Conv2d(c2, c2, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=c2),
            nn.Conv2d(c2, c2, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=c2),
            # nn.Conv2d(c2, c2, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=c2,
            #           dilation=2),
            # nn.Conv2d(c2, c2, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=c2,
            #           dilation=2),
            nn.Conv2d(c2, c2, kernel_size=(1, 11), stride=(1, 1), padding=(0, 15), groups=c2,
                                        dilation=3),
            nn.Conv2d(c2, c2, kernel_size=(11, 1), stride=(1, 1), padding=(15, 0), groups=c2,
                                        dilation=3),
            # nn.Conv2d(c2, c2, kernel_size=3, dilation=3, padding=3),
        )
        self.conv1 = nn.Conv2d(in_channels=c2,out_channels=c2,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=c2,out_channels=c2,kernel_size=3,stride=1,padding=1)
        # 第三条分支
        # 1x1卷积用于通道数变换
        # self.conv1x1 = nn.Conv2d(3 * c2, c2, 1)

    def forward(self, x):
        # 分别对三条分支进行前向传播
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4= out1+out2+out3
        result = self.conv2(self.relu(self.conv1(out4)))
        result = result + x + out4
        return result
x = torch.rand(1,256,32,32)
r = RFBconv(256)
print(r(x).shape)