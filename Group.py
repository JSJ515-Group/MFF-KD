import torch.nn as nn
import torch
import numpy as np
# 输入值
im = torch.randn(1, 4, 5, 5)
# 分组卷积使用
groups = 2 # 组数
c=nn.Conv2d(4, 2, kernel_size=2, stride=2,
                     padding=2, groups=groups, bias=False)
output = c(im)
# 输出
print("输入：\n",im.shape)
print("输出：\n",output.shape)
print("卷积核参数：\n",list(c.parameters()))