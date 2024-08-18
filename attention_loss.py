import torch

class LAM:
    def __init__(self, T=1.0):
        self.T = T

    def compute_Gc(self, A):
        # 计算通道注意力图 Gc(A)
        Gc = torch.mean(torch.abs(A), dim=(1, 2))  # 沿着高度和宽度维度求绝对值的平均值
        return Gc

    def compute_Gs(self, A):
        # 计算空间注意力图 Gs(A)
        Gs = torch.mean(torch.abs(A), dim=1)  # 沿着通道维度求绝对值的平均值
        return Gs

    def compute_Ms(self, Gs_A, Gs_T):
        # 计算空间注意力掩码 Ms
        spatial_numerator = torch.sum(torch.softmax(Gs_T / self.T, dim=(1, 2)) * Gs_A, dim=0)
        Ms = spatial_numerator / torch.sum(spatial_numerator)
        return Ms

    def compute_Mc(self, Gc_A, Gc_T):
        # 计算通道注意力掩码 Mc
        channel_numerator = torch.sum(torch.softmax(Gc_T / self.T, dim=0) * Gc_A, dim=0)
        Mc = channel_numerator / torch.sum(channel_numerator)
        return Mc

    def compute_LAM(self, AT, AS, Ms, Mc):
        # 计算 LAM 损失
        diff_squared = (AT - AS) ** 2
        masked_diff_squared = diff_squared * Ms.unsqueeze(0).unsqueeze(0) * Mc.unsqueeze(1).unsqueeze(1)
        LAM = torch.sqrt(torch.sum(masked_diff_squared))
        return LAM

    def compute_LAT(self, Gs_AS, Gs_AT, Gc_AS, Gc_AT):
        # 计算 LAT 损失
        spatial_L2 = torch.norm(Gs_AS - Gs_AT)
        channel_L2 = torch.norm(Gc_AS - Gc_AT)
        LAT = spatial_L2 + channel_L2
        return LAT

    def forward(self, t, s):
        # 计算 Gs(T) 和 Gc(T)
        Gs_T = self.compute_Gs(t)
        Gc_T = self.compute_Gc(t)

        # 计算 Gs(S) 和 Gc(S)
        Gs_S = self.compute_Gs(s)
        Gc_S = self.compute_Gc(s)

        # 计算 Ms 和 Mc
        Ms = self.compute_Ms(Gs_S, Gs_T)
        Mc = self.compute_Mc(Gc_S, Gc_T)

        # 计算 LAM 和 LAT 损失
        LAM_loss = self.compute_LAM(t, s, Ms, Mc)
        LAT_loss = self.compute_LAT(Gs_S, Gs_T, Gc_S, Gc_T)

        return LAM_loss, LAT_loss


# 示例用法
C, H, W = 3, 4, 4
teacher_feature = torch.randn(C, H, W)
student_feature = torch.randn(C, H, W)

lam_model = LAM(T=1.0)
lam_loss, lat_loss = lam_model.forward(teacher_feature, student_feature)

print("LAM Loss:", lam_loss)
print("LAT Loss:", lat_loss)
