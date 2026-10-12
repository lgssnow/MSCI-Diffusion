import torch
import torch.nn as nn
import torch.nn.functional as F

class MCAM_V2(nn.Module):
    def __init__(self,
                 in_channels,  # C1
                 content_channel,  # C2
                 out_channels,  # 输出通道数
                 inter_in_channels=None,
                 inter_content_channel=None,
                 dimension=2,
                 sub_sample=True,
                 bn_layer=True):
        super(MCAM_V2, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.content_channel = content_channel

        # 中间通道数默认设为原通道数的一半
        if inter_in_channels is None:
            inter_in_channels = max(1, in_channels // 2)
        if inter_content_channel is None:
            inter_content_channel = max(1, content_channel // 2)

        self.inter_in_channels = inter_in_channels
        self.inter_content_channel = inter_content_channel

        if out_channels is None:
            self.out_channels = in_channels  # 或设为两者平均
        else:
            self.out_channels = out_channels

        # 卷积/BN模块选择
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d

        # Value 分支
        self.g_one = conv_nd(in_channels, inter_in_channels, kernel_size=1)
        self.g_two = conv_nd(content_channel, inter_content_channel, kernel_size=1)

        # Query/Key 分支
        self.theta_one = conv_nd(in_channels, inter_in_channels, kernel_size=1)
        self.phi_one   = conv_nd(in_channels, inter_in_channels, kernel_size=1)

        self.theta_two = conv_nd(content_channel, inter_content_channel, kernel_size=1)
        self.phi_two   = conv_nd(content_channel, inter_content_channel, kernel_size=1)

        if sub_sample:
            self.g_one = nn.Sequential(self.g_one, max_pool_layer)
            self.g_two = nn.Sequential(self.g_two, max_pool_layer)
            self.phi_one = nn.Sequential(self.phi_one, max_pool_layer)
            self.phi_two = nn.Sequential(self.phi_two, max_pool_layer)

        # 融合后映射回 out_channels
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(inter_in_channels, self.out_channels, kernel_size=1),
                bn(self.out_channels)
            )
        else:
            self.W = conv_nd(inter_in_channels, self.out_channels, kernel_size=1)

    def forward(self, one, two):
        B = one.size(0)

        # one 分支（C1）
        g1 = self.g_one(one).view(B, self.inter_channels_one, -1).permute(0, 2, 1)   # [B, N, C']
        theta1 = self.theta_one(one).view(B, self.inter_channels_one, -1).permute(0, 2, 1)
        phi1 = self.phi_one(one).view(B, self.inter_channels_one, -1)

        f1 = torch.matmul(theta1, phi1)  # [B, N, N]
        attn1 = F.softmax(f1, dim=-1)

        # two 分支（C2）
        g2 = self.g_two(two).view(B, self.inter_channels_two, -1).permute(0, 2, 1)
        theta2 = self.theta_two(two).view(B, self.inter_channels_two, -1).permute(0, 2, 1)
        phi2 = self.phi_two(two).view(B, self.inter_channels_two, -1)

        f2 = torch.matmul(theta2, phi2)
        attn2 = F.softmax(f2, dim=-1)

        # 合并注意力（这里用点乘法，也可以尝试加权平均）
        attn = attn1 * attn2  # [B, N, N]

        # 将注意力权重分别作用于两路 Value
        out1 = torch.matmul(attn, g1)  # [B, N, C']
        out2 = torch.matmul(attn, g2)  # [B, N, C'']

        # 若需融合，可以映射两者至同一通道数后相乘或相加
        if self.inter_channels_one != self.inter_channels_two:
            # 调整 g2 输出通道维度对齐 g1
            out2 = out2.permute(0, 2, 1)  # [B, C'', N]
            out2 = F.interpolate(out2, size=out1.shape[-1], mode='nearest')
            out2 = out2.permute(0, 2, 1)

        fused = out1 * out2  # 或 fused = out1 + out2
        fused = fused.permute(0, 2, 1).contiguous()
        fused = fused.view(B, self.inter_channels_one, *one.size()[2:])
        fused = self.W(fused)
        return fused
