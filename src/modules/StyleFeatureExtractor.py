# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from .imdn import IMDN
# from style_encoder import StyleEncoder, style_encoder_textedit_addskip_arch
#
#
# class StyleFeatureExtractor(nn.Module):
#     def __init__(
#             self,
#             G_ch=64,  # 基本通道数
#             G_wide=True,  # 是否使用宽网络层
#             resolution=128,  # 输入图像的分辨率
#             G_kernel_size=3,  # 卷积核大小
#             G_attn='64_32_16_8',  # 注意力机制的配置
#             n_classes=1000,  # 输出类别数
#             num_G_SVs=1,  # Spectral Normalization的奇异值个数
#             num_G_SV_itrs=1,  # Spectral Normalization的迭代次数
#             G_activation=nn.ReLU(inplace=False),  # 激活函数，默认为ReLU
#             SN_eps=1e-12,  # Spectral Normalization的epsilon值
#             output_dim=1,  # 输出维度
#             G_fp16=False,  # 是否使用FP16精度，进行快速训练
#             G_init='N02',  # 权重初始化方法
#             G_param='SN',  # 网络参数化方法，'SN'表示使用Spectral Normalization
#             nf_mlp=512,  # MLP层的特征数量
#             nEmbedding=256,  # 嵌入维度
#             input_nc=3,  # 输入图像的通道数（默认3，表示RGB图像）
#             output_nc=3,  # 输出图像的通道数（默认为3，表示RGB图像）
#             upscale=2
#     ):
#         """
#         双分支样式编码器，结合超分辨率和原始编码。
#
#         Args:
#             imdn (nn.Module): 超分辨率模块 IMDN
#             encoder (nn.Module): 样式编码器 StyleEncoder
#         """
#         super(StyleFeatureExtractor, self).__init__()
#
#         self.upscale = upscale
#
#         self.branch1_encoder = StyleEncoder(
#             G_ch=G_ch,
#             G_wide=G_wide,
#             resolution=resolution*2,
#             G_kernel_size=G_kernel_size,
#             G_attn=G_attn,
#             n_classes=n_classes,
#             num_G_SVs=num_G_SVs,
#             num_G_SV_itrs=num_G_SV_itrs,
#             G_activation=G_activation,
#             SN_eps=SN_eps,
#             output_dim=output_dim,
#             G_fp16=G_fp16,
#             G_init=G_init,
#             G_param=G_param,
#             nf_mlp=nf_mlp,
#             nEmbedding=nEmbedding,
#             input_nc=input_nc,
#             output_nc=output_nc
#         )
#         self.branch2_encoder = StyleEncoder(
#             G_ch=G_ch,
#             G_wide=G_wide,
#             resolution=resolution,
#             G_kernel_size=G_kernel_size,
#             G_attn=G_attn,
#             n_classes=n_classes,
#             num_G_SVs=num_G_SVs,
#             num_G_SV_itrs=num_G_SV_itrs,
#             G_activation=G_activation,
#             SN_eps=SN_eps,
#             output_dim=output_dim,
#             G_fp16=G_fp16,
#             G_init=G_init,
#             G_param=G_param,
#             nf_mlp=nf_mlp,
#             nEmbedding=nEmbedding,
#             input_nc=input_nc,
#             output_nc=output_nc
#         )
#
#         self.imdn = IMDN(self.upscale)
#
#         # 计算风格编码器的输出通道数
#         feature_dim = self.branch1_encoder.arch['out_channels'][-1] * 4 * 4
#         self.gate = nn.Linear(2 * feature_dim, feature_dim)  # 输入拼接后为2*D，输出 D
#
#     def forward(self, style_img):
#         """
#         Args:
#             style_img (Tensor): 输入样式图像, 形状 (B, 3, H, W)，归一化到 [-1,1]
#         Returns:
#             final_style_emd: 融合后的风格嵌入，与单独使用StyleEncoder的输出形状相同
#             fused_global: 融合后的全局特征向量 (B, D)
#             residual_features: 分支1的 residual features (list)
#         """
#         # 分支1：超分辨率分支
#         # 这里 IMDN 接受[-1,1]数据直接处理，将128x128图像提升至256x256
#         high_res_style_img = self.imdn(style_img)  # 形状: (B, 3, 256, 256)
#
#         # 将超分分支的图像输入风格编码器
#         style_emd1, global_feat1, residual_features1 = self.branch1_encoder(high_res_style_img)
#
#         # 分支2：直接编码分支，处理128x128图像
#         style_emd2, global_feat2, residual_features2 = self.branch2_encoder(style_img)
#
#         # 得到的全局特征 (global_feat) 的形状为 (B, D) ，D = C * 16
#         # 融合全局特征：将两个分支拼接后通过门控层得到融合后的全局特征
#         combined_global = torch.cat([global_feat1, global_feat2], dim=1)  # (B, 2*D)
#         gate_weight = torch.sigmoid(self.gate(combined_global))  # (B, D)
#         fused_global = gate_weight * global_feat1 + (1 - gate_weight) * global_feat2  # (B, D)
#
#         print(gate_weight)
#
#         # 对于 style_emd 和 residual_features，保留分支1的输出（也可尝试其他融合策略）
#         final_style_emd = style_emd1
#         final_residual_features = residual_features1
#
#         return final_style_emd, fused_global, final_residual_features
