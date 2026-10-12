import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)

from src.modules.imdn import IMDN
from collections import OrderedDict

# 定义一个投影函数
def proj(x, y):
    # 计算y在x方向的投影：
    # 1. 先计算x的转置与y的矩阵乘法，即点积：torch.mm(y, x.t())
    # 2. 将结果乘以y，然后除以y与自身的点积（相当于归一化）
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
    # 对输入向量 x 进行 Gram-Schmidt 正交化，
    # 使其与列表 ys 中的所有向量正交。
  for y in ys:
      x = x - proj(x, y)
  return x


def power_iteration(W, u_, update=True, eps=1e-12):
    """
        定义 power_iteration 函数，用于对矩阵 W 进行幂迭代，计算其主要特征值及对应的特征向量
        参数说明：
        W: 待处理的矩阵
        u_: 初始向量列表（或张量列表），作为迭代初始值
        update: 是否更新初始向量（默认为 True）
        eps: 在归一化过程中避免除 0 的小常数
    """
    us, vs, svs = [], [], []
    # 遍历每个初始向量 u，并使用其索引 i
    for i, u in enumerate(u_):
        # 在不需要梯度计算的上下文中进行操作，加快计算并节省内存
        with torch.no_grad():
            # 计算 v = u * W，得到 W 作用于 u 后的向量
            v = torch.matmul(u, W)
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            vs += [v]
            u = torch.matmul(v, W.t())   # 转置矩阵 W.t()
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            us += [u]
            if update:
                u_[i][:] = u
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    return svs, us, vs


# 用于构建一个包含全连接层、归一化层以及激活函数的线性模块
class LinearBlock(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        norm='none',   # 归一化类型，可选 'bn'（BatchNorm1d）、'in'（InstanceNorm1d）或 'none'
        act='relu', 
        use_sn=False  # 是否使用谱归一化（Spectral Normalization）
    ):
        super(LinearBlock, self).__init__()
        use_bias = True
        # 定义一个全连接层，将 in_dim 映射到 out_dim
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        # 如果指定了使用谱归一化，则对全连接层进行谱归一化处理
        if use_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

        # 初始化归一化层
        # 归一化的维度通常为全连接层的输出维度 out_dim
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # 初始化激活函数
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class MLP(nn.Module):
    def __init__(
        self, 
        nf_in, 
        nf_out, 
        nf_mlp,   # MLP 隐藏层的特征数（隐藏层尺寸）
        num_blocks,    # MLP 层数（包含输入层和输出层）
        norm, 
        act, 
        use_sn =False
    ):
        super(MLP,self).__init__()
        self.model = nn.ModuleList()
        nf = nf_mlp

        # 添加第一个模块：将输入维度 nf_in 映射到隐藏层维度 nf
        self.model.append(LinearBlock(nf_in, nf, norm = norm, act = act, use_sn = use_sn))

        # 添加中间的多个隐藏层模块（共 num_blocks-2 个模块）
        # 每个模块都是从 nf 映射到 nf，保持相同的隐藏层尺寸
        for _ in range((num_blocks - 2)):
            self.model.append(LinearBlock(nf, nf, norm=norm, act=act, use_sn=use_sn))

        # 添加最后一个模块：将隐藏层维度 nf 映射到输出维度 nf_out
        # 此处不使用归一化和激活函数，因此 norm='none' 和 act='none'
        self.model.append(LinearBlock(nf, nf_out, norm='none', act ='none', use_sn = use_sn))

        # 将 ModuleList 中的各层封装为一个 nn.Sequential 模块，使得前向传播时依次执行各层
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


#谱归一化
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        self.num_itrs = num_itrs
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)
    
    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward_wo_sn(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
    
    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                preactivation=False, activation=None, downsample=None,):
        super(DBlock, self).__init__()
    
        self.in_channels, self.out_channels = in_channels, out_channels
        
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation  # 是否进行激活函数前置
        self.activation = activation
        self.downsample = downsample  # 下采样操作

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)

        # 如果输入和输出通道数不同，或者需要下采样，则启用可学习的捷径卷积
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                          kernel_size=1, padding=0)
    def shortcut(self, x):
        # 定义捷径连接（ResNet 的核心）
        if self.preactivation: # 如果是 preactivation 模式，则先进行激活和卷积处理
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
    
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        # 返回卷积结果与捷径连接相加的结果（残差连接）
        return h + self.shortcut(x)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d,which_bn= nn.BatchNorm2d, activation=None,
                 upsample=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv,self.which_bn =which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        # 定义批归一化层
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    
    def forward(self, x):
        # 对输入进行第一次批归一化并应用激活函数
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class GBlock2(nn.Module):
    def __init__(self, in_channels, out_channels,
                which_conv=nn.Conv2d, activation=None,
                upsample=None, skip_connection = True):
        super(GBlock2, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv = which_conv
        self.activation = activation
        self.upsample = upsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                          kernel_size=1, padding=0)
        # upsample layers
        self.upsample = upsample
        self.skip_connection = skip_connection

    def forward(self, x):
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        
        h = self.activation(h)
        h = self.conv2(h)
        
        if self.learnable_sc:
            x = self.conv_sc(x)
        if self.skip_connection:
            out = h + x
        else:
            out = h
        return out


def style_encoder_textedit_addskip_arch(ch =64,out_channel_multiplier = 1, input_nc = 3):
    #存储不同分辨率的网络架构信息。
    arch = {}
    n=2
    arch[96] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2,4,8]],
                'out_channels' : [item * ch for item in [1,2,4,8,16]],
                'resolution': [48,24,12,6,3]}

    arch[192] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'resolution': [64,32,16,8,4]}

    arch[128] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2,4,8]],
                                'out_channels' : [item * ch for item in [1,2,4,8,16]],
                                'resolution': [64,32,16,8,4]}
    
    arch[256] = {'in_channels':[input_nc]+[ch*item for item in [1,2,4,8,8]],
                                'out_channels':[item*ch for item in [1,2,4,8,8,16]],
                                'resolution': [128,64,32,16,8,4]}
    return arch


class StyleEncoder(ModelMixin, ConfigMixin):
    """
    This class is to encode the style image to image embedding.
    Downsample scale is 32.
    For example:
        Input: Shape[Batch, 3, 128, 128]
        Output: Shape[Batch, 255, 4, 4]
    """
    @register_to_config
    def __init__(
        self,
            G_ch=64,  # 基本通道数
            G_wide=True,  # 是否使用宽网络层
            resolution=128,  # 输入图像的分辨率
            G_kernel_size=3,  # 卷积核大小
            G_attn='64_32_16_8',  # 注意力机制的配置
            n_classes=1000,  # 输出类别数
            num_G_SVs=1,  # Spectral Normalization的奇异值个数
            num_G_SV_itrs=1,  # Spectral Normalization的迭代次数
            G_activation=nn.ReLU(inplace=False),  # 激活函数，默认为ReLU
            SN_eps=1e-12,  # Spectral Normalization的epsilon值
            output_dim=1,  # 输出维度
            G_fp16=False,  # 是否使用FP16精度，进行快速训练
            G_init='N02',  # 权重初始化方法
            G_param='SN',  # 网络参数化方法，'SN'表示使用Spectral Normalization
            nf_mlp=512,  # MLP层的特征数量
            nEmbedding=256,  # 嵌入维度
            input_nc=3,  # 输入图像的通道数（默认3，表示RGB图像）
            output_nc=3  # 输出图像的通道数（默认为3，表示RGB图像）
    ):
        super(StyleEncoder, self).__init__()

        self.ch = G_ch
        self.G_wide = G_wide
        self.resolution = resolution
        self.kernel_size = G_kernel_size
        self.attention = G_attn
        self.n_classes = n_classes
        self.activation = G_activation
        self.init = G_init
        self.G_param = G_param
        self.SN_eps = SN_eps
        self.fp16 = G_fp16

        # 根据分辨率，决定需要保存哪些特征层
        if self.resolution == 96:
            self.save_featrues = [0,1,2,3,4]
        elif self.resolution == 192:
            self.save_featrues = [0,1,2,3,4]
        elif self.resolution == 128:
            self.save_featrues = [0,1,2,3,4]
        elif self.resolution == 256:
            self.save_featrues = [0,1,2,3,4,5]

        # 输出通道的倍数
        self.out_channel_nultipiler = 1
        self.arch = style_encoder_textedit_addskip_arch(
          self.ch, 
          self.out_channel_nultipiler,
          input_nc
        )[resolution]# 根据分辨率选择相应的架构配置

        if self.G_param == 'SN':
            self.which_conv = functools.partial(
              SNConv2d,
              kernel_size=3, padding=1,
              num_svs=num_G_SVs, 
              num_itrs=num_G_SV_itrs,
              eps=self.SN_eps
            )
            self.which_linear = functools.partial(
              SNLinear,
              num_svs=num_G_SVs, 
              num_itrs=num_G_SV_itrs,
              eps=self.SN_eps
            )

        # 定义网络中的多个块（DBlock），这些块负责进行风格编码
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            # 每一层使用DBlock，指定输入输出通道、卷积方式、是否宽层等
            self.blocks += [[DBlock(
              in_channels=self.arch['in_channels'][index],
              out_channels=self.arch['out_channels'][index],
              which_conv=self.which_conv,
              wide=self.G_wide,
              activation=self.activation,
              preactivation=(index > 0),
              downsample=nn.AvgPool2d(2)
            )]]

        self.blocks = nn.ModuleList([
          nn.ModuleList(block) for block in self.blocks
        ])
        # 最后一层处理，用于风格编码的输出
        last_layer = nn.Sequential(
          nn.InstanceNorm2d(self.arch['out_channels'][-1]),
          self.activation,
          nn.Conv2d(
            self.arch['out_channels'][-1],
            self.arch['out_channels'][-1],
            kernel_size=1,
            stride=1
          )
        )
        self.blocks.append(last_layer)
        self.init_weights()  # 初始化权重

    # 初始化网络中的所有可训练参数
    def init_weights(self):
        self.param_count = 0   # 初始化参数计数
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                # 计算当前层的参数数量
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self,x):        
        h = x
        residual_features = []  # 用于保存每个层的特征
        residual_features.append(h)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)  # 对输入图像进行一系列卷积和激活操作
            if index in self.save_featrues[:-1]:
                residual_features.append(h)
        h = self.blocks[-1](h)  # 经过最后一层处理
        style_emd = h  # 风格嵌入（style embedding）
        h = F.adaptive_avg_pool2d(h, (1, 1))  # 对输出进行自适应平均池化，得到1x1的输出
        h = h.view(h.size(0), -1)  # 将输出展平为一维向量
        
        return style_emd,h,residual_features


class StyleFeatureExtractor(nn.Module):
    def __init__(
            self,
            G_ch=64,  # 基本通道数
            G_wide=True,  # 是否使用宽网络层
            resolution=128,  # 输入图像的分辨率
            G_kernel_size=3,  # 卷积核大小
            G_attn='64_32_16_8',  # 注意力机制的配置
            n_classes=1000,  # 输出类别数
            num_G_SVs=1,  # Spectral Normalization的奇异值个数
            num_G_SV_itrs=1,  # Spectral Normalization的迭代次数
            G_activation=nn.ReLU(inplace=False),  # 激活函数，默认为ReLU
            SN_eps=1e-12,  # Spectral Normalization的epsilon值
            output_dim=1,  # 输出维度
            G_fp16=False,  # 是否使用FP16精度，进行快速训练
            G_init='N02',  # 权重初始化方法
            G_param='SN',  # 网络参数化方法，'SN'表示使用Spectral Normalization
            nf_mlp=512,  # MLP层的特征数量
            nEmbedding=256,  # 嵌入维度
            input_nc=3,  # 输入图像的通道数（默认3，表示RGB图像）
            output_nc=3,  # 输出图像的通道数（默认为3，表示RGB图像）
            upscale=2
    ):
        """
        双分支样式编码器，结合超分辨率和原始编码。

        Args:
            imdn (nn.Module): 超分辨率模块 IMDN
            encoder (nn.Module): 样式编码器 StyleEncoder
        """
        super(StyleFeatureExtractor, self).__init__()

        self.upscale = upscale

        self.branch1_encoder = StyleEncoder(
            G_ch=G_ch,
            G_wide=G_wide,
            resolution=resolution*2,
            G_kernel_size=G_kernel_size,
            G_attn=G_attn,
            n_classes=n_classes,
            num_G_SVs=num_G_SVs,
            num_G_SV_itrs=num_G_SV_itrs,
            G_activation=G_activation,
            SN_eps=SN_eps,
            output_dim=output_dim,
            G_fp16=G_fp16,
            G_init=G_init,
            G_param=G_param,
            nf_mlp=nf_mlp,
            nEmbedding=nEmbedding,
            input_nc=input_nc,
            output_nc=output_nc
        )
        self.branch2_encoder = StyleEncoder(
            G_ch=G_ch,
            G_wide=G_wide,
            resolution=resolution,
            G_kernel_size=G_kernel_size,
            G_attn=G_attn,
            n_classes=n_classes,
            num_G_SVs=num_G_SVs,
            num_G_SV_itrs=num_G_SV_itrs,
            G_activation=G_activation,
            SN_eps=SN_eps,
            output_dim=output_dim,
            G_fp16=G_fp16,
            G_init=G_init,
            G_param=G_param,
            nf_mlp=nf_mlp,
            nEmbedding=nEmbedding,
            input_nc=input_nc,
            output_nc=output_nc
        )

        self.imdn = IMDN(upscale=self.upscale)
        pretrained_dict = torch.load('ckpt/IMDN_x2.pth')
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 去掉 "module." 前缀
            else:
                new_state_dict[k] = v

        # 加载到模型
        self.imdn.load_state_dict(new_state_dict)
        for param in self.imdn.parameters():
            param.requires_grad = False

        # 计算风格编码器的输出通道数
        feature_dim = self.branch1_encoder.arch['out_channels'][-1]
        #print(f"feature_dim shape: {feature_dim}")

        self.gate = nn.Linear(2 * feature_dim, feature_dim)  # 输入拼接后为2*D，输出 D

    def forward(self, style_img):
        """
        Args:
            style_img (Tensor): 输入样式图像, 形状 (B, 3, H, W)，归一化到 [-1,1]
        Returns:
            final_style_emd: 融合后的风格嵌入，与单独使用StyleEncoder的输出形状相同
            fused_global: 融合后的全局特征向量 (B, D)
            residual_features: 分支1的 residual features (list)
        """
        # 分支1：超分辨率分支
        # 这里 IMDN 接受[-1,1]数据直接处理，将128x128图像提升至256x256
        high_res_style_img = self.imdn(style_img)
        #high_res_style_img = self.imdn(style_img)  # 形状: (B, 3, 256, 256)
        high_res_style_img = F.adaptive_avg_pool2d(high_res_style_img, (96, 96))  # 调整回128x128

        # 将超分分支的图像输入风格编码器
        style_emd1, global_feat1, residual_features1 = self.branch1_encoder(high_res_style_img)

        # 分支2：直接编码分支，处理128x128图像
        style_emd2, global_feat2, residual_features2 = self.branch2_encoder(style_img)

        # 得到的全局特征 (global_feat) 的形状为 (B, D) ，D = C * 16
        # 融合全局特征：将两个分支拼接后通过门控层得到融合后的全局特征
        combined_global = torch.cat([global_feat1, global_feat2], dim=1)  # (B, 2*D)
        gate_weight = torch.sigmoid(self.gate(combined_global))  # (B, D)
        fused_global = gate_weight * global_feat1 + (1 - gate_weight) * global_feat2  # (B, D)

        #print(f"fused_global shape: {fused_global.shape}")

        # 对于 style_emd 和 residual_features，保留分支1的输出（也可尝试其他融合策略）
        final_style_emd = style_emd1
        final_residual_features = residual_features1

        return final_style_emd, fused_global, final_residual_features

