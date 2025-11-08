###使用复数卷积进行频域特征提取，光照分量


import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


#class ChannelGate(nn.Module):
    #def __init__(self, in_channels):
        #super(ChannelGate, self).__init__()
        #self.gate = nn.Sequential(
            #nn.AdaptiveAvgPool2d(1),
            #nn.Conv2d(in_channels, in_channels, 1),
            #nn.BatchNorm2d(in_channels),
            #nn.Hardswish(),
            #nn.Sigmoid()
        #)

    #def forward(self, x):
        #weight = self.gate(x)
        #return x * weight


class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ComplexConv, self).__init__()
        self.real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, real, imag):
        r_r = self.real(real)
        i_i = self.imag(imag)
        i_r = self.imag(real)
        r_i = self.real(imag)
        return r_r - i_i, i_r + r_i


class ComplexLeakyReLU(nn.Module):
    def __init__(self):
        super(ComplexLeakyReLU, self).__init__()
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, real, imag):
        return self.act(real), self.act(imag)


class FourierReflectedConvolution(nn.Module):
    def __init__(self, kernel_nums=8, kernel_size=3):
        super(FourierReflectedConvolution, self).__init__()
        self.kernel_nums = kernel_nums
        self.kernel_size = kernel_size

        # 空间域色差部分
        self.rg_bn_sp = nn.BatchNorm2d(kernel_nums)
        self.gb_bn_sp = nn.BatchNorm2d(kernel_nums)
        self.rb_bn_sp = nn.BatchNorm2d(kernel_nums)
        self.spatial_filter = nn.Parameter(torch.randn(kernel_nums, 1, kernel_size, kernel_size))

        # 堆叠三层复数卷积+激活
        self.complex_conv1 = ComplexConv(1, kernel_nums, 3, 1, 1)
        self.complex_act1 = ComplexLeakyReLU()
        self.complex_conv2 = ComplexConv(kernel_nums, kernel_nums, 3, 1, 1)
        self.complex_act2 = ComplexLeakyReLU()
        self.complex_conv3 = ComplexConv(kernel_nums, kernel_nums, 3, 1, 1)
        self.complex_act3 = ComplexLeakyReLU()

        # 最后一层恢复为单通道
        self.complex_conv_out = ComplexConv(kernel_nums, 1, 1, 1, 0)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.spatial_filter)
        for bn in [self.rg_bn_sp, self.gb_bn_sp, self.rb_bn_sp]:
            torch.nn.init.constant_(bn.weight, 0.01)
            torch.nn.init.constant_(bn.bias, 0)

    def mean_constraint(self, kernel):
        bs, cin, kw, kh = kernel.shape
        kernel_mean = torch.mean(kernel.view(bs, -1), dim=1, keepdim=True)
        kernel = kernel.view(bs, -1) - kernel_mean
        return kernel.view(bs, cin, kw, kh)

    def fft_feature(self, x):
        # 1. 傅里叶变换
        x_fft = torch.fft.rfft2(x, norm='ortho')
        x_r, x_i = x_fft.real, x_fft.imag

        # 2. 多层复数卷积 + 激活
        r, i = self.complex_conv1(x_r, x_i)
        r, i = self.complex_act1(r, i)
        r, i = self.complex_conv2(r, i)
        r, i = self.complex_act2(r, i)
        r, i = self.complex_conv3(r, i)
        r, i = self.complex_act3(r, i)
        # 可选：最后一层降回单通道（可保留/可省略，视网络结构而定）
        # r, i = self.complex_conv_out(r, i)

        # 3. 合成复数
        f_complex = torch.complex(r, i)
        return f_complex

    def forward(self, img):
        zeroMasks = torch.zeros_like(img)
        zeroMasks[img == 0] = 1

        log_img = torch.log(img + 1e-7)
        r, g, b = log_img[:, 0:1], log_img[:, 1:2], log_img[:, 2:3]

        # 每个通道单独做多层复数卷积
        r_fft = self.fft_feature(r)
        g_fft = self.fft_feature(g)
        b_fft = self.fft_feature(b)

        # 逆变换回空间域
        red_chan   = torch.fft.irfft2(r_fft, s=r.shape[-2:], norm='ortho')
        green_chan = torch.fft.irfft2(g_fft, s=g.shape[-2:], norm='ortho')
        blue_chan  = torch.fft.irfft2(b_fft, s=b.shape[-2:], norm='ortho')

        # 色差卷积（空间域）
        norm_filter = self.mean_constraint(self.spatial_filter)
        filt_rg = self.rg_bn_sp(
            F.conv2d(red_chan, norm_filter, padding=self.kernel_size // 2,groups=self.kernel_nums) +
            F.conv2d(green_chan, -norm_filter, padding=self.kernel_size // 2,groups=self.kernel_nums)
        )
        filt_gb = self.gb_bn_sp(
            F.conv2d(green_chan, norm_filter, padding=self.kernel_size // 2,groups=self.kernel_nums) +
            F.conv2d(blue_chan, -norm_filter, padding=self.kernel_size // 2,groups=self.kernel_nums)
        )
        filt_rb = self.rb_bn_sp(
            F.conv2d(red_chan, norm_filter, padding=self.kernel_size // 2,groups=self.kernel_nums) +
            F.conv2d(blue_chan, -norm_filter, padding=self.kernel_size // 2,groups=self.kernel_nums)
        )

        # 掩码处理
        rg = torch.where(zeroMasks[:, 0:1].expand(-1, self.kernel_nums, -1, -1) == 1, 0, filt_rg)
        gb = torch.where(zeroMasks[:, 1:2].expand(-1, self.kernel_nums, -1, -1) == 1, 0, filt_gb)
        rb = torch.where(zeroMasks[:, 2:3].expand(-1, self.kernel_nums, -1, -1) == 1, 0, filt_rb)

        out = torch.cat([rg, gb, rb], dim=1)
        return out



class fIIBlock(nn.Module):
    def __init__(self, kernel_nums=8, kernel_size=3, Gtheta=[0.6, 0.8]):
        super(fIIBlock, self).__init__()
        self.Gtheta = Gtheta
        self.feat_projector = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU()
        )
        self.fuse_net = nn.Sequential(
            nn.Conv2d(48, 32, 3, 1, 1, groups=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        self.final_fuse = nn.Sequential(
            nn.Conv2d(56, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        self.iim = FourierReflectedConvolution(kernel_nums, kernel_size)

    def forward(self, x):
        x_gma = torch.pow(x, np.random.uniform(self.Gtheta[0], self.Gtheta[1]))
        x_gma = torch.clamp(x_gma, 0, 1)

        feat_ii_f = self.iim(x)
        feat_ii_gma_f = self.iim(x_gma)

        # ---- 使用 illumination mapper 提取光照分量 ----
        #illum_maps = []
        #for img in x:
            #img_np = img.permute(1, 2, 0).cpu().numpy()
           # illum_gray = self.illumination_mapper.generate_illumination_map(img_np)
            #illum_tensor = torch.from_numpy(illum_gray).unsqueeze(0)
            #illum_maps.append(illum_tensor)

       # illum_stack = torch.stack(illum_maps, dim=0).to(x.device)
        #illum_stack = illum_stack.repeat(1, 3, 1, 1)

        feat_proj = self.feat_projector(x)
        x_cat = torch.cat((feat_proj, feat_ii_f), dim=1)

        feat_enhance = self.fuse_net(x_cat)
        #x_out = self.final_fuse(torch.cat([feat_enhance, feat_ii_f], dim=1))
        #final_input = torch.cat((feat_enhance, illum_stack), dim=1)
        # x_out = self.final_fuse
        #x_out = self.final_fuse(final_input)

        return feat_enhance, (feat_ii_f, feat_ii_gma_f)
