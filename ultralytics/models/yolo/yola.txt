###添加 Gate 模块（ChannelGate + Hardswish）,无引导滤波

from ultralytics.models.yolo.illm import IlluminationMapper
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2


class ChannelGate(nn.Module):
    def __init__(self, in_channels):
        super(ChannelGate, self).__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),               # [B, C, 1, 1]
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.gate(x)                      # [B, C, 1, 1]
        return x * weight                          # gating per channel


class ReflectedConvolution(nn.Module):
    def __init__(self, kernel_nums=8, kernel_size=3):
        super(ReflectedConvolution, self).__init__()
        self.kernel_nums = kernel_nums
        self.kernel_size = kernel_size
        self.rg_bn = nn.BatchNorm2d(kernel_nums)
        self.gb_bn = nn.BatchNorm2d(kernel_nums)
        self.rb_bn = nn.BatchNorm2d(kernel_nums)
        self.filter = torch.nn.Parameter(torch.randn(self.kernel_nums, 1, self.kernel_size, self.kernel_size))
        self.gate = ChannelGate(kernel_nums * 3)  # gate after concat
        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.filter)
        torch.nn.init.constant_(self.rg_bn.weight, 0.01)
        torch.nn.init.constant_(self.rg_bn.bias, 0)
        torch.nn.init.constant_(self.gb_bn.weight, 0.01)
        torch.nn.init.constant_(self.gb_bn.bias, 0)
        torch.nn.init.constant_(self.rb_bn.weight, 0.01)
        torch.nn.init.constant_(self.rb_bn.bias, 0)

    def mean_constraint(self, kernel):
        bs, cin, kw, kh = kernel.shape
        kernel_mean = torch.mean(kernel.view(bs, -1), dim=1, keepdim=True)
        kernel = (kernel.view(bs, -1) - kernel_mean)
        return kernel.view(bs, cin, kw, kh)

    def forward(self, img):
        zeroMasks = torch.zeros_like(img)
        zeroMasks[img == 0] = 1
        log_img = torch.log(img + 1e-7)

        red_chan = log_img[:, 0:1, :, :]
        green_chan = log_img[:, 1:2, :, :]
        blue_chan = log_img[:, 2:3, :, :]
        normalized_filter = self.mean_constraint(self.filter)

        filt_rg = self.rg_bn(
            F.conv2d(red_chan, normalized_filter, padding=self.kernel_size // 2) +
            F.conv2d(green_chan, -normalized_filter, padding=self.kernel_size // 2)
        )
        filt_gb = self.gb_bn(
            F.conv2d(green_chan, normalized_filter, padding=self.kernel_size // 2) +
            F.conv2d(blue_chan, -normalized_filter, padding=self.kernel_size // 2)
        )
        filt_rb = self.rb_bn(
            F.conv2d(red_chan, normalized_filter, padding=self.kernel_size // 2) +
            F.conv2d(blue_chan, -normalized_filter, padding=self.kernel_size // 2)
        )

        rg = torch.where(zeroMasks[:, 0:1].expand(-1, self.kernel_nums, -1, -1) == 1, 0, filt_rg)
        gb = torch.where(zeroMasks[:, 1:2].expand(-1, self.kernel_nums, -1, -1) == 1, 0, filt_gb)
        rb = torch.where(zeroMasks[:, 2:3].expand(-1, self.kernel_nums, -1, -1) == 1, 0, filt_rb)

        out = torch.cat([rg, gb, rb], dim=1)
        # 获取 gate 权重
        #with torch.no_grad():
            #weight = self.gate.gate(out)  # [B, C, 1, 1]
            #min_w = weight.min().item()
            #max_w = weight.max().item()
            #mean_w = weight.mean().item()
            #std_w = weight.std().item()
            #print(f"[Gate Weights] min: {min_w:.4f}, max: {max_w:.4f}, mean: {mean_w:.4f}, std: {std_w:.4f}")

        #return self.gate(out)       ###加入通道注意力机制，对cat后的输出out进行筛选
        return out


class IIBlock(nn.Module):
    def __init__(self, kernel_nums=8, kernel_size=3, Gtheta=[0.6, 0.8]):
        super(IIBlock, self).__init__()
        self.Gtheta = Gtheta
        self.feat_projector = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU()
        )
        self.fuse_net = nn.Sequential(*[
            nn.Conv2d(48, 32, 3, 1, 1, groups=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        ])
        self.final_fuse = nn.Sequential(
            nn.Conv2d(56, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        self.iim = ReflectedConvolution(kernel_nums, kernel_size)
        self.illumination_mapper = IlluminationMapper()

    def forward(self, x):
        x_gma = torch.pow(x, np.random.uniform(self.Gtheta[0], self.Gtheta[1]))
        x_gma = torch.clamp(x_gma, min=0, max=1)

        feat_ii, feat_ii_gma = list(map(self.iim, [x, x_gma]))

        # ---- 使用 illumination mapper 提取光照分量 ----
        illum_maps = []
        for img in x:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            illum_gray = self.illumination_mapper.generate_illumination_map(img_np)
            illum_tensor = torch.from_numpy(illum_gray).unsqueeze(0)
            illum_maps.append(illum_tensor)

        illum_stack = torch.stack(illum_maps, dim=0).to(x.device)
        #illum_stack = illum_stack.repeat(1, 3, 1, 1)

        # 语义特征
        feat_proj = self.feat_projector(x)
        x_cat = torch.cat((feat_proj, feat_ii), dim=1)
        feat_enhance = self.fuse_net(x_cat)

        #final_input = torch.cat((feat_enhance, illum_stack), dim=1)
        #x_out = self.final_fuse
        #x_out = self.final_fuse(final_input)

        return feat_enhance, (feat_ii, feat_ii_gma)
    #def forward(self, x):
        #x_gma = torch.pow(x, np.random.uniform(self.Gtheta[0], self.Gtheta[1]))
        #x_gma = torch.clamp(x_gma, min=0, max=1)

        #feat_ii, feat_ii_gma = list(map(self.iim, [x, x_gma]))

        #feat_proj = self.feat_projector(x)
        #x_cat = torch.cat((feat_proj, feat_ii), dim=1)
        #feat_enhance = self.fuse_net(x_cat)

        # 这里动态获取拼接后的通道数
        #fuse_input = torch.cat([feat_enhance, feat_ii], dim=1)
        #c_in = fuse_input.shape[1]

        # 直接在 forward 中构造 final_fuse（更灵活，也可提前写到 __init__ 中做判断）
        #final_fuse_layer = nn.Sequential(
            #nn.Conv2d(c_in, 32, 3, 1, 1),
#nn.BatchNorm2d(32),
            #nn.LeakyReLU(),
            #nn.Conv2d(32, 3, 3, 1, 1)
        #).to(fuse_input.device)

        #x_out = final_fuse_layer(fuse_input)

        #return x_out, (feat_ii, feat_ii_gma)