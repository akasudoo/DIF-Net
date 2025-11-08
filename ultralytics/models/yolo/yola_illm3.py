import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kornia.color import rgb_to_lab
from kornia.filters import guided_blur
from ultralytics.models.yolo.illm import IlluminationMapper

class ReflectedConvolution(nn.Module):
    def __init__(self, kernel_nums=8, kernel_size=3):
        super(ReflectedConvolution, self).__init__()
        self.kernel_nums = kernel_nums
        self.kernel_size = kernel_size
        self.rg_bn = nn.BatchNorm2d(kernel_nums)
        self.gb_bn = nn.BatchNorm2d(kernel_nums)
        self.rb_bn = nn.BatchNorm2d(kernel_nums)
        self.filter = nn.Parameter(torch.randn(self.kernel_nums, 1, self.kernel_size, self.kernel_size))
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

        red_chan = log_img[:, 0, :, :].unsqueeze(1)
        green_chan = log_img[:, 1, :, :].unsqueeze(1)
        blue_chan = log_img[:, 2, :, :].unsqueeze(1)
        normalized_filter = self.mean_constraint(self.filter)

        filt_r1 = F.conv2d(red_chan, weight=normalized_filter, padding=self.kernel_size // 2)
        filt_g1 = F.conv2d(green_chan, weight=-normalized_filter, padding=self.kernel_size // 2)
        filt_rg = self.rg_bn(filt_r1 + filt_g1)

        filt_g2 = F.conv2d(green_chan, weight=normalized_filter, padding=self.kernel_size // 2)
        filt_b1 = F.conv2d(blue_chan, weight=-normalized_filter, padding=self.kernel_size // 2)
        filt_gb = self.gb_bn(filt_g2 + filt_b1)

        filt_r2 = F.conv2d(red_chan, weight=normalized_filter, padding=self.kernel_size // 2)
        filt_b2 = F.conv2d(blue_chan, weight=-normalized_filter, padding=self.kernel_size // 2)
        filt_rb = self.rb_bn(filt_r2 + filt_b2)

        rg = torch.where(zeroMasks[:, 0:1, ...].expand(-1, self.kernel_nums, -1, -1) == 1, 0, filt_rg)
        gb = torch.where(zeroMasks[:, 1:2, ...].expand(-1, self.kernel_nums, -1, -1) == 1, 0, filt_gb)
        rb = torch.where(zeroMasks[:, 2:3, ...].expand(-1, self.kernel_nums, -1, -1) == 1, 0, filt_rb)

        return torch.cat([rg, gb, rb], dim=1)

class IIBlock(nn.Module):
    def __init__(self, kernel_nums=8, kernel_size=3, Gtheta=[0.6, 0.8]):
        super(IIBlock, self).__init__()
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
            nn.Conv2d(32, 12, 3, 1, 1)  # 输出12通道，仅作示例，可按需调整
        )
        self.final_fuse = None  # 动态初始化
        self.iim = ReflectedConvolution(kernel_nums, kernel_size)
        self.illumination_mapper = IlluminationMapper()

    def forward(self, x):
        x_gma = torch.pow(x, np.random.uniform(self.Gtheta[0], self.Gtheta[1]))
        x_gma = torch.clamp(x_gma, min=0, max=1)

        feat_ii, feat_ii_gma = list(map(self.iim, [x, x_gma]))
        lab = rgb_to_lab(x)
        ab = lab[:, 1:, :, :]
        feat_ii_refined = guided_blur(ab, feat_ii, kernel_size=3, eps=1e-2)

        illum_maps = []
        for img in x:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            illum_gray = self.illumination_mapper.generate_illumination_map(img_np)
            illum_tensor = torch.from_numpy(illum_gray)
            if illum_tensor.ndim == 2:
                illum_tensor = illum_tensor.unsqueeze(0)
            elif illum_tensor.ndim == 3 and illum_tensor.shape[0] != 1:
                illum_tensor = illum_tensor.permute(2, 0, 1)
            illum_maps.append(illum_tensor)

        illum_stack = torch.stack(illum_maps, dim=0).to(x.device)
        if illum_stack.ndim == 3:
            illum_stack = illum_stack.unsqueeze(1)

        # 确保illum_stack与feat_enhance通道数一致（如有需要可repeat）
        # 这里repeat次数需与fuse_net最后输出通道一致
        feat_proj = self.feat_projector(x)
        x_cat = torch.cat((feat_proj, feat_ii_refined), dim=1)
        feat_enhance = self.fuse_net(x_cat)  # 如默认输出12通道
        if illum_stack.shape[1] != feat_enhance.shape[1]:
            illum_stack = illum_stack.repeat(1, feat_enhance.shape[1], 1, 1)

        final_input = torch.cat((feat_enhance, illum_stack), dim=1)  # [B, 2*通道, H, W]
        # 动态初始化final_fuse
        if self.final_fuse is None or self.final_fuse[0].in_channels != final_input.shape[1]:
            self.final_fuse = nn.Sequential(
                nn.Conv2d(final_input.shape[1], 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 3, 3, 1, 1)
            ).to(final_input.device)
        x_out = self.final_fuse(final_input)
        return x_out, (feat_ii, feat_ii_gma)
