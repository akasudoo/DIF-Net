####加入isp合成低光图片对

import torch
import torch.nn as nn
import numpy as np
from ultralytics.models.yolo.yola3_illm import IIBlock
from ultralytics.models.yolo.yola5_illm3 import fIIBlock
from ultralytics.models.yolo.illm import IlluminationMapper
from ultralytics.models.yolo.darkisp import batch_low_illumination_degrading  # 合成低光模块


class AdaptiveFusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, bias=False)
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, spatial_feat, freq_feat):
        assert isinstance(spatial_feat, torch.Tensor), f"spatial_feat type: {type(spatial_feat)}"
        assert isinstance(freq_feat, torch.Tensor), f"freq_feat type: {type(freq_feat)}"

        x_cat = torch.cat([spatial_feat, freq_feat], dim=1)
        weights = self.conv(x_cat)
        B, C2, H, W = weights.shape
        C = C2 // 2
        w_s, w_f = weights[:, :C], weights[:, C:]
        ws, wf = torch.softmax(torch.stack([w_s, w_f], dim=1), dim=1).unbind(dim=1)
        fused = spatial_feat * ws + freq_feat * wf
        out = self.residual(fused) + fused
        return out


class ImageEnhancer(nn.Module):
    def __init__(self, kernel_nums=8, kernel_size=3, Gtheta=[0.6, 0.8], channels=3):
        super().__init__()
        self.iim = IIBlock(kernel_nums=kernel_nums, kernel_size=kernel_size, Gtheta=Gtheta)
        self.fiim = fIIBlock(kernel_nums=kernel_nums, kernel_size=kernel_size, Gtheta=Gtheta)

        self.fusion = AdaptiveFusionModule(channels=channels)             # 空间 + 频域
        self.illum_fusion = AdaptiveFusionModule(channels=channels)       # x_fused + illum_stack

        self.illum_mapper = IlluminationMapper()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, imgs):
        """
        输入: imgs (B, 3, H, W) - 原始正常图像
        输出:
            x_fused: 空间与频域融合特征 (B, 3, H, W)
            x_out: 融合illum后的特征 (B, 3, H, W) —— 经自适应融合后仍为3通道
            features_all: 用于一致性损失的4组特征
        """
        device = imgs.device

        # 1. 合成低光照图像
        with torch.no_grad():
            x_low = batch_low_illumination_degrading(imgs).to(device)

        # 2. 提取illumination map（来自低光图像）
        illum_maps = []
        for img in x_low:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            illum_gray = self.illum_mapper.generate_illumination_map(img_np)
            illum_tensor = torch.from_numpy(illum_gray).unsqueeze(0).float().to(device)
            illum_maps.append(illum_tensor)
        illum_stack = torch.stack(illum_maps, dim=0)  # (B, 1, H, W)
        illum_stack = illum_stack.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        # 3. 空间域增强（基于低光图像）
        x_spatial, (feat_ii) = self.iim(x_low)
        x_spatial2, (feat_ii2) = self.iim(imgs)

        # 4. 频域增强（基于低光图像）
        x_freq, (feat_ii_f) = self.fiim(x_low)
        x_freq2, (feat_ii_f2) = self.fiim(imgs)

        # 5. 空频融合
        x_fused = self.fusion(x_spatial, x_freq)
        x_fused2 = self.fusion(x_spatial2, x_freq2)
        x_fused = self.dropout(x_fused)
        x_fused2 = self.dropout(x_fused2)

        # 6. 使用illum_stack进一步融合
        x_out = self.illum_fusion(x_fused, illum_stack)
        x_out2 = self.illum_fusion(x_fused2, illum_stack)

        features_all = (feat_ii, feat_ii2, feat_ii_f, feat_ii_f2)
        return x_fused, x_out, features_all,x_out2
