import torch
import torch.nn as nn
import numpy as np
from ultralytics.models.yolo.yola3_illm import IIBlock
from ultralytics.models.yolo.yola5_illm3 import fIIBlock
from ultralytics.models.yolo.illm import IlluminationMapper


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

        print(f"[DEBUG] spatial_feat.shape: {spatial_feat.shape}, freq_feat.shape: {freq_feat.shape}")
        #x_cat = torch.cat([spatial_feat, freq_feat], dim=1)

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
        输入: imgs (B, 3, H, W)
        输出:
            x_fused: 空间与频域融合特征 (B, 3, H, W)
            x_out: 融合illum后的特征 (B, 3, H, W) —— 经自适应融合后仍为3通道
            features_all: 用于一致性损失的4组特征
        """
        device = imgs.device

        # 1. 提取illumination map（灰度图 → 3通道）
        illum_maps = []
        for img in imgs:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            illum_gray = self.illum_mapper.generate_illumination_map(img_np)
            illum_tensor = torch.from_numpy(illum_gray).unsqueeze(0).float().to(device)
            illum_maps.append(illum_tensor)
        illum_stack = torch.stack(illum_maps, dim=0)  # (B, 1, H, W)
        illum_stack = illum_stack.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        # 2. 空间域增强
        x_spatial, (feat_ii, feat_ii_gma) = self.iim(imgs)

        # 3. 频域增强
        x_freq, (feat_ii_f, feat_ii_gma_f) = self.fiim(imgs)

        # 4. 空频融合
        x_fused = self.fusion(x_spatial, x_freq)
        x_fused = self.dropout(x_fused)

        # 5. 使用illum_stack进一步自适应融合（替代拼接）
        x_out = self.illum_fusion(x_fused, illum_stack)

        features_all = (feat_ii, feat_ii_gma, feat_ii_f, feat_ii_gma_f)
        return x_fused, x_out, features_all

