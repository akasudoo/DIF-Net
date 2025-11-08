import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.illm import IlluminationMapper
from ultralytics.models.yolo.yola3_illm import IIBlock
from ultralytics.models.yolo.yola5_illm3 import fIIBlock
#from ultralytics.models.yolo.darkisp import batch_low_illumination_degrading
from torch.amp import custom_fwd


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
        print("[Enhancer] 正在增强图像中...", spatial_feat.shape,freq_feat.shape)

        x_cat = torch.cat([spatial_feat, freq_feat], dim=1)
        weights = self.conv(x_cat)
        B, C2, H, W = weights.shape
        C = C2 // 2
        w_s, w_f = weights[:, :C], weights[:, C:]
        ws, wf = torch.softmax(torch.stack([w_s, w_f], dim=1), dim=1).unbind(dim=1)
        fused = spatial_feat * ws + freq_feat * wf
        out = self.residual(fused) + fused
        return out


def enhance_illum(illum_rgb):
    illum_log = torch.log1p(illum_rgb)
    illum_blur = F.avg_pool2d(illum_log, kernel_size=7, stride=1, padding=3)
    enhanced = illum_log - illum_blur
    return torch.sigmoid(enhanced * 4)


class IlluminationFusion(nn.Module):
    def __init__(self, in_channels=3):
        super(IlluminationFusion, self).__init__()
        self.conv_attn = nn.Sequential(
            nn.Conv2d(in_channels * 2, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x_fused_gray, illum_rgb):
        illum_rgb = enhance_illum(illum_rgb)
        alpha = self.conv_attn(torch.cat([x_fused_gray, illum_rgb], dim=1))
        return alpha * illum_rgb + (1 - alpha) * x_fused_gray


class LearnablePseudoColorMapper(nn.Module):
    """
    可学习伪彩色映射模块：增强结构图 + 原图颜色特征 => 彩色增强图
    """
    def __init__(self, in_channels=6, mid_channels=16, out_channels=3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        )

    def forward(self, gray_img, color_img):
        x = torch.cat([gray_img, color_img], dim=1)  # (B, 6, H, W)
        return torch.sigmoid(self.fusion(x))  # 输出 RGB 图像


class ImageEnhancer(nn.Module):
    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def __init__(self):
        super(ImageEnhancer, self).__init__()
        self.illum_mapper = IlluminationMapper()
        self.iim = IIBlock()
        self.fiim = fIIBlock()
        self.fusion = AdaptiveFusionModule(channels=3)
        self.dropout = nn.Dropout2d(0.1)
        self.illum_fusion = IlluminationFusion()
        self.color_mapper = LearnablePseudoColorMapper()

    def forward(self, imgs):
        with torch.cuda.amp.autocast(enabled=False):
            imgs = imgs.float()
            device = imgs.device
            orig_rgb = imgs.clone()

            # Step 1: 合成低光图
            # with torch.no_grad():
            # x_low = batch_low_illumination_degrading(imgs).to(device)

            # Step 2: 生成光照图
            illum_maps = []
            for img in imgs:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                illum_rgb = self.illum_mapper.generate_illumination_map(img_np)
                illum_tensor = torch.from_numpy(illum_rgb).permute(2, 0, 1).float().to(device)
                illum_maps.append(illum_tensor)
            illum_stack = torch.stack(illum_maps, dim=0)  # (B, 3, H, W)

            # Step 3-4: 空频域特征融合
            x_spatial, (feat_ii, feat_ii_gma) = self.iim(imgs)
            x_freq, (feat_ii_f, feat_ii_gma_f) = self.fiim(imgs)
            x_fused = self.fusion(x_spatial, x_freq)
            #x_fused = self.dropout(x_fused)

            # Step 5: Retinex风格融合生成结构图
            x_fused_gray = 0.299 * x_fused[:, 0:1] + 0.587 * x_fused[:, 1:2] + 0.114 * x_fused[:, 2:3]
            x_fused_gray = x_fused_gray.expand(-1, 3, -1, -1)

            # Step 6: 光照引导融合
            x_out = self.illum_fusion(x_fused_gray, illum_stack)

            # Step 7: 使用原图颜色引导生成伪彩色增强图
            x_colored = self.color_mapper(x_out, orig_rgb)

            features_all = (feat_ii, feat_ii_gma, feat_ii_f, feat_ii_gma_f)

            return x_fused_gray, x_colored, features_all, feat_ii, feat_ii_gma, feat_ii_f, feat_ii_gma_f

        """
        imgs: 原图 (B, 3, H, W)
        returns:
            x_fused_gray: 灰度结构图
            x_colored: 彩色增强图
            features_all: 结构特征（空间 + 频域）
            x_low: 合成低光图（可用于训练）
        """

