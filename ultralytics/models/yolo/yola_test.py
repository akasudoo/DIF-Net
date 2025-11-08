import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from enfeat2 import ImageEnhancer

# 路径配置
input_dir = r'D:\yolav11\yolov11\datasets\exdark\images\val'
output_dir = r'D:\yolav11\yolov11\datasets\exdark\images\iim'
os.makedirs(output_dir, exist_ok=True)
ckpt_path = r'D:\ultralytics-main\ultralytics\models\yolo\detect\runs\detect\yola_with_enhancer3\weights\best.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ImageEnhancer().to(device)

def extract_state_dict(obj):
    """
    自动递归提取最终的 state_dict (dict)，
    兼容Ultralytics所有保存形式。
    """
    if isinstance(obj, dict):
        # 常见Ultralytics结构
        if 'model' in obj:
            return extract_state_dict(obj['model'])
        elif 'state_dict' in obj:
            return extract_state_dict(obj['state_dict'])
        else:
            return obj
    elif hasattr(obj, 'state_dict'):
        return obj.state_dict()
    else:
        raise ValueError(f"找不到可用的state_dict, 类型: {type(obj)}")

# 加载权重
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = extract_state_dict(ckpt)
print("最终提取到的state_dict类型:", type(state_dict))
print("dict keys:", list(state_dict.keys())[:10])

# 严格/非严格加载都支持，missing/unexpected自动提示
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("权重加载完成，missing:", missing, "unexpected:", unexpected)
model.eval()

img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

with torch.no_grad():
    for fname in tqdm(img_files, desc="Extracting IIM (iim) features"):
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f'Warning: cannot load {img_path}')
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

        # 获取 i.i.m（IIBlock）输出的空间特征
        _, _, _, feat_ii, *_ = model(img_tensor)
        feat_ii = feat_ii[0].cpu().numpy()  # [C, H, W]

        # 取前三通道作为RGB，分别归一化
        feat_rgb = feat_ii[:3]
        for c in range(3):
            channel = feat_rgb[c]
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)
            feat_rgb[c] = channel
        feat_rgb_img = (feat_rgb * 255).astype(np.uint8).transpose(1, 2, 0)  # [H, W, 3]

        out_path = os.path.join(output_dir, f'{os.path.splitext(fname)[0]}_iim_rgb.png')
        cv2.imwrite(out_path, cv2.cvtColor(feat_rgb_img, cv2.COLOR_RGB2BGR))

print('所有图片已处理并保存！')
