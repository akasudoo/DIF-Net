import os
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from enfaet import ImageEnhancer  # 请确保这个模块路径正确

# 加载图像
def load_image(img_path, img_size=640):
    img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
    img = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # shape: (1, 3, H, W)
    return img_tensor

# 保存图像
def save_tensor_image(tensor_img, save_path):
    img = tensor_img.squeeze(0).cpu().clamp(0, 1)  # (3, H, W)
    torchvision.utils.save_image(img, save_path)

def main():
    # 🔧 数据路径配置
    input_dir = r'D:\yolav11\yolov11\datasets\exdark\images\val'
    output_dir = r'D:\yolav11\yolov11\datasets\exdark\images\enfeat'
    os.makedirs(output_dir, exist_ok=True)

    # ✅ 加载增强器模型和权重
    enhancer = ImageEnhancer().eval().cuda()
    weights_path = r'D:\yolav11\yolov11\ultralytics\models\yolo\enhancer_only.pth'
    state_dict = torch.load(weights_path, map_location='cuda')
    enhancer.load_state_dict(state_dict)
    print(f"✅ Loaded trained weights from {weights_path}")

    # 🔄 遍历图片并增强
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Found {len(image_files)} images to enhance.")

    for img_name in tqdm(image_files, desc="Enhancing images"):
        img_path = os.path.join(input_dir, img_name)
        save_path = os.path.join(output_dir, img_name)

        # 加载图像
        img_tensor = load_image(img_path).cuda()

        # 增强
        with torch.no_grad():
            x_fused, _, _ = enhancer(img_tensor)  # 只保存融合后的输出图像

        # 保存增强图像
        save_tensor_image(x_fused, save_path)

    print(f"✅ All enhanced images saved to: {output_dir}")

if __name__ == '__main__':
    main()
