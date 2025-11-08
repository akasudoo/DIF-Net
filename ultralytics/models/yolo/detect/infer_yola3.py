import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os

def normalize_and_save(tensor, name, out_dir):
    arr = tensor[0, 0].detach().cpu().numpy()
    vis = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255.0
    cv2.imwrite(os.path.join(out_dir, name), vis.astype(np.uint8))

def main():
    # 配置路径
    weights_path = r"D:\ultralytics-main\ultralytics\models\yolo\detect\runs\detect\yola_with_enhancer3\weights\best.pt"
    image_dir = r"D:\yolav11\yolov11\datasets\exdark\images\val"
    output_root = r"D:\yolav11\yolov11\datasets\enhancer_predict_outputs_all"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和增强器
    yolov8_model = YOLO(weights_path)
    model = yolov8_model.model
    enhancer = model.enhancer.to(device)
    yolov8_model.fuse()

    # 创建输出根目录
    os.makedirs(output_root, exist_ok=True)

    # 遍历所有图片
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(image_dir, fname)
        name_no_ext = os.path.splitext(fname)[0]
        output_dir = os.path.join(output_root, name_no_ext)
        os.makedirs(output_dir, exist_ok=True)

        # 读取图片
        img_cv = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)

        # 检测
        results = yolov8_model(image_path)
        results[0].save(filename=os.path.join(output_dir, "detection.jpg"))

        # 增强器中间结果
        with torch.no_grad():
            x_fused_gray, x_colored, _, feat_ii, _, feat_ii_f, _ = enhancer(img_tensor)

        # 保存 x_colored
        x_colored_np = x_colored.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        x_colored_np = np.clip(x_colored_np * 255.0, 0, 255).astype(np.uint8)
        x_colored_bgr = cv2.cvtColor(x_colored_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, "x_colored.jpg"), x_colored_bgr)

        # 可视化特征图
        normalize_and_save(feat_ii, "feat_ii_vis.jpg", output_dir)
        normalize_and_save(feat_ii_f, "feat_ii_f_vis.jpg", output_dir)

        # 保存特征为 numpy
        np.save(os.path.join(output_dir, "feat_ii.npy"), feat_ii[0].detach().cpu().numpy())
        np.save(os.path.join(output_dir, "feat_ii_f.npy"), feat_ii_f[0].detach().cpu().numpy())

        print(f"✅ 处理完成: {fname} → {output_dir}")

if __name__ == "__main__":
    main()