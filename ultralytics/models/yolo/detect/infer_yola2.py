import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import os

def main():
    # ===================== 配置路径 =====================
    weights_path =r"D:\ultralytics-main\ultralytics\models\yolo\detect\runs\detect\yola_with_enhancer3\weights\best.pt"
    image_path = r"D:\yolav11\yolov11\datasets\exdark\images\val\2015_01456.jpg"
    output_dir = r"D:\yolav11\yolov11\datasets\enhancer_predict_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # ===================== 加载模型 =====================
    yolov8_model = YOLO(weights_path)
    model = yolov8_model.model
    enhancer = model.enhancer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enhancer.to(device)

    # ===================== 图像预处理 =====================
    img_cv = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # ===================== 推理 =====================
    results = yolov8_model(image_path)
    detection_output_path = os.path.join(output_dir, "detection_result.jpg")
    results[0].save(filename=detection_output_path)

    # ===================== 获取中间变量 =====================
    with torch.no_grad():
        x_fused_gray, x_colored, _, feat_ii, _, feat_ii_f, _ = enhancer(img_tensor)

    # 保存增强图 x_colored
    x_colored_np = x_colored.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    x_colored_np = np.clip(x_colored_np * 255.0, 0, 255).astype(np.uint8)
    x_colored_bgr = cv2.cvtColor(x_colored_np, cv2.COLOR_RGB2BGR)
    colored_output_path = os.path.join(output_dir, "x_colored.jpg")
    cv2.imwrite(colored_output_path, x_colored_bgr)

    # 保存 feat_ii 第一个通道
    feat_ii_np = feat_ii[0, 0].detach().cpu().numpy()
    feat_ii_img = (feat_ii_np - feat_ii_np.min()) / (feat_ii_np.max() - feat_ii_np.min() + 1e-6) * 255.0
    cv2.imwrite(os.path.join(output_dir, "feat_ii_vis.jpg"), feat_ii_img.astype(np.uint8))

    # 保存 feat_ii_f 第一个通道
    feat_ii_f_np = feat_ii_f[0, 0].detach().cpu().numpy()
    feat_ii_f_img = (feat_ii_f_np - feat_ii_f_np.min()) / (feat_ii_f_np.max() - feat_ii_f_np.min() + 1e-6) * 255.0
    cv2.imwrite(os.path.join(output_dir, "feat_ii_f_vis.jpg"), feat_ii_f_img.astype(np.uint8))

    # 保存中间特征原始数组为 .npy
    np.save(os.path.join(output_dir, "feat_ii.npy"), feat_ii[0].detach().cpu().numpy())
    np.save(os.path.join(output_dir, "feat_ii_f.npy"), feat_ii_f[0].detach().cpu().numpy())

    print(f"✅ 检测结果已保存到: {detection_output_path}")
    print(f"✅ 增强图像已保存到: {colored_output_path}")
    print(f"✅ 中间特征图像与数组已输出至: {output_dir}")

if __name__ == "__main__":
    main()
