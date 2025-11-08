###yolo+iim模块的结果可视化（加载图片，输出非光照分量）


import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.yola3_illm  import IIBlock
from ultralytics.models.yolo.yola5_illm3   import fIIBlock
import os


def main():
    # 加载训练好的YOLO模型
    weights_path = r"D:\yolav11\yolov11\ultralytics\models\yolo\detect\runs\train\exdark_iim_finetune_frep24\weights\best.pt"
    yolov8_model = YOLO(weights_path)  # 这会自动加载对应的模型结构

    # 图片路径
    image_path = r"D:\yolav11\yolov11\datasets\exdark\images\val0\2015_02087.jpg"
    output_dir = r"D:\yolav11\yolov11\datasets"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 推理并保存检测结果
    results = yolov8_model(image_path)

    # 修正：直接指定 filename 参数保存图像
    detection_output_path = os.path.join(output_dir, "1456_detection_result.jpg")
    results[0].save(filename=detection_output_path)  # 使用 filename 参数

    # 加载IIBlock进行预处理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iim_block = IIBlock(kernel_nums=8, kernel_size=3, Gtheta=[0.6, 0.8]).to(device)
    fiim_block = fIIBlock(kernel_nums=8, kernel_size=3, Gtheta=[0.6, 0.8]).to(device)

    # 读取图像
    img_cv = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)  # 确保 img_tensor 在 device 上

    # IIM预处理
    x_proc, _ = iim_block(img_tensor)
    x_procf, _ = fiim_block(img_tensor)
    # 修正：添加 .detach() 和 .cpu() 确保可以转换为 NumPy
    iim_processed_img = x_proc.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
    iim_processed_img = iim_processed_img.astype(np.uint8)
    iim_processed_img = cv2.cvtColor(iim_processed_img, cv2.COLOR_RGB2BGR)

    iim_processed_imgf = x_procf.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
    iim_processed_imgf = iim_processed_imgf.astype(np.uint8)
    iim_processed_imgf = cv2.cvtColor(iim_processed_imgf, cv2.COLOR_RGB2BGR)

    # 保存IIM预处理后的图像
    iim_output_path = os.path.join(output_dir, "20872_iim_processed.jpg")
    cv2.imwrite(iim_output_path, iim_processed_img)

    iim_output_path = os.path.join(output_dir, "20872_fiim_processed.jpg")
    cv2.imwrite(iim_output_path, iim_processed_imgf)

    print(f"检测结果已保存到: {detection_output_path}")
    print(f"IIM预处理后的图像已保存到: {iim_output_path}")


if __name__ == "__main__":
    main()