###进行推理的代码，目标检测输出

import os
from ultralytics import YOLO
import cv2
from pathlib import Path

# --------------------------
# 配置路径
# --------------------------
#MODEL_PATH = r"D:\ultralytics-main\ultralytics\models\yolo\detect\runs\detect\yola_with_enhancer3\weights\best.pt"
MODEL_PATH = r"D:\ultralytics-main\ultralytics\models\yolo\detect\runs\pre.pt"
IMAGE_DIR = r"D:\ultralytics-main\datasets\exdark\images\val"
OUTPUT_DIR = r"D:\ultralytics-main\ultralytics\datasets\exdark\images\77.8"

# --------------------------
# 加载模型
# --------------------------
model = YOLO(MODEL_PATH)  # 自动加载你训练好的自定义 YOLA 模型（包含增强模块）
model.fuse()              # 可选，加速推理

# --------------------------
# 推理图像
# --------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for image_name in image_files:
    image_path = os.path.join(IMAGE_DIR, image_name)

    # Run prediction
    results = model.predict(source=image_path, save=False, imgsz=512, conf=0.25, device=0)

    # 取结果图（结果自动处理好 BGR 可视化图）
    result_img = results[0].plot()  # 获取结果图像（np.ndarray）

    # 保存结果图像
    save_path = os.path.join(OUTPUT_DIR, image_name)
    cv2.imwrite(save_path, result_img)

print(f"\n✅ 所有结果图像已保存至：{OUTPUT_DIR}")
