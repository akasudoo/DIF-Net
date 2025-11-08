import torch
from enfaet import ImageEnhancer  # 替换为你保存的增强模块路径

def extract_enhancer_weights(full_model_ckpt_path, save_path):
    # 1. 加载完整模型权重（YOLOWithIIM 的 state_dict）
    print(f"Loading weights from {full_model_ckpt_path}...")
    full_state_dict = torch.load(full_model_ckpt_path, map_location='cpu')

    # 2. 构造增强模块模型
    enhancer = ImageEnhancer()
    enhancer_state_dict = enhancer.state_dict()

    # 3. 从 full_state_dict 中提取与增强模块对应的键值
    extracted_state_dict = {
        k: v for k, v in full_state_dict.items()
        if k in enhancer_state_dict and enhancer_state_dict[k].shape == v.shape
    }

    # 4. 检查提取结果
    print(f"Extracted {len(extracted_state_dict)} parameters out of {len(enhancer_state_dict)} for enhancer.")

    # 5. 加载这些权重（以确保没有错）
    missing_keys, unexpected_keys = enhancer.load_state_dict(extracted_state_dict, strict=False)
    print("Missing keys (ignored):", missing_keys)
    print("Unexpected keys (ignored):", unexpected_keys)

    # 6. 保存为单独的增强模块权重
    torch.save(enhancer.state_dict(), save_path)
    print(f"Saved enhancer weights to {save_path}")

if __name__ == "__main__":
    # 修改为你的权重路径
    full_model_ckpt_path = r"D:\yolav11\yolov11\ultralytics\models\yolo\detect\runs\train\exdark_iim_finetune_frep24\weights\best.pt"
    save_path = "enhancer_only.pth"
    extract_enhancer_weights(full_model_ckpt_path, save_path)
