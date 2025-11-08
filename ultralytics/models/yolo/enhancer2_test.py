import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

from ultralytics.models.yolo.enfeat2 import ImageEnhancer


def load_image(image_path, img_size=640):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)
    return image_tensor, image


def save_tensor_image(tensor, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tensor = tensor.squeeze(0).clamp(0, 1).cpu()
    image = transforms.ToPILImage()(tensor)
    image.save(save_path)


def enhance_all_images(input_dir, output_dir, img_size=640):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageEnhancer().to(device).eval()

    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_paths = [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if os.path.splitext(fname)[-1].lower() in img_exts
    ]

    print(f"共发现 {len(image_paths)} 张图像，开始增强处理...")
    for image_path in tqdm(image_paths, desc="增强图像处理中"):
        try:
            input_tensor, _ = load_image(image_path, img_size=img_size)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                _, enhanced_tensor, _, _ = model(input_tensor)  # 保留4个返回值

            relative_name = os.path.basename(image_path)
            save_path = os.path.join(output_dir, relative_name)
            save_tensor_image(enhanced_tensor, save_path)

        except Exception as e:
            print(f"[错误] 图像处理失败: {image_path}，错误信息: {e}")

    print(f"全部图像处理完成，增强图像已保存至: {output_dir}")


if __name__ == "__main__":
    input_dir = r"D:\yolav11\yolov11\datasets\exdark\images\val"
    output_dir = r"D:\yolav11\yolov11\datasets\exdark\images\enfeat2_pseudo"
    enhance_all_images(input_dir, output_dir, img_size=640)
