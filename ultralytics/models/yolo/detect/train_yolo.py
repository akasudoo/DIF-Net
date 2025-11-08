from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\ultralytics-main\ultralytics\models\yolo\detect\yolov8n.pt')
    model.train(
        data='coco.yaml',  # 替换成你的数据集
        epochs=300,
        imgsz=640,
        batch=16,
        workers=8,
        project='runs/train',
        name='yolov8n-coco-finetune',
        device=0,
        freeze=None,
    )
