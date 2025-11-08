from ultralytics import YOLO

if __name__ == '__main__':
    #weights = r'D:\ultralytics-main\ultralytics\models\yolo\detect\runs\detect\yola_with_enhancer3\weights\best.pt'
    weights = r'D:\ultralytics-main\ultralytics\models\yolo\detect\runs\pre.pt'
    data_yaml = 'exdark.yaml'
    # 只要你的模型和Trainer在训练时用的自定义注册方式还在，这样就能自动使用你的集成模型
    model = YOLO(weights)
    model.val(
        data=data_yaml,
        imgsz=512,
        batch=8,
        device=0
    )
