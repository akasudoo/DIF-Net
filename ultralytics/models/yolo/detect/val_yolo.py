from ultralytics import YOLO

if __name__ == '__main__':
    weights = r'D:\ultralytics-main\ultralytics\models\yolo\detect\runs\best.pt'
    data_yaml = 'exdark.yaml'   # 你的数据集yaml路径
    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,     # 验证集配置
        imgsz=640,          # 输入分辨率（可调）
        batch=8,            # batch size
        device=0            # GPU编号, 或 'cpu'
    )
    print(metrics)
