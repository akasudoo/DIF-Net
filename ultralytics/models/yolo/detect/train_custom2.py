# train_custom.py

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.yolo_enfeat import YOLA  # 请替换为你实际放置模型类的路径
import torch
from torch.optim.lr_scheduler import MultiStepLR

class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, **kwargs):
        # data为dict, nc为类别数
        nc = self.data.get('nc', 1)
        model = YOLA(cfg=cfg or 'yolov8.yaml', nc=nc)
        if weights:
            model.load(weights)
            for param in model.parameters():
                param.requires_grad = False
            print('>>>>> 已冻结所有参数！')
        return model

    def setup_optimizer(self):
        """自定义优化器和学习率调度器"""
        # 1. 优化器：SGD，初始学习率 1e-3
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=1e-3,
            momentum=0.937,
            weight_decay=0.0005
        )
        # 2. 多步学习率衰减：第18和23 epoch时降为1/10
        self.lr_scheduler = MultiStepLR(
            self.optimizer,
            milestones=[18, 23],
            gamma=0.1
        )

if __name__ == '__main__':
    YOLO(r'D:\ultralytics-main\ultralytics\models\yolo\detect\runs\detect\yola_with_enhancer\weights\best.pt').train(
        trainer=CustomTrainer,
        data='exdark.yaml',
        epochs=24,                  # 训练24个epoch
        lr0=1e-3,                   # 初始学习率
        weight_decay=0.0005,
        workers=2,
        device='0',
        imgsz=512,
        batch=8,
        name='yola_with_enhancer',
        amp=False,
        resume=False,
        optimizer='SGD',
        # 注意！不要再写 lr_scheduler/lr_factor/lr_milestones 相关参数
    )
