# train_custom.py

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.yolo_enfeat import YOLA  # 请替换为你实际放置模型类的路径
from torch.optim.lr_scheduler import MultiStepLR

class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None,**kwargs):
        # data为dict, nc为类别数
        nc = self.data.get('nc', 1)
        model = YOLA(cfg=cfg or 'yolov8.yaml', nc=nc)
        if weights:
            model.load(weights)
            #for param in model.parameters():
                #param.requires_grad = False
            #print('>>>>> 已冻结所有参数！')

        return model

if __name__ == '__main__':
    #YOLO(r'D:\ultralytics-main\ultralytics\models\yolo\detect\runs\detect\yola_with_enhancer').train(
    YOLO(r'D:\ultralytics-main\ultralytics\models\yolo\detect\yolov8n.pt').train(
        trainer=CustomTrainer,
        data='coco.yaml',
        #epochs=100,
        #lr0=1e-5,
        lrf=0.01,
        #imgsz=640,
        weight_decay=0.0005,
        workers=2,
        device='0',
        imgsz=512,
        batch=16,
        name='yola_with_enhancer',
        amp = False,
        resume=False,
        optimizer='SGD',
        epochs=276,
        lr0=1e-2,
        #optimizer='SGD',
        #lr_scheduler='multistep',   # 指定MultiStepLR
        #lr_factor=0.1,              # 衰减因子
        #lr_milestones=[18,23],      # 18、23轮衰减
    )