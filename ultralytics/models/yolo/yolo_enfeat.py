import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel

# (Assume ImageEnhancer and its dependencies (IlluminationMapper, IIBlock, fIIBlock, etc.)
# are defined elsewhere and imported correctly. For example:)
from ultralytics.models.yolo.enfeat2 import ImageEnhancer  # replace with actual import path of ImageEnhancer


class YOLA(DetectionModel):
    """YOLOv8 DetectionModel with integrated ImageEnhancer and consistency loss."""

    def __init__(self, cfg='yolov8.yaml', ch=3, nc=None, consistency_weight=10.0, verbose=True):
        """
        Initialize the YOLA model.
        :param cfg: YOLO model configuration (YAML path or dict).
        :param ch: Number of input channels (default 3 for RGB).
        :param nc: Number of detection classes (overrides cfg if provided).
        :param consistency_weight: Weight for the consistency L1 loss.
        :param verbose: Whether to print model info.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # build the YOLOv8 detection model
        # Insert the ImageEnhancer module at the input
        self.enhancer = ImageEnhancer()
        # Consistency loss weight (for L1 loss between spatial and frequency features)
        self.consistency_weight = consistency_weight
        # Define L1 (Smooth L1) loss for consistency computation
        # Using SmoothL1Loss as specified (can also use nn.L1Loss for pure L1)
        self.consistency_loss_fn = nn.SmoothL1Loss(reduction='sum')

    def forward(self, x, *args, **kwargs):
        # 训练阶段：x是batch dict
        if isinstance(x, dict):
            return self.loss(x)
        # 推理/验证阶段：x是图像tensor，允许传入多余的关键字参数
        if hasattr(self, 'enhancer') and isinstance(x, torch.Tensor) and x.shape[1] == 3:
            x_fused_gray, x_colored, features_all, feat_ii, feat_ii_gma, feat_ii_f, feat_ii_gma_f = self.enhancer(x)
            preds = super().forward(x_colored, *args, **kwargs)
            return preds
        else:
            # 结构推理等特殊情况
            return super().forward(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        # batch['img'] 是图片张量
        images = batch['img']
        x_fused_gray, x_colored, features_all, feat_ii, feat_ii_gma, feat_ii_f, feat_ii_gma_f = self.enhancer(images)
        if preds is None:
            preds = super().forward(x_colored)
        # 3. Compute the standard YOLOv8 detection loss using the built-in criterion
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()  # initialize v8DetectionLoss
        det_loss, det_items = self.criterion(preds,
                                             batch)  # det_items: tensor of [loss_box, loss_cls, loss_dfl] (detached)
        # 4. Compute consistency L1 loss between spatial and frequency features from the enhancer
        loss_cons_1 = self.consistency_loss_fn(feat_ii, feat_ii_f)  # L1 loss between feat_ii and feat_ii_f
        loss_cons_2 = self.consistency_loss_fn(feat_ii_gma,
                                               feat_ii_gma_f)  # L1 loss between feat_ii_gma and feat_ii_gma_f
        consistency_loss = loss_cons_1 + loss_cons_2
        # 5. Combine losses: total = detection loss + consistency loss (weighted)
        total_loss = det_loss + self.consistency_weight * consistency_loss
        # 6. Prepare loss outputs (for Ultralytics trainer compatibility):
        # Detachment for logging each component (box, cls, dfl, consistency)
        consistency_item = (self.consistency_weight * consistency_loss).detach()
        # If detection loss items were returned, append consistency for logging
        if isinstance(det_items, torch.Tensor):
            # det_items is e.g. a tensor of shape [3] for (box, cls, dfl)
            loss_items = torch.cat([det_items, consistency_item.unsqueeze(0)])
        else:
            # In case det_items is not a tensor (should not happen in v8DetectionLoss), handle gracefully
            loss_items = torch.tensor([consistency_item], device=consistency_item.device)
        # Return total loss and individual components (the trainer can use this for backprop and logging)
        return total_loss, loss_items
