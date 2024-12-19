# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature

import torch

#from mmcv.runner import force_fp32, auto_fp16
from mmengine.registry import build_from_cfg
from mmdet.models import (
    BaseDetector,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["Sparse4D"]

from mmdet.models.backbones.resnet import ResNet
from mmdet.models.necks.fpn import FPN
from projects.mmdet3d_plugin.models.sparse4d_head import Sparse4DHead
from mmdet.registry import MODELS
from mmengine.model.utils import detect_anomalous_params

@MODELS.register_module()
class Sparse4D(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(Sparse4D, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        self.head = MODELS.build(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = MODELS.build(depth_branch)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    def extract_feat(self, img, return_depth=False, metas=None):
        with torch.cuda.amp.autocast():
            bs = img.shape[0]
            #img = img.to(dtype=torch.float16)
            if img.dim() == 5:  # multi-view
                num_cams = img.shape[1]
                img = img.flatten(end_dim=1)
            else:
                num_cams = 1
            if self.use_grid_mask:
                img = self.grid_mask(img)
            if "metas" in signature(self.img_backbone.forward).parameters:
                feature_maps = self.img_backbone(img, num_cams, metas=metas)
            else:
                feature_maps = self.img_backbone(img)
            if self.img_neck is not None:
                feature_maps = list(self.img_neck(feature_maps))
            for i, feat in enumerate(feature_maps):
                feature_maps[i] = torch.reshape(
                    feat, (bs, num_cams) + feat.shape[1:]
                )
            if return_depth and self.depth_branch is not None:
                depths = self.depth_branch(feature_maps, metas.get("focal"))
            else:
                depths = None
            if self.use_deformable_func:
                feature_maps = feature_maps_format(feature_maps)
            if return_depth:
                return feature_maps, depths
            return feature_maps

    def _forward(self, img, **data):
        if self.training:
            return self.forward_train(img.to(dtype=torch.float32), **data)
        else:
            return self.forward_test(img.to(dtype=torch.float32), **data)
    
    def loss(self, x, y):
        # Implement loss computation
        pass

    def predict(self, x):
        # Implement prediction
        pass

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)

    def train_step(self, data, optim_wrapper):
        # move tensors to gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data["focal"] = torch.stack(data["focal"]).to(device)
        data["img"] = torch.stack(data["img"]).to(device)
        data["projection_mat"] = torch.stack(data["projection_mat"]).to(device)
        data["image_wh"] = torch.stack(data["image_wh"]).to(device)
        data["timestamp"] = torch.tensor(data["timestamp"]).to(device)
        data["gt_labels_3d"] = [t.to(device) for t in data["gt_labels_3d"]]
        data["gt_bboxes_3d"] = [t.to(device) for t in data["gt_bboxes_3d"]]
        data["instance_id"] = [t.to(device) for t in data["instance_id"]]
        data["gt_depth"] = [t.to(device) for t in data["gt_depth"]]

        with optim_wrapper.optim_context(self):
            feature_maps, depths = self.extract_feat(data["img"], True, data)
            model_outs = self.head(feature_maps, data)
            output = self.head.loss(model_outs, data)
            if depths is not None and "gt_depth" in data:
                output["loss_dense_depth"] = self.depth_branch.loss(
                    depths, data["gt_depth"]
                )
        parsed_loss, log_vars = self.parse_losses(output)
        optim_wrapper.update_params(parsed_loss)
        detect_anomalous_params(parsed_loss, model=self)
        return log_vars
