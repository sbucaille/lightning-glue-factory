"""PyTorch implementation of the SuperPoint model,
   derived from the TensorFlow re-implementation (2018).
   Authors: Rémi Pautrat, Paul-Edouard Sarlin
   https://github.com/rpautrat/SuperPoint
   The implementation of this model and its trained weights are made
   available under the MIT license.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from lightning_gluefactory.models.base_model import BaseModel, BaseModelConfig
from gluefactory.models.utils.misc import pad_and_stack


class SuperPointOpenConfig(BaseModelConfig):
    """
    Configuration for the SuperPoint model.
    """
    _target_: str = "lightning_gluefactory.models.extractors.superpoint_open.SuperPoint"
    descriptor_dim: int = 256
    nms_radius: int = 4
    max_num_keypoints: Optional[int] = None
    force_num_keypoints: bool = False
    detection_threshold: float = 0.015
    remove_borders: int = 4
    channels: List[int] = field(default_factory=lambda: [64, 64, 128, 128, 256, 256, 512, 512])
    dense_outputs: bool = False
    checkpoint_url: str = "https://github.com/rpautrat/SuperPoint/raw/master/weights/superpoint_v6_from_tf.pth"


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


def batched_nms(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding
        )
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        bn = nn.BatchNorm2d(c_out, eps=0.001)
        super().__init__(
            OrderedDict(
                [
                    ("conv", conv),
                    ("activation", activation),
                    ("bn", bn),
                ]
            )
        )


class SuperPoint(BaseModel):

    def __init__(
            self,
            descriptor_dim: int,
            nms_radius: int,
            max_num_keypoints: int,
            force_num_keypoints: bool,
            detection_threshold: float,
            remove_borders: int,
            channels: List[int],
            dense_outputs: bool,
            checkpoint_url: str,
            **kwargs
    ):
        self.descriptor_dim = descriptor_dim
        self.nms_radius = nms_radius
        self.max_num_keypoints = max_num_keypoints
        self.force_num_keypoints = force_num_keypoints
        self.detection_threshold = detection_threshold
        self.remove_borders = remove_borders
        self.channels = channels
        self.dense_outputs = dense_outputs
        self.checkpoint_url = checkpoint_url
        super().__init__(**kwargs)

    def _init(self):
        self.stride = 2 ** (len(self.channels) - 2)
        channels = [1, *self.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride ** 2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.descriptor_dim, 1, relu=False),
        )

        state_dict = torch.hub.load_state_dict_from_url(self.checkpoint_url)
        self.load_state_dict(state_dict)

    def _forward(self, data):
        image = data["image"]
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        features = self.backbone(image)
        descriptors_dense = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )

        # Decode the detection scores
        scores = self.detector(features)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )
        scores = batched_nms(scores, self.nms_radius)

        # Discard keypoints near the image borders
        if self.remove_borders:
            pad = self.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        if b > 1:
            idxs = torch.where(scores > self.detection_threshold)
            mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
        else:  # Faster shortcut
            scores = scores.squeeze(0)
            idxs = torch.where(scores > self.detection_threshold)

        # Convert (i, j) to (x, y)
        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = scores[idxs]

        keypoints = []
        scores = []
        for i in range(b):
            if b > 1:
                k = keypoints_all[mask[i]]
                s = scores_all[mask[i]]
            else:
                k = keypoints_all
                s = scores_all
            if self.max_num_keypoints is not None:
                k, s = select_top_k_keypoints(k, s, self.max_num_keypoints)

            keypoints.append(k)
            scores.append(s)

        if self.force_num_keypoints:
            keypoints = pad_and_stack(
                keypoints,
                self.max_num_keypoints,
                -2,
                mode="random_c",
                bounds=(
                    0,
                    data.get("image_size", torch.tensor(image.shape[-2:])).min().item(),
                ),
            )
            scores = pad_and_stack(
                scores, self.max_num_keypoints, -1, mode="zeros"
            )
        else:
            keypoints = torch.stack(keypoints, 0)
            scores = torch.stack(scores, 0)

        if len(keypoints) == 1 or self.force_num_keypoints:
            # Batch sampling of the descriptors
            desc = sample_descriptors(keypoints, descriptors_dense, self.stride)
        else:
            desc = [
                sample_descriptors(k[None], d[None], self.stride)[0]
                for k, d in zip(keypoints, descriptors_dense)
            ]

        pred = {
            "keypoints": keypoints + 0.5,
            "keypoint_scores": scores,
            "descriptors": desc.transpose(-1, -2),
        }
        if self.dense_outputs:
            pred["dense_descriptors"] = descriptors_dense

        return pred
