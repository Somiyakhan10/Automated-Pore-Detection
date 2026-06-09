"""
models/model_manager.py
Backbone loading and caching. Supports ResNet-50 and DINOv2 ViT-S/14.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DEVICE, RESNET_EMBED_DIM, DINO_EMBED_DIM


def load_backbone(prefer_dino: bool = False) -> Tuple[nn.Module, int, str]:
    """
    Returns (backbone, embed_dim, name).

    Priority when prefer_dino=True: DINOv2 → ResNet-50.
    Default: ResNet-50 (always available offline).
    """
    if prefer_dino:
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", verbose=False
            )
            model = model.to(DEVICE).eval()
            return model, DINO_EMBED_DIM, "dinov2"
        except Exception as e:
            print(f"[model_manager] DINOv2 unavailable ({e}). Falling back to ResNet-50.")

    resnet   = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(resnet.children())[:-1])   # drop fc head
    backbone = backbone.to(DEVICE).eval()
    return backbone, RESNET_EMBED_DIM, "resnet50"


def get_backbone_info(name: str) -> str:
    infos = {
        "resnet50": (
            "ResNet-50 (ImageNet1K-V2)  |  embed_dim=2048  |  "
            "Global average-pooled feature vector from the pre-classification layer."
        ),
        "dinov2": (
            "DINOv2 ViT-S/14 (self-supervised)  |  embed_dim=384  |  "
            "CLS token from a Vision Transformer trained via self-distillation "
            "without labels — excellent for texture and structural similarity."
        ),
    }
    return infos.get(name, "Unknown model")
