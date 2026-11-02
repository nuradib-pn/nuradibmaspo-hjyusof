"""Shared-backbone multi-task network: wound type + T (tissue) + I (infection) + M (moisture)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
}


def _strip_classifier(net: nn.Module) -> tuple[nn.Module, int]:
    if hasattr(net, "fc"):  # ResNet
        dim = net.fc.in_features
        net.fc = nn.Identity()
    else:  # EfficientNet / ConvNeXt keep the final Linear as classifier[-1]
        dim = net.classifier[-1].in_features
        net.classifier[-1] = nn.Identity()
    return net, dim


class WoundTIMNet(nn.Module):
    def __init__(
        self,
        head_sizes: dict[str, int],
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
        weights_path: str | None = None,
    ):
        super().__init__()
        ctor, weights = _BACKBONES[backbone]
        net = ctor(weights=weights if pretrained and not weights_path else None)
        if weights_path:
            # A local ImageNet checkpoint, e.g. when download.pytorch.org is unreachable.
            # timm ResNet checkpoints use the same parameter names as torchvision.
            state = torch.load(weights_path, map_location="cpu")
            state = state.get("state_dict", state.get("model", state))
            missing, unexpected = net.load_state_dict(state, strict=False)
            if [k for k in missing if not k.startswith(("fc.", "classifier."))]:
                raise ValueError(f"{weights_path} does not match {backbone}: missing {missing[:5]}")
        self.backbone, dim = _strip_classifier(net)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({name: nn.Linear(dim, n) for name, n in head_sizes.items()})

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.dropout(self.backbone(x))
        return {name: head(feats) for name, head in self.heads.items()}
