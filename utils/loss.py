# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """sqrt(MSE)"""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred, target) + self.eps)


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE: mean(w * (pred-target)^2)
    - w can be a tensor broadcastable to pred
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        err2 = (pred - target) ** 2
        if w is None:
            return err2.mean()
        return (w * err2).sum() / (w.sum() + self.eps)


class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target)


class HuberLoss(nn.Module):
    """
    SmoothL1 / Huber loss. Good for suppressing outliers.
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.loss = nn.SmoothL1Loss(beta=delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute RMSE / MAE / R2
    pred, target: shape (...,)
    """
    pred = pred.detach().flatten().float()
    target = target.detach().flatten().float()

    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse + 1e-8)
    mae = torch.mean(torch.abs(pred - target))

    # R2 = 1 - SSE/SST
    sse = torch.sum((pred - target) ** 2)
    sst = torch.sum((target - torch.mean(target)) ** 2) + 1e-12
    r2 = 1.0 - (sse / sst)

    return {"rmse": rmse.item(), "mae": mae.item(), "r2": r2.item()}


@dataclass
class LossPack:
    """
    If you want a single place to choose loss type:
        loss_pack = LossPack(name="rmse")
        loss = loss_pack(pred, target)
    """
    name: str = "rmse"   # "rmse" | "mse" | "mae" | "huber"
    huber_delta: float = 1.0

    def __post_init__(self):
        n = self.name.lower()
        if n == "rmse":
            self._loss = RMSELoss()
        elif n == "mse":
            self._loss = nn.MSELoss()
        elif n == "mae":
            self._loss = MAELoss()
        elif n == "huber":
            self._loss = HuberLoss(delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss name: {self.name}")

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred, target)
