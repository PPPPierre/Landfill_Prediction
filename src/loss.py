import torch
import torch.nn as nn

class BCEWithLogitsLoss:
    def __init__(self) -> None:
        self.criterion = nn.BCEWithLogitsLoss()

    def __call__(self, preds: torch.Tensor, labels: torch.Tensor):
        return self.forward(preds, labels)

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        return self.criterion(preds, labels)