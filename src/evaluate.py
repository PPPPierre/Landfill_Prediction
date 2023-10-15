import os
import torch
import logging

from torch.utils.data import DataLoader

from .metrics import compute_metrics
from .model import  get_model_from_cfg
from .loss import BCEWithLogitsLoss
from .dataset import get_dataloader_from_cfg, SatelliteDataset

def evaluate(
        cfg: dict, 
        model: torch.nn.Module, 
        test_loader: DataLoader, 
        criterion: torch.nn.Module, 
        device: torch.device):
    
    model.eval()  # Set the model to evaluation mode
    
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1).float()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # convert outputs to probabilities and classify
            probs = torch.sigmoid(outputs)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    metrics = compute_metrics(all_preds, all_labels)
    metrics['evaluate_loss'] = total_loss / len(test_loader.dataset)
    return metrics


if __name__ == "__main__":

    cfg = {
        "name": "ResNet18",
        "params": {}
    }

    model = get_model_from_cfg(cfg)
    BCEloss = BCEWithLogitsLoss()

    cfg = {
        "geojson_path": "./data/raw/train.geojson",
        "datetime": "2022-01-01/2022-12-30",
        "transforms": [
            {
                "name": "to_tensor", 
                "params": {}
            },
            {
                "name": "resize", 
                "params": {"size": (224, 224)}
            },
            {
                "name": "normalize", 
                "params": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            },
        ],
        "batch_size": 1,
        "shuffle": False
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset, dataloader = get_dataloader_from_cfg(cfg)
    model = model.to(device)
    result = evaluate({}, model, dataloader, BCEloss, device)
    print(result)