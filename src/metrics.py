from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> Dict[str, float]:
    accuracy = accuracy_score(preds, gts)
    auc = roc_auc_score(preds, gts)
    precision = precision_score(preds, gts)
    recall = recall_score(preds, gts)
    f1 = f1_score(preds, gts)
    
    return {
        'accuracy': accuracy,
        'AUC': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


if __name__ == '__main__':
    import torch

    preds = torch.randint(0, 2, (10,), dtype=torch.float32)
    labels = torch.randint(0, 2, (10,), dtype=torch.float32)
    
    metrics = compute_metrics(preds.cpu().numpy(), labels.cpu().numpy())
    print(metrics)