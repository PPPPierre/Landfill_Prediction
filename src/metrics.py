from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc

def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> Dict[str, float]:
    fpr, tpr, thresholds = roc_curve(gts, preds)
    auc = roc_auc_score(gts, preds)
    
    # Determine the best threshold using Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Convert probabilities to binary predictions based on optimal threshold
    binary_preds = np.where(preds >= optimal_threshold, 1, 0)

    accuracy = accuracy_score(gts, binary_preds)
    precision = precision_score(gts, binary_preds)
    recall = recall_score(gts, binary_preds)
    f1 = f1_score(gts, binary_preds)
    
    return {
        'AUC': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'optimal_threshold': optimal_threshold
    }


if __name__ == '__main__':
    import torch

    preds = torch.rand(10) # Random float values between 0 and 1
    labels = torch.randint(0, 2, (10,))
    
    metrics = compute_metrics(preds.cpu().numpy(), labels.cpu().numpy())
    print(metrics)