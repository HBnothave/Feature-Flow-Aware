import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(pred, target):
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    target = target.cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(target, pred),
        'precision': precision_score(target, pred, average='weighted', zero_division=0),
        'recall': recall_score(target, pred, average='weighted', zero_division=0),
        'f1': f1_score(target, pred, average='weighted', zero_division=0)
    }
    
    return metrics