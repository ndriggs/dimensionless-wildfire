
import torch

def tversky_loss(y_pred, y_true, alpha=0.7, beta=0.3):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    true_pos = torch.sum(y_true * y_pred)
    false_neg = torch.sum(y_true * (1 - y_pred))
    false_pos = torch.sum((1 - y_true) * y_pred)
    tversky = true_pos / (true_pos + alpha * false_neg + beta * false_pos)
    return 1 - tversky
