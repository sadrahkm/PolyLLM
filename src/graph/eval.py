from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from functions import average_precision_at_k

def calculate_metrics(preds, ground_truths):
    train_pred = torch.cat(preds, dim=0)
    train_ground_truth = torch.cat(ground_truths, dim=0)

    metric = BinaryAUROC()
    auc2 = metric(train_pred, train_ground_truth)

    train_pred = train_pred.cpu().detach().numpy()
    train_ground_truth = train_ground_truth.cpu().detach().numpy()

    auc = roc_auc_score(train_ground_truth, train_pred)
    auprc = average_precision_score(train_ground_truth, train_pred)
    ap_at_k = average_precision_at_k(train_ground_truth, train_pred, k=50)

    return (auc, auc2), auprc, ap_at_k