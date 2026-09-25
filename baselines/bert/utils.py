import os
import random
import logging

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss
from cost_matric import compute_cost_matrix, hierarchy, leaf_labels
import ot

M = compute_cost_matrix(hierarchy, leaf_labels)
M_tensor = torch.tensor(M, dtype=torch.float32)
def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def emd_compute(labels, logits):
    total_emd = 0
    global M_tensor  # 确保M_tensor已初始化
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    batch_size = logits_tensor.size(0)

    for i in range(batch_size):
        if torch.sum(labels_tensor[i]) != torch.sum(logits_tensor[i]):
            labels_sum = torch.sum(labels_tensor[i])
            logits_sum = torch.sum(logits_tensor[i])

            if labels_sum > 0 and logits_sum > 0:
                labels_tensor[i] = labels_tensor[i] / labels_sum
                logits_tensor[i] = logits_tensor[i] / logits_sum
            else:
                labels_tensor[i] = torch.zeros_like(labels_tensor[i])
                logits_tensor[i] = torch.zeros_like(logits_tensor[i])

    for i in range(batch_size):
        single_logit = logits_tensor[i]
        single_label = labels_tensor[i]

        if torch.sum(single_logit) > 0 and torch.sum(single_label) > 0:
            emd = ot.emd2(single_logit, single_label, M_tensor)
            total_emd += emd

    final_emd = total_emd / batch_size  
    return final_emd


def compute_metrics(labels, preds, logits=None):
    assert logits is not None, "Logits must be provided for EMD calculation"
    assert len(preds) == len(labels), "Preds and labels must have the same length"

    results = dict()

    results["accuracy"] = accuracy_score(labels, preds)
    results["macro_precision"], results["macro_recall"], results[
        "macro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    results["micro_precision"], results["micro_recall"], results[
        "micro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0)
    results["weighted_precision"], results["weighted_recall"], results[
        "weighted_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0)
    results["hamming_loss"] = hamming_loss(labels, preds)

    results["emd"] = emd_compute(labels, logits)

    return results
