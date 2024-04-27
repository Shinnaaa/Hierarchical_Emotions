import os
import random
import logging

import torch
import numpy as np
import ot
from cost_matric import compute_cost_matrix, hierarchy, leaf_labels

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss

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
    global M_tensor  # 确保 M_tensor 已初始化

    # 将 labels 和 logits 转换为 tensor
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    batch_size = min(len(labels_tensor), len(logits_tensor))  # 确保 batch_size 不超出任一张量的大小

    # 确保预测和真实标签的和相等
    for i in range(batch_size):  # 确保不超出范围
        if i >= labels_tensor.size(0) or i >= logits_tensor.size(0):
            raise IndexError(f"Index {i} out of bounds")

        # 对 labels 进行归一化，确保和为 1
        labels_sum = torch.sum(labels_tensor[i])

        if labels_sum > 0:
            labels_tensor[i] = labels_tensor[i] / labels_sum  # 归一化
        else:
            labels_tensor[i] = torch.zeros_like(labels_tensor[i])  # 零和处理

    # 计算 EMD
    for i in range(batch_size):
        single_logit = logits_tensor[i]
        single_label = labels_tensor[i]

        if torch.sum(single_logit) > 0 and torch.sum(single_label) > 0:
            # 使用 M_tensor 计算 EMD
            emd = ot.emd2(single_logit, single_label, M_tensor)
            total_emd += emd

    final_emd = total_emd / batch_size  # 计算平均 EMD
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

    # 使用logits计算EMD
    results["emd"] = emd_compute(labels, logits)

    return results