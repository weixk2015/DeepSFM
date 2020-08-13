from __future__ import division
import torch
from torch import nn
import numpy as np
from demon_metrics import *


def compute_errors_test(gt, pred):
    gt = gt.numpy()
    pred = pred.numpy()
    # same scale
    scale = np.sum(gt) / np.sum(pred)
    pred = pred * scale
    n = float(np.size(gt))
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_diff = np.mean(np.abs(gt - pred))
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    pred = pred * scale
    l1_inv = np.mean(np.abs(np.reciprocal(gt) - np.reciprocal(pred)))
    pred = pred * scale
    log_diff = np.log(gt) - np.log(pred)
    sc_inv = np.sqrt(np.sum(np.square(log_diff)) / n - np.square(np.sum(log_diff)) / np.square(n))
    return abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3, l1_inv, sc_inv


def compute_errors_train(gt, pred, valid):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size = gt.size(0)

    for current_gt, current_pred, current_valid in zip(gt, pred, valid):
        valid_gt = current_gt[current_valid]
        valid_pred = current_pred[current_valid]

        if len(valid_gt) == 0:
            continue
        else:
            thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
            a1 += (thresh < 1.25).float().mean()
            a2 += (thresh < 1.25 ** 2).float().mean()
            a3 += (thresh < 1.25 ** 3).float().mean()

            abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
            abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

            sq_rel += torch.mean(((valid_gt - valid_pred) ** 2) / valid_gt)

    return [metric / batch_size for metric in [abs_rel, abs_diff, sq_rel, a1, a2, a3]]
