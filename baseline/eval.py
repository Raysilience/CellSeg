import logging
from os import PathLike
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.segmentation
from PIL import Image
from tqdm import tqdm

from baseline.utils.misc import load_img


def intersection_over_union(ground_truth, prediction):
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))

    # Compute intersection
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects, pred_objects))
    intersection = h[0]

    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]

    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]

    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection / union

    return IOU


def measures_at(threshold, IOU):
    matches = IOU > threshold

    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects

    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))

    TP, FP, FN = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

    f1 = 2 * TP / (2 * TP + FP + FN + 1e-9)
    official_score = TP / (TP + FP + FN + 1e-9)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)

    return f1, TP, FP, FN, official_score, precision, recall


def avg_f1(labels: List[Union[str, PathLike]], preds: List[Union[str, PathLike]], thresh: float = 0.5):
    if len(labels) != len(preds):
        logging.error("labels and preds don't match")
        return None
    f1 = 0.0
    labels = sorted(labels)
    preds = sorted(preds)
    for i in tqdm(range(len(labels))):
        gt = load_img(labels[i])
        pr = load_img(preds[i])
        iou = intersection_over_union(gt, pr)
        curr, *_ = measures_at(thresh, iou)
        # print(curr)
        f1 = f1 + (curr - f1) / (i + 1)

    return f1


# Compute Average Precision for all IoU thresholds

def compute_af1_results(ground_truth, prediction, results, image_name):
    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    if IOU.shape[0] > 0:
        jaccard = np.max(IOU, axis=0).mean()
    else:
        jaccard = 0.0

    # Calculate F1 score at all thresholds
    for t in np.arange(0.5, 1.0, 0.05):
        f1, tp, fp, fn, os, prec, rec = measures_at(t, IOU)
        res = {"Image": image_name, "Threshold": t, "F1": f1, "Jaccard": jaccard,
               "TP": tp, "FP": fp, "FN": fn, "Official_Score": os, "Precision": prec, "Recall": rec}
        row = len(results)
        results.loc[row] = res

    return results


# Count number of False Negatives at 0.7 IoU

def get_false_negatives(ground_truth, prediction, results, image_name, threshold=0.7):
    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)

    true_objects = len(np.unique(ground_truth))
    if true_objects <= 1:
        return results

    area_true = np.histogram(ground_truth, bins=true_objects)[0][1:]
    true_objects -= 1

    # Identify False Negatives
    matches = IOU > threshold
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects

    data = np.asarray([
        area_true.copy(),
        np.array(false_negatives, dtype=np.int32)
    ])

    results = pd.concat([results, pd.DataFrame(data=data.T, columns=["Area", "False_Negative"])])

    return results


# Count the number of splits and merges

def get_splits_and_merges(ground_truth, prediction, results, image_name):
    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)

    matches = IOU > 0.1
    merges = np.sum(matches, axis=0) > 1
    splits = np.sum(matches, axis=1) > 1
    r = {"Image_Name": image_name, "Merges": np.sum(merges), "Splits": np.sum(splits)}
    results.loc[len(results) + 1] = r
    return results


if __name__ == '__main__':
    # label_file = "/root/CellSeg/data/Train_Pre_3class/labels/cell_00001_label.png"
    gt_dir = "/root/CellSeg/data/Train_Labeled/labels/"
    pr_dir = "/root/CellSeg/outputs/swinunetr/"
    gt_lst = list(Path(gt_dir).glob('*'))
    pr_lst = list(Path(pr_dir).glob('*'))

    print(avg_f1(gt_lst, pr_lst))

    gt = np.zeros((5, 5))
    gt[1:3, 1:3] = 1
    # print(gt)
    pred = np.zeros_like(gt)
    pred[:2, :2] = 2
    pred[2:, 2:] = 1
    pred[3, 3] = 0
    # print(pred)
    gt = skimage.morphology.label(gt)
    pred = skimage.morphology.label(pred)

    res = intersection_over_union(gt, pred)
    # print(gt, pred)
    # kernel = skimage.morphology.square(1)
    # pred = skimage.morphology.dilation(pred, kernel)
    # print(kernel)
    # print(pred)
    # intersection_over_union(gt, pred)
    # gt = skimage.segmentation.relabel_sequential(gt)[0]
    # pred = skimage.segmentation.relabel_sequential(pred)[0]
    print(res)
    measures_at(0, res)
    # res = skimage.morphology.label(gt)

