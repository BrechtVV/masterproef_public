import os
import cv2
import numpy as np

from plot import plot_evaluation
from get_data import get_mask
from constants import MASK_BOX_EXPAND


def compare_libs_masks(plot_folder, dataset_folder, img_set=None):
    """
    Compares the results of the libraries in the libs_folder and saves the comparison as a plot in the plot_folder map.
    For validation and evaluation the ground truth data is used. If the data consists of a certain set it can also be choosen.

    param plot_folder:          the folder where the results will be saved
    param dataset_folder:       the folder containing a subfolder with ground truth data and a folder with the output of different person segmentation libraries
    img_set:                    [opt] list of filenames that can be provided to evaluate a certain set of images
    returns:                    no return value but the plots for accuracy and distance are saved to the plot_folder
    """

    libs_folder = os.path.join(dataset_folder, "SEGMENTATION")
    ground_truth_folder = os.path.join(dataset_folder, "GROUND_TRUTH/SEGMENTATION")
    metrics = {}
    for f in sorted(os.listdir(libs_folder))[:]:
        if os.path.isdir(os.path.join(libs_folder,f)):
            print(f)
            test_data_folder = os.path.join(libs_folder, f)
            met = evaluate_masks(ground_truth_folder, test_data_folder, img_set)
            metrics[f] = met
    if len(metrics.keys()) > 0:
        metric_names = list(metrics[list(metrics.keys())[0]])
        #print(metric_names)
        #print(metrics)
        dataset_name = os.path.basename(os.path.dirname(dataset_folder))
        title_set = ""
        plot_name = dataset_name + "_seg_eval.png"
        if img_set is not None:
            title_set = "(img set = "  + ", ".join(img_set) + ")"
            plot_name = dataset_name + "_seg_eval_" + "".join(img_set) + ".png"
        plot_evaluation(os.path.join(plot_folder, plot_name), metric_names, metrics, "Segmentation evaluation " + title_set, factor=100)


def evaluate_masks(ground_truth_folder, data_folder, img_set=None):
    """
    Evaluates the data of the library to be tested against the ground truth data.
    ACC, TPR, FPR, FNR, PPV and the predicted area of the mask is calculated. 
    For normalization a wider bounding box of the ground truth mask is used.

    param ground_truth_folder:  the folder where the ground truth data can be find 
    param data_folder:          the folder with the result data of the library
    img_set:                    [opt] list of filenames that can be provided to evaluate a certain set of images
    returns:                    a map of calculated metrics for the library
    """
    metrics = {}
    files = sorted(os.listdir(data_folder))[:]
    if img_set is not None:
        files = [f for f in files if any(ext in f for ext in img_set)]

    for f in files:
        gt_mask_path = os.path.join(ground_truth_folder, f)
        data_mask_path = os.path.join(data_folder, f)    
        print(gt_mask_path, data_mask_path)
        met = evaluate_mask(gt_mask_path, data_mask_path)
        for m in met:
            if met[m] != -1:
                if m in metrics:
                    metrics[m] += met[m]
                else:
                    metrics[m] = met[m]
    for m in metrics:
        metrics[m] /= len(files)

    return metrics

def evaluate_mask(gt_mask_path, data_mask_path):
    """
    Calculates the metrics of one mask against the corresponding ground truth mask
    
    param gt_mask_path:         the filename of the ground truth mask which is a black and white image
    param data_mask_path:       the filename of the predicted mask which is also a black and white image
    returns:                    calculated metrics in a dictionary: ACC, TPR, TNR, FPR, FNR, PPV, AREA_COMP_BOX, AREA_COMP_FULL
    """
    gt_mask = get_mask(gt_mask_path)
    points = cv2.findNonZero(gt_mask)
    box = cv2.boundingRect(points)
    r_x, r_y, r_w, r_h = box
    data_mask = get_mask(data_mask_path)
    x1 = r_x
    x2 = r_x + r_w
    y1 = r_y
    y2 = r_y + r_h

    ## The bouding box is expanded to include more interesting pixels around te border of the segmentation of the cyclist
    ## Using the constant MASK_BOX_EXPAND (can be changed in the constants files)
    e = MASK_BOX_EXPAND
    x1 = max(r_x - int(e*r_w), 0)
    y1 = max(r_y - int(e*r_h), 0)
    x2 = min(r_x + int((1+e)*r_w), gt_mask.shape[1])
    y2 = min(r_y + int((1+e)*r_h), gt_mask.shape[0])
    r_h = y2-y1
    r_w = x2-x1
    ##

    gt_box = gt_mask[y1:y2, x1:x2]      
    data_box = data_mask[y1:y2, x1:x2]      
    total = r_h * r_w

    p, n, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0   
    for gt_r, data_r in zip(gt_box, data_box):
        for gt, data in zip(gt_r, data_r):
            if gt == 255:
                p += 1
            elif gt == 0:
                n += 1
            if gt == 255 and data == 255:
                tp += 1       
            elif gt == 0 and data == 0:
                tn += 1
            elif gt == 0 and data == 255:
                fp += 1
            elif gt == 255 and data == 0:
                fn += 1

    tpr = tp / p
    tnr = tn / n
    fpr = 1 - tpr
    fnr = 1 - tnr
    if tp + fp == 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    acc = (tp + tn) / total

    area_comp_box  = (tp + fp) / p
    area_comp_full = sum(map(sum, data_mask)) / sum(map(sum, gt_mask))
    
    return {    "ACC"                   :   acc, 
                "TPR"                   :   tpr, 
                "TNR"                   :   tnr, 
                "FPR"                   :   fpr, 
                "FNR"                   :   fnr,
                "PPV"                   :   ppv,
                "AREA_COMP_BOX"         :   area_comp_box,
                "AREA_COMP_FULL"         :   area_comp_full,
            }