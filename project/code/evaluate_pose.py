import numpy as np
import os

from constants import KPS_GROUPS, KEYPOINTS
from plot import plot_evaluation
from vis import *
from get_data import get_keypoints, bounding_box_based_on_keypoints, calculate_distances


def compare_libs_keypoints_distances(plot_folder, dataset_folder, threshold, img_set=None):
    """
    Compares the results of the libraries in the libs_folder and saves the comparison as a plot in the plot_folder map.
    For validation and evaluation the ground truth data is used. If the data consists of a certain set it can also be choosen.

    param plot_folder:          the folder where the results will be saved
    param dataset_folder:       the folder containing a subfolder with ground truth data and a folder with the output of different pose estimation libraries
    img_set:                    [opt] list of filenames that can be provided to evaluate a certain set of images
    returns:                    no return value but the plots for accuracy and distance are saved to the plot_folder
    """

    libs_folder = os.path.join(dataset_folder, "POSE_ESTIMATION")
    ground_truth_folder = os.path.join(dataset_folder, "GROUND_TRUTH/POSE_ESTIMATION")
    accuracy = {}
    avg_distance = {}
    for f in os.listdir(libs_folder):
        if os.path.isdir(os.path.join(libs_folder,f)):
            print(f)
            test_data_folder = os.path.join(libs_folder, f)
            acc, avg_dist = evaluate_keypoints_distances(ground_truth_folder, test_data_folder, threshold, img_set)
            accuracy[f] = acc
            avg_distance[f] = avg_dist 

    labels = [group for group in KPS_GROUPS]
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)
    
    title_set = ""
    dataset_name = os.path.basename(os.path.dirname(dataset_folder))
    plot_name_kps = dataset_name + "_kps_acc.png"
    plot_name_dst = dataset_name + "_kps_dst.png"
    if img_set is not None:
        title_set = ", img set = "  + ", ".join(img_set)
        plot_name_kps = dataset_name + "_kps_acc_" + "".join(img_set) + ".png"
        plot_name_dst = dataset_name + "_kps_dst_" + "".join(img_set) + ".png"

    plot_evaluation(os.path.join(plot_folder, plot_name_kps), labels, accuracy, "Accuracy keypoints (threshold = " + str(threshold) + title_set + ")", factor=100, vlines=[0.5,1.5])
    plot_evaluation(os.path.join(plot_folder, plot_name_dst), labels, avg_distance, "Average distance to ground truth point", vlines=[0.5,1.5])


def evaluate_keypoints_distances(ground_truth_folder, data_folder, threshold, img_set):
    """
    Evaluates the data of the library to be tested against the ground truth data.
    Accuracy and normalized average distance between the keypoints is calculated. For the normalization the diagonal of the bounding box is used.

    param ground_truth_folder:  the folder where the ground truth data can be find 
    param data_folder:          the folder with the result data of the library
    img_set:                    [opt] list of filenames that can be provided to evaluate a certain set of images
    returns:                    accuracy, avg_distance: two dicts of keypoints mapped on the value of the calculation
    """
    accuracy = {}
    avg_distance = {}

    for group in KPS_GROUPS:
        accuracy[group] = 0
        avg_distance[group] = 0
    
    for kp in KEYPOINTS:
        acc, avg_dist = evaluate_keypoint_distances(kp, ground_truth_folder, data_folder, threshold, img_set)
        for group in KPS_GROUPS:
            if kp in KPS_GROUPS[group]:
                accuracy[group] += acc
                avg_distance[group] += avg_dist

    for group in KPS_GROUPS:
        accuracy[group] /= len(KPS_GROUPS[group])
        avg_distance[group] /= len(KPS_GROUPS[group])

    return accuracy, avg_distance



def evaluate_keypoint_distances(keypoint, ground_truth_folder, data_folder, threshold, img_set):
    """
    Evaluates the data for one keypoint. 
    The distances are normalized by deviding the value by the length of the diagonal of the bounding box of the keypoints.
    For the accuracy a threshold is used to decide if the keypoint is estimated correctly.
    
    param keypoint:             the name of the keypoint to be evaluated
    param ground_truth_folder:  the folder where the ground truth data can be find 
    param data_folder:          the folder with the result data of the library
    img_set:                    [opt] list of filenames that can be provided to evaluate a certain set of images
    threshold:                  [opt, default: 5] value that is used to decide if a keypoint is estimated correctly when the distance is lower than this threshold
    returns:                    acc, avg_dist: the accuracy of the correct estimated keypoints and the average distance
    """
    ground_list = []
    data_list = []
    diagonal_bounding_box = []
    files = sorted(os.listdir(ground_truth_folder))

    if img_set is not None:
        files = [f for f in files if f.split("_")[0] in img_set]

    ground_truth_n = 0 
    # Counter for the number of keypoints that are expected to be predicted

    for f in files:
        json_ground = os.path.join(ground_truth_folder, f)
        json_data = os.path.join(data_folder, f)
        ground_kps = get_keypoints(json_ground)
        data_kps = get_keypoints(json_data)
        
        if ground_kps[keypoint][:2] == [0, 0]:   
            # The keypoint is not visible in the picture according to ground truth
            continue
        ground_truth_n += 1
        if data_kps is None or keypoint not in data_kps or data_kps[keypoint][:2] == [0, 0]:     
            # The keypoint is not predicted in the picture by the pose estimation algorithm
            continue
        
        ground_list.append(ground_kps[keypoint])
        data_list.append(data_kps[keypoint])
        
        b, w, h, diagonal, a = bounding_box_based_on_keypoints(ground_kps)
        diagonal_bounding_box.append(diagonal)


    distances = calculate_distances(ground_list, data_list)
    normalized_distances = np.divide(distances,diagonal_bounding_box)
    distances = 100 * normalized_distances

    if len(distances) == 0:
        return 0, -1

    avg_dist = sum(distances) / len(distances)
    
    correct = 0
    total = len(ground_list)
    correct = sum([1 for d in distances if d < threshold])
    
    acc = correct/ground_truth_n
    
    return acc, avg_dist