import os
import argparse
from evaluate_pose import compare_libs_keypoints_distances
from evaluate_segmentation import compare_libs_masks


"""----------------------------- Main options -----------------------------"""
parser = argparse.ArgumentParser(description="Compare pose estimation and person segmentation libraries (for cyclists)")
parser.add_argument("dataset", type=str)
parser.add_argument("--compare-pose", action="store_true", default=False)
parser.add_argument("--compare-segmentation", action="store_true", default=False)
parser.add_argument("--kps-threshold", type=int, default=5, dest="KPS_THRESHOLD")
parser.add_argument("--plot-folder", default="plots/evaluation", dest="PLOT_FOLDER")
parser.add_argument("--img-set", nargs="*", default=None)
args = parser.parse_args()

if args.compare_pose:
    print("\nPOSE ESTIMATION COMPARISON")
    POSE_ESTIMATION_DATA_FOLDER = os.path.join(args.dataset,"POSE_ESTIMATION")
    PLOT_FOLDER_POSE = os.path.join(args.PLOT_FOLDER,"pose")
    if not os.path.exists(PLOT_FOLDER_POSE):
        os.makedirs(PLOT_FOLDER_POSE)
    compare_libs_keypoints_distances(PLOT_FOLDER_POSE, args.dataset, args.KPS_THRESHOLD, args.img_set)


if args.compare_segmentation:
    print("\nSEGMENTATION COMPARISON")
    SEGMENTATION_DATA_FOLDER = os.path.join(args.dataset,"SEGMENTATION")
    PLOT_FOLDER_SEG = os.path.join(args.PLOT_FOLDER,"seg")
    if not os.path.exists(PLOT_FOLDER_SEG):
        os.makedirs(PLOT_FOLDER_SEG)
    compare_libs_masks(PLOT_FOLDER_SEG, args.dataset, args.img_set)


#import sys
#sys.path.append("../help")
#from vis import visualize_keypoints_data
#visualize_keypoints_data(GROUND_TRUTH_FOLDER + "images/", GROUND_TRUTH_FOLDER + "POSE_ESTIMATION/", POSE_ESTIMATION_DATA_FOLDER)

