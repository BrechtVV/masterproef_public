import os
import argparse
from CycloComparer import *
from config import ANALYSE_RATE


"""----------------------------- Main options -----------------------------"""
parser = argparse.ArgumentParser(description="Analyse bike positions of cyclists based on a dataset or using detectron2")
parser.add_argument("dataset", type=str)
parser.add_argument("--pose-folder", default=None)
parser.add_argument("--seg-folder", default=None)
parser.add_argument('--mask-interest-folder', default=None)
parser.add_argument("--plot-folder", default="plots/analysis", dest="PLOT_FOLDER")
parser.add_argument("--analyse-rate", default=ANALYSE_RATE)
parser.add_argument("--img-set", nargs="*", default=None)


args = parser.parse_args()


com = CycloComparer()
com.setup(data_folder=args.dataset, analyse_rate=args.analyse_rate, \
    pose_estimation_folder=args.pose_folder, segmentation_folder=args.seg_folder, mask_interest_folder=args.mask_interest_folder, \
        img_set=args.img_set)
com.compare(visualize=True, plot_path=args.PLOT_FOLDER)