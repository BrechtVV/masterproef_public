import os
import cv2
import sys

from plot import plot_evaluation, plot_track, plot_single_values
from CycloDetector import *


class CycloComparer:
    detector                    =   None
    data_folder                 =   None
    pose_estimation_folder      =   None
    segmentation_folder         =   None
    mask_interest_folder        =   None
    analyse_rate                =   None


    def __init__(self):
        self.detector = CycloDetector()

    
    def setup(self, data_folder, analyse_rate, pose_estimation_folder=None, segmentation_folder=None, mask_interest_folder=None):        
        self.data_folder = data_folder
        self.pose_estimation_folder = pose_estimation_folder
        self.segmentation_folder = segmentation_folder
        self.mask_interest_folder = mask_interest_folder
        self.analyse_rate = analyse_rate

    def compare(self, visualize=True, plot_path="plots/"):
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

        import time
        start_comp_time = time.time()
        area = {}
        tracing = {}
        tracing_points = {}
        
        files = sorted(os.listdir(self.data_folder))
        
        for f in files:
            name = f.split(".")[0]
            path = os.path.join(self.data_folder, f)
            print(name)
            self.detector.setup(path,self.analyse_rate, self.pose_estimation_folder, self.segmentation_folder, self.mask_interest_folder)
            start_an_time = time.time()
            orientation, a, tracks, angles, vis = self.detector.analyse(vis_path=os.path.join(plot_path,"vis_"+f))
            end_an_time = time.time()
            print('\t {:10s} : {:10f}'.format(name, round(end_an_time-start_an_time, 2)))
            if a is not None:
                area[name] = a
            
            if not orientation in tracing:
                tracing[orientation] = {}
            tracing[orientation][name] = {}
            for tr in tracks:
                tracing[orientation][name][tr] = tracks[tr]["std"]

            track_pts = {tr : tracks[tr]["points"] for tr in tracks}
            plot_track(os.path.join(plot_path,"plot_" + orientation + "_" + name + ".png"), track_pts, "Keypoint Tracing Analysis", fps=self.analyse_rate, xlabel="time (seconds)", ylabel="position")

            if len(angles) > 0:
                plot_single_values(os.path.join(plot_path,"plot_angles" + "_" + name + ".png"), angles, "Angle Analysis", fps=self.analyse_rate, xlabel="time (seconds)", ylabel="angle (degrees)")

            if visualize:
                cv2.imwrite(os.path.join(plot_path, name + ".jpg"), vis)
        
        if len(area) > 0:
            metric_names = list(area[list(area.keys())[0]])
            plot_evaluation(os.path.join(plot_path, "plot_area.png"), metric_names, area, "Front Area Analysis")

        for orientation in tracing:
            metric_names = list(tracing[orientation][list(tracing[orientation].keys())[0]])
            plot_evaluation(os.path.join(plot_path,"plot_" + orientation + ".png"), metric_names, tracing[orientation], "KPS stability analysis (" + orientation + ")")

        end_comp_time = time.time()
        print('COMP_TIME : {:10f}'.format(round(end_comp_time-start_comp_time, 2)))


#com = CycloComparer()

#dataset = "../../DATA/dataset_videos/"
#com.setup(  dataset + "videos/", \
#            dataset + "POSE_ESTIMATION/alphapose/", \
#            dataset + "SEGMENTATION/detectron2/")
#com.compare()

#com.setup(  dataset + "videos/")
#com.compare()

#dataset = "../../DATA/dataset_example/"
#com.setup(  dataset + "GROUND_TRUTH/images/", \
#            dataset + "POSE_ESTIMATION/alphapose/", \
#            dataset + "SEGMENTATION/detectron2/")
#com.compare()