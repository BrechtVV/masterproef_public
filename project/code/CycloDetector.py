import os
import cv2
import numpy as np
import math
import sys
import json
import shutil

from config import DETECTRON_PATH
from get_data import get_mask, get_keypoints, get_orientation, get_area, get_angle
from vis import draw_skeleton, draw_angle


IMG_TYPES = [".jpg", ".png", ".jpeg"]
VID_TYPES = [".mp4", ".mov"]


class CycloDetector:
    file_path                   =   None
    file_basename               =   None
    pose_estimation_folder      =   None
    segmentation_folder         =   None
    
    skip_frames                 =   None
    frame_number                =   None
    video_cap                   =   None
    ANALYSE_RATE                =   None
    detectron_pose_path         =   os.path.join(DETECTRON_PATH, "pose_estimation")
    detectron_seg_path          =   os.path.join(DETECTRON_PATH, "segmentation")
    detectron                   =   None
    pose_file_path              =   None
    pose_kps                    =   None
    seg_folder_path             =   None
    
    def __init__(self):
        super().__init__()


    def setup(self, file_path, analyse_rate, pose_estimation_folder=None, segmentation_folder=None, mask_interest_folder=None):
        self.file_path = file_path
        self.file_basename = os.path.basename(file_path).split(".")[0]
        self.pose_estimation_folder = pose_estimation_folder
        self.segmentation_folder = segmentation_folder 
        self.mask_interest_folder = mask_interest_folder
        self.analyse_rate = int(analyse_rate)
        self.mask_interest = None 
        
        self.frame_number = 0
        self.skip_frames = 1
        self.video_cap = None
        if os.path.splitext(self.file_path)[1] in VID_TYPES:
            self.video_cap = cv2.VideoCapture(self.file_path)
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self.skip_frames = max(1, round(fps / self.analyse_rate))
        
        self.pose_kps = None
        self.pose_file_path = None
        self.seg_folder_path = None

        if self.pose_estimation_folder is None or self.segmentation_folder is None:
            try:
                from detectron import Detectron
                self.detectron = Detectron()
            except:
                print("\n\nPLEASE INSTALL DETECTRON2 BEFORE CONTINUING, see the github for more information. Or select pose and segmentation data as a folder.\n")
                sys.exit(1)
            
        if self.pose_estimation_folder is None:
            if not os.path.exists(self.detectron_pose_path):
                os.makedirs(self.detectron_pose_path)
            self.pose_file_path = os.path.join(self.detectron_pose_path, self.file_basename + ".json")
            if os.path.exists(self.pose_file_path):
                with open(self.pose_file_path) as json_file:
                    self.pose_kps = json.load(json_file) 
            else:
                self.pose_kps = {}
        if self.segmentation_folder is None:
            self.seg_folder_path = os.path.join(self.detectron_seg_path, self.file_basename)
            if not os.path.exists(self.seg_folder_path):
                os.makedirs(self.seg_folder_path)
            
    
    def analyse(self, vis_path=None):
        frame, kps, mask = self.get_frame_data()

        temp = draw_skeleton(frame, kps, mask)
        orientation = get_orientation(kps, temp)
    
        first_frame = np.copy(frame)
        first_frame = draw_skeleton(first_frame, kps, mask, self.mask_interest)
        
        area_bool = False
        kps_to_track = []
        angles_to_track = []
        if self.mask_interest_folder is not None:
            self.mask_interest = get_mask(os.path.join(self.mask_interest_folder, orientation + ".jpg"))
        if orientation == "FRONT":
            area_bool = True
            kps_to_track = ["Neck", "LShoulder", "RShoulder", "RKnee", "LKnee"]
        elif orientation == "L-SIDE":
            kps_to_track = ["LHip", "LShoulder", "LKnee"]
            angles_to_track = ["LHip_Angle", "LKnee_Angle", "LShoulder_Angle", "LElbow_Angle"]
        else:
            kps_to_track = ["RHip", "RShoulder", "RKnee"]
            angles_to_track = ["RHip_Angle", "RKnee_Angle", "RShoulder_Angle", "RElbow_Angle"]
        
        angles = {}
        track = {}
        for kp in kps_to_track:
            track[kp] = []
        for angle in angles_to_track:
            angles[angle] = []
        
        self.start_visualize(vis_path)

        results = []
        while frame is not None:

            res, vis = self.analyse_frame(frame, kps, mask, area_bool, angles_to_track, kps_to_track)
            results.append(res)
            self.append_visualize(vis)
            frame, kps, mask = self.get_frame_data(get_kps_flag=True, get_mask_flag=area_bool)    

        self.stop_visualize()
        
        
        area_r = None
        if area_bool:
            area = {
                "area": [r["area"] for r in results],
                "area_interest": [r["area_interest"] for r in results],
            }
            area_r, first_frame = self.analyse_area_end(area, first_frame)

        t = np.array([r["kps_to_track"] for r in results])
        track = {kps_to_track[i] : t[:,i] for i in range(len(kps_to_track))}
        t = np.array([r["angles_to_track"] for r in results])
        angles = {angles_to_track[i] : t[:,i] for i in range(len(angles_to_track))}
        
        kps_r, first_frame = self.analyse_keypoints_end(track, first_frame)
        cv2.putText(first_frame, self.file_basename, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 1)

        self.save_kps()

        return orientation, area_r, kps_r, angles, first_frame


    def get_frame_data(self, get_kps_flag=True, get_mask_flag=True):
        frame, kps, mask = None, None, None
        
        if self.video_cap is None and self.frame_number == 0:
            frame = cv2.imread(self.file_path)
        elif self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            ret, frame = self.video_cap.read()
        if frame is None:
            return None, None, None

        # kps
        if get_kps_flag:
            if self.pose_estimation_folder is not None:
                path = os.path.join(self.pose_estimation_folder, self.file_basename, str(self.frame_number) + ".json")
                if self.video_cap is None:
                    path = os.path.join(self.pose_estimation_folder, self.file_basename + ".json")
                kps = get_keypoints(path)
            else:
                key = str(self.frame_number).zfill(10)
                if key in self.pose_kps:
                    kps = self.pose_kps[key]
                else:
                    kps = self.predict_kps(frame) 
                    self.pose_kps[key] = kps
        
        # mask
        if get_mask_flag:
            if self.segmentation_folder is not None:
                path = os.path.join(self.segmentation_folder, self.file_basename, "frame" + str(self.frame_number).zfill(10) + ".jpg")
                if self.video_cap is None:
                    path = os.path.join(self.segmentation_folder, self.file_basename + ".jpg")
                mask = get_mask(path)
            else:
                mask_path = os.path.join(self.seg_folder_path, str(self.frame_number).zfill(10) + ".jpg")
                if os.path.exists(mask_path):
                    mask = get_mask(mask_path)
                else:
                    mask = self.predict_mask(frame)
                    cv2.imwrite(mask_path, mask)

        self.frame_number += self.skip_frames
        return frame, kps, mask


    def analyse_frame(self, frame, kps, mask, area_bool=True, angles_to_track=[], kps_to_track=[], visualize=True):
        res = {}
        if area_bool:
            a, a_i = self.analyse_area_mask(mask)
            res["area"] = a
            res["area_interest"] = a_i
        res["kps_to_track"] = [ [round(kps[kp][0], 0), round(kps[kp][1], 0)] for kp in kps_to_track ]
        res["angles_to_track"] = [ get_angle(angle, kps) for angle in angles_to_track ]   
        
        if visualize:
            cv2.putText(frame, self.file_basename, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 1)
            for angle in angles_to_track:
                frame = draw_angle(angle, kps, frame)
            #frame = draw_skeleton(frame, kps, mask, self.mask_interest)


        return res, frame


    def predict_kps(self, frame):
        return self.detectron.predict_keypoints(frame)
    
    def save_kps(self):
        if self.pose_file_path is not None:
            import json
            with open(self.pose_file_path, 'w') as json_file:
                json.dump(self.pose_kps, json_file)

    def predict_mask(self, frame):
        return self.detectron.predict_mask_panoptic(frame)

    def analyse_area_mask(self, mask):
        a = get_area(mask)
        a_i = 0
        if self.mask_interest is not None:
            interest = self.mask_interest & mask
            a_i = get_area(interest)
        return a, a_i

    def analyse_area_end(self, area, img):
        res = {}
        if len(area["area"]) > 0:
            res["AREA_MEAN"] = np.mean(area["area"])
            res["AREA_STD"] = np.std(area["area"])
        if len(area["area_interest"]) > 0:
            res["AREA_INTEREST_MEAN"] = int(round(np.mean(area["area_interest"])))
            res["AREA_INTEREST_STD"] = np.std(area["area_interest"])

        if img is not None:
            cv2.putText(img, "Area Mean:   {:7d}".format(int(res["AREA_MEAN"])), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
            cv2.putText(img, "Area STD:    {:7d}".format(int(res["AREA_STD"])), (10, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
            cv2.putText(img, "Area I Mean: {:7d}".format(int(res["AREA_INTEREST_MEAN"])), (10, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
            cv2.putText(img, "Area I STD:  {:7d}".format(int(res["AREA_INTEREST_STD"])), (10, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        return res, img

    def analyse_keypoints_end(self, track, img):
        tracks = {}
        for kp in track:
            points = np.array(track[kp])
            center = [np.mean(points[:,0]) , np.mean(points[:,1])] 
            dist = []
            for p in points:
                d = np.linalg.norm(center-p)
                dist.append(d)
            mean = np.mean(dist)
            std = np.std(dist)
            tracks[kp] = { "center" : center, "mean" : mean, "std" : std, "points": points}
            if img is not None:
                img = cv2.circle(img, (int(round(center[0])), int(round(center[1]))), int(math.ceil(max(dist))), (255, 255, 255), 2)

            
            points = points.astype(int)
            points = np.array([ [p] for p in points])
            cv2.drawContours(img, points, -1, (0, 0, 255), 3)

        return tracks, img
    
    def start_visualize(self, vis_path):
        self.writer = None
        self.vis_path = vis_path
        if self.vis_path is not None and self.video_cap is not None:
            frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.analyse_rate
            self.writer = cv2.VideoWriter(vis_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            
    def append_visualize(self, frame):
        if self.vis_path is not None:
            if self.writer is not None:
                self.writer.write(frame)
            else:
                cv2.imwrite(self.vis_path, frame)

    def stop_visualize(self):
        if self.vis_path is not None and self.writer is not None:
            self.writer.release()


#det = CycloDetector()

#dataset = "../../DATA/dataset_example/"
#det.setup(  dataset + "GROUND_TRUTH/images/front_000.jpg", \
#            dataset + "POSE_ESTIMATION/alphapose/", \
#            dataset + "SEGMENTATION/detectron2_pan/")
#det.analyse("vis.jpg")

#dataset = "../../DATA/dataset_videos/"
#det.setup(  dataset + "videos/vid_front_001.mp4", \
#            dataset + "POSE_ESTIMATION/alphapose/vid_front_001/", \
#            dataset + "SEGMENTATION/detectron2/vid_front_001/")
#det.analyse("vis.mp4")
