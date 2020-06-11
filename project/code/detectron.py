# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

#import other utilities
import os

from constants import DETECTRON_DEVICE

class Detectron:

    mask_predictor      = None
    mask_cfg            = None
    kps_predictor       = None
    kps_cfg             = None
    pan_predictor       = None
    pan_cfg             = None
    device              = None

    def __init__(self):
        super().__init__()
        self.device = DETECTRON_DEVICE
        self.setup_kps_predictor()
        #self.setup_mask_predictor()
        self.setup_pan_predictor()

    def setup_mask_predictor(self):
        ## INSTANCE SEGMENTATION
        self.mask_cfg = get_cfg()
        self.mask_cfg.MODEL.DEVICE=self.device
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.mask_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.mask_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.mask_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.mask_predictor = DefaultPredictor(self.mask_cfg)


    def setup_kps_predictor(self):
        ## KEYPOINTS
        # Inference with a keypoint detection model
        self.kps_cfg = get_cfg()
        self.kps_cfg.MODEL.DEVICE=self.device
        self.kps_cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.kps_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        self.kps_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.kps_predictor = DefaultPredictor(self.kps_cfg)


    def setup_pan_predictor(self):
        ## PANOPTIC SEGMENTATION
        # Inference with a panoptic segmentation model
        self.pan_cfg = get_cfg()
        self.pan_cfg.MODEL.DEVICE=self.device
        self.pan_cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.pan_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.pan_predictor = DefaultPredictor(self.pan_cfg)


    def predict_keypoints(self, img):
        outputs = self.kps_predictor(img)
        v2 = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.kps_cfg.DATASETS.TRAIN[0]), scale=1)
        v2 = v2.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2.imwrite(output_path, v2.get_image()[:,:,::-1])
        persons = np.array(outputs["instances"].pred_keypoints.to("cpu"), dtype=float)
        if len(persons) > 0:
            return convert_keypoints_one_person(persons[0])
        else:
            return None


    def predict_mask_panoptic(self, img):
        panoptic_seg, segments_info = self.pan_predictor(img)["panoptic_seg"]
        v3 = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.pan_cfg.DATASETS.TRAIN[0]), scale=1)
        v3 = v3.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        
        ##create list of id's who are the person...
        ids = [seg["id"] for seg in segments_info if seg["category_id"] == 0]
        pan = np.array(panoptic_seg)
        mask = np.zeros(img.shape, dtype=int)
        for c in range(3):
            mask[:,:,c] = np.where(np.isin(pan, ids), 255.0, 0.0)
        cv2.imwrite("temp.jpg", mask)
        mask = cv2.imread("temp.jpg", cv2.IMREAD_GRAYSCALE)
        ret, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        return mask



import json
from enum import Enum

class CocoPart(Enum):
    Nose = 0 
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6 
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10 
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16 


def convert_keypoints_one_person(detectron_kps_one):
    kps = {}
    for i in range(len(detectron_kps_one)):
        x, y, c = detectron_kps_one[i]
        kps[CocoPart(i).name] = [x, y, c]
    neck_x = (kps["LShoulder"][0] + kps["RShoulder"][0])/2
    neck_y = (kps["LShoulder"][1] + kps["RShoulder"][1])/2
    neck_c = (kps["LShoulder"][2] + kps["RShoulder"][2])/2
    kps["Neck"] = [neck_x, neck_y, neck_c]
    
    return kps