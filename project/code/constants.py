STD_FRONT_THRESHOLD = 10
DEFAULT_MASK_SHAPE = (720, 1280, 1)
#SKIP_FRAMES = 10
KPS_DIST_THRESHOLD = 5

MASK_BOX_EXPAND = 0.05

KEYPOINTS  =    ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar","LEar"]
keypoints = KEYPOINTS

eval_keypoints = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle"]


upper_body =    ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist"]
lower_body =    ["RHip","RKnee","RAnkle","LHip","LKnee","LAnkle"]
stable_body =   upper_body + ["RHip", "LHip"]

#kps_groups =    { "All" : keypoints, "UBody" : upper_body, "LBody" : lower_body, "SBody": stable_body }
KPS_GROUPS =    { "All" : keypoints, "SBody": stable_body }
kps_groups = KPS_GROUPS

for kp in eval_keypoints:
    kps_groups[kp] = [kp]

kps_skeleton = [
                ["LWrist", "LElbow", "LShoulder", "Neck"],
                ["RWrist", "RElbow", "RShoulder", "Neck"],
                ["LAnkle", "LKnee", "LHip", "Neck"],
                ["RAnkle", "RKnee", "RHip", "Neck"],
                ["LEar", "LEye", "Nose", "Neck"],
                ["REar", "REye", "Nose", "Neck"]
                ]
KPS_LEFT =      ["LWrist", "LElbow", "LShoulder", "LAnkle", "LKnee", "LHip", "LEar", "LEye"]
KPS_RIGHT =     ["RWrist", "RElbow", "RShoulder", "RAnkle", "RKnee", "RHip", "REar", "REye"]
KPS_CENTER =    ["Neck", "Nose"]

kps_left = KPS_LEFT
kps_right = KPS_RIGHT
kps_center = KPS_CENTER


KPS_ANGLES = {
    "LHip_Angle"        : ["LKnee", "LHip", "LShoulder"],
    "LKnee_Angle"       : ["LAnkle", "LKnee", "LHip"],
    "LShoulder_Angle"   : ["LHip", "LShoulder", "LElbow"],
    "LElbow_Angle"      : ["LShoulder", "LElbow", "LWrist"],
    "RHip_Angle"        : ["RKnee", "RHip", "RShoulder"],
    "RKnee_Angle"       : ["RAnkle", "RKnee", "RHip"],
    "RShoulder_Angle"   : ["RHip", "RShoulder", "RElbow"],
    "RElbow_Angle"      : ["RShoulder", "RElbow", "RWrist"],
}
kps_angles = KPS_ANGLES


def get_vis_color(lib):
    if "keypoints" in lib: 
        return (255, 0, 255) 
    elif "openpose" in lib:
        #return (0, 0, 0)
        return (255, 0, 0)
    elif "alphapose" in lib:
        #return (0, 0, 0)
        return (0, 255, 0)
    elif "tf-pose-estimation" in lib:
        #return (0, 0, 0)
        return (0, 0, 255)
    else:
        return (255, 255, 255)