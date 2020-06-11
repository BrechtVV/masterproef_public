import json
import numpy as np
import cv2
from constants import KEYPOINTS, KPS_LEFT, KPS_RIGHT, KPS_CENTER, STD_FRONT_THRESHOLD, KPS_ANGLES


def get_keypoints(json_path):
    """
    Reads a json file and returns a dictionary with the keypoints of the first person
    :param json_path:       the path of the json file with the keypoints
    :returns:               kps: dict with name of keypoint mapped on the [x, y] point or sometimes [x, y, c] with the certainty as value of c
    """
    with open(json_path) as json_file:
        data = json.load(json_file)
    if len(data['people']) == 0:
        return None
    else:
        kps = data['people'][0]
        kps = { kp : kps[kp] for kp in kps if kp in KEYPOINTS }
        return kps    


def bounding_box_based_on_keypoints(kps):
    """
    Calculates the bounding box of a keypoints dict
    :param kps:             dict with (x, y) points as values
    :returns:               box [minX, maxX, minY, maxY], width, height, diagonal, area of the
    """
    minX, maxX, minY, maxY = 1000000, 0, 1000000, 0
    for kp in kps:
        if kps[kp][0] > 0 and kps[kp][0] < minX:
            minX = kps[kp][0]
        elif kps[kp][0] > maxX:
            maxX = kps[kp][0]
        if kps[kp][1] > 0 and kps[kp][1] < minY:
            minY = kps[kp][1]
        elif kps[kp][1] > maxY:
            maxY = kps[kp][1]
    box = [minX, maxX, minY, maxY]
    width = maxX-minX
    height = maxY-minY
    diagonal = np.sqrt(width*width + height*height)
    area = width*height
    return box, width, height, diagonal, area


def calculate_distances(ground_list, data_list):
    """
    Calculates all distances between (x,y) points in the ground_list and data_list
    :param ground_list:     list of (x, y) points
    :param data_list:       list of (x, y) points corresponding to ground_list
    :return:                euclidean distances between the points in the lists of corresponding indexes             
    """
    distances = []
    for i in range(len(ground_list)):
        ground = ground_list[i]
        data = data_list[i]
        pt1 = np.array(ground[:2])
        pt2 = np.array(data[:2])
        dist = np.linalg.norm(pt1-pt2)
        distances.append(dist)
    return distances


def get_orientation(kps, img=None):
    """
    Estimates the orientation of the cyclist in the picture based on the keypoints:
    :param kps:             keypoints of the person
    :return:                string value in [FRONT, L-SIDE, R-SIDE]
    """
    box, width, height, diagonal, area = bounding_box_based_on_keypoints(kps)
    center_points = []
    for l, r in zip(KPS_LEFT, KPS_RIGHT):
        #print(l, kps[l][:2], r, kps[r][:2])
        if l in kps and r in kps and kps[l][:2] != [0, 0] and kps[r][:2] != [0, 0]:
            x = (kps[l][0] + kps[r][0]) / 2
            y = (kps[l][1] + kps[r][1]) / 2
            pt = [x, y]
            center_points.append(pt)
    for c in KPS_CENTER:
        if c in kps:
            center_points.append(kps[c][:2])
    center_points = np.array(center_points)
    x_coords = center_points[:,0]
    std = np.std(x_coords)*100/width

    #if img is not None:
    #    maxY = img.shape[0]
    #    for x in x_coords:
    #        img = cv2.line(img, (int(x),0), (int(x),maxY), (0, 0, 255), 1)
    #    
    #    cv2.putText(img, "STD x-coords: " + str(round(std,3)), (20, maxY-100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
    #    cv2.imwrite("/home/brecht/Pictures/temp.jpg", img)
    #    cv2.imshow("frame", img)
    #    cv2.waitKey(0)
    

    if std < STD_FRONT_THRESHOLD:
        return "FRONT"
    else:
        m_wrist = np.array(np.array(kps["LWrist"][:2]) + np.array(kps["RWrist"][:2]))/2
        m_hip = np.array(np.array(kps["LHip"][:2]) + np.array(kps["RHip"][:2]))/2
        if (m_wrist-m_hip)[0] < 0:
            return "L-SIDE"
        else:
            return "R-SIDE"
    #print(min(x_coords), max(x_coords), diff_x*100)


def get_mask(mask_path):
    """
    Opens an image that represents a mask and ensures that the result is a binary one channel image with values 0 and 255
    :param mask_path:       the path of the image mask
    :return:                one channel image
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ret, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    return mask


def get_area(mask):
    """
    Calculates the sum of all black pixels representing (if the image only contains black and white pixels)
    :param mask:            a black and white image representing a segmentation mask
    :return:                the number of pixels or area of the segmentation in the mask
    """
    return int(sum(map(sum, mask)) / 255)


def get_angle(angle, kps):
    """
    Calculates the angle in degrees between for given pixel coordinates in an image
    :param angle:       the name of the angle specified in the constants file, it is used as a key to get the 3 points that form the angle
    :param kps:         all keypoints of a person (dict)
    :return:            the requested angle in degrees
    """
    a = np.array([ kps[KPS_ANGLES[angle][0]][0] , kps[KPS_ANGLES[angle][0]][1] ])
    b = np.array([ kps[KPS_ANGLES[angle][1]][0] , kps[KPS_ANGLES[angle][1]][1] ])
    c = np.array([ kps[KPS_ANGLES[angle][2]][0] , kps[KPS_ANGLES[angle][2]][1] ])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle_value = np.arccos(cosine_angle)
    return np.degrees(angle_value)