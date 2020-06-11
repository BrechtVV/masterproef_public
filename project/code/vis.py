import cv2
import os
import numpy as np
from get_data import get_keypoints, bounding_box_based_on_keypoints
from constants import get_vis_color, KPS_SKELETON, KPS_DIST_THRESHOLD, KPS_ANGLES


def draw_keypoints(img, kps):
    img = np.copy(img)
    for line in KPS_SKELETON:
        for a, b in zip(line, line[1:]):
            if a in kps and b in kps:
                pt1 = (int(kps[a][0]),int(kps[a][1]))
                pt2 = (int(kps[b][0]),int(kps[b][1]))
                if pt1 != (0, 0) and pt2 != (0,0):
                    img = cv2.line(img, pt1, pt2, (255, 0, 0), 1)
    for kp in kps:
        img = cv2.circle(img, (int(kps[kp][0]), int(kps[kp][1])), 2, (0, 255, 0), -1)
    return img


def draw_mask(img, mask, mask_i=None):
    if mask is not None:
        img = cv2.addWeighted(img, 0.8, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.2, 0)
    if mask_i is not None:
        img = cv2.addWeighted(img, 0.8, cv2.cvtColor(mask_i, cv2.COLOR_GRAY2BGR), 0.2, 0)
    return img


def draw_skeleton(img, kps, mask, mask_i=None):
    img = draw_mask(img, mask, mask_i)
    img = draw_keypoints(img, kps)
    return img


def visualize_keypoints_data(images_folder, ground_data_folder, libs_folder):
    libs = [ground_data_folder]
    for f in os.listdir(libs_folder):
        if os.path.isdir(os.path.join(libs_folder,f)):
            test_data_folder = os.path.join(libs_folder, f)
            libs.append(test_data_folder)
    
    for f in sorted(os.listdir(images_folder)):
        img_path = os.path.join(images_folder, f)
        img = cv2.imread(img_path)

        for l in libs:
            json_path = os.path.join(l, f[:-3] + "json")
            kps = get_keypoints(json_path)
            if kps is not None:
                for k in kps:
                    x, y = kps[k][:2]
                    img = cv2.circle(img, (int(x), int(y)), 2, get_vis_color(l), -1)
            if "GROUND_TRUTH" in l:
                b, w, h, diagonal, a = bounding_box_based_on_keypoints(kps)
                pt1 = (b[0], b[2])
                pt2 = (b[1], b[3])
                cv2.rectangle(img, pt1, pt2, (255, 255, 0), 1, cv2.LINE_AA)
                img = cv2.line(img, pt1, pt2, (255, 0, 255), 1, cv2.LINE_8)
                dist = KPS_DIST_THRESHOLD * diagonal / 100
                #print(dist)
                dist = int(round(dist))
                for kp in kps:
                    img = cv2.circle(img, (int(kps[kp][0]), int(kps[kp][1])), dist, (0, 255, 255), 1)

        cv2.putText(img, f[:-4], (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
        cv2.imshow("img", img)
        k = cv2.waitKey(0)
        if k == 27:
            break


def visualize_masks(gt, data, box):
    r_x, r_y, r_w, r_h = box
    pt1 = (r_x, r_y)
    pt2 = (r_x + r_w, r_y + r_h)

    #gt_i = np.zeros([gt.shape[0], gt.shape[1], 3])
    #gt_i[:,:,0] = gt
    #gt_i[:,:,1] = gt
    #gt_i[:,:,1] = gt

    #data_i = np.zeros([data.shape[0], data.shape[1], 3])
    #data_i[:,:,0] = data
    #data_i[:,:,1] = data
    #data_i[:,:,1] = data

    vis = cv2.addWeighted(gt, 0.5, data, 0.5, 0)
    vis = cv2.rectangle(vis, pt1, pt2, (255, 0, 0), 3)
    cv2.imshow("vis", vis)
    cv2.waitKey(1)


def draw_angle(angle, kps, frame):
    a = np.array([ int(kps[KPS_ANGLES[angle][0]][0]) , int(kps[KPS_ANGLES[angle][0]][1]) ])
    b = np.array([ int(kps[KPS_ANGLES[angle][1]][0]) , int(kps[KPS_ANGLES[angle][1]][1]) ])
    c = np.array([ int(kps[KPS_ANGLES[angle][2]][0]) , int(kps[KPS_ANGLES[angle][2]][1]) ])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle_value = np.arccos(cosine_angle)
    degrees = np.degrees(angle_value)
    a = tuple(a)
    b = tuple(b)
    c = tuple(c)

    frame = cv2.line(frame, a, b, (255, 0, 0), 2)
    frame = cv2.line(frame, b, c, (255, 0, 0), 2)
    frame = cv2.circle(frame, a, 4, (0, 255, 0), -1)
    frame = cv2.circle(frame, b, 4, (0, 255, 0), -1)
    frame = cv2.circle(frame, c, 4, (0, 255, 0), -1)
    cv2.putText(frame, str((round(degrees,1))), (b[0]+10, b[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)

    return frame