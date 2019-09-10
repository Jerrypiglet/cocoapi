import numpy as np
import matplotlib.pyplot as plt

def merge_bboxes(bboxes):
    max_x1y1x2y2 = [np.inf, np.inf, -np.inf, -np.inf]
    for bbox in bboxes:
        max_x1y1x2y2 = [min(max_x1y1x2y2[0], bbox[0]), min(max_x1y1x2y2[1], bbox[1]),
                        max(max_x1y1x2y2[2], bbox[2]+bbox[0]), max(max_x1y1x2y2[3], bbox[3]+bbox[1])]
    return [max_x1y1x2y2[0], max_x1y1x2y2[1], max_x1y1x2y2[2]-max_x1y1x2y2[0], max_x1y1x2y2[3]-max_x1y1x2y2[1]]

def check_clear(ann, vis=False, debug=False):
    kps = np.asarray(ann['keypoints']).reshape(-1, 3)
    if debug:
        print(np.hstack((np.arange(kps.shape[0]).reshape((-1, 1)), kps)))

    if vis:
        plt.figure(figsize=(20, 20))
        plt.imshow(I); plt.axis('off')
        for idx, kp in enumerate(kps):
            plt.scatter(kp[0], kp[1], )
            plt.text(kp[0], kp[1], '%d'%idx, weight='bold')

    eyes_ys = kps[1:5, 1]
    eyes_ys_valid_idx = eyes_ys!=0
    eyes_ys_valid = eyes_ys[eyes_ys_valid_idx]
    ankles_ys = kps[15:17, 1]
    ankles_ys_valid_idx = ankles_ys!=0
    ankles_ys_valid = ankles_ys[ankles_ys_valid_idx]
    if eyes_ys_valid.size==0 or ankles_ys_valid.size==0:
        return False

    should_min_y_idx = np.argmin(eyes_ys_valid) # two eyes
    should_max_y_idx = np.argmax(ankles_ys_valid) # two ankles

    kps_valid = kps[kps[:, 1]!=0, :]

    if debug:
        print(eyes_ys_valid[should_min_y_idx], np.min(kps_valid[:, 1]), kps[15:17, 1][should_max_y_idx], np.max(kps_valid[:, 1]), kps[1:5, 2], kps[15:17, 2])

    return eyes_ys_valid[should_min_y_idx]==np.min(kps_valid[:, 1]) and ankles_ys_valid[should_max_y_idx]==np.max(kps_valid[:, 1]) \
        and np.any(np.logical_or(kps[1:5, 2]==1, kps[1:5, 2]==2)) and np.any(np.logical_or(kps[15:17, 2]==1, kps[15:17, 2]==2))

def check_valid_surface(cats):
    green_cats_exception = {'water':'', 'ground':'', 'solid':'', 'vegetation':['-', 'flower', 'tree'], 'floor':'', 'plant':['+', 'grass']}
    if_green = False
    for super_cat in green_cats_exception.keys():
        if cats[0] == super_cat:
            sub_cats = green_cats_exception[super_cat]
            if sub_cats == '':
                if_green = True
            elif sub_cats[0] == '-':
                if cats[1] not in sub_cats[1:]:
                    if_green = True
            elif sub_cats[0] == '+':
                if cats[1] in sub_cats[1:]:
                    if_green = True
    return if_green
