# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform
import pdb
import math

DEBUG = False

def cal_iou(box, truth):
    xmin = max(box[1], truth[1])
    ymin = max(box[2], truth[2])
    xmax = min(box[3], truth[3])
    ymax = min(box[4], truth[4])

    w = xmax - xmin
    h = ymax - ymin

    if w < 0 or h < 0:
        inter_s = 0
    else:
        inter_s = w * h

    outer_s = (box[3] - box[1]) * (box[4] - box[2]) + (truth[3] - truth[1]) * (truth[4] - truth[2])
    iou = inter_s * 1.0 / (outer_s - inter_s);
    return iou

def edge_box_layer(rois, im_info):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    n_boxes = len(rois) #128, 256
    # allow boxes to sit over the edge by a small amount
    #_allowed_border =  0
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]
    
    #print ">>>>>>>>>>>>>>>>>>>>>>>>union_boxes"
    #print ">>>>>>>>>>>>>>>>>>>>>>>>",len(rois) 
    rois = rois.tolist()
    im_info = im_info.tolist()

    union_boxes = []
    im_info = im_info[0]
    #print im_info
    for i in range(n_boxes):
        for j in range(n_boxes):
	    if i == j:
                iou = 1.0
            else:
                iou = cal_iou(rois[i], rois[j])

            if iou < 0.6:
                box = []

                cx1 = (rois[i][1] + rois[i][3]) * 0.5
                cy1 = (rois[i][2] + rois[i][4]) * 0.5
                w1 = (rois[i][3] - rois[i][1]) * 1.0
                h1 = (rois[i][4] - rois[i][2]) * 1.0

                if w1 < 0:
                    w1 = 0
                if h1 < 0:
                    h1 = 0

                s1 = w1 * h1

                cx2 = (rois[j][1] + rois[j][3]) * 0.5
                cy2 = (rois[j][2] + rois[j][4]) * 0.5
                w2 = (rois[j][3] - rois[j][1]) * 1.0
                h2 = (rois[j][4] - rois[j][2]) * 1.0

                if w2 < 0:
                    w2 = 0
                if h2 < 0:
                    h2 = 0

                s2 = w2 * h2

                box.append(w1 / (im_info[0] + 1))
                box.append(h1 / (im_info[1] + 1))
                box.append(s1 / ((im_info[0] + 1) * (im_info[1] + 1)))

                box.append(w2 / (im_info[0] + 1))
                box.append(h2 / (im_info[1] + 1))
                box.append(s2 / ((im_info[0] + 1) * (im_info[1] + 1)))

                box.append((cx1 - cx2) / (w2 + 1))
                box.append((cy1 - cy2) / (h2 + 1))

                box.append(pow((cx1 - cx2) / (w2 + 1), 2))
                box.append(pow((cy2 - cy2) / (h2 + 1), 2))

                box.append(math.log((w1 + 1) / (w2 + 1)))
                box.append(math.log((h1 + 1) / (h2 + 1)))

            else:
                box = [0] * 12
                #index += 1

            union_boxes.append(box)
       
    edge_boxes = np.array(union_boxes).astype(np.float32)
    return edge_boxes
