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

DEBUG = False

def union_box_layer(rois, im_info):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    n_boxes = 0
    # allow boxes to sit over the edge by a small amount
    #_allowed_border =  0
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]
    
    #print ">>>>>>>>>>>>>>>>>>>>>>>>union_boxes"
    
    union_boxes = []
    im_info = im_info[0]
    #print im_info
    for i in range(n_boxes):
        for j in range(n_boxes):
	    box = []
	    #xmin = min(rois[i][1], rois[j][1])
	    #ymin = min(rois[i][2], rois[j][2])
	    #xmax = max(rois[i][3], rois[j][3])
	    #ymax = max(rois[i][4], rois[j][4])
	    cx1 = (rois[i][1] + rois[i][3]) * 1.0 / (2 * im_info[0])
	    cy1 = (rois[i][2] + rois[i][4]) * 1.0 / (2 * in_info[1])
	    w1 = (rois[i][3] - rois[i][1]) * 1.0 / im_info[0]
	    h1 = (rois[i][4] - rois[i][2]) * 1.0 / im_info[1]
	    s1 = w1 * h1		

	    cx2 = (rois[j][1] + rois[j][3]) * 1.0 / (2 * im_info[0])
	    cy2 = (rois[j][2] + rois[j][4]) * 1.0 / (2 * im_info[1])
	    w2 = (rois[j][3] - rois[j][1]) * 1.0 / (2 * im_info[0])
	    h2 = (rois[j][4] - rois[j][2]) * 1.0 / (2 * im_info[1])
	    s2 = w2 * h2	   	 
	   
	    box.append(cx1)
	    box.append(cy1)
	    box.append(w1)
	    box.append(h1)
	    box.append(s1)

	    box.append(cx2)
	    box.append(cy2)
	    box.append(w2)
	    box.append(h2)
	    box.append(s2)
	    if i == j:
		box = [0] * 10		
	
            union_boxes.append(box)
      	     
    scene = [[0, 0, 0, im_info[0], im_info[1]]]
    
    #union_boxes = np.array(union_boxes).astype(np.float32)
    scene = np.array(scene).astype(np.float32)
    
    #print scene

    return scene
