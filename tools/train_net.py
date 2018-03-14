#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys
import pdb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='kitti_train', type=str)
    #parser.add_argument('--imdbval', dest='imdb_val_name',
    #                    help='dataset to validate on',
    #                    default='voc_2007_test', type=str)
    parser.add_argument('--tag', dest='tb_dir', default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    #print argparse.REMAINDER
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


if __name__ == '__main__':
    #print "coming....................."
    args = parse_args()

    print('Called with args:')
    print(args)
    #print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>args."
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
	
    #print ">>>>>>>>>>>>>>>>>>>>>>>>>>>config."

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
    
    #imdb = get_imdb(args.imdb_name)
    imdb, roidb = combined_roidb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    #roidb = get_training_roidb(imdb)
	
    #imdb_val = get_imdb(args.imdb_val_name)
    #print 'Loaded dataset `{:s}` for training'.format(imdb_val.name)
    #roidb_val = get_training_roidb(imdb_val)
	
    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    tb_dir = get_output_tb_dir(imdb, args.tb_dir)
    print 'event will be saved to `{:s}`'.format(tb_dir)

    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print device_name

    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)
    print args.pretrained_model
    train_net(network, imdb, roidb, output_dir, tb_dir, pretrained_model=args.pretrained_model, max_iters=args.max_iters)
