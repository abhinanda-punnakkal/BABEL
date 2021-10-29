#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.

import os, sys
import os.path as osp

import random
import numpy as np
import math
import torch
from torch.nn.functional import interpolate as intrp

import subprocess
import shutil
import uuid
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pdb

import dutils


"""
Visualize input and output motion sequences and labels
"""

def get_smpl_skeleton():
    '''Skeleton ordering so that you traverse joints in this order:
        Left lower, Left upper, Spine, Neck, Head, Right lower, Right upper.
    '''
    return np.array(
        [
            # Left lower
            [ 0, 1 ],
            [ 1, 4 ],
            [ 4, 7 ],
            [ 7, 10],

            # Left upper
            [ 9, 13],
            [13, 16],
            [16, 18],
            [18, 20],
            # [20, 22],

            # Spinal column
            [ 0, 3 ],
            [ 3, 6 ],
            [ 6, 9 ],
            [ 9, 12],
            [12, 15],

            # Right lower
            [ 0, 2 ],
            [ 2, 5 ],
            [ 5, 8 ],
            [ 8, 11],

            # Right upper
            [ 9, 14],
            [14, 17],
            [17, 19],
            [19, 21],
            # [21, 23],
        ])

def get_nturgbd_joint_names():
    '''From paper:
    1-base of the spine 2-middle of the spine 3-neck 4-head 5-left shoulder 6-left elbow 7-left wrist 8- left hand 9-right shoulder 10-right elbow 11-right wrist 12- right hand 13-left hip 14-left knee 15-left ankle 16-left foot 17- right hip 18-right knee 19-right ankle 20-right foot 21-spine 22- tip of the left hand 23-left thumb 24-tip of the right hand 25- right thumb
    '''
    # Joint names by AC, based on SMPL names
    joint_names_map = {
        0: 'Pelvis',

        12: 'L_Hip',
        13: 'L_Knee',
        14: 'L_Ankle',
        15: 'L_Foot',

        16: 'R_Hip',
        17: 'R_Knee',
        18: 'R_Ankle',
        19: 'R_Foot',

        1: 'Spine1',
        # 'Spine2',
        20: 'Spine3',
        2: 'Neck',
        3: 'Head',

        # 'L_Collar',
        4: 'L_Shoulder',
        5: 'L_Elbow',
        6: 'L_Wrist',
        7: 'L_Hand',
        21: 'L_HandTip',  # Not in SMPL
        22: 'L_Thumb',  # Not in SMPL

        # 'R_Collar',
        8: 'R_Shoulder',
        9: 'R_Elbow',
        10: 'R_Wrist',
        11: 'R_Hand',
        23: 'R_HandTip',  # Not in SMPL
        24: 'R_Thumb',  # Not in SMPL
    }

    return [joint_names_map[idx] for idx in range(len(joint_names_map))]

def get_smpl_joint_names():
    # Joint names from SMPL Wiki
    joint_names_map = {
        0: 'Pelvis',

        1: 'L_Hip',
        4: 'L_Knee',
        7: 'L_Ankle',
        10: 'L_Foot',

        2: 'R_Hip',
        5: 'R_Knee',
        8: 'R_Ankle',
        11: 'R_Foot',

        3: 'Spine1',
        6: 'Spine2',
        9: 'Spine3',
        12: 'Neck',
        15: 'Head',

        13: 'L_Collar',
        16: 'L_Shoulder',
        18: 'L_Elbow',
        20: 'L_Wrist',
        22: 'L_Hand',
        14: 'R_Collar',
        17: 'R_Shoulder',
        19: 'R_Elbow',
        21: 'R_Wrist',
        23: 'R_Hand'}

    # Return all joints except indices 22 (L_Hand), 23 (R_Hand)
    return [joint_names_map[idx] for idx in range(len(joint_names_map)-2)]

def get_nturgbd_skeleton():
    ''' Skeleton ordering such that you traverse joints in this order:
        Left lower, Left upper, Spine, Neck, Head, Right lower, Right upper.
    '''
    return np.array(
        [
            # Left lower
            [0, 12],
            [12, 13],
            [13, 14],
            [14, 15],

            # Left upper
            [4, 20],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 21],
            [7, 22],  # --> L Thumb

            # Spinal column
            [0, 1],
            [1, 20],
            [20, 2],
            [2, 3],

            # Right lower
            [0, 16],
            [16, 17],
            [17, 18],
            [18, 19],

            # Right upper
            [20, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 24],
            # [24, 11] --> R Thumb

            [21, 22],

            [23, 24],

        ]
    )

def get_joint_colors(joint_names):
    '''Return joints based on a color spectrum. Also, joints on
    L and R should have distinctly different colors.
    '''
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(joint_names))]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    return colors

def calc_angle_from_x(sk):
    '''Given skeleton, calc. angle from x-axis'''
    # Hip bone
    id_l_hip = get_smpl_joint_names().index('L_Hip')
    id_r_hip = get_smpl_joint_names().index('R_Hip')
    pl, pr = sk[id_l_hip], sk[id_r_hip]
    bone = np.array(pr-pl)
    unit_v =  bone / np.linalg.norm(bone)
    # Angle with x-axis
    pdb.set_trace()
    x_ax = np.array([1, 0, 0])
    x_angle = math.degrees(np.arccos(np.dot(x_ax, unit_v)))

    '''
    l_hip_z = seq[0, joint_names.index('L_Hip'), 2]
    r_hip_z = seq[0, joint_names.index('R_Hip'), 2]
    az = 0 if (l_hip_z > zroot and zroot > r_hip_z) else 180
    '''
    if bone[1] > 0:
        x_angle = - x_angle

    return x_angle

def calc_angle_from_y(sk):
    '''Given skeleton, calc. angle from x-axis'''
    # Hip bone
    id_l_hip = get_smpl_joint_names().index('L_Hip')
    id_r_hip = get_smpl_joint_names().index('R_Hip')
    pl, pr = sk[id_l_hip], sk[id_r_hip]
    bone = np.array(pl-pr)
    unit_v =  bone / np.linalg.norm(bone)
    print(unit_v)
    # Angle with x-axis
    pdb.set_trace()
    y_ax = np.array([0, 1, 0])
    y_angle = math.degrees(np.arccos(np.dot(y_ax, unit_v)))

    '''
    l_hip_z = seq[0, joint_names.index('L_Hip'), 2]
    r_hip_z = seq[0, joint_names.index('R_Hip'), 2]
    az = 0 if (l_hip_z > zroot and zroot > r_hip_z) else 180
    '''
    # if bone[1] > 0:
    #    y_angle = - y_angle
    seq_y_proj = bone * np.cos(np.deg2rad(y_angle))
    print('Bone projected onto y-axis: ', seq_y_proj)

    return y_angle

def viz_skeleton(seq, folder_p, sk_type='smpl', radius=1, lcolor='#ff0000', rcolor='#0000ff', action='', debug=False):
    ''' Visualize skeletons for given sequence and store as images.

    Args:
        seq (np.array): Array (frames) of joint positions.
        Size depends on sk_type (see below).
            if sk_type is 'smpl' then assume:
                1. first 3 dims = translation.
                2. Size = (# frames, 69)
            elif sk_type is 'nturgbd', then assume:
                1. no translation.
                2. Size = (# frames, 25, 3)
        folder_p (str): Path to root folder containing visualized frames.
            Frames are dumped to the path: folder_p/frames/*.jpg
        radius (float): Space around the subject?

    Returns:
        Stores skeleton sequence as jpg frames.
    '''
    joint_names = get_nturgbd_joint_names() if 'nturgbd' == sk_type \
                                    else get_smpl_joint_names()
    n_j = n_j = len(joint_names)

    az = 90
    if 'smpl' == sk_type:
        # SMPL kinematic chain, joint list.
        # NOTE that hands are skipped.
        kin_chain = get_smpl_skeleton()
        # Reshape flat pose features into (frames, joints, (x,y,z)) (skip trans)
        seq = seq[:, 3:].reshape(-1, n_j, 3).cpu().detach().numpy()

    elif 'nturgbd' == sk_type:
        kin_chain = get_nturgbd_skeleton()
        az = 0

    # Get color-spectrum for skeleton
    colors = get_joint_colors(joint_names)
    labels = [(joint_names[jidx[0]], joint_names[jidx[1]]) for jidx in kin_chain]

    # xroot, yroot, zroot = 0.0, 0.0, 0.0
    xroot, yroot, zroot = seq[0, 0, 0], seq[0, 0, 1], seq[0, 0, 2]
    # seq = seq - seq[0, :, :]

    # Change viewing angle so that first frame is in frontal pose
    # az = calc_angle_from_x(seq[0]-np.array([xroot, yroot, zroot]))
    # az = calc_angle_from_y(seq[0]-np.array([xroot, yroot, zroot]))

    # Viz. skeleton for each frame
    for t in range(seq.shape[0]):

        # Fig. settings
        fig = plt.figure(figsize=(7, 6)) if debug else \
              plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

        for i, (j1, j2) in enumerate(kin_chain):
            # Store bones
            x = np.array([seq[t, j1, 0], seq[t, j2, 0]])
            y = np.array([seq[t, j1, 1], seq[t, j2, 1]])
            z = np.array([seq[t, j1, 2], seq[t, j2, 2]])
            # Plot bones in skeleton
            ax.plot(x, y, z, c=colors[i], marker='o', linewidth=2, label=labels[i])

        # More figure settings
        ax.set_title(action)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # xroot, yroot, zroot = seq[t, 0, 0], seq[t, 0, 1], seq[t, 0, 2]

        # pdb.set_trace()
        ax.set_xlim3d(-radius + xroot, radius + xroot)
        ax.set_ylim3d([-radius + yroot, radius + yroot])
        ax.set_zlim3d([-radius + zroot, radius + zroot])

        if True==debug:
            ax.axis('on')
            ax.grid(b=True)
        else:
            ax.axis('off')
            ax.grid(b=None)
        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])

        cv2.waitKey(0)

        # ax.view_init(-75, 90)
        # ax.view_init(elev=20, azim=90+az)
        ax.view_init(elev=20, azim=az)

        if True==debug:
            ax.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
            pass

        fig.savefig(osp.join(folder_p, 'frames', '{0}.jpg'.format(t)))
        plt.close(fig)

        # break

def write_vid_from_imgs(folder_p, fps):
    '''Collate frames into a video sequence.

    Args:
        folder_p (str): Frame images are in the path: folder_p/frames/<int>.jpg
        fps (float): Output frame rate.

    Returns:
        Output video is stored in the path: folder_p/video.mp4
    '''
    vid_p = osp.join(folder_p, 'video.mp4')
    cmd = ['ffmpeg', '-r', str(int(fps)), '-i',
                    osp.join(folder_p, 'frames', '%d.jpg'), '-y', vid_p]
    FNULL = open(os.devnull, 'w')
    retcode = subprocess.call(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
    if not 0 == retcode:
        print('*******ValueError(Error {0} executing command: {1}*********'.format(retcode, ' '.join(cmd)))
    shutil.rmtree(osp.join(folder_p, 'frames'))

def viz_seq(seq, folder_p, sk_type, orig_fps=30.0, debug=False):
    '''1. Dumps sequence of skeleton images for the given sequence of joints.
    2. Collates the sequence of images into an mp4 video.

    Args:
        seq (np.array): Array of joint positions.
        folder_p (str): Path to root folder that will contain frames folder.
        sk_type (str): {'smpl', 'nturgbd'}

    Return:
        None. Path of mp4 video: folder_p/video.mp4
    '''
    # Delete folder if exists
    if osp.exists(folder_p):
        print('Deleting existing folder ', folder_p)
        shutil.rmtree(folder_p)

    # Create folder for frames
    os.makedirs(osp.join(folder_p, 'frames'))

    # Dump frames into folder. Args: (data, radius, frames path)
    viz_skeleton(seq, folder_p=folder_p, sk_type=sk_type, radius=1.2, debug=debug)
    write_vid_from_imgs(folder_p, orig_fps)

    return None

def viz_rand_seq(X, Y, dtype, epoch, wb, urls=None,
                 k=3, pred_labels=None):
    '''
    Args:
        X (np.array): Array (frames) of SMPL joint positions.
        Y (np.array): Multiple labels for each frame in x \in X.
        dtype (str): {'input', 'pred'}
        k (int): # samples to viz.
        urls (tuple): Tuple of URLs of the rendered videos from original mocap.
        wb (dict): Wandb log dict.
    Returns:
        viz_ds (dict): Data structure containing all viz. info so far.
    '''
    import wandb
    # `idx2al`: idx --> action label string
    al2idx = dutils.read_json('data/action_label_to_idx.json')
    idx2al = {al2idx[k]: k for k in al2idx}

    # Sample k random seqs. to viz.
    for s_idx in random.sample(list(range(X.shape[0])), k):
        # Visualize a single seq. in path `folder_p`
        folder_p = osp.join('viz', str(uuid.uuid4()))
        viz_seq(seq=X[s_idx], folder_p=folder_p)
        title='{0} seq. {1}: '.format(dtype, s_idx)
        acts_str = ', '.join([idx2al[l] for l in torch.unique(Y[s_idx])])
        wb[title+urls[s_idx]] = wandb.Video(osp.join(folder_p, 'video.mp4'),
                                           caption='Actions: '+acts_str)

        if 'pred' == dtype or 'preds'==dtype:
            raise NotImplementedError

    print('Done viz. {0} seqs.'.format(k))
    return wb
