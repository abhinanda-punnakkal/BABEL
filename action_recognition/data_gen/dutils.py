#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.

import sys, os, pdb
import os.path as osp
from os.path import join as ospj
from os.path import basename as ospb
from os.path import dirname as ospd

import numpy as np
import torch

import json, pickle, csv
from collections import Counter
from tqdm import tqdm

from smplx import SMPLH

import viz


def read_json(json_filename):
	'''Return contents of JSON file'''
	jc = None
	with open(json_filename) as infile:
		jc = json.load(infile)
	return jc

def read_pkl(pkl_filename):
	'''Return contents of pikcle file'''
	pklc = None
	with open(pkl_filename, 'rb') as infile:
		pklc = pickle.load(infile)
	return pklc

def write_json(contents, filename):
	with open(filename, 'w') as outfile:
		json.dump(contents, outfile, indent=2)

def write_pkl(contents, filename):
	with open(filename, 'wb') as outfile:
		pickle.dump(contents, outfile)

def smpl_to_nturgbd(model_type='smplh', out_format='nturgbd'):
	''' Borrowed from https://gitlab.tuebingen.mpg.de/apunnakkal/2s_agcn/-/blob/master/data_gen/smpl_data_utils.py
	NTU mapping
	-----------
	0 --> ?
	1-base of the spine
	2-middle of the spine
	3-neck
	4-head
	5-left shoulder
	6-left elbow
	7-left wrist
	8-left hand
	9-right shoulder
	10-right elbow
	11-right wrist
	12-right hand
	13-left hip
	14-left knee
	15-left ankle
	16-left foot
	17-right hip
	18-right knee
	19-right ankle
	20-right foot
	21-spine
	22-tip of the left hand
	23-left thumb
	24-tip of the right hand
	25-right thumb

	:param model_type:
	:param out_format:
	:return:
	'''
	if model_type == 'smplh' and out_format == 'nturgbd':
		'22 and 37 are approximation for hand (base of index finger)'
		return np.array([0, 3, 12, 15,
						 16, 18, 20, 22,		#left hand
						 17, 19, 21, 37,		   # right hand
						 1, 4, 7, 10,			#left leg
						 2, 5, 8, 11,			#right hand
						 9,
						 63, 64 , 68, 69
						 ],
						dtype=np.int32)

class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

def store_counts(label_fp):
	"""Compute # samples per class, from stored labels

	Args:
		label_fp <str>: Path to label file

	Writes (to same path as label file):
		out_fp <dict>: # samples per class = {<idx>: <count>, ...}
	"""
	Y_tup = read_pkl(label_fp)
	Y_idxs = Y_tup[1][0]
	print('# Samples in set = ', len(Y_idxs))

	label_count = Counter(Y_idxs)
	print('File ', label_fp, 'len',len(label_count))

	out_fp = label_fp.replace('.pkl', '_count.pkl')
	write_pkl(label_count, out_fp)

def load_babel_dataset(d_folder='../../data/babel_v1.0_release'):
	'''Load the BABEL dataset'''
	# Data folder
	l_babel_dense_files = ['train', 'val', 'test']
	l_babel_extra_files = ['extra_train', 'extra_val']

	# BABEL Dataset
	babel = {}
	for fn in l_babel_dense_files + l_babel_extra_files:
		babel[fn] = json.load(open(ospj(d_folder, fn+'.json')))

	return babel

def store_seq_fps(amass_p):
	'''Get fps for each seq. in BABEL
	Arguments:
	---------
		amass_p <str>: Path where you download AMASS to.
	Save:
	-----
		featp_2_fps.json <dict>: Key: feat path <str>, value: orig. fps
	in AMASS <float>. E.g.,: {'KIT/KIT/4/RightTurn01_poses.npz': 100.0, ...}
	'''
	# Get BABEL dataset
	babel = load_babel_dataset()

	# Loop over each BABEL seq, store frame-rate
	ft_p_2_fps = {}
	for fn in babel:
		for sid in tqdm(babel[fn]):
			ann = babel[fn][sid]
			if ann['feat_p'] not in ft_p_2_fps:
				fps = np.load(ospj(amass_p, ann['feat_p']))['mocap_framerate']
				ft_p_2_fps[ann['feat_p']] = float(fps)
	dest_fp = '../data/featp_2_fps.json'
	write_json(ft_p_2_fps, dest_fp)
	return None

def store_ntu_jpos(smplh_model_p, dest_jpos_p, amass_p):
	'''Store joint positions of kfor NTU-RGBD skeleton
	'''
	# Model to forward-pass through, to store joint positions
	smplh = SMPLH(smplh_model_p, create_transl=False, ext='pkl',
							gender='male', use_pca=False, batch_size=1)

	# Load paths to all BABEL features
	featp_2_fps = read_json('../data/featp_2_fps.json')

	# Loop over all BABEL data, verify that joint positions are stored on disk
	l_m_ft_p = []
	for ft_p in featp_2_fps:

		# Get the correct dataset folder name
		ddir_n = ospb(ospd(ospd(ft_p)))
		ddir_map = {'BioMotionLab_NTroje': 'BMLrub', 'DFaust_67': 'DFaust'}
		ddir_n = ddir_map[ddir_n] if ddir_n in ddir_map else ddir_n
		# Get the subject folder name
		sub_fol_n = ospb(ospd(ft_p))

		# Sanity check
		fft_p = ospj(dest_jpos_p, ddir_n, sub_fol_n, ospb(ft_p))
		if not os.path.exists(fft_p):
			l_m_ft_p.append((ft_p, fft_p))
	print('Total # missing NTU RGBD skeleton features = ', len(l_m_ft_p))

	# Loop over missing joint positions and store them on disk
	for i, (ft_p, ntu_jpos_p) in enumerate(tqdm(l_m_ft_p)):
		jrot_smplh = np.load(ospj(amass_p, ft_p))['poses']
		# Break joints down into body parts
		smpl_body_jrot = jrot_smplh[:, 3:66]
		left_hand_jrot = jrot_smplh[:, 66:111]
		right_hand_jrot = jrot_smplh[:, 111:]
		root_orient = jrot_smplh[:, 0:3].reshape(-1, 3)

		# Forward through model to get a superset of required joints
		T = jrot_smplh.shape[0]
		ntu_jpos = np.zeros((T, 219))
		for t in range(T):
			res = smplh(body_pose=torch.Tensor(smpl_body_jrot[t:t+1, :]),
						global_orient=torch.Tensor(root_orient[t: t+1, :]),
						left_hand_pose = torch.Tensor(left_hand_jrot[t: t+1, :]),
						right_hand_pose=torch.Tensor(right_hand_jrot[t: t+1, :]),
						# transl=torch.Tensor(transl)
						)
			jpos = res.joints.detach().cpu().numpy()[:, :, :].reshape(-1)
			ntu_jpos[t, :] = jpos

		# Save to disk
		if not os.path.exists(ospd(ntu_jpos_p)):
			os.makedirs(ospd(ntu_jpos_p))
		np.savez(ntu_jpos_p, joint_pos=ntu_jpos, allow_pickle=True)

	return

def viz_ntu_jpos(jpos_p, l_ft_p):
	'''Visualize sequences of NTU-skeleton joint positions'''
	# Load paths to all BABEL features
	featp_2_fps = read_json('../data/featp_2_fps.json')
	# Indices that are in the NTU RGBD skeleton
	smpl2nturgbd = smpl_to_nturgbd()
	# Iterate over each
	for ft_p in l_ft_p:
		x = np.load(ospj(jpos_p, ft_p))['joint_pos']
		T, ft_sz = x.shape
		x = x.reshape(T, ft_sz//3, 3)
		# print('Data shape = {0}'.format(x.shape))
		x = x[:, smpl2nturgbd, :]
		# print('Data shape = {0}'.format(x.shape))
		# x = x[:,:,:, 0].transpose(1, 2, 0)	# (3, 150, 22, 1) --> (150, 22, 3)
		print('Data shape = {0}'.format(x.shape))
		viz.viz_seq(seq=x, folder_p='test_viz/test_ntu_w_axis', sk_type='nturgbd', debug=True)
		print('-'*50)


def main():
	'''Store preliminary stuff'''
	amass_p= '/ps/project/conditional_action_gen/data/AMASS_March2021/'

	# Save feature paths --> fps (released in babel/action_recognition/data/)
	# store_seq_fps(amass_p)

	# Save joint positions in NTU-RGBD skeleton format
	smplh_model_p = '/ps/project/conditional_action_gen/body_models/mano_v1_2/models_cleaned_merged/SMPLH_male.pkl'
	jpos_p = '/ps/project/conditional_action_gen/amass/babel_joint_pos'
	# store_ntu_jpos(smplh_model_p, jpos_p, amass_p)

	#  Viz. saved seqs.
	# l_ft_p = ['KIT/917/Experiment3a_09_poses.npz']
	# viz_ntu_jpos(jpos_p, l_ft_p)

if __name__ == '__main__':
	main()

