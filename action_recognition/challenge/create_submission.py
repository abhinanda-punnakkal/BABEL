#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.

import sys, os, pdb, glob
import uuid
from os.path import join as ospj
from os.path import dirname as ospd
import json, pickle
import argparse
from tqdm import tqdm
from collections import *

import numpy as np
import pandas as pd
from pandas.core.common import flatten
from fnmatch import fnmatch
import re


def load_test_scores(test_scores_fp):
	'''
	Load the score prediction file, validate.

	Format of data structure stored in prediction file:
			score_dict = list(zip(
					self.data_loader[ln].dataset.label[1],	# sid
					self.data_loader[ln].dataset.sample_name,  # seg_id
					self.data_loader[ln].dataset.label[2],	# chunk_id
											score))
	'''
	# load test set predictions from model
	test_scores = pickle.load(open(test_scores_fp, 'rb'))

	# GT labels (-1 for test set), seg_id, chunk_id, score
	_, seg_ids, chunk_ids, scores = zip(*test_scores)

	# Validate the shape of predictions
	scores = np.array(scores)
	n_samples, n_classes = scores.shape
	assert n_classes in (60, 120)

	return list(zip(seg_ids, chunk_ids, scores)), n_classes


def load_test_samples(n_classes):
	'''Load the GT samples corresponding to the BABEL subset (# classes) used.

	GT labels data structure format:
			List of seg_id, (label, sid, chunk_n, anntr_id)

	Arguments:
			scores: np.array (n_samples, n_classes) contains predicted scores for samples.
	'''
	# load test set samples
	samples_filename = f'test_label_{n_classes}.pkl'
	test_samples = pickle.load(open(f'../data/release/{samples_filename}', 'rb'))

	# GT labels (-1 for test set), sid, chunk_id, anntr_id
	seg_ids, (_, _, chunk_ids, _) = test_samples

	return list(zip(seg_ids, chunk_ids))


def create_submission(test_samples, test_pred_scores, n_classes):
	'''Create a submission with the same ordering of samples
	as provided in the `test_label_{60, 120}.pkl` file.
	'''
	submission = []
	perfect_map = True

	# Ideal scenario -- 1:1 map between samples in two files
	for i, ((seg_id, chunk_id), (pred_seg_id, pred_chunk_id, _)) in \
							enumerate(zip(test_samples, test_pred_scores)):
		if seg_id != pred_seg_id or chunk_id != pred_chunk_id:
			perfect_map = False

	if True == perfect_map:
		submission = np.array(list(zip(*test_pred_scores))[2])
	else:
		# For each sample, find its predicted score
		for i, (seg_id, chunk_id) in enumerate(test_samples):
			for pred_seg_id, pred_chunk_id, score in test_pred_scores:
				if pred_seg_id == seg_id and pred_chunk_id == chunk_id:
					submission.append(score)
					break
		submission = np.array(submission)
		if 60 == n_classes:
			assert 15647 == submission.shape[0]
		elif 120 == n_classes:
			assert 16839 == submission.shape[0]

	return submission


def save_submission(submission, filepath):
	'''Save predicted scores for test samples in .npz format for
	submission to BABEL Action Recognition Challenge.
	'''
	np.savez(filepath, submission)
	print(f'Successfully saved submission in: {filepath}')

	return None


if __name__ == '__main__':
	# Add args
	parser = argparse.ArgumentParser(
							description='Predicted test scores --> Submission to server')
	parser.add_argument(
			'--pred_path',
			default='./epoch1_test_score.pkl',
			help='Path to file containing model predictions (saved to disk by train_test.py.')
	parser.add_argument(
			'--sub_path',
			default='./test_sub.npz',
			help='Path to write submission file.')

	# Parse args
	args = parser.parse_args()

	# Process scores into submission file
	test_pred_scores, n_classes = load_test_scores(args.pred_path)
	test_samples = load_test_samples(n_classes)
	submission = create_submission(test_samples, test_pred_scores, n_classes)
	save_submission(submission, args.sub_path)
