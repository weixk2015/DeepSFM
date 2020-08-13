from __future__ import division

import datetime

import numpy as np
import torch
import transforms3d
from minieigen import Quaternion
from path import Path


def save_path_formatter(args, parser):
	def is_default(key, value):
		return value == parser.get_default(key)

	args_dict = vars(args)
	data_folder_name = str(Path(args_dict['data']).normpath().name)
	folder_string = [data_folder_name]
	# if not is_default('epochs', args_dict['epochs']):
	#     folder_string.append('{}epochs'.format(args_dict['epochs']))
	# keys_with_prefix = OrderedDict()
	# keys_with_prefix['epoch_size'] = 'epoch_size'
	# keys_with_prefix['batch_size'] = 'b'
	# keys_with_prefix['lr'] = 'lr'
	#
	# for key, prefix in keys_with_prefix.items():
	#     value = args_dict[key]
	#     if not is_default(key, value):
	#         folder_string.append('{}{}'.format(prefix, value))
	save_path = Path(','.join(folder_string))
	timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
	return save_path / timestamp


def tensor2array(tensor, max_value=255, colormap='rainbow'):
	if max_value is None:
		max_value = tensor.max()
	if tensor.ndimension() == 2 or tensor.size(0) == 1:
		try:
			import cv2
			if cv2.__version__.startswith('2'):  # 2.4
				color_cvt = cv2.cv.CV_BGR2RGB
			else:
				color_cvt = cv2.COLOR_BGR2RGB
			if colormap == 'rainbow':
				colormap = cv2.COLORMAP_RAINBOW
			elif colormap == 'bone':
				colormap = cv2.COLORMAP_BONE
			array = (255 * tensor.squeeze().numpy() / max_value).clip(0, 255).astype(np.uint8)
			colored_array = cv2.applyColorMap(array, colormap)
			array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32) / 255
			array = array.transpose(2, 0, 1)
		except ImportError:
			if tensor.ndimension() == 2:
				tensor.unsqueeze_(2)
			array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy() / max_value).clip(0, 1)

	elif tensor.ndimension() == 3:
		assert (tensor.size(0) == 3)
		array = 0.5 + tensor.numpy() * 0.5
	return array


def save_checkpoint(save_path, dpsnet_state, epoch, filename='checkpoint.pth.tar'):
	file_prefixes = ['dpsnet']
	states = [dpsnet_state]
	for (prefix, state) in zip(file_prefixes, states):
		torch.save(state, save_path / '{}_{}_{}'.format(prefix, epoch, filename))


def adjust_learning_rate(args, optimizer, epoch):
	lr = args.lr * (0.8 ** (epoch // 10))
	print("learning rate:", lr)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def get_angle(v1, v2):
	product = np.dot(v1, v2)
	angle = np.arccos(product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
	return angle


def euler2axangle(eangle):
	vec, theta = transforms3d.euler.euler2axangle(eangle[0], eangle[1], eangle[2])
	return Quaternion(theta, vec)
