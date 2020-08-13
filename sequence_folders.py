import numpy
import os
import torch.utils.data as data
import numpy as np
from transforms3d.euler import mat2euler
from scipy.misc import imread
from path import Path
import random


def load_as_float(path):
	return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
	"""A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

	def __init__(self, root, seed=None, ttype='train.txt', sequence_length=2, transform=None,
	             target_transform=None, add_geo=False, depth_source="p", dataset="", gt_source='g',
	             pose_source='', scale=False, req_angle=False, size=0, req_gt=False, get_path=False):
		print(dataset + pose_source)
		np.random.seed(seed)
		random.seed(seed)
		self.root = Path(root)
		scene_list_path = self.root / ttype
		scenes = [self.root / folder[:-1] for folder in open(scene_list_path) if folder.startswith(dataset)]
		# if size > 0:
		# 	scenes = random.sample(scenes, size * sequence_length)
		self.size = size
		self.pose_source = pose_source
		self.ttype = ttype
		self.scenes = sorted(scenes)
		self.scale = scale
		self.transform = transform
		self.geo = add_geo
		self.gt_source = gt_source
		self.avg_scale = 0
		self.max_scale = 0
		self.counter = 0
		self.req_angle = req_angle
		self.depth_source = depth_source
		self.req_gt = req_gt
		self.get_path = get_path
		self.crawl_folders(sequence_length)

	def crawl_folders(self, sequence_length):
		sequence_set = []
		demi_length = sequence_length // 2
		p_num = 0
		g_num = 0
		scale_sum = 0
		l1counter = 0
		for scene in self.scenes:

			intrinsics = np.genfromtxt(scene / 'cam.txt').astype(np.float32).reshape((3, 3))
			source = False
			if self.pose_source and os.path.exists(scene / self.pose_source):
				poses = np.genfromtxt(scene / self.pose_source).astype(np.float32)
				source = True
				imgs = sorted(scene.files('*.jpg'))
				if len(imgs) >= 20:
					print(scene)
			else:
				poses = np.genfromtxt(scene / 'poses.txt').astype(np.float32)
			if self.req_gt:
				poses_gt = np.genfromtxt(scene / 'poses.txt').astype(np.float32)
			imgs = sorted(scene.files('*.jpg'))
			# print(len(imgs))

			if len(imgs) < sequence_length:
				continue
			for i in range(len(imgs)):
				if i < demi_length:
					shifts = list(range(0, sequence_length))
					shifts.pop(i)
				elif i >= len(imgs) - demi_length:
					shifts = list(range(len(imgs) - sequence_length, len(imgs)))
					shifts.pop(i - len(imgs))
				else:
					shifts = list(range(i - demi_length, i + (sequence_length + 1) // 2))
					shifts.pop(demi_length)

				img = imgs[i]
				depth = img.dirname() / img.name[:-4] + '.npy'
				if self.gt_source == 'p':
					depth = img.dirname() / img.name[:-4] + '_p.npy'
				pose_tgt = np.concatenate((poses[i, :].reshape((3, 4)), np.array([[0, 0, 0, 1]])), axis=0)
				if self.req_gt:
					pose_tgt_gt = np.concatenate((poses_gt[i, :].reshape((3, 4)), np.array([[0, 0, 0, 1]])), axis=0)
				sample = {'intrinsics': intrinsics, 'tgt': img, 'tgt_depth': depth, 'ref_imgs': [],
				          'ref_poses': [], 'ref_poses_gt': [], 'ref_depths': [], 'scale': 1.0, 'gt_angle': [],
				          'source': source}
				for j in shifts:
					sample['ref_imgs'].append(imgs[j])
					if self.geo:
						if self.depth_source == 'g':
							sample['ref_depths'].append(imgs[j].dirname() / imgs[j].name[:-4] + '.npy')
						elif self.depth_source == 'p':
							path = imgs[j].dirname() / imgs[j].name[:-4] + '_p.npy'
							if (os.path.exists(path)):
								sample['ref_depths'].append(path)
								p_num += 1
							else:
								sample['ref_depths'].append(imgs[j].dirname() / imgs[j].name[:-4] + '.npy')
								g_num += 1
						else:
							path = imgs[j].dirname() / imgs[j].name[:-4] + '_' + self.depth_source + '.npy'
							if (os.path.exists(path)):
								sample['ref_depths'].append(path)
								p_num += 1
							else:
								sample['ref_depths'].append(imgs[j].dirname() / imgs[j].name[:-4] + '.npy')
								g_num += 1
					pose_src = np.concatenate((poses[j, :].reshape((3, 4)), np.array([[0, 0, 0, 1]])), axis=0)
					pose_rel = pose_src @ np.linalg.inv(pose_tgt)
					if self.req_gt:
						pose_src_gt = np.concatenate((poses_gt[j, :].reshape((3, 4)), np.array([[0, 0, 0, 1]])), axis=0)
						pose_rel_gt = pose_src_gt @ np.linalg.inv(pose_tgt_gt)
					if self.req_angle:
						angle = mat2euler(pose_rel[:3, :3])
						sample['gt_angle'] = angle
					pose = pose_rel[:3, :].reshape((1, 3, 4)).astype(np.float32)
					if self.req_gt:
						pose_gt = pose_rel_gt[:3, :].reshape((1, 3, 4)).astype(np.float32)
						sample['ref_poses_gt'].append(pose_gt)
					if self.scale:
						self.counter = self.counter + 1
						scale = (pose[0, 0, 3] ** 2 + pose[0, 1, 3] ** 2 + pose[0, 2, 3] ** 2) ** 0.5
						scale_sum += scale
						self.avg_scale = scale_sum / self.counter
						self.max_scale = max(self.max_scale, scale)
						sample['scale'] = scale
						pose[0, 0, 3] /= scale
						pose[0, 1, 3] /= scale
						pose[0, 2, 3] /= scale
						if scale < 0.5:
							l1counter += 1

					sample['ref_poses'].append(pose)
				sequence_set.append(sample)
		if self.size > 0:
			sequence_set = random.sample(sequence_set, self.size)
		if self.ttype == 'train.txt':
			random.shuffle(sequence_set)

		print("pn:", p_num, "  gn:", g_num)
		self.samples = [sq for sq in sequence_set if str(sq['tgt']).split('/')[3].startswith('')]

	def __getitem__(self, index):
		sample = self.samples[index]
		tgt_img = load_as_float(sample['tgt'])
		tgt_depth = np.load(sample['tgt_depth'])
		if not sample["source"]:
			print("warning")
		nanmask = tgt_depth != tgt_depth
		num = np.sum(nanmask)
		if num != 0:
			print('tgt depth nan')
		tgt_depth[nanmask] = 1
		tgt_depth = tgt_depth / sample['scale']

		ref_depths = []
		for path in sample['ref_depths']:
			ref_depth = np.load(path)
			nanmask = ref_depth != ref_depth
			num = np.sum(nanmask)
			if (num != 0):
				print('ref depth nan')
			ref_depth[nanmask] = 1
			ref_depth = ref_depth / sample['scale']
			ref_depths.append(ref_depth)

		ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
		ref_poses = sample['ref_poses']
		if self.transform is not None:

			imgs, depths, intrinsics = self.transform([tgt_img] + ref_imgs, [tgt_depth] + ref_depths,
			                                          np.copy(sample['intrinsics']))
			tgt_img = imgs[0]
			tgt_depth = depths[0]
			ref_imgs = imgs[1:]
			ref_depths = depths[1:]

		else:
			intrinsics = np.copy(sample['intrinsics'])
		if self.get_path:
			return tgt_img, ref_imgs, ref_poses, sample['ref_poses_gt'], intrinsics, np.linalg.inv(
				intrinsics), tgt_depth, ref_depths, sample['tgt']
		if self.req_angle:
			return tgt_img, ref_imgs, ref_poses, np.array([a for a in sample['gt_angle']])
		if self.req_gt:
			return tgt_img, ref_imgs, ref_poses, sample['ref_poses_gt'], intrinsics, np.linalg.inv(
				intrinsics), tgt_depth, ref_depths

		return tgt_img, ref_imgs, ref_poses, intrinsics, np.linalg.inv(intrinsics), tgt_depth, ref_depths

	def __len__(self):
		return len(self.samples)
