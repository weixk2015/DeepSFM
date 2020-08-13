import argparse
import os.path as Path
import warnings

import custom_transforms
import time
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from logger import AverageMeter
from transforms3d.axangles import mat2axangle

from convert import *
from demon_metrics import compute_motion_errors
from models import PoseNet
from pose_sequence_folders import SequenceFolder

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=2)

parser.add_argument('-b', '--batch-size', default=1, type=int,  # 6
                    metavar='N', help='mini-batch size')
parser.add_argument('--geo', '--geo-cost', default=True, type=bool,
                    metavar='GC', help='whether add geometry cost')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained-dps', dest='pretrained_dps',
                    default='pose_checkpoint.pth.tar',
                    metavar='PATH',
                    help='path to pre-trained model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--save', default="I0", type=str, help='save prefix')

parser.add_argument('--ttype', default='test.txt', type=str, help='Text file indicates input data')
parser.add_argument('-f', '--training-output-freq', type=int,
                    help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=100)
parser.add_argument('--nlabel', type=int, default=10, help='number of label')
parser.add_argument('--std_tr', type=float, default=0.27, help='translation')
parser.add_argument('--std_rot', type=float, default=0.12, help='rotation')
parser.add_argument('--pose_init', default='demon', help='path to init pose')
parser.add_argument('--depth_init', default='demon', help='path to init depth')

n_iter = 0
warnings.filterwarnings('ignore')


# NOTE: test set for testing

def main():
	global n_iter
	args = parser.parse_args()

	# Data loading code
	normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
	                                        std=[0.5, 0.5, 0.5])
	train_transform = custom_transforms.Compose([
		# custom_transforms.RandomScaleCrop(),
		custom_transforms.ArrayToTensor(),
		normalize
	])

	print("=> fetching scenes in '{}'".format(args.data))
	train_set = SequenceFolder(
		args.data,
		transform=train_transform,
		seed=args.seed,
		ttype=args.ttype,
		add_geo=args.geo,
		depth_source=args.depth_init,
		sequence_length=args.sequence_length,
		gt_source='g',
		std=args.std_tr,
		pose_init=args.pose_init,
		dataset="",
		get_path=True
	)

	print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
	val_loader = torch.utils.data.DataLoader(
		train_set, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	# create model
	print("=> creating model")
	pose_net = PoseNet(args.nlabel, args.std_tr, args.std_rot, add_geo_cost=args.geo, depth_augment=False).cuda()

	if args.pretrained_dps:
		# freeze feature extra layers
		# for param in pose_net.feature_extraction.parameters():
		#     param.requires_grad = False

		print("=> using pre-trained weights for DPSNet")
		model_dict = pose_net.state_dict()
		weights = torch.load(args.pretrained_dps)['state_dict']
		pretrained_dict = {k: v for k, v in weights.items() if
		                   k in model_dict and weights[k].shape == model_dict[k].shape}

		model_dict.update(pretrained_dict)

		pose_net.load_state_dict(model_dict)

	else:
		pose_net.init_weights()

	cudnn.benchmark = True
	pose_net = torch.nn.DataParallel(pose_net)

	global n_iter
	data_time = AverageMeter()

	pose_net.eval()
	end = time.time()

	errors = np.zeros((2, 2, int(np.ceil(len(val_loader)))), np.float32)
	with torch.no_grad():
		for i, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth, ref_depths,
		        ref_noise_poses, initial_pose, tgt_path, ref_paths) in enumerate(val_loader):

			data_time.update(time.time() - end)
			tgt_img_var = Variable(tgt_img.cuda())
			ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
			ref_poses_var = [Variable(pose.cuda()) for pose in ref_poses]
			ref_noise_poses_var = [Variable(pose.cuda()) for pose in ref_noise_poses]
			initial_pose_var = Variable(initial_pose.cuda())

			ref_depths_var = [Variable(dep.cuda()) for dep in ref_depths]
			intrinsics_var = Variable(intrinsics.cuda())
			intrinsics_inv_var = Variable(intrinsics_inv.cuda())
			tgt_depth_var = Variable(tgt_depth.cuda())

			pose = torch.cat(ref_poses_var, 1)

			noise_pose = torch.cat(ref_noise_poses_var, 1)

			pose_norm = torch.norm(noise_pose[:, :, :3, 3], dim=-1, keepdim=True)  # b * n* 1

			p_angle, p_trans, rot_c, trans_c = pose_net(tgt_img_var, ref_imgs_var, initial_pose_var, noise_pose,
			                                            intrinsics_var,
			                                            intrinsics_inv_var,
			                                            tgt_depth_var,
			                                            ref_depths_var, trans_norm=pose_norm)

			batch_size = p_angle.shape[0]
			p_angle_v = torch.sum(F.softmax(p_angle, dim=1).view(batch_size, -1, 1) * rot_c, dim=1)
			p_trans_v = torch.sum(F.softmax(p_trans, dim=1).view(batch_size, -1, 1) * trans_c, dim=1)
			p_matrix = Variable(torch.zeros((batch_size, 4, 4)).float()).cuda()
			p_matrix[:, 3, 3] = 1
			p_matrix[:, :3, :] = torch.cat([angle2matrix(p_angle_v), p_trans_v.unsqueeze(-1)], dim=-1)  # 2*3*4

			p_rel_pose = torch.ones_like(noise_pose)
			for bat in range(batch_size):
				path = tgt_path[bat]
				dirname = Path.dirname(path)

				orig_poses = np.genfromtxt(Path.join(dirname, args.pose_init + "_poses.txt"))
				for j in range(len(ref_imgs)):
					p_rel_pose[:, j] = torch.matmul(noise_pose[:, j], inv(p_matrix))

					seq_num = int(Path.basename(ref_paths[bat][j])[:-4])
					orig_poses[seq_num] = p_rel_pose[bat, j, :3, :].data.cpu().numpy().reshape(12, )

					p_aa = mat2axangle(p_rel_pose[bat, j, :3, :3].data.cpu().numpy())
					gt_aa = mat2axangle(pose[bat, j, :3, :3].data.cpu().numpy(), unit_thresh=1e-2)

					n_aa = mat2axangle(noise_pose[bat, j, :3, :3].data.cpu().numpy(), unit_thresh=1e-2)
					p_t = p_rel_pose[bat, j, :3, 3].data.cpu().numpy()
					gt_t = pose[bat, j, :3, 3].data.cpu().numpy()
					n_t = noise_pose[bat, j, :3, 3].data.cpu().numpy()
					p_aa = p_aa[0] * p_aa[1]
					n_aa = n_aa[0] * n_aa[1]
					gt_aa = gt_aa[0] * gt_aa[1]
					error = compute_motion_errors(np.concatenate([n_aa, n_t]), np.concatenate([gt_aa, gt_t]), True)
					error_p = compute_motion_errors(np.concatenate([p_aa, p_t]), np.concatenate([gt_aa, gt_t]), True)
					print("%d n r%.6f, t%.6f" % (i, error[0], error[2]))
					print("%d p r%.6f, t%.6f" % (i, error_p[0], error_p[2]))
					errors[0, 0, i] += error[0]
					errors[0, 1, i] += error[2]
					errors[1, 0, i] += error_p[0]
					errors[1, 1, i] += error_p[2]
				errors[:, :, i] /= len(ref_imgs)
				if args.save and not Path.exists(Path.join(dirname, args.save + "_poses.txt")):
					np.savetxt(Path.join(dirname, args.save + "_poses.txt"), orig_poses)

		mean_error = errors.mean(2)
		error_names = ['rot', 'trans']
		print("%s Results : " % args.pose_init)
		print(
			"{:>10}, {:>10}".format(
				*error_names))
		print("{:10.4f}, {:10.4f}".format(*mean_error[0]))

		print("new Results : ")
		print(
			"{:>10}, {:>10}".format(
				*error_names))
		print("{:10.4f}, {:10.4f}".format(*mean_error[1]))


if __name__ == '__main__':
	main()
