import argparse
import os

import numpy as np
import time
import torch
import torch.optim
import torch.utils.data
from path import Path
from scipy.misc import imsave
from torch.autograd import Variable

import custom_transforms
from loss_functions import compute_errors_test
from models import PSNet as PSNet
from sequence_folders import SequenceFolder
from utils import tensor2array

parser = argparse.ArgumentParser(description='DeepSFM depth subnet test script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=2)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--pretrained-dps', dest='pretrained_dps',
                    default='depth_checkpoint.pth.tar', metavar='PATH',
                    help='path to pre-trained depth_net model')
parser.add_argument('--save', default="I6", type=str, help='save prefix')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--output-dir', default='result_tmp', type=str, help='Output directory for saving predictions '
                                                                         'in a big 3D numpy file')
parser.add_argument('--ttype', default='test.txt', type=str, help='Text file indicates input data')
parser.add_argument('--nlabel', type=int, default=64, help='number of label')
parser.add_argument('--mindepth', type=float, default=0.5, help='minimum depth')
parser.add_argument('--maxdepth', type=float, default=100, help='maximum depth')
parser.add_argument('--output-print', action='store_true', help='print output depth')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--geo', '--geo-cost', default=True, type=bool,
                    metavar='GC', help='whether add geometry cost')
parser.add_argument('--pose_init', default='demon', help='path to init pose')
parser.add_argument('--depth_init', default='demon', help='path to init depth')


def main():
	args = parser.parse_args()

	normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
	                                        std=[0.5, 0.5, 0.5])
	valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
	val_set = SequenceFolder(
		args.data,
		transform=valid_transform,
		seed=args.seed,
		ttype=args.ttype,
		dataset='',
		sequence_length=args.sequence_length,
		add_geo=args.geo,
		depth_source=args.depth_init,
		pose_source='%s_poses.txt' % args.pose_init if args.pose_init else 'poses.txt',
		scale=False,
		size=0,
		req_gt=True,
		get_path=True
	)

	print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
	val_loader = torch.utils.data.DataLoader(
		val_set, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	depth_net = PSNet(args.nlabel, args.mindepth, add_geo_cost=args.geo).cuda()
	weights = torch.load(args.pretrained_dps)
	depth_net.load_state_dict(weights['state_dict'])
	depth_net.eval()

	output_dir = Path(args.output_dir)
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	errors = np.zeros((2, 13, int(np.ceil(len(val_loader) / args.print_freq))), np.float32)
	with torch.no_grad():

		for ii, (tgt_img, ref_imgs, ref_poses, poses_gt, intrinsics, intrinsics_inv, tgt_depth, ref_depths, tgt_path
		         ) in enumerate(val_loader):
			if ii % args.print_freq == 0:
				i = int(ii / args.print_freq)
				tgt_img_var = Variable(tgt_img.cuda())
				ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
				ref_poses_var = [Variable(pose.cuda()) for pose in ref_poses]
				poses_gt_var = [Variable(pose_gt.cuda()) for pose_gt in poses_gt]

				ref_depths_var = [Variable(dep.cuda()) for dep in ref_depths]
				intrinsics_var = Variable(intrinsics.cuda())
				intrinsics_inv_var = Variable(intrinsics_inv.cuda())

				# compute output
				pose = torch.cat(ref_poses_var, 1)
				poses_gt = torch.cat(poses_gt_var, 1)
				rel_pose = poses_gt.squeeze().data.cpu().numpy()

				scale = float(np.sqrt(rel_pose[:3, 3].dot(rel_pose[:3, 3])))

				start = time.time()
				output_depth = depth_net(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var,
				                         ref_depths_var)

				elps = time.time() - start
				tgt_disp = args.mindepth * args.nlabel / tgt_depth
				output_disp = args.mindepth * args.nlabel / output_depth

				mask = (tgt_depth <= args.maxdepth) & (tgt_depth >= 0.5) & (tgt_depth == tgt_depth)

				tgt_depth = tgt_depth / scale
				output_depth_scaled = output_depth / scale
				output_disp_ = torch.squeeze(output_disp.data.cpu(), 1)
				output_depth_ = torch.squeeze(output_depth_scaled.data.cpu(), 1)
				if args.save:
					output_depth_n = torch.squeeze(output_depth.data.cpu(), 1).numpy()[0]
					save_path = tgt_path[0][:-4] + "_" + args.save + ".npy"
					if not os.path.exists(save_path):
						np.save(save_path, output_depth_n)
				errors[0, :10, i] = compute_errors_test(tgt_depth[mask], output_depth_[mask])
				errors[1, :10, i] = compute_errors_test(tgt_disp[mask], output_disp_[mask])

				print('iter{}, Elapsed Time {} Abs Error {:.10f}'.format(i, elps, errors[0, 0, i]))

				if args.output_print:
					output_disp_n = (output_disp_).numpy()[0]
					np.save(output_dir / '{:08d}{}'.format(i, '.npy'), output_disp_n)
					disp = (255 * tensor2array(torch.from_numpy(output_disp_n), max_value=args.nlabel,
					                           colormap='bone')).astype(np.uint8)
					disp = disp.transpose(1, 2, 0)
					imsave(output_dir / '{:08d}_disp{}'.format(i, '.png'), disp)

	mean_errors = errors.mean(2)
	error_names = ['abs_rel', 'abs_diff', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3', 'L1-inv', "sc-inv", 'ra', 'rd',
	               'ta']
	print("{}".format(args.output_dir))
	print("Depth & angle Results : ")
	print(
		"{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
			*error_names))
	print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, "
	      "{:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

	np.savetxt(output_dir / 'errors.csv', mean_errors, fmt='%1.4f', delimiter=',')


if __name__ == '__main__':
	main()
