from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn.functional as F

pixel_coords = None


def set_id_grid(depth):
	global pixel_coords
	b, h, w = depth.size()
	i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1, h, w)).type_as(depth)  # [1, H, W]
	j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1, h, w)).type_as(depth)  # [1, H, W]
	ones = Variable(torch.ones(1, h, w)).type_as(depth)

	pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
	condition = [input.ndimension() == len(expected)]
	for i, size in enumerate(expected):
		if size.isdigit():
			condition.append(input.size(i) == int(size))
	assert (all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected),
	                                                                          list(input.size()))


def pixel2cam(depth, intrinsics_inv):
	global pixel_coords
	"""Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
	b, h, w = depth.size()
	if (pixel_coords is None) or pixel_coords.size(2) < h:
		set_id_grid(depth)
	current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).contiguous().view(b, 3,
	                                                                                       -1).cuda()  # [B, 3, H*W]
	cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)
	return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode, rounded=False):
	"""Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
	b, _, h, w = cam_coords.size()
	cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
	if proj_c2p_rot is not None:
		pcoords = proj_c2p_rot.bmm(cam_coords_flat)
	else:
		pcoords = cam_coords_flat

	if proj_c2p_tr is not None:
		pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
	X = pcoords[:, 0]
	Y = pcoords[:, 1]
	Z = pcoords[:, 2].clamp(min=1e-3)
	if rounded:
		X_norm = torch.round(2 * (X / Z)) / (
					w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
		Y_norm = torch.round(2 * (Y / Z)) / (h - 1) - 1  # Idem [B, H*W]
	else:
		X_norm = 2 * (X / Z) / (
					w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
		Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]

	if padding_mode == 'zeros':
		X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
		X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
		Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
		Y_norm[Y_mask] = 2

	pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
	return pixel_coords.view(b, h, w, 2)


def cam2pixel_cost(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
	"""Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: [B, 3, H, W]
        proj_c2p_rot: rotation -- b * NNN* 3 * 3
        proj_c2p_tr: translation -- b * NNN * 3 * 1
    Returns:
        array of [-1,1] coordinates -- [B, NNN, 2, H, W]
    """
	b, _, h, w = cam_coords.size()
	n = proj_c2p_rot.shape[1]
	cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
	# if proj_c2p_rot is not None:
	pcoords = proj_c2p_rot.matmul(cam_coords_flat.view(b, 1, 3, h * w))  # b * NNN * 3 * (h*w)

	if proj_c2p_tr is not None:
		pcoords = pcoords + proj_c2p_tr  # [B, NNN, 3, H*W]
	X = pcoords[:, :, 0]  # [B, NNN, H*W]
	Y = pcoords[:, :, 1]
	Z = pcoords[:, :, 2].clamp(min=1e-3)

	X_norm = 2 * (X / Z) / (w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
	Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
	if padding_mode == 'zeros':
		X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
		X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
		Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
		Y_norm[Y_mask] = 2

	pixel_coords = torch.stack([X_norm, Y_norm], dim=3)  # [B, NNN, H*W, 2]
	return pixel_coords.view(b, -1, h, w, 2)


def cam2depth(cam_coords, proj_c2p_rot, proj_c2p_tr):
	"""Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        depth -- [B, H, W]
    """
	b, _, h, w = cam_coords.size()
	cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
	if proj_c2p_rot is not None:
		pcoords = proj_c2p_rot.bmm(cam_coords_flat)
	else:
		pcoords = cam_coords_flat

	if proj_c2p_tr is not None:
		pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
	z = pcoords[:, 2, :].contiguous()
	return z.view(b, h, w)


def cam2depth_cost(cam_coords, proj_c2p_rot, proj_c2p_tr):
	"""Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
        proj_c2p_rot: rotation matrix of cameras -- b * nnn* 3 * 3
        proj_c2p_tr: translation vectors of cameras -- b * nnn* 3 * 1
    Returns:
        depth -- [B, nnn, H, W]
    """
	b, _, h, w = cam_coords.size()
	n = proj_c2p_rot.shape[1]
	cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
	# if proj_c2p_rot is not None:
	pcoords = proj_c2p_rot.matmul(cam_coords_flat.resize(b, 1, 3, h * w))  # b, nnn, 3, h*w
	# else:
	#     pcoords = cam_coords_flat

	if proj_c2p_tr is not None:
		pcoords = pcoords + proj_c2p_tr  # b, nnn, 3, h*w
	z = pcoords[:, :, 2, :].contiguous()
	return z.view(b, n, h, w)


def depth_warp(fdepth, depth, pose, intrinsics, intrinsics_inv, padding_mode='zeros'):
	"""
    warp a target depth to the source image plane.

    Args:
        fdepth: the source depth (where to sample pixels) -- [B, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        target depth warped to the source image plane
    """
	check_sizes(depth, 'depth', 'BHW')
	check_sizes(pose, 'pose', 'B34')
	check_sizes(intrinsics, 'intrinsics', 'B33')
	check_sizes(intrinsics_inv, 'intrinsics', 'B33')
	assert (intrinsics_inv.size() == intrinsics.size())

	batch_size, feat_height, feat_width = depth.size()

	cam_coords = pixel2cam(depth, intrinsics_inv)
	pose_mat = pose
	pose_mat = pose_mat.cuda()

	# Get projection matrix for tgt camera frame to source pixel frame
	proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]
	src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:],
	                             padding_mode, rounded=True)  # [B,H,W,2]
	projected_depth = cam2depth(cam_coords, pose_mat[:, :, :3], pose_mat[:, :, -1:])
	# projected_depth = projected_depth.clamp(min=-1e1, max=1e3)
	fdepth_expand = fdepth.unsqueeze(1)
	fdepth_expand = torch.nn.functional.upsample(fdepth_expand, [feat_height, feat_width], mode='bilinear')

	warped_depth = torch.nn.functional.grid_sample(fdepth_expand, src_pixel_coords, mode="nearest",
	                                               padding_mode=padding_mode)
	warped_depth = warped_depth.view(batch_size, feat_height, feat_width)
	# [B, H, W]
	projected_depth = projected_depth.clamp(min=1e-3, max=float(torch.max(warped_depth) + 10))
	return projected_depth, warped_depth


def depth_warp_cost(fdepth, depth, pose, intrinsics, intrinsics_inv, padding_mode='zeros'):
	"""
    warp a target depth to the source image plane.

    Args:
        fdepth: the source depth (where to sample pixels) -- [B, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- b * n * n * n * 3 * 4
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        target depth warped to the source image plane
    """
	check_sizes(depth, 'depth', 'BHW')
	# check_sizes(pose, 'pose', 'BNN34')
	check_sizes(intrinsics, 'intrinsics', 'B33')
	check_sizes(intrinsics_inv, 'intrinsics', 'B33')
	assert (intrinsics_inv.size() == intrinsics.size())

	batch_size, feat_height, feat_width = depth.size()
	pose = pose.view(batch_size, -1, 3, 4)  # [B,NNN, 3, 4]
	cost_n = pose.shape[1]

	cam_coords = pixel2cam(depth, intrinsics_inv)
	pose_mat = pose
	pose_mat = pose_mat.cuda()

	# Get projection matrix for tgt camera frame to source pixel frame
	intrinsics = intrinsics.resize(batch_size, 1, 3, 3)
	proj_cam_to_src_pixel = intrinsics.matmul(pose_mat)  # b * nnn * 3 * 4
	src_pixel_coords = cam2pixel_cost(cam_coords, proj_cam_to_src_pixel[:, :, :, :3],
	                                  proj_cam_to_src_pixel[:, :, :, -1:],
	                                  padding_mode).view(-1, feat_height, feat_width, 2)  # [B,nnn,H,W,2]
	projected_depth = cam2depth_cost(cam_coords, pose_mat[:, :, :, :3], pose_mat[:, :, :, -1:])  # b nnn h w
	fdepth_expand = fdepth.unsqueeze(1)

	fdepth_expand = fdepth_expand.resize(batch_size, 1, feat_height, feat_width).repeat(
		1, cost_n, 1, 1).view(-1, 1, feat_height, feat_width)
	warped_depth = torch.nn.functional.grid_sample(fdepth_expand, src_pixel_coords, mode='nearest',
	                                               padding_mode=padding_mode)
	warped_depth = warped_depth.view(-1, 1, feat_height, feat_width)

	projected_depth = projected_depth.clamp(min=1e-3, max=float(torch.max(warped_depth) + 10))
	return projected_depth.view(-1, 1, cost_n, feat_height, feat_width), \
	       warped_depth.view(-1, 1, cost_n, feat_height, feat_width)  # b *nnn * 1 * h * w


def inverse_warp(feat, depth, pose, intrinsics, intrinsics_inv, padding_mode='zeros'):
	"""
    Inverse warp a source image to the target image plane.

    Args:
        feat: the source feature (where to sample pixels) -- [B, CH, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
	check_sizes(depth, 'depth', 'BHW')
	check_sizes(pose, 'pose', 'B34')
	check_sizes(intrinsics, 'intrinsics', 'B33')
	check_sizes(intrinsics_inv, 'intrinsics', 'B33')

	assert (intrinsics_inv.size() == intrinsics.size())

	batch_size, _, feat_height, feat_width = feat.size()

	cam_coords = pixel2cam(depth, intrinsics_inv)

	pose_mat = pose
	pose_mat = pose_mat.cuda()

	# Get projection matrix for tgt camera frame to source pixel frame
	proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

	src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:],
	                             padding_mode, rounded=True)  # [B,H,W,2]

	projected_feat = torch.nn.functional.grid_sample(feat, src_pixel_coords, mode='nearest', padding_mode=padding_mode)

	return projected_feat


def inverse_warp_cost(feat, depth, pose, intrinsics, intrinsics_inv, padding_mode='zeros'):
	"""
    ref -> targets

    Args:
        feat: b * c * h * w
        depth: b * h * w
        pose: b * n (* n * n) * 3 * 4
        intrinsics: [B, 3, 3]
        intrinsics_inv: [B, 3, 3]
    """

	check_sizes(depth, 'depth', 'BHW')
	check_sizes(intrinsics, 'intrinsics', 'B33')
	check_sizes(intrinsics_inv, 'intrinsics', 'B33')

	assert (intrinsics_inv.size() == intrinsics.size())

	batch_size, channal, feat_height, feat_width = feat.size()

	cam_coords = pixel2cam(depth, intrinsics_inv)  # [B, 3, H, W]
	pose = pose.view(batch_size, -1, 3, 4)  # [B,NNN, 3, 4]
	cost_n = pose.shape[1]
	pose_mat = pose
	pose_mat = pose_mat.cuda()

	# Get projection matrix for tgt camera frame to source pixel frame
	intrinsics = intrinsics.view(batch_size, 1, 3, 3)
	proj_cam_to_src_pixel = intrinsics.matmul(pose_mat)  # b * NNN  * 3 * 4

	src_pixel_coords = cam2pixel_cost(cam_coords, proj_cam_to_src_pixel[:, :, :, :3],
	                                  proj_cam_to_src_pixel[:, :, :, -1:],
	                                  padding_mode)  # [B,NNN,H,W,2]
	feat = feat.view(batch_size, 1, channal, feat_height, feat_width).repeat(1, cost_n, 1, 1, 1).view(-1, channal,
	                                                                                                  feat_height,
	                                                                                                  feat_width)
	projected_feat = torch.nn.functional.grid_sample(feat, src_pixel_coords.view(-1, feat_height, feat_width, 2),
	                                                 padding_mode=padding_mode)

	return projected_feat  # (bNNN) * c * h * w
