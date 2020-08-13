from __future__ import print_function

import torch.nn.functional as F
import torch.utils.data

from convert import *
from inverse_warp import inverse_warp, inverse_warp_cost, depth_warp_cost
from models.submodule import *


def convtext(in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
	return nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
		          padding=((kernel_size - 1) * dilation) // 2, bias=False),
		nn.LeakyReLU(0.1, inplace=True)
	)


class PoseNet(nn.Module):
	def __init__(self, nlabel, std_tr, std_rot, add_geo_cost=False, depth_augment=False):
		super(PoseNet, self).__init__()
		self.nlabel = int(nlabel)
		self.std_tr = std_tr
		self.std_rot = std_rot
		self.add_geo = add_geo_cost
		self.depth_augment = depth_augment
		self.feature_extraction = feature_extraction(pool=True)

		if add_geo_cost:
			self.n_dres0 = nn.Sequential(convbn_3d(66, 32, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
			                             nn.LeakyReLU(inplace=True),
			                             conv_3d(32, 64, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
			                             nn.LeakyReLU(inplace=True),
			                             nn.AvgPool3d((4, 2, 2), stride=(4, 2, 2)),
			                             convbn_3d(64, 128, 3, 1, 1),
			                             nn.LeakyReLU(inplace=True))
			self.n_dres0_trans = nn.Sequential(convbn_3d(66, 64, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
			                                   nn.LeakyReLU(inplace=True),
			                                   nn.AvgPool3d((4, 2, 2), stride=(4, 2, 2)),
			                                   convbn_3d(64, 128, 3, 1, 1),
			                                   nn.LeakyReLU(inplace=True))
		else:
			self.dres0 = nn.Sequential(conv_3d(64, 32, 3, 1, 1),
			                           nn.LeakyReLU(inplace=True),
			                           nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
			                           convbn_3d(32, 32, 3, 1, 1),
			                           nn.LeakyReLU(inplace=True))
			self.dres0_trans = nn.Sequential(conv_3d(64, 32, 3, 1, 1),
			                                 nn.LeakyReLU(inplace=True),
			                                 nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
			                                 convbn_3d(32, 32, 3, 1, 1),
			                                 nn.LeakyReLU(inplace=True))
		self.dres1 = nn.Sequential(conv_3d(128, 128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
		                           nn.LeakyReLU(inplace=True),
		                           convbn_3d(128, 128, 3, 1, 1)
		                           )
		self.dres1_trans = nn.Sequential(convbn_3d(128, 128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
		                                 nn.LeakyReLU(inplace=True),
		                                 convbn_3d(128, 128, 3, 1, 1))

		self.dres2 = nn.Sequential(conv_3d(128, 128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
		                           nn.LeakyReLU(inplace=True),
		                           convbn_3d(128, 128, 3, 1, 1))
		self.dres2_trans = nn.Sequential(convbn_3d(128, 128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
		                                 nn.LeakyReLU(inplace=True),
		                                 convbn_3d(128, 128, 3, 1, 1))
		self.AvgPool3d = nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2))
		self.dres3 = nn.Sequential(  # nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)),
			conv_3d(128, 128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
			nn.LeakyReLU(inplace=True),
			convbn_3d(128, 128, 3, 1, 1)
		)
		self.dres3_trans = nn.Sequential(  # nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)),
			convbn_3d(128, 128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
			nn.LeakyReLU(inplace=True),
			convbn_3d(128, 128, 3, 1, 1)
		)

		self.dres4 = nn.Sequential(nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)),
		                           conv_3d(128, 128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
		                           nn.LeakyReLU(inplace=True),
		                           convbn_3d(128, 128, 3, 1, 1)
		                           )
		self.dres4_trans = nn.Sequential(  # nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)),
			convbn_3d(128, 128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
			nn.LeakyReLU(inplace=True),
			convbn_3d(128, 128, 3, 1, 1)
		)
		self.classify = nn.Sequential(
			conv_3d(128, 512, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
			nn.LeakyReLU(inplace=True),
			nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)),
			convbn_3d(512, 512, (1, 1, 1), (1, 1, 1), 0),
			nn.LeakyReLU(inplace=True),
			nn.AdaptiveAvgPool3d((16, 1, 1)),
		)
		self.fc = nn.Linear(512 * 16, self.nlabel ** 3)
		self.classify_trans = nn.Sequential(convbn_3d(128, 512, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
		                                    nn.LeakyReLU(inplace=True),
		                                    nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)),
		                                    convbn_3d(512, 512, (1, 1, 1), (1, 1, 1), 0),
		                                    nn.LeakyReLU(inplace=True),
		                                    nn.AdaptiveAvgPool3d((16, 1, 1)), )
		self.fc_trans = nn.Linear(512 * 16, self.nlabel ** 3)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.Conv3d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.GroupNorm):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				m.bias.data.zero_()

	# for m in self.modules():
	#     if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
	#         nn.init.xavier_uniform(m.weight)

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.xavier_uniform(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()

	def get_geo_loss_cost(self, refimg_fea, ref_depth, targetimg_fea, targets_depth, pose, intrinsics4,
	                      intrinsics_inv4):
		b, c, h, w = refimg_fea.size()
		# NOTE large memory
		targetimg_fea_t = inverse_warp_cost(targetimg_fea, ref_depth, pose, intrinsics4,
		                                    intrinsics_inv4)  # bNNN * c * h * w

		targetimg_fea_t = targetimg_fea_t.view(b, -1, c, h, w).transpose(1, 2).contiguous()
		n = targetimg_fea_t.size()[2]
		refimg_fea = refimg_fea.view(b, c, 1, h, w).repeat(1, 1, n, 1, 1)

		if self.add_geo:
			projected_depth, warped_depth = depth_warp_cost(targets_depth, ref_depth,
			                                                pose, intrinsics4, intrinsics_inv4)
			return torch.cat([refimg_fea, targetimg_fea_t, projected_depth, warped_depth], dim=1)
		return torch.cat([refimg_fea, targetimg_fea_t], dim=1)

	def get_geo_loss(self, refimg_fea, ref_depth, targetimg_fea, targets_depth, pose, intrinsics4,
	                 intrinsics_inv4):
		# NOTE large memory
		targetimg_fea_t = inverse_warp(targetimg_fea, ref_depth, pose, intrinsics4,
		                               intrinsics_inv4)  # b * c * h * w
		abs_diff = torch.abs(refimg_fea - targetimg_fea_t)
		abs_diff[targetimg_fea_t == 0] = 0
		non_zero_c = torch.sum((targetimg_fea_t != 0).float())
		abs_diff[abs_diff > 1] = 1
		loss = torch.sum(abs_diff) / non_zero_c
		return loss

	def sample_poses(self, ref_pose, trans_norm):
		batch_size = ref_pose.size()[0]
		ref_trans = ref_pose[:, :3, 3].float()
		trans = Variable(torch.Tensor(np.array(range(int(-self.nlabel / 2), int(self.nlabel / 2)))).cuda(),
		                 requires_grad=False)
		trans = - (trans + 0.5) / trans[0] * self.std_tr
		trans1 = trans.view(1, self.nlabel, 1, 1, 1).repeat(batch_size, 1, self.nlabel, self.nlabel,
		                                                    1)  # b * n * n * n * 1
		trans2 = trans.view(1, 1, self.nlabel, 1, 1).repeat(batch_size, self.nlabel, 1, self.nlabel, 1)
		trans3 = trans.view(1, 1, 1, self.nlabel, 1).repeat(batch_size, self.nlabel, self.nlabel, 1, 1)
		trans_vol = torch.cat((trans1, trans2, trans3), 4) * trans_norm.view(batch_size, 1, 1, 1,
		                                                                     1)  # b * n * n * n * 3
		trans_volume = ref_trans.view(batch_size, 1, 1, 1, 3).repeat(1, self.nlabel, self.nlabel, self.nlabel, 1) \
		               + trans_vol
		trans_disp = ref_pose.clone()
		trans_disp = trans_disp.view(batch_size, 1, 1, 1, 4, 4).repeat(1, self.nlabel, self.nlabel, self.nlabel,
		                                                               1, 1)
		trans_disp[:, :, :, :, :3, 3] = trans_volume  # b * n * n * n * 4 * 4
		trans_vol = trans_vol.view(batch_size, -1, 3)

		rot = Variable(torch.Tensor(np.array(range(int(-self.nlabel / 2),
		                                           int(self.nlabel / 2)))).cuda(), requires_grad=False)
		rot = - (rot + 0.5) / rot[0] * self.std_rot
		rot1 = rot.view(1, self.nlabel, 1, 1, 1).repeat(batch_size, 1, self.nlabel, self.nlabel,
		                                                1)  # b * n * n * n * 1
		rot2 = rot.view(1, 1, self.nlabel, 1, 1).repeat(batch_size, self.nlabel, 1, self.nlabel, 1)
		rot3 = rot.view(1, 1, 1, self.nlabel, 1).repeat(batch_size, self.nlabel, self.nlabel, 1, 1)
		angle_vol = torch.cat((rot1, rot2, rot3), 4)  # b * n * n * n * 3
		angle_matrix = angle2matrix(angle_vol)  # b * n * n * n * 3 * 3
		rot_volume = torch.matmul(angle_matrix,
		                          ref_pose[:, :3, :3].view(batch_size, 1, 1, 1, 3,
		                                                   3).repeat(1, self.nlabel, self.nlabel, self.nlabel, 1, 1))
		angle_vol = angle_vol.view(batch_size, -1, 3)
		rot_disp = ref_pose.clone()
		rot_disp = rot_disp.view(batch_size, 1, 1, 1, 4, 4).repeat(1, self.nlabel, self.nlabel, self.nlabel,
		                                                           1, 1)
		rot_disp[:, :, :, :, :3, :3] = rot_volume  # b * n * n * n * 4 * 4

		return rot_disp.float(), trans_disp.float(), angle_vol, trans_vol

	def forward(self, ref, targets, ref_pose, tgt_poses, intrinsics, intrinsics_inv, ref_depth=None,
	            targets_depths=None, gt_poses=None, mode=0, trans_norm=None):

		intrinsics4 = intrinsics.clone()
		intrinsics_inv4 = intrinsics_inv.clone()
		intrinsics4[:, :2, :] = intrinsics4[:, :2, :] / 8
		intrinsics_inv4[:, :2, :2] = intrinsics_inv4[:, :2, :2] * 8

		refimg_fea = self.feature_extraction(ref)
		batch_size, channel, fea_h, fea_w = refimg_fea.size()

		rot_pose_vol, trans_pose_vol, angle_vol, trans_vol = self.sample_poses(ref_pose, trans_norm[:, 0])
		ref_depth = ref_depth.unsqueeze(1)
		ref_depth = torch.nn.functional.upsample(ref_depth, [fea_h, fea_w], mode='bilinear')
		ref_depth = ref_depth.squeeze(1)
		for j, target in enumerate(targets):
			targetimg_fea = self.feature_extraction(target)

			tgt_depth = targets_depths[j].unsqueeze(1)
			tgt_depth = torch.nn.functional.upsample(tgt_depth, [fea_h, fea_w], mode='bilinear')
			tgt_depth = tgt_depth.squeeze(1)

			tgt_pose = tgt_poses[:, j].unsqueeze(1).float().repeat(1, self.nlabel * self.nlabel * self.nlabel,
			                                                       1, 1).view(-1, 4, 4)  # bnnn* 4 * 4
			rel_trans_vol = torch.bmm(tgt_pose, inv(trans_pose_vol.view(-1, 4, 4)))[:, :3, :4].contiguous().view(
				batch_size, self.nlabel, self.nlabel, self.nlabel, 3, 4)
			rel_rot_vol = torch.bmm(tgt_pose, inv(rot_pose_vol.view(-1, 4, 4)))[:, :3, :4].contiguous().view(
				batch_size, self.nlabel, self.nlabel, self.nlabel, 3, 4)

			if self.add_geo:
				trans_cost = self.get_geo_loss_cost(refimg_fea, ref_depth, targetimg_fea, tgt_depth,
				                                    rel_trans_vol, intrinsics4, intrinsics_inv4)  # B*NNN
				rot_cost = self.get_geo_loss_cost(refimg_fea, ref_depth, targetimg_fea, tgt_depth,
				                                  rel_rot_vol, intrinsics4, intrinsics_inv4)  # B*NNN
			else:
				rot_cost = self.get_geo_loss_cost(refimg_fea, ref_depth, targetimg_fea, None,
				                                  rel_rot_vol, intrinsics4, intrinsics_inv4)  # B*NNN
				trans_cost = self.get_geo_loss_cost(refimg_fea, ref_depth, targetimg_fea, None,
				                                    rel_trans_vol, intrinsics4, intrinsics_inv4)  # B*NNN

			if mode % 2 == 0:
				trans_cost = trans_cost.contiguous()
				if self.add_geo:
					trans_cost0 = self.n_dres0_trans(trans_cost)
				else:
					trans_cost0 = self.dres0_trans(trans_cost)
				trans_cost0 = self.dres1_trans(trans_cost0) + trans_cost0
				trans_cost0 = self.dres2_trans(trans_cost0) + trans_cost0
				trans_cost0 = self.dres3_trans(trans_cost0) + trans_cost0
				trans_cost0 = self.dres4_trans(trans_cost0) + trans_cost0
				trans_cost0 = self.classify_trans(trans_cost0)
			if mode < 2:
				rot_cost = rot_cost.contiguous()
				if self.add_geo:
					rot_cost0 = self.n_dres0(rot_cost)
				else:
					rot_cost0 = self.dres0(rot_cost)
				rot_cost0 = self.dres1(rot_cost0) + rot_cost0
				rot_cost0 = self.dres2(rot_cost0) + rot_cost0
				rot_cost0 = self.dres3(rot_cost0) + rot_cost0
				rot_cost0 = self.dres4(rot_cost0) + self.AvgPool3d(rot_cost0)
				rot_cost0 = self.classify(rot_cost0)
			if j == 0:
				trans_costs = trans_cost0

				rot_costs = rot_cost0

			else:
				trans_costs = trans_cost0 + trans_costs

				rot_costs = rot_costs + rot_cost0

		if mode % 2 == 0:
			trans_costs = (trans_costs / len(targets))
			trans_costs = trans_costs.view(batch_size, -1)
			trans_costs = self.fc_trans(trans_costs)

			pred_trans = trans_costs

		if mode < 2:
			rot_costs = (rot_costs / len(targets))
			rot_costs = rot_costs.view(batch_size, -1)

			rot_costs = self.fc(rot_costs)
			pred_rot = rot_costs

		return pred_rot, pred_trans, angle_vol, trans_vol
