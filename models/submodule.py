from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation=1):
	return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
	                               padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
	                     nn.BatchNorm2d(out_planes))


def convbn_3d_o(in_planes, out_planes, kernel_size, stride, pad):
	return nn.Sequential(
		nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
		nn.BatchNorm3d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
	return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride),
	                     nn.GroupNorm(num_groups=8, num_channels=out_planes))


def conv_3d(in_planes, out_planes, kernel_size, stride, pad):
	return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride))


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
		super(BasicBlock, self).__init__()

		self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
		                           nn.ReLU(inplace=True))

		self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)

		if self.downsample is not None:
			x = self.downsample(x)

		out += x

		return out


class matchshifted(nn.Module):
	def __init__(self):
		super(matchshifted, self).__init__()

	def forward(self, left, right, shift):
		batch, filters, height, width = left.size()
		shifted_left = F.pad(
			torch.index_select(left, 3, Variable(torch.LongTensor([i for i in range(shift, width)])).cuda()),
			(shift, 0, 0, 0))
		shifted_right = F.pad(
			torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width - shift)])).cuda()),
			(shift, 0, 0, 0))
		out = torch.cat((shifted_left, shifted_right), 1).view(batch, filters * 2, 1, height, width)
		return out


class disparityregression(nn.Module):
	def __init__(self, maxdisp):
		super(disparityregression, self).__init__()
		self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda(),
		                     requires_grad=False)

	def forward(self, x):
		disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
		out = torch.sum(x * disp, 1)
		return out


class disparityregression_p(nn.Module):
	def __init__(self, nlabel, std=0.2):
		super(disparityregression_p, self).__init__()
		disp = Variable(torch.Tensor(np.array(range(int(-nlabel / 12), int(nlabel / 12)))).cuda() + 0.5,
		                requires_grad=False)
		self.disp = -disp / disp[0] * std * 2.5
		# self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

	def forward(self, x):
		disp = self.disp.repeat(x.size()[0], 6, 1)
		out = torch.sum(x * disp, 2)
		return out


class disparityregression_p1(nn.Module):
	def __init__(self, nlabel, std=0.2):
		super(disparityregression_p1, self).__init__()
		disp = Variable(torch.Tensor(np.array(range(int(-nlabel / 2), int(nlabel / 2)))).cuda() + 0.5,
		                requires_grad=False)
		self.disp = -disp / disp[0] * std * 2.5
		# self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

	def forward(self, x):
		disp = self.disp.repeat(x.size()[0], 1, 1)
		out = torch.sum(x * disp, 2)
		return out


class PoseRegression(nn.Module):
	def __init__(self, nlabel, std=0.2):
		super(PoseRegression, self).__init__()
		self.nlabel = nlabel
		trans = Variable(torch.Tensor(np.array(range(int(-nlabel / 2), int(nlabel / 2)))).cuda() + 0.5,
		                 requires_grad=False)
		trans = - trans / trans[0] * std * 2.5
		trans1 = trans.resize(1, 1, nlabel, 1, 1).repeat(1, 1, 1, nlabel, nlabel)
		trans2 = trans.resize(1, 1, 1, nlabel, 1).repeat(1, 1, nlabel, 1, nlabel)
		trans3 = trans.resize(1, 1, 1, 1, nlabel).repeat(1, 1, nlabel, nlabel, 1)
		self.trans = torch.cat((trans1, trans2, trans3), 1)

	def forward(self, x):
		trans = self.trans.repeat(x.size()[0], 1, 1, 1, 1)
		trans = (x * trans).resize(x.size()[0], 3, self.nlabel ** 3)
		out = torch.sum(trans, 2)
		return out


class feature_extraction(nn.Module):
	def __init__(self, pool=False):
		super(feature_extraction, self).__init__()
		self.inplanes = 32
		self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
		                               nn.ReLU(inplace=True),
		                               convbn(32, 32, 3, 1, 1, 1),
		                               nn.ReLU(inplace=True),
		                               convbn(32, 32, 3, 1, 1, 1),
		                               nn.ReLU(inplace=True))

		self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
		self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
		self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
		self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

		self.branch1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
		                             convbn(128, 32, 1, 1, 0, 1),
		                             nn.ReLU(inplace=True))

		self.branch2 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
		                             convbn(128, 32, 1, 1, 0, 1),
		                             nn.ReLU(inplace=True))

		self.branch3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
		                             convbn(128, 32, 1, 1, 0, 1),
		                             nn.ReLU(inplace=True))

		self.branch4 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
		                             convbn(128, 32, 1, 1, 0, 1),
		                             nn.ReLU(inplace=True))
		if pool:
			self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
			                              nn.ReLU(inplace=True),
			                              nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False),
			                              nn.AvgPool2d((2, 2), stride=(2, 2)))
		else:
			self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
			                              nn.ReLU(inplace=True),
			                              nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

	def _make_layer(self, block, planes, blocks, stride, pad, dilation):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion), )

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

		return nn.Sequential(*layers)

	def forward(self, x):
		output = self.firstconv(x)
		output = self.layer1(output)
		output_raw = self.layer2(output)
		output = self.layer3(output_raw)
		output_skip = self.layer4(output)

		output_branch1 = self.branch1(output_skip)
		output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

		output_branch2 = self.branch2(output_skip)
		output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

		output_branch3 = self.branch3(output_skip)
		output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

		output_branch4 = self.branch4(output_skip)
		output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

		output_feature = torch.cat(
			(output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
		output_feature = self.lastconv(output_feature)

		return output_feature
