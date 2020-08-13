import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def depth2normal(d_im):
	zy, zx = np.gradient(d_im)
	# You may also consider using Sobel to get a joint Gaussian smoothing and differentation
	# to reduce noise
	# zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
	# zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

	normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
	n = np.linalg.norm(normal, axis=2)
	normal[:, :, 0] /= n
	normal[:, :, 1] /= n
	normal[:, :, 2] /= n

	return normal


def imgrad(img):
	img = torch.mean(img, 1, True)
	fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
	conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
	weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
	if img.is_cuda:
		weight = weight.cuda()
	conv1.weight = nn.Parameter(weight, requires_grad=False)
	grad_x = conv1(img)

	fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
	weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
	if img.is_cuda:
		weight = weight.cuda()
	conv2.weight = nn.Parameter(weight, requires_grad=False)
	grad_y = conv2(img)

	#     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))

	return grad_y, grad_x


class GradLoss(nn.Module):
	def __init__(model):
		super(GradLoss, model).__init__()

	# L1 norm
	def forward(model, grad_fake, grad_real):
		return torch.sum(torch.mean(torch.abs(grad_real - grad_fake)))


def imgrad_yx(img):
	N, C, h, w = img.size()
	grad_y, grad_x = imgrad(img)
	return torch.cat((grad_y.view(N, h, w), grad_x.view(N, h, w)), dim=1)


def matrix2angle(matrix):
	"""
    ref: https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    input size: ... * 3 * 3
    output size:  ... * 3
    """
	i = 0
	j = 1
	k = 2
	dims = [dim for dim in matrix.shape]
	M = matrix.contiguous().view(-1, 3, 3)

	cy = torch.sqrt(M[:, i, i] * M[:, i, i] + M[:, j, i] * M[:, j, i])

	if torch.max(cy).item() > 1e-15 * 4:
		ax = torch.atan2(M[:, k, j], M[:, k, k])
		ay = torch.atan2(-M[:, k, i], cy)
		az = torch.atan2(M[:, j, i], M[:, i, i])
	else:
		ax = torch.atan2(-M[:, j, k], M[:, j, j])
		ay = torch.atan2(-M[:, k, i], cy)
		az = torch.zero(matrix.shape[:-1])
	return torch.cat([torch.unsqueeze(ax, -1), torch.unsqueeze(ay, -1), torch.unsqueeze(az, -1)], -1).view(dims[:-1])


def angle2matrix(angle):
	"""
    ref: https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    input size:  ... * 3
    output size: ... * 3 * 3
    """
	dims = [dim for dim in angle.shape]
	angle = angle.view(-1, 3)

	i = 0
	j = 1
	k = 2
	ai = angle[:, 0]
	aj = angle[:, 1]
	ak = angle[:, 2]
	si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
	ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
	cc, cs = ci * ck, ci * sk
	sc, ss = si * ck, si * sk

	M = torch.eye(3)
	M = M.view(1, 3, 3)
	M = Variable(M.repeat(angle.shape[0], 1, 1).cuda())

	M[:, i, i] = cj * ck
	M[:, i, j] = sj * sc - cs
	M[:, i, k] = sj * cc + ss
	M[:, j, i] = cj * sk
	M[:, j, j] = sj * ss + cc
	M[:, j, k] = sj * cs - sc
	M[:, k, i] = -sj
	M[:, k, j] = cj * si
	M[:, k, k] = cj * ci

	return M.view(dims + [3])


def b_inv(A):
	eye = A.new_ones(A.size(-1)).diag().expand_as(A)
	b_inv, _ = torch.gesv(eye, A)
	return b_inv


def inv(A, eps=1e-10):
	assert len(A.shape) == 3 and \
	       A.shape[1] == A.shape[2]
	n = A.shape[1]
	U = A.clone()
	L = Variable(torch.zeros(A.shape).cuda(), requires_grad=False)
	L[:, range(n), range(n)] = 1
	L_inv = L.clone()

	# A = LU
	# [A I] = [LU I] -> [U L^{-1}]

	for i in range(n - 1):
		L[:, i + 1:, i:i + 1] = U[:, i + 1:, i:i + 1] / (U[:, i:i + 1, i:i + 1] + eps)
		L_inv[:, i + 1:, :] = L_inv[:, i + 1:, :] - L[:, i + 1:, i:i + 1].matmul(L_inv[:, i:i + 1, :])
		U[:, i + 1:, :] = U[:, i + 1:, :] - L[:, i + 1:, i:i + 1].matmul(U[:, i:i + 1, :])

	# [U L^{-1}] -> [I U^{-1}L^{-1}] = [I (LU)^{-1}]
	A_inv = L_inv.clone()

	for i in range(n - 1, -1, -1):
		A_inv[:, i:i + 1, :] = A_inv[:, i:i + 1, :] / (U[:, i:i + 1, i:i + 1] + eps)
		U[:, i:i + 1, :] = U[:, i:i + 1, :] / (U[:, i:i + 1, i:i + 1] + eps)

		if i > 0:
			A_inv[:, :i, :] = A_inv[:, :i, :] - U[:, :i, i:i + 1].matmul(A_inv[:, i:i + 1, :])
			U[:, :i, :] = U[:, :i, :] - U[:, :i, i:i + 1].matmul(U[:, i:i + 1, :])

	return A_inv
