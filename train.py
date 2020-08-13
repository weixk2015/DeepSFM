import argparse
import csv

import time
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

import custom_transforms
from convert import *
from logger import AverageMeter
from loss_functions import compute_errors_train
from models import PSNet as PSNet
from sequence_folders import SequenceFolder
from utils import tensor2array, save_checkpoint, save_path_formatter, adjust_learning_rate

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=3, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=4e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--geo', '--geo-cost', default=True, type=bool,
                    metavar='GC', help='whether add geometry cost')
parser.add_argument('--sn', '--sn-cost', default=False, type=bool,
                    metavar='SN', help='whether add geometry cost')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained-dps', dest='pretrained_dps',
                    default='', metavar='PATH',
                    help='path to pre-trained dispnet model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('--ttype', default='train.txt', type=str, help='Text file indicates input data')
parser.add_argument('--ttype2', default='val.txt', type=str, help='Text file indicates input data')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=100)
parser.add_argument('--nlabel', type=int ,default=64, help='number of label')
parser.add_argument('--mindepth', type=float ,default=0.5, help='minimum depth')
parser.add_argument('--pose_init', default='demon', help='path to init pose')
parser.add_argument('--depth_init', default='demon', help='path to init depth')

n_iter = 0


def main():
    global n_iter
    args = parser.parse_args()
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        ttype=args.ttype,
        add_geo=args.geo,
        depth_source=args.depth_init,
        pose_source='%s_poses.txt'%args.pose_init if args.pose_init else 'poses.txt',
        scale=False
    )
    val_set = SequenceFolder(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        ttype=args.ttype2,
        add_geo=args.geo,
        depth_source=args.depth_init,
        pose_source='%s_poses.txt' % args.pose_init if args.pose_init else 'poses.txt',
        scale=False
    )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    depth_net = PSNet(args.nlabel, args.mindepth, add_geo_cost=args.geo,
                   depth_augment=False, add_sn_cost=args.sn).cuda()

    if args.pretrained_dps:
        # for param in depth_net.feature_extraction.parameters():
        #     param.requires_grad = False


        print("=> using pre-trained weights for DPSNet")
        model_dict = depth_net.state_dict()
        weights = torch.load(args.pretrained_dps)['state_dict']
        pretrained_dict = {k: v for k, v in weights.items() if k in model_dict}

        model_dict.update(pretrained_dict)

        depth_net.load_state_dict(model_dict)

    else:
        depth_net.init_weights()

    cudnn.benchmark = True
    depth_net = torch.nn.DataParallel(depth_net)

    print('=> setting adam solver')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, depth_net.parameters()), args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss'])


    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train_loss = train(args, train_loader, depth_net, optimizer, args.epoch_size, training_writer)

        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': depth_net.module.state_dict()
            },
            epoch)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss])


def train(args, train_loader, depth_net, optimizer, epoch_size, train_writer):
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    # switch to train mode
    depth_net.train()

    end = time.time()

    for i, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth, ref_depths) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img_var = Variable(tgt_img.cuda())
        ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
        ref_poses_var = [Variable(pose.cuda()) for pose in ref_poses]
        ref_depths_var = [Variable(dep.cuda()) for dep in ref_depths]
        intrinsics_var = Variable(intrinsics.cuda())
        intrinsics_inv_var = Variable(intrinsics_inv.cuda())
        tgt_depth_var = Variable(tgt_depth.cuda()).cuda()

        # compute output
        pose = torch.cat(ref_poses_var,1)

        # get mask
        mask = (tgt_depth_var <= args.nlabel*args.mindepth) & (tgt_depth_var >= args.mindepth) & (tgt_depth_var == tgt_depth_var)
        mask.detach_()

        depths = depth_net(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var, ref_depths_var)
        disps = [args.mindepth*args.nlabel/(depth) for depth in depths]

        loss = 0.

        for l, depth in enumerate(depths):
            output = torch.squeeze(depth,1)

            loss += F.smooth_l1_loss(output[mask], tgt_depth_var[mask], size_average=True) * pow(0.7, len(depths)-l-1)


        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('total_loss', loss.data[0], n_iter)
        if n_iter > 0 and n_iter % 5000 == 0:
            save_checkpoint(
                args.save_path, {
                    'epoch': n_iter + 1,
                    'state_dict': depth_net.module.state_dict()
                },
                n_iter)
        if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:

            train_writer.add_image('train Input', tensor2array(tgt_img[0]), n_iter)

            depth_to_show = tgt_depth_var.data[0].cpu()
            depth_to_show[depth_to_show > args.nlabel*args.mindepth] = args.nlabel*args.mindepth
            disp_to_show = (args.nlabel*args.mindepth/depth_to_show)
            disp_to_show[disp_to_show > args.nlabel] = 0
            train_writer.add_image('train Dispnet GT Normalized',
                                   tensor2array(disp_to_show, max_value=args.nlabel, colormap='bone'),
                                   n_iter)
            train_writer.add_image('train Depth GT Normalized',
                                   tensor2array(depth_to_show, max_value=args.nlabel*args.mindepth*0.3),
                                   n_iter)

            for k,scaled_depth in enumerate(depths):
                train_writer.add_image('train Dispnet Output Normalized {}'.format(k),
                                       tensor2array(disps[k].data[0].cpu(), max_value=args.nlabel, colormap='bone'),
                                       n_iter)
                train_writer.add_image('train Depth Output Normalized {}'.format(k),
                                       tensor2array(depths[k].data[0].cpu(), max_value=args.nlabel*args.mindepth*0.3),
                                       n_iter)

        # record loss and EPE
        losses.update(loss.data[0], args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.data[0]])
        if i % args.print_freq == 0:
            print('Train {}: Time {} Data {} Loss {}'.format(i, batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


def validate_with_gt(args, val_loader, depth_net, epoch, output_writers=[]):
    batch_time = AverageMeter()
    error_names = ['abs_rel', 'abs_diff', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    depth_net.eval()

    end = time.time()
    for i, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth, ref_depths) in enumerate(val_loader):
        tgt_img_var = Variable(tgt_img.cuda(), volatile=True)
        ref_imgs_var = [Variable(img.cuda(), volatile=True) for img in ref_imgs]
        ref_poses_var = [Variable(pose.cuda(), volatile=True) for pose in ref_poses]
        intrinsics_var = Variable(intrinsics.cuda(), volatile=True)
        intrinsics_inv_var = Variable(intrinsics_inv.cuda(), volatile=True)
        tgt_depth_var = Variable(tgt_depth.cuda(), volatile=True)
        ref_depths_var = [Variable(dep.cuda()) for dep in ref_depths]

        pose = torch.cat(ref_poses_var,1)

        output_depth = depth_net(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var, ref_depths_var)
        output_disp = args.nlabel*args.mindepth/(output_depth)

        mask = (tgt_depth <= args.nlabel*args.mindepth) & (tgt_depth >= args.mindepth) & (tgt_depth == tgt_depth)

        output = torch.squeeze(output_depth.data.cpu(),1)

        if log_outputs and i % 100 == 0 and i/100 < len(output_writers):
            index = int(i//100)
            if epoch == 0:
                output_writers[index].add_image('val Input', tensor2array(tgt_img[0]), 0)
                depth_to_show = tgt_depth_var.data[0].cpu()
                depth_to_show[depth_to_show > args.nlabel*args.mindepth] = args.nlabel*args.mindepth
                disp_to_show = (args.nlabel*args.mindepth/depth_to_show)
                disp_to_show[disp_to_show > args.nlabel] = 0

                output_writers[index].add_image('val target Disparity Normalized', tensor2array(disp_to_show, max_value=args.nlabel, colormap='bone'), epoch)
                output_writers[index].add_image('val target Depth Normalized', tensor2array(depth_to_show, max_value=args.nlabel*args.mindepth*0.3), epoch)

            output_writers[index].add_image('val Dispnet Output Normalized', tensor2array(output_disp.data[0].cpu(), max_value=args.nlabel, colormap='bone'), epoch)
            output_writers[index].add_image('val Depth Output', tensor2array(output_depth.data[0].cpu(), max_value=args.nlabel*args.mindepth*0.3), epoch)

        errors.update(compute_errors_train(tgt_depth, output, mask))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))

    return errors.avg, error_names


if __name__ == '__main__':
    main()
