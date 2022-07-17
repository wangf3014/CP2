import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from mmcv.utils import Config

import loader
import builder

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('./log_cp2.txt')
handler.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

parser = argparse.ArgumentParser(description='Copy-Paste Contrastive Pretraining on ImageNet')
parser.add_argument('--config', help='path to configuration file')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num-images', default=1281167, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='total batch size over all GPUs')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--optim', default='sgd', help='optimizer')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--scalar-freq', default=100, type=int,
                    help='metrics writing frequency')
parser.add_argument('--ckpt-freq', default=1, type=int,
                    help='checkpoint saving frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multiple GPUs by default')
parser.set_defaults(multiprocessing_distributed=True)

parser.add_argument('--output-stride', default=16, type=int,
                    help='output stride of encoder')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cfg = Config.fromfile(args.config)
    data_dir = args.data
    args.gpu = gpu

    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    model = builder.CP2_MOCO(cfg)
    print(model)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                      weight_decay=0.01)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("Only sgd and adamw optimizers are supported.")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    traindir = os.path.join(data_dir, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    
    # simply use RandomErasing for Copy-Paste implementation:
    # erase a random block of background image and replace the erased positions by foreground
    augmentation_bg = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=1., scale=(0.5, 0.8), ratio=(0.8, 1.25), value=0.)
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        loader.TwoCropsTransform(transforms.Compose(augmentation)))
    train_dataset_bg = datasets.ImageFolder(
        traindir,
        transforms.Compose(augmentation_bg))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=0)
        train_sampler_bg0 = torch.utils.data.distributed.DistributedSampler(train_dataset_bg, seed=1024)
        train_sampler_bg1 = torch.utils.data.distributed.DistributedSampler(train_dataset_bg, seed=2048)
    else:
        train_sampler = None
        train_sampler_bg0 = None
        train_sampler_bg1 = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    train_loader_bg0 = torch.utils.data.DataLoader(
        train_dataset_bg, batch_size=args.batch_size, shuffle=(train_sampler_bg0 is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler_bg0, drop_last=True)
    train_loader_bg1 = torch.utils.data.DataLoader(
        train_dataset_bg, batch_size=args.batch_size, shuffle=(train_sampler_bg1 is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler_bg1, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            train_sampler_bg0.set_epoch(epoch)
            train_sampler_bg1.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train([train_loader, train_loader_bg0, train_loader_bg1], model, criterion, optimizer, epoch, args)
        if epoch % args.ckpt_freq == args.ckpt_freq - 1:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader_list, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    loss_i = AverageMeter('Loss_ins', ':.4f')
    loss_d = AverageMeter('Loss_den', ':.4f')
    acc_ins = AverageMeter('Acc_ins', ':6.2f')
    acc_seg = AverageMeter('Acc_seg', ':6.2f')
    train_loader, train_loader_bg0, train_loader_bg1 = train_loader_list
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_i, loss_d, acc_ins, acc_seg],
        prefix="Epoch: [{}]".format(epoch))

    # cre_dense = nn.LogSoftmax(dim=1)

    model.train()

    end = time.time()
    for i, ((images, _), (bg0, _), (bg1, _)) in enumerate(zip(train_loader, train_loader_bg0, train_loader_bg1)):
        # data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            bg0 = bg0.cuda(args.gpu, non_blocking=True)
            bg1 = bg1.cuda(args.gpu, non_blocking=True)
            # mask_q = mask_q.cuda(args.gpu, non_blocking=True)
            # mask_k = mask_k.cuda(args.gpu, non_blocking=True)

        mask_q, mask_k = (bg0[:, 0] == 0).float(), (bg1[:, 0] == 0).float()
        image_q = images[0] * mask_q.unsqueeze(1) + bg0
        image_k = images[1] * mask_k.unsqueeze(1) + bg1

        # compute output
        stride = args.output_stride
        output_instance, output_dense, target_instance, target_dense, mask_dense = model(
            image_q, image_k,
            mask_q[:, stride//2::stride, stride//2::stride],
            mask_k[:, stride//2::stride, stride//2::stride])
        loss_instance = criterion(output_instance, target_instance)

        # dense loss of softmax
        output_dense_log = (-1.) * nn.LogSoftmax(dim=1)(output_dense)
        output_dense_log = output_dense_log.reshape(output_dense_log.shape[0], -1)
        loss_dense = torch.mean(
            torch.mul(output_dense_log, target_dense).sum(dim=1) / target_dense.sum(dim=1))

        loss = loss_instance + loss_dense * .2

        acc1, acc5 = accuracy(output_instance, target_instance, topk=(1, 5))
        acc_dense_pos = output_dense.reshape(output_dense.shape[0], -1).argmax(dim=1)
        acc_dense = target_dense[torch.arange(0, target_dense.shape[0]), acc_dense_pos].float().mean() * 100.
        loss_i.update(loss_instance.item(), images[0].size(0))
        loss_d.update(loss_dense.item(), images[0].size(0))
        acc_ins.update(acc1[0], images[0].size(0))
        acc_seg.update(acc_dense.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('    '.join(entries))
        if torch.distributed.get_rank() == 0:
            logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
