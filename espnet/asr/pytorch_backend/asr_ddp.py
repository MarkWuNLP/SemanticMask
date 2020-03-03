#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training/decoding definition for the speech recognition task."""

import copy
import json
import logging
import math
import os
import sys
import random
import time
import shutil

from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
from chainer.training.updater import StandardUpdater
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.e2e_asr import pad_list
from chainer.datasets import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset

import matplotlib
matplotlib.use('Agg')

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest

class RedirectStdout:
    def __init__(self):
        self.content = ''
        self.savedStdout = sys.stdout
        self.fileObj = None
    def write(self, outStr):
        self.content += outStr

    def toFile(self, filename):
        self.fileObj = open(filename, 'a+', 1)
        sys.stdout = self.fileObj  
  
    def restore(self):
        self.content = ''
        self.fileObj.close()

class AverageMeter(object):
    """Compute and storesthe average and current value"""
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
        fmtstr = '{name} {val' + self.fmt +'} ({avg' + self.fmt +'})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.warning('\t'.join(entries))
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches)+']'


class TrainingConverter(object):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self,device, subsampling_factor=1, dtype=torch.float32):
        """Construct a MaskConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype
        self.device = device


    def __call__(self, batch):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        # batch should be located in list

        xs, ys = batch
        ys = list(ys)
        if len(xs) != len(ys):
            print("error uttr")
            print(xs[0])
            pass

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])
        ilens = torch.from_numpy(ilens).to(self.device)
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(self.device, dtype=self.dtype)

        ys_pad = pad_list([torch.from_numpy(y[2]) for y in ys], self.ignore_id).long().to(self.device)

        return xs_pad, ilens, ys_pad

class CustomConverter(object):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, device, subsampling_factor=1, dtype=torch.float32, task='asr'):
        """Construct a CustomConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype
        self.device = device
        self.task = task

    def __call__(self, batch):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        # batch should be located in list
        xs, ys = batch
        ys = list(ys)

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        # currently only support real number
        if xs[0].dtype.kind == 'c':
            xs_pad_real = pad_list(
                [torch.from_numpy(x.real).float() for x in xs], 0).to(self.dtype).cuda(self.device, non_blocking=True)
            xs_pad_imag = pad_list(
                [torch.from_numpy(x.imag).float() for x in xs], 0).to(self.dtype).cuda(self.device, non_blocking=True)
            # Note(kamo):
            # {'real': ..., 'imag': ...} will be changed to ComplexTensor in E2E.
            # Don't create ComplexTensor and give it E2E here
            # because torch.nn.DataParellel can't handle it.
            xs_pad = {'real': xs_pad_real, 'imag': xs_pad_imag}
        else:
            xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(self.dtype).cuda(self.device, non_blocking=True)

        ilens = torch.from_numpy(ilens).cuda(self.device, non_blocking=True)
        # NOTE: this is for multi-task learning (e.g., speech translation)
        ys_pad = pad_list([torch.from_numpy(np.array(y[0]) if isinstance(y, tuple) else y).long()
                          for y in ys], self.ignore_id).cuda(self.device, non_blocking=True)
        if self.task == "asr":
            return xs_pad, ilens, ys_pad
        elif self.task == "st":
            ys_pad_asr = pad_list([torch.from_numpy(np.array(y[1])).long()
                                  for y in ys], 0).cuda(self.device, non_blocking=True)
            return xs_pad, ilens, ys_pad, ys_pad_asr
        else:
            raise ValueError('Support only asr and st data')

def dist_train(gpu, args):
    """Initialize torch.distributed."""
    args.gpu = gpu
    args.rank = gpu
    logging.warning("Hi Master, I am gpu {0}".format(gpu))
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    redirObj = RedirectStdout()
    sys.stdout = redirObj
    redirObj.toFile(args.outdir + 'log')
    best_acc = 0
    if args.gpu is not None:
        print('Use GPU: {} for training'.format(args.gpu))

    init_method = "tcp://localhost:{port}".format(port=args.port)

    torch.distributed.init_process_group(
        backend='nccl', world_size=args.ngpu, rank=args.gpu,
        init_method=init_method)
    torch.cuda.set_device(args.gpu)

    converter = TrainingConverter(args.gpu)

    validconverter = CustomConverter(args.gpu)
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][-1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][-1])

    print('#input dims : ' + str(idim))
    print('#output dims: ' + str(odim))
    print('initialize model on gpu: {}'.format(args.gpu))
    if args.enc_init is not None or args.dec_init is not None:
        model = load_trained_modules(idim, odim, args)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(idim, odim, args)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        print('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)
    logging.warning("Init")
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=False,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout)
    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True},word_mask_ratio=args.maskratio  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )
    logging.warning("start loading data. ")
    train_dataset = TransformDataset(train, lambda data: converter(load_tr(data)))
    valid_dataset = TransformDataset(valid, lambda data: validconverter(load_cv(data)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False)

    start_epoch = 0
    latest = [int(f.split('.')[-1]) for f in os.listdir(args.outdir) if 'snapshot.ep' in f]
    if not args.resume and len(latest):
        latest_snapshot = os.path.join(args.outdir, 'snapshot.ep.{}'.format(str(max(latest))))
        args.resume = latest_snapshot

    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        start_epoch = 0
    logging.warning("start training ")
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_epoch(train_loader, model, optimizer, epoch, args)
        acc = validate(valid_loader, model, args)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        if args.rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model_module,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=os.path.join(args.outdir, 'snapshot.ep.{}'.format(epoch)))
    


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model.acc.best')

def train_epoch(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.5f')
    acc_meter = AverageMeter('Acc', ':6.5f')
    lr_meter = AverageMeter("Lr", ":6.5f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, acc_meter,lr_meter],
        prefix="Epoch: [{}] GPU: [{}]".format(epoch, args.gpu))
    model.train()
    start = time.time()
    for i, batch in enumerate(train_loader):
        x = tuple(arr[0] for arr in batch)
        loss, acc = model(*x, return_acc=True)
        losses.update(loss.item())
        acc_meter.update(acc)
        lr_meter.update(optimizer.get_rate())
        loss = loss / args.accum_grad
        #optimizer.zero_grad()
        loss.backward()
        if i % args.accum_grad == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip)
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                optimizer.step()
            optimizer.zero_grad()

        if i % 5000 == 0 and args.rank == 0:
            torch.save({
                'epoch': epoch,
                'arch': args.model_module,
                'state_dict': model.state_dict(),
            }, os.path.join(args.outdir, 'snapshot.iter.{1}.epoch{0}'.format(epoch, i)))
        if i % args.report_interval_iters == 0 and args.rank == 0:
            batch_time.update(time.time() - start)
            start = time.time()
            progress.display(i)

def validate(valid_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    acc_meter = AverageMeter('Acc', ':6.6f')
    progress = ProgressMeter(
        len(valid_loader),
        [batch_time, losses, acc_meter],
        prefix="Test: GPU: [{}]".format(args.gpu))
    model.eval()
    with torch.no_grad():
        start = time.time()
        for i, batch in enumerate(valid_loader):
            batch = tuple(arr[0] for arr in batch)
            loss, acc = model(*batch, return_acc=True)
            losses.update(loss.item())
            acc_meter.update(acc)
            batch_time.update(time.time()-start)
    progress.display(len(valid_loader))
    return acc_meter.avg
            
        
def train(args):
    """Main training program."""
    if args.ngpu == 0:
        logging.warning('distributed training only supported for GPU training')

    args.ngpu = torch.cuda.device_count()
    if args.ngpu == 0:
        logging.warning('no gpu detected')
        exit(0)
    args.port = random.randint(10000, 20000)
    mp.spawn(dist_train, nprocs=args.ngpu, args=(args,))

