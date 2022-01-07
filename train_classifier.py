import argparse
import os
import random
from collections import OrderedDict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.optimizers as optimizers

def mixup_batch_data(x_spt, y_spt, x_qry, y_qry, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_spt.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x_spt = lam * x_spt + (1 - lam) * x_spt[index, :]
    y_spt_a, y_spt_b = y_spt, y_spt[index]
    mixed_x_qry = lam * x_qry + (1 - lam) * x_qry[index, :]
    y_qry_a, y_qry_b = y_qry, y_qry[index]
    return mixed_x_spt, y_spt_a, y_spt_b, mixed_x_qry, y_qry_a, y_qry_b, lam


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def main(config):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  ckpt_name = args.name
  if ckpt_name is None:
    ckpt_name = config['encoder']
  if args.tag is not None:
    ckpt_name += '_' + args.tag
  if args.mix:
    ckpt_name += '_mix'
  print('保存的路径名',ckpt_name)

  ckpt_path = os.path.join('./save', ckpt_name)
  utils.ensure_path(ckpt_path)
  utils.set_log_path(ckpt_path)
  writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
  yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

  ##### Dataset #####

  # train
  train_set = datasets.make(config['dataset'], **config['train'])
  train_loader = DataLoader(train_set, config['batch_size'], shuffle=True,
                              num_workers=1, pin_memory=True)
  utils.log('train dataset: {} (x{}), {}'.format(
            train_set[0][0].shape, len(train_set),
            train_set.n_classes))
  #val
  eval_val = False
  if config.get('val'):
    eval_val = True
    val_set = datasets.make(config['dataset'], **config['val'])
    val_loader = DataLoader(train_set, config['batch_size'], shuffle=True,
                                num_workers=1, pin_memory=True)
    utils.log('val dataset: {} (x{}), {}'.format(
            val_set[0][0].shape, len(val_set),
            val_set.n_classes))

  # meta-val
  meta_val = False
  if config.get('meta_val'):
    meta_val = True
    meta_val_set = datasets.make('meta-'+config['dataset'], **config['meta_val'])
    utils.log('meta-val set: {} (x{}), {}'.format(
      val_set[0][0].shape, len(meta_val_set), meta_val_set.n_classes))
    meta_val_loader = DataLoader(
      meta_val_set, config['meta_val']['n_episode'],
      collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)


  ##### Model and Optimizer #####

  inner_args = utils.config_inner_args(config.get('inner_args'))
  if config.get('load'):
    ckpt = torch.load(config['load'])
    config['encoder'] = ckpt['encoder']
    config['encoder_args'] = ckpt['encoder_args']

    config['classifier_args'] = config.get('classifier_args') or dict()
    config['classifier_args']['n_way'] = config['n_class'] 
    model = models.load(ckpt,load_clf=False,clf_name=config['classifier'],clf_args=config['classifier_args'])

    if meta_val:
      config['classifier_args']['n_way'] = config['meta_val']['n_way']
      meta_model=models.make(config['encoder'], config['encoder_args'],
                          config['classifier'], config['classifier_args'])
      meta_model.encoder=model.encoder

    optimizer, lr_scheduler = optimizers.load(ckpt, model.parameters())
    start_epoch = ckpt['training']['epoch'] + 1
    max_va = ckpt['training']['max_va']

  else:
    config['encoder_args'] = config.get('encoder_args') or dict()
    config['classifier_args'] = config.get('classifier_args') or dict()
    config['classifier_args']['n_way'] = config['n_class']
    model = models.make(config['encoder'], config['encoder_args'],
                        config['classifier'], config['classifier_args'])
    optimizer, lr_scheduler = optimizers.make(
      config['optimizer'], model.parameters(), **config['optimizer_args'])
    if meta_val:
      config['classifier_args']['n_way'] = config['meta_val']['n_way']
      meta_model=models.make(config['encoder'], config['encoder_args'],
                          config['classifier'], config['classifier_args'])
      meta_model.encoder=model.encoder

    start_epoch = 1
    max_va = 0.


  if args.efficient:
    model.go_efficient()
  if config.get('_parallel'):
    model = nn.DataParallel(model)
  utils.log('num params: {}'.format(utils.compute_n_params(model)))
  timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

  ##### Training and evaluation #####
    
  # 'tl': meta-train loss
  # 'ta': meta-train accuracy
  # 'vl': meta-val loss
  # 'va': meta-val accuracy
  aves_keys = ['tl', 'ta', 'vl', 'va','mvl','mva']
  trlog = dict()
  for k in aves_keys:
    trlog[k] = []

  for epoch in range(start_epoch, config['epoch'] + 1):
    timer_epoch.start()
    aves = {k: utils.AverageMeter() for k in aves_keys} 

    # meta-train
    model.train()
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    np.random.seed(epoch)

    for data in tqdm(train_loader, desc='train', leave=False):
      x_shot, labels = data
      x_shot, labels = x_shot.cuda(), labels.cuda()
      
      if inner_args['reset_classifier']:
        if config.get('_parallel'):
          model.module.reset_classifier()
        else:
          model.reset_classifier()

      if args.mix:
        x_shot, y_shot_a, y_shot_b,lam = mixup_batch_data(x_shot, labels,alpha=1.0)
        
        logits = model.cls_forward(x_shot, x_query)

        pred = torch.argmax(logits, dim=-1)
        acc1 = utils.compute_acc(pred, y_shot_a.flatten())
        acc2 = utils.compute_acc(pred, y_shot_b.flatten())
        acc=lam*acc1+(1-lam)*acc2

        loss1 = F.cross_entropy(logits,y_shot_a)
        loss2 = F.cross_entropy(logits,y_shot_b)
        loss=lam*loss1+(1-lam)*loss2
      else:
        logits = model.cls_forward(x_shot)
        #logits = logits.flatten(0, 1)
        #labels = y_shot.flatten()
        
        
        pred = torch.argmax(logits, dim=-1)
        acc = utils.compute_acc(pred, labels)
        loss = F.cross_entropy(logits, labels)

      #print(loss.item(),acc)
      aves['tl'].update(loss.item(), 1)
      aves['ta'].update(acc, 1)
      
      optimizer.zero_grad()
      loss.backward()
      for param in optimizer.param_groups[0]['params']:
        nn.utils.clip_grad_value_(param, 10)
      optimizer.step()
    
    if eval_val:
      model.eval()
      np.random.seed(0)
      for data in tqdm(train_loader, desc='val', leave=False):
        x_shot, labels = data
        x_shot, labels = x_shot.cuda(), labels.cuda()
        if inner_args['reset_classifier']:
          if config.get('_parallel'):
            model.module.reset_classifier()
          else:
            model.reset_classifier()
        logits = model.cls_forward(x_shot)
        pred = torch.argmax(logits, dim=-1)
        acc = utils.compute_acc(pred, labels)
        loss = F.cross_entropy(logits, labels)
        aves['vl'].update(loss.item(), 1)
        aves['va'].update(acc, 1)

    # meta-val
    if meta_val:
      meta_model.eval()
      np.random.seed(0)

      for data in tqdm(meta_val_loader, desc='meta-val', leave=False):
        x_shot, x_query, y_shot, y_query = data
        x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
        x_query, y_query = x_query.cuda(), y_query.cuda()

        if inner_args['reset_classifier']:
          if config.get('_parallel'):
            model.module.reset_classifier()
          else:
            model.reset_classifier()

        logits = meta_model(x_shot, x_query, y_shot, inner_args, meta_train=False)
        logits = logits.flatten(0, 1)
        labels = y_query.flatten()
        
        pred = torch.argmax(logits, dim=-1)
        acc = utils.compute_acc(pred, labels)
        loss = F.cross_entropy(logits, labels)
        aves['mvl'].update(loss.item(), 1)
        aves['mva'].update(acc, 1)

    if lr_scheduler is not None:
      lr_scheduler.step()

    for k, avg in aves.items():
      aves[k] = avg.item()
      trlog[k].append(aves[k])

    t_epoch = utils.time_str(timer_epoch.end())
    t_elapsed = utils.time_str(timer_elapsed.end())
    t_estimate = utils.time_str(timer_elapsed.end() / 
      (epoch - start_epoch + 1) * (config['epoch'] - start_epoch + 1))

    # formats output
    log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
      str(epoch), aves['tl'], aves['ta'])
    writer.add_scalars('loss', {'train': aves['tl']}, epoch)
    writer.add_scalars('acc', {'train': aves['ta']}, epoch)

    if eval_val:
      log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
      writer.add_scalars('loss', {'val': aves['vl']}, epoch)
      writer.add_scalars('acc', {'val': aves['va']}, epoch)

    if meta_val:
      log_str += ', meta-val {:.4f}|{:.4f}'.format(aves['mvl'], aves['mva'])
      writer.add_scalars('loss', {'meta-val': aves['mvl']}, epoch)
      writer.add_scalars('acc', {'meta-val': aves['mva']}, epoch)

    log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
    utils.log(log_str)

    # saves model and meta-data
    if config.get('_parallel'):
      model_ = model.module
    else:
      model_ = model

    training = {
      'epoch': epoch,
      'max_va': max(max_va, aves['va'],aves['mva']),

      'optimizer': config['optimizer'],
      'optimizer_args': config['optimizer_args'],
      'optimizer_state_dict': optimizer.state_dict(),
      'lr_scheduler_state_dict': lr_scheduler.state_dict() 
        if lr_scheduler is not None else None,
    }
    ckpt = {
      'file': __file__,
      'config': config,

      'encoder': config['encoder'],
      'encoder_args': config['encoder_args'],
      'encoder_state_dict': model_.encoder.state_dict(),
      'training': training,
    }

    # 'epoch-last.pth': saved at the latest epoch
    # 'max-va.pth': saved when validation accuracy is at its maximum
    torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))
    torch.save(trlog, os.path.join(ckpt_path, 'trlog.pth'))

    if aves['va'] > max_va:
      max_va = aves['va']
      torch.save(ckpt, os.path.join(ckpt_path, 'max-va.pth'))

    writer.flush()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--mix', 
                       action='store_true')
  parser.add_argument('--name', 
                      help='model name', 
                      type=str, default=None)
  parser.add_argument('--tag', 
                      help='auxiliary information', 
                      type=str, default=None)
  parser.add_argument('--gpu', 
                      help='gpu device number', 
                      type=str, default='0')
  parser.add_argument('--efficient', 
                      help='if True, enables gradient checkpointing',
                      action='store_true')
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

  utils.set_gpu(args.gpu)
  main(config)