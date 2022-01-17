import argparse
import random

import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
import datasets
import models
import utils


def main(config):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  ##### Dataset #####

  

  ##### Model #####
  '''
  ckpt = torch.load(args.load)
  inner_args = utils.config_inner_args(config.get('inner_args'))
  model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
  '''
  if args.test_type=='com':
    ckpt = torch.load(config['load'])
    logging.info(config['load'])
    inner_args = utils.config_inner_args(config.get('inner_args'))
    model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
  elif args.test_type=='mix':
    ckpt = torch.load(config['mixload'])
    logging.info(config['mixload'])
    inner_args = utils.config_inner_args(config.get('inner_args'))
    model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
  elif args.test_type=='hmix':
    ckpt = torch.load(config['hmixload'])
    logging.info(config['hmixload'])
    inner_args = utils.config_inner_args(config.get('inner_args'))
    model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
  elif args.test_type=='re':
    ckpt = torch.load(config['reload'])
    inner_args = utils.config_inner_args(config.get('inner_args'))
    model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
  elif args.test_type=='cls':
    ckpt = torch.load(config['load_encoder'])
    config['classifier_args'] = config.get('classifier_args') or dict()
    config['classifier_args']['n_way'] = config['test']['n_way']
    model = models.load(ckpt,load_clf=False,clf_name=config['classifier'],clf_args=config['classifier_args'])#只读取encoder
    inner_args = utils.config_inner_args(config.get('inner_args'))
  elif args.test_type=='cls_base':
    ckpt = torch.load(config['base_encoder'])
    ckpt['encoder']=config['encoder']
    ckpt['encoder_args'] = config.get('encoder_args') or dict()
    ckpt['encoder_args']['bn_args']['n_episode'] = config['test']['n_episode']

    config['classifier_args'] = config.get('classifier_args') or dict()
    config['classifier_args']['n_way'] = config['test']['n_way']
    model = models.load(ckpt,load_clf=False,clf_name=config['classifier'],clf_args=config['classifier_args'])#只读取encoder
    inner_args = utils.config_inner_args(config.get('inner_args'))
  else:
    raise NotImplementedError
  
  #print(inner_args,inner_args['reset_classifier'])
  print(ckpt['training']['epoch'],ckpt['training']['max_va'])

  if args.efficient:
    model.go_efficient()

  if config.get('_parallel'):
    model = nn.DataParallel(model)



  ##### Evaluation #####

  model.eval()
  aves_va = utils.AverageMeter()
  va_lst = []

  for epoch in range(1, config['epoch'] + 1):
    np.random.seed(epoch)
    dataset = datasets.make(config['dataset'], **config['test'])
    loader = DataLoader(dataset, config['test']['n_episode'],
    collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)

    for data in tqdm(loader, leave=False):
      x_shot, x_query, y_shot, y_query = data
      #print(x_shot.shape)
      x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
      x_query, y_query = x_query.cuda(), y_query.cuda()

      if inner_args['reset_classifier']:
        if config.get('_parallel'):
          model.module.reset_classifier()
        else:
          model.reset_classifier()

      logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
      logits = logits.view(-1, config['test']['n_way'])
      labels = y_query.view(-1)
      
      pred = torch.argmax(logits, dim=1)
      acc = utils.compute_acc(pred, labels)
      aves_va.update(acc, 1)
      va_lst.append(acc)
      #print(acc)

    print('test epoch {}: acc={:.2f} +- {:.2f} (%)'.format(
      epoch, aves_va.item() * 100, 
      utils.mean_confidence_interval(va_lst) * 100))
    logging.info(args)
    logging.info('test epoch {}: acc={:.2f} +- {:.2f} (%)'.format(
      epoch, aves_va.item() * 100, 
      utils.mean_confidence_interval(va_lst) * 100))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--gpu', 
                      help='gpu device number', 
                      type=str, default='0')
  parser.add_argument('--efficient', 
                      help='if True, enables gradient checkpointing',
                      action='store_true')
  parser.add_argument('--seed', 
                      help='auxiliary information',  
                      type=int, default='666')
  parser.add_argument('--test_type', 
                      help='auxiliary information', 
                      type=str, default='com')
  parser.add_argument('--load', 
                      help='auxiliary information', 
                      type=str, default='./save/resnet12_5_5/max-va.pth')
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
  
  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu
  log_name='save/test/test.log'
  logging.basicConfig(format=None,
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=log_name)
  logging.info('')
  
  utils.set_gpu(args.gpu)
  main(config)