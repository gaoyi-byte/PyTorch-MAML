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
import sys

import datasets
import models
import utils
import utils.optimizers as optimizers
import pandas as pd

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


def mixup_task_data(x_spt, x_test_spt,x_qry,x_test_qry,use_cuda=True, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    x_spt = x_spt.reshape(5, -1, 3, 84, 84)  # (n,k,3,84,84)
    x_qry = x_qry.reshape(5, -1, 3, 84, 84)  # (n,k,3,84,84)
    x_test_spt = x_test_spt.reshape(5, -1, 3, 84, 84)  # (n,k,3,84,84)
    
    x_spt = lam * x_spt + (1 - lam) * x_test_spt
    x_qry = lam * x_qry + (1 - lam) * x_test_qry

    x_spt=x_spt.reshape(1,-1,3,84,84)
    x_qry=x_qry.reshape(1,-1,3,84,84)

    return x_spt,x_qry, lam

def mixup_in_data(x_spt, y_spt,use_cuda=True, alpha=1.0):  # 对任务内的数据进行混合，对测试数据创造更多的数据。

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    x_spt = x_spt.reshape(5, 5, 3, 84, 84)  # (n,k,3,84,84)
    y_spt = y_spt.reshape(5, -1)  # (n,k)
    #print(x_spt.shape,y_spt.shape)
    x_spt=x_spt.repeat(1,3,1,1,1)
    y_spt = y_spt.repeat(1, 3)
    #print(x_spt.shape,y_spt.shape)

    if use_cuda:
        index = torch.randperm(15).cuda()
    else:
        index = torch.randperm(15)

    x_qry = lam * x_spt + (1 - lam) * x_spt[:,index, :]

    return x_qry,y_spt.reshape(1,-1)

def params_change_gc(db_test, db,maml,config): 

  inner_args = utils.config_inner_args(config.get('inner_args'))
  inner_args['n_step']=args.update
  inner_args['alpha']=args.alpha
  maml.eval()
  #计算单个test任务和训练任务的相似度
  result = torch.zeros((len(db_test.dataset), len(db.dataset)))
  for step, (x_spt, x_qry,y_spt,  y_qry) in enumerate(tqdm(db_test, desc='test-params', leave=False)):
    x_spt, y_spt = x_spt.cuda(), y_spt.cuda()
        #x_qry, y_qry = x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
    params_test = maml.get_params_change(x_spt, y_spt,inner_args)
      
    for step1, (x_spt, x_qry,y_spt,  y_qry) in enumerate(db):
      x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()
      grad = maml.get_params_change(x_spt, y_spt,inner_args)
      result[step][step1]=torch.cosine_similarity(grad, params_test,dim=0)
      # print(result)
  
  task = torch.argsort(result,dim=1,descending=True).cpu().numpy()  # 从小到大排
  csv_name = f"save/retrain/{args.seed}_{config['train']['n_batch']}_{config['test']['n_batch']}/{config['encoder']}/{args.train_type}{args.sim_type}/task_{args.update}_{args.alpha}.csv"
  pd.DataFrame(task).to_csv(csv_name)
  #sys.exit()
  return pd.DataFrame(task)

def params_change(db_test, db,maml,config): 

  inner_args = utils.config_inner_args(config.get('inner_args'))
  inner_args['n_step']=args.update
  inner_args['alpha']=args.alpha
  maml.eval()
  if args.sim_type == 'gc':
    # 直接计算
    params = []
    for step, (x_spt, x_qry,y_spt,  y_qry) in enumerate(tqdm(db, desc='train-params', leave=False)):
      x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()
        #grad = maml.get_params_change(x_spt, y_spt, x_qry, y_qry)
      grad = maml.get_params_change(x_spt, y_spt,inner_args)
      params.append(grad)
      

    params = torch.stack(params).detach()
        # pd.DataFrame(params.cpu().numpy()).to_csv(f"params_qry_{args.test_update}.csv")
  elif args.sim_type == 'sim_cos':
    params = []
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(tqdm(db, desc='train-params', leave=False)):
      x_spt = x_spt.cuda()
      #print(x_spt.shape)
      params.append(maml.sim_emb(x_spt,inner_args))

    params = torch.stack(params).detach()
    params = params.view(len(params), config['train']['n_way'],
                            config['train']['n_shot'], -1)  # (num,n,k,-1)
      # print(params)
    params = params.mean(2)  # (num,n,-1)
    params = params.unsqueeze(1)  # (num,1,n,d)

  print('计算相似度')
  task = np.zeros((len(db_test.dataset), len(params)))
  for step, (x_spt, x_qry,y_spt,  y_qry) in enumerate(tqdm(db_test, desc='test-params', leave=False)):
    x_spt, y_spt = x_spt.cuda(), y_spt.cuda()
        #x_qry, y_qry = x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
    if args.sim_type == 'gc':
      params_test = maml.get_params_change(x_spt, y_spt,inner_args)
      result = torch.cosine_similarity(params, params_test, -1)
      
    elif args.sim_type == 'sim_cos':
      params_test = maml.sim_emb(x_spt,inner_args).view(
                config['train']['n_way'], config['train']['n_shot'], -1).mean(1).unsqueeze(1)  # (n,1,d)
      result1 = torch.cosine_similarity(params, params_test, -1)  # (N,N)
      result1, _ = result1.max(-1)
      result = result1.sum(-1).squeeze(0)
    

        # print(result)
    tmp = torch.argsort(result).cpu().numpy()  # 从小到大排
    tmp = tmp[::-1]  # 从大到小排
    task[step] = tmp
  if args.sim_type == 'gc':
    csv_name = f"save/retrain/{args.seed}_{config['train']['n_batch']}_{config['test']['n_batch']}/{config['encoder']}/{args.train_type}{args.sim_type}/task_{args.update}_{args.alpha}.csv"
  else:
    csv_name = f"save/retrain/{args.seed}_{config['train']['n_batch']}_{config['test']['n_batch']}/{config['encoder']}/{args.train_type}{args.sim_type}/task.csv"
  pd.DataFrame(task).to_csv(csv_name)
  #sys.exit()
  return pd.DataFrame(task)

def main(config):

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  #torch.backends.cudnn.deterministic = True
  #torch.backends.cudnn.benchmark = False

 

  ##### Dataset #####
  path=f"save/retrain/{args.seed}_{config['train']['n_batch']}_{config['test']['n_batch']}/{config['encoder']}/{args.train_type}{args.sim_type}"
  if not os.path.exists(path):
    os.makedirs(path)
  utils.set_log_path(path,retrain=True,args=args)
  
  
  # meta-test
  test_set = datasets.make(config['dataset'], **config['test'])
  #utils.log('meta-test set: {} (x{}), {}'.format(test_set[0][0].shape, len(test_set), test_set.n_classes))
  test_loader = DataLoader(test_set, config['test']['n_episode'],
    collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)

  # meta-train
  train_set = datasets.make(config['dataset'], **config['train'])
  #utils.log('meta-train set: {} (x{}), {}'.format(train_set[0][0].shape, len(train_set), train_set.n_classes))
  train_loader = DataLoader(
    train_set, config['train']['n_episode'],
    collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)

  


  
  ##### Model and Optimizer #####
  #调用模型
  if args.train_type=='mix':
    print('调用模型：',config['mixload'])
    ckpt = torch.load(config['mixload'])
  elif args.train_type=='hmix':
    print('调用模型：',config['hmixload'])
    ckpt = torch.load(config['hmixload'])
  else:
    print('调用模型：',config['load'])
    ckpt = torch.load(config['load'])

  inner_args = utils.config_inner_args(config.get('inner_args'))
  model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
  optimizer, lr_scheduler = optimizers.make(
      config['optimizer'], model.parameters(), **config['optimizer_args'])
  #optimizer, lr_scheduler = optimizers.load(ckpt, model.parameters())

 

  if args.efficient:
    model.go_efficient()
  if config.get('_parallel'):
    model = nn.DataParallel(model)

  #utils.log('num params: {}'.format(utils.compute_n_params(model)))
  #utils.log('')
  
  timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

  if args.sim_type != 'random':
    if args.sim_type == 'gc':
      path=f"save/retrain/{args.seed}_{config['train']['n_batch']}_{config['test']['n_batch']}/{config['encoder']}/{args.train_type}{args.sim_type}/task_{args.update}_{args.alpha}.csv"
    elif args.sim_type=='sim_cos':
      path=f"save/retrain/{args.seed}_{config['train']['n_batch']}_{config['test']['n_batch']}/{config['encoder']}/{args.train_type}{args.sim_type}/task.csv"
    print(path)
    if os.path.exists(path):
      print('已有',path)
      task = pd.read_csv(path, index_col=0)
    else:
      if config['encoder']=='resnet12' and args.sim_type=='gc':
        task= params_change_gc(test_loader, train_loader,model,config)
      else:
        task= params_change(test_loader, train_loader,model,config)
  #sys.exit()
  

  aves_va_b = utils.AverageMeter()
  va_lst_b = []
  aves_va_a = utils.AverageMeter()
  va_lst_a = []
  #tqdm(test_loader, desc='meta-test', leave=False)
  loss_task=[]
  suc_task=[]
  
  for (id,data) in enumerate(test_loader):
    #加载模型
    model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
    optimizer, lr_scheduler = optimizers.make(
      config['optimizer'], model.parameters(), **config['optimizer_args'])

    model.eval()
    if inner_args['reset_classifier']:
      if config.get('_parallel'):
        model.module.reset_classifier()
      else:
        model.reset_classifier()
    
    #retrain前测试
    x_shot, x_query, y_shot, y_query = data
    x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
    x_query, y_query = x_query.cuda(), y_query.cuda()

    
    inner_args['n_step']=args.test_step
    logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
    logits = logits.view(-1, config['test']['n_way'])
    labels = y_query.view(-1)
      
    pred = torch.argmax(logits, dim=1)
    acc = utils.compute_acc(pred, labels)
    aves_va_b.update(acc, 1)
    va_lst_b.append(acc)
    

    #retrain
    if args.sim_type == 'random':
      train_id = np.random.choice(list(range(1000)), args.num, False)
    elif args.task_type == '+-':
        train_id = list(task.iloc[id, :args.num//2]) + \
            list(task.iloc[id, -args.num//2:])
    elif args.task_type == '+':
        train_id = list(task.iloc[id, :args.num])
    elif args.task_type == '-':
        train_id = list(task.iloc[id, -args.num:])
    else:
        raise NotImplementedError
    #print(args.sim_type,args.task_type,train_id)
    np.random.shuffle(train_id)
    #
    model.train()
    inner_args['n_step']=args.train_step
    if args.retrain_type=='com':
      for i,task1 in enumerate(train_id) : 
        x_shot_train, x_query_train, y_shot_train, y_query_train = train_loader.dataset[int(task1)]
        #print(x_shot_train.shape)
        x_shot_train, y_shot_train = x_shot_train.unsqueeze(0).cuda(), y_shot_train.unsqueeze(0).cuda()
        x_query_train, y_query_train = x_query_train.unsqueeze(0).cuda(), y_query_train.unsqueeze(0).cuda()

        if inner_args['reset_classifier']:
          if config.get('_parallel'):
            model.module.reset_classifier()
          else:
            model.reset_classifier()

        logits = model(x_shot_train, x_query_train, y_shot_train,inner_args, meta_train=True)
        logits = logits.flatten(0, 1)
        labels = y_query.flatten()
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        for param in optimizer.param_groups[0]['params']:
          nn.utils.clip_grad_value_(param, 10)
        optimizer.step()

    elif args.retrain_type=='mix':#训练任务和测试任务的mix
      x_tmp_query,y_tmp_query=mixup_in_data( x_shot, y_shot )#生成大量的测试集数据。
      for i,task1 in enumerate(train_id) : 
        if inner_args['reset_classifier']:
          if config.get('_parallel'):
            model.module.reset_classifier()
          else:
            model.reset_classifier()
        x_shot_train, x_query_train, y_shot_train, y_query_train = train_loader.dataset[int(task1)]
        x_shot_train, y_shot_train = x_shot_train.unsqueeze(0).cuda(), y_shot_train.unsqueeze(0).cuda()
        x_query_train, y_query_train = x_query_train.unsqueeze(0).cuda(), y_query_train.unsqueeze(0).cuda()

        
        x_mix_spt, x_mix_qry,lam=mixup_task_data(x_shot_train, x_shot,x_query_train,x_tmp_query)#测试任务和训练任务来mix

        logits = model.mix_forward(x_mix_spt,x_mix_qry, y_shot_train,  y_shot, lam,inner_args, meta_train=True)
        logits = logits.flatten(0, 1)
        labels = y_query.flatten()
        loss1 = F.cross_entropy(logits, y_query_train.flatten())
        loss2 = F.cross_entropy(logits, y_tmp_query.flatten())
        loss=lam*loss1+(1-lam)*loss2
        optimizer.zero_grad()
        loss.backward()
        for param in optimizer.param_groups[0]['params']:
          nn.utils.clip_grad_value_(param, 10)
        optimizer.step()

    elif args.retrain_type=='hmix':#训练任务和测试任务的mix
      x_tmp_query,y_tmp_query=mixup_in_data( x_shot, y_shot )#生成大量的测试集数据。
      x_tmp_query=x_tmp_query.reshape(1,-1,x_tmp_query.size(2),x_tmp_query.size(3),x_tmp_query.size(4))
      y_tmp_query=y_tmp_query.reshape(1,-1)

      for i,task1 in enumerate(train_id) : 
        if inner_args['reset_classifier']:
          if config.get('_parallel'):
            model.module.reset_classifier()
          else:
            model.reset_classifier()
        x_shot_train, x_query_train, y_shot_train, y_query_train = train_loader.dataset[int(task1)]
        x_shot_train, y_shot_train = x_shot_train.unsqueeze(0).cuda(), y_shot_train.unsqueeze(0).cuda()
        x_query_train, y_query_train = x_query_train.unsqueeze(0).cuda(), y_query_train.unsqueeze(0).cuda()
        #x_mix_spt, x_mix_qry,lam=mixup_task_data(x_shot_train, x_shot,x_query_train,x_tmp_query)#测试任务和训练任务来mix
        x_spt=torch.cat([x_shot_train, x_shot], dim=0)
        x_qry=torch.cat([x_query_train, x_tmp_query], dim=0)
        y_spt=torch.cat([y_shot,y_shot_train], dim=0)
        y_qry=torch.cat([y_query_train,y_tmp_query], dim=0)

        logits,lam,_ = model.hmix_forward(x_spt, x_qry, y_spt,y_qry,inner_args, meta_train=True,retrain=True)

        logits = logits.flatten(0, 1)
        loss1 = F.cross_entropy(logits, y_query_train.flatten())
        loss2 = F.cross_entropy(logits, y_tmp_query.flatten())
        loss=lam*loss1+(1-lam)*loss2
        optimizer.zero_grad()
        loss.backward()
        for param in optimizer.param_groups[0]['params']:
          nn.utils.clip_grad_value_(param, 10)
        optimizer.step()

    #retrain完再次测试：
    model.eval()
    if inner_args['reset_classifier']:
      if config.get('_parallel'):
        model.module.reset_classifier()
      else:
        model.reset_classifier()

    inner_args['n_step']=args.test_step
    logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
    logits = logits.view(-1, config['test']['n_way'])
    labels = y_query.view(-1)
      
    pred = torch.argmax(logits, dim=1)
    acc = utils.compute_acc(pred, labels)
    aves_va_a.update(acc, 1)
    va_lst_a.append(acc)
    if va_lst_a[id]>va_lst_b[id]:
      print(id,va_lst_b[id],va_lst_a[id],'*****')
      suc_task.append(id)
    else:
      print(id,va_lst_b[id],va_lst_a[id])
      loss_task.append(id)
  
  utils.log('#########################################################')
  utils.log(f"model:{args.train_type},retrain:{args.retrain_type},task:{args.task_type},num:{args.num},候选：{config['train']['n_batch']}")
  if args.sim_type=='gc':
    utils.log(f'update:{args.update},retrain_update:{args.train_step},test_update:{args.test_step},alpha:{args.alpha}')
    
  utils.log('retrain 前的准确率为：{:.2f} +- {:.2f} (%),retrain后的准确率为：{:.2f} +- {:.2f} (%),整体变化：{}'.format(
      aves_va_b.item() * 100, 
      utils.mean_confidence_interval(va_lst_b) * 100,
      aves_va_a.item() * 100, 
      utils.mean_confidence_interval(va_lst_a) * 100,
      aves_va_a.item() * 100-aves_va_b.item() * 100))

  utils.log(f'成功的任务数量：{len(suc_task)},成功的任务是：{suc_task}')
  utils.log(f'失败的任务数量：{len(loss_task)},成功的任务是：{loss_task}\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--name', 
                      help='model name', 
                      type=str, default=None)
  parser.add_argument('--sim_type', 
                      help='auxiliary information', 
                      type=str, default='sim_cos')
  parser.add_argument('--alpha', 
                      help='auxiliary information', 
                      type=float, default='0.1')
  parser.add_argument('--num', 
                      help='auxiliary information', 
                      type=int, default='20')
  parser.add_argument('--seed', 
                      help='auxiliary information', 
                      type=int, default=666)
  parser.add_argument('--retrain_type', 
                      help='auxiliary information', 
                      type=str, default='com')
  parser.add_argument('--train_type', 
                      help='auxiliary information', 
                      type=str, default='mix')
  parser.add_argument('--update', 
                      help='auxiliary information', 
                      type=int, default=5)
  parser.add_argument('--train_step', 
                      help='auxiliary information', 
                      type=int, default=5)
  parser.add_argument('--test_step', 
                      help='auxiliary information', 
                      type=int, default=10)
  parser.add_argument('--task_type', 
                      help='auxiliary information', 
                      type=str, default='+-')
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