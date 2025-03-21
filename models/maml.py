from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.checkpoint as cp
import numpy as np
from . import encoders
from . import classifiers
from .modules import get_child_dict, Module, BatchNorm2d


def make(enc_name, enc_args, clf_name, clf_args):
  """
  Initializes a random meta model.

  Args:
    enc_name (str): name of the encoder (e.g., 'resnet12').
    enc_args (dict): arguments for the encoder.
    clf_name (str): name of the classifier (e.g., 'meta-nn').
    clf_args (dict): arguments for the classifier.

  Returns:
    model (MAML): a meta classifier with a random encoder.
  """
  enc = encoders.make(enc_name, **enc_args)
  clf_args['in_dim'] = enc.get_out_dim()
  clf = classifiers.make(clf_name, **clf_args)
  model = MAML(enc, clf)
  return model


def load(ckpt, load_clf=False, clf_name=None, clf_args=None):
  """
  Initializes a meta model with a pre-trained encoder.

  Args:
    ckpt (dict): a checkpoint from which a pre-trained encoder is restored.
    load_clf (bool, optional): if True, loads a pre-trained classifier.
      Default: False (in which case the classifier is randomly initialized)
    clf_name (str, optional): name of the classifier (e.g., 'meta-nn')
    clf_args (dict, optional): arguments for the classifier.
    (The last two arguments are ignored if load_clf=True.)

  Returns:
    model (MAML): a meta model with a pre-trained encoder.
  """
  enc = encoders.load(ckpt)
  if load_clf:
    clf = classifiers.load(ckpt)
  else:
    if clf_name is None and clf_args is None:
      clf = classifiers.make(ckpt['classifier'], **ckpt['classifier_args'])
    else:
      clf_args['in_dim'] = enc.get_out_dim()
      clf = classifiers.make(clf_name, **clf_args)
  model = MAML(enc, clf)
  return model


class MAML(Module):
  def __init__(self, encoder, classifier):
    super(MAML, self).__init__()
    self.encoder = encoder
    self.classifier = classifier

  def cls_forward(self,x):
    params = OrderedDict(self.named_parameters())
    feat = self.encoder(x, get_child_dict(params, 'encoder'))
    logits = self.classifier(feat, get_child_dict(params, 'classifier'))
    return logits

  def reset_classifier(self):
    self.classifier.reset_parameters()

  def _inner_forward(self, x, params, episode):
    """ Forward pass for the inner loop. """
    feat = self.encoder(x, get_child_dict(params, 'encoder'), episode)
    logits = self.classifier(feat, get_child_dict(params, 'classifier'))
    return logits

  def _inner_iter(self, x, y, params, mom_buffer, episode, inner_args, detach):
    """ 
    Performs one inner-loop iteration of MAML including the forward and 
    backward passes and the parameter update.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): the model parameters BEFORE the update.
      mom_buffer (dict): the momentum buffer BEFORE the update.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      detach (bool): if True, detachs the graph for the current iteration.

    Returns:
      updated_params (dict): the model parameters AFTER the update.
      mom_buffer (dict): the momentum buffer AFTER the update.
    """
    with torch.enable_grad():
      # forward pass
      logits = self._inner_forward(x, params, episode)
      loss = F.cross_entropy(logits, y)
      # backward pass
      grads = autograd.grad(loss, params.values(), 
        create_graph=(not detach and not inner_args['first_order']),
        only_inputs=True, allow_unused=True)
      # parameter update
      updated_params = OrderedDict()
      for (name, param), grad in zip(params.items(), grads):
        if grad is None:
          updated_param = param
        else:
          if inner_args['weight_decay'] > 0:
            grad = grad + inner_args['weight_decay'] * param
          if inner_args['momentum'] > 0:
            grad = grad + inner_args['momentum'] * mom_buffer[name]
            mom_buffer[name] = grad
          if 'encoder' in name:
            lr = inner_args['encoder_lr']
          elif 'classifier' in name:
            lr = inner_args['classifier_lr']
          else:
            raise ValueError('invalid parameter name')
          updated_param = param - lr * grad
        if detach:#测试的话detach
          updated_param = updated_param.detach().requires_grad_(True)
        updated_params[name] = updated_param

    return updated_params, mom_buffer

  def _adapt(self, x, y, params, episode, inner_args, meta_train):
    """
    Performs inner-loop adaptation in MAML.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
        (T: transforms, C: channels, H: height, W: width)
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): a dictionary of parameters at meta-initialization.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      params (dict): model paramters AFTER inner-loop adaptation.
    """
    assert x.dim() == 4 and y.dim() == 1
    assert x.size(0) == y.size(0)

    # Initializes a dictionary of momentum buffer for gradient descent in the 
    # inner loop. It has the same set of keys as the parameter dictionary.
    mom_buffer = OrderedDict()
    if inner_args['momentum'] > 0:
      for name, param in params.items():
        mom_buffer[name] = torch.zeros_like(param)
    params_keys = tuple(params.keys())
    mom_buffer_keys = tuple(mom_buffer.keys())

    for m in self.modules():
      if isinstance(m, BatchNorm2d) and m.is_episodic():
        m.reset_episodic_running_stats(episode)

    def _inner_iter_cp(episode, *state):
      """ 
      Performs one inner-loop iteration when checkpointing is enabled. 
      The code is executed twice:
        - 1st time with torch.no_grad() for creating checkpoints.
        - 2nd time with torch.enable_grad() for computing gradients.
      """
      params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
      mom_buffer = OrderedDict(
        zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))

      detach = not torch.is_grad_enabled()  # detach graph in the first pass
      self.is_first_pass(detach)
      params, mom_buffer = self._inner_iter(
        x, y, params, mom_buffer, int(episode), inner_args, detach)
      state = tuple(t if t.requires_grad else t.clone().requires_grad_(True)
        for t in tuple(params.values()) + tuple(mom_buffer.values()))
      return state

    for step in range(inner_args['n_step']):
      if self.efficient:  # checkpointing
        state = tuple(params.values()) + tuple(mom_buffer.values())
        state = cp.checkpoint(_inner_iter_cp, torch.as_tensor(episode), *state)
        params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
        mom_buffer = OrderedDict(
          zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))
      else:
        params, mom_buffer = self._inner_iter(
          x, y, params, mom_buffer, episode, inner_args, not meta_train)
        
    return params

  def forward(self, x_shot, x_query, y_shot, inner_args, meta_train):
    """
    Args:
      x_shot (float tensor, [n_episode, n_way * n_shot, C, H, W]): support sets.
      x_query (float tensor, [n_episode, n_way * n_query, C, H, W]): query sets.
        (T: transforms, C: channels, H: height, W: width)
      y_shot (int tensor, [n_episode, n_way * n_shot]): support set labels.
      inner_args (dict, optional): inner-loop hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      logits (float tensor, [n_episode, n_way * n_shot, n_way]): predicted logits.
    """
    assert self.encoder is not None
    assert self.classifier is not None
    assert x_shot.dim() == 5 and x_query.dim() == 5
    assert x_shot.size(0) == x_query.size(0)

    # a dictionary of parameters that will be updated in the inner loop
    #初始参数
    params = OrderedDict(self.named_parameters())
    for name in list(params.keys()):
      if not params[name].requires_grad or \
        any(s in name for s in inner_args['frozen'] + ['temp']):
        params.pop(name)

    logits = []
    for ep in range(x_shot.size(0)):
      # inner-loop training
      self.train()
      if not meta_train:#测试的话执行这个
        for m in self.modules():
          if isinstance(m, BatchNorm2d) and not m.is_episodic():
            m.eval()
      #内层循环更新后的参数
      updated_params = self._adapt(
        x_shot[ep], y_shot[ep], params, ep, inner_args, meta_train)

      # inner-loop validation
      with torch.set_grad_enabled(meta_train):
        self.eval()
        logits_ep = self._inner_forward(x_query[ep], updated_params, ep)
      logits.append(logits_ep)

    self.train(meta_train)
    logits = torch.stack(logits)
    return logits


  def hmix_inner_forward(self, x, params, ep,mix_ep):
    """ Forward pass for the inner loop. """
    feat1 = self.encoder(x[ep], get_child_dict(params, 'encoder'), ep)
    feat2 = self.encoder(x[mix_ep], get_child_dict(params, 'encoder'), ep)
    lam = np.random.beta(1.0, 1.0)
    feat=lam*feat1+(1-lam)*feat2
    logits = self.classifier(feat, get_child_dict(params, 'classifier'))
    return logits,lam

  def hmix_inner_iter(self, x, y, params, mom_buffer, ep, mix_ep, inner_args, detach):
    """ 
    Performs one inner-loop iteration of MAML including the forward and 
    backward passes and the parameter update.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): the model parameters BEFORE the update.
      mom_buffer (dict): the momentum buffer BEFORE the update.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      detach (bool): if True, detachs the graph for the current iteration.

    Returns:
      updated_params (dict): the model parameters AFTER the update.
      mom_buffer (dict): the momentum buffer AFTER the update.
    """
    with torch.enable_grad():
      logits,lam = self.hmix_inner_forward(x, params, ep,mix_ep)
      loss1 = F.cross_entropy(logits, y[ep])
      loss2 = F.cross_entropy(logits, y[mix_ep])
      loss=lam*loss1+(1-lam)*loss2
      # backward pass
      grads = autograd.grad(loss, params.values(), 
        create_graph=(not detach and not inner_args['first_order']),
        only_inputs=True, allow_unused=True)
      # parameter update
      updated_params = OrderedDict()
      for (name, param), grad in zip(params.items(), grads):
        if grad is None:
          updated_param = param
        else:
          if inner_args['weight_decay'] > 0:
            grad = grad + inner_args['weight_decay'] * param
          if inner_args['momentum'] > 0:
            grad = grad + inner_args['momentum'] * mom_buffer[name]
            mom_buffer[name] = grad
          if 'encoder' in name:
            lr = inner_args['encoder_lr']
          elif 'classifier' in name:
            lr = inner_args['classifier_lr']
          else:
            raise ValueError('invalid parameter name')
          updated_param = param - lr * grad
        if detach:#测试的话detach
          updated_param = updated_param.detach().requires_grad_(True)
        updated_params[name] = updated_param

    return updated_params, mom_buffer

  def hmix_adapt(self, x, y, params, ep,ep_mix,inner_args, meta_train):
    """
    Performs inner-loop adaptation in MAML.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
        (T: transforms, C: channels, H: height, W: width)
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): a dictionary of parameters at meta-initialization.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      params (dict): model paramters AFTER inner-loop adaptation.
    """
    assert x.dim() == 5 and y.dim() == 2
    assert x.size(1) == y.size(1)

    # Initializes a dictionary of momentum buffer for gradient descent in the 
    # inner loop. It has the same set of keys as the parameter dictionary.
    mom_buffer = OrderedDict()
    if inner_args['momentum'] > 0:
      for name, param in params.items():
        mom_buffer[name] = torch.zeros_like(param)
    params_keys = tuple(params.keys())
    mom_buffer_keys = tuple(mom_buffer.keys())

    for m in self.modules():
      if isinstance(m, BatchNorm2d) and m.is_episodic():
        m.reset_episodic_running_stats(ep)

    def _inner_iter_cp(episode, *state):
      """ 
      Performs one inner-loop iteration when checkpointing is enabled. 
      The code is executed twice:
        - 1st time with torch.no_grad() for creating checkpoints.
        - 2nd time with torch.enable_grad() for computing gradients.
      """
      params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
      mom_buffer = OrderedDict(
        zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))

      detach = not torch.is_grad_enabled()  # detach graph in the first pass
      self.is_first_pass(detach)
      params, mom_buffer = self._inner_iter(
        x, y, params, mom_buffer, int(episode), inner_args, detach)
      state = tuple(t if t.requires_grad else t.clone().requires_grad_(True)
        for t in tuple(params.values()) + tuple(mom_buffer.values()))
      return state

    for step in range(inner_args['n_step']):
      if self.efficient:  # checkpointing
        state = tuple(params.values()) + tuple(mom_buffer.values())
        state = cp.checkpoint(_inner_iter_cp, torch.as_tensor(ep), *state)
        params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
        mom_buffer = OrderedDict(
          zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))
      else:
        params, mom_buffer = self.hmix_inner_iter(
          x, y, params, mom_buffer, ep, ep_mix,inner_args, not meta_train)
        
    return params

  def hmix_forward(self, x_shot, x_query, y_shot, y_query,inner_args, meta_train,retrain=False):
    """
    Args:
      x_shot (float tensor, [n_episode, n_way * n_shot, C, H, W]): support sets.
      x_query (float tensor, [n_episode, n_way * n_query, C, H, W]): query sets.
        (T: transforms, C: channels, H: height, W: width)
      y_shot (int tensor, [n_episode, n_way * n_shot]): support set labels.
      inner_args (dict, optional): inner-loop hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      logits (float tensor, [n_episode, n_way * n_shot, n_way]): predicted logits.
    """
    assert self.encoder is not None
    assert self.classifier is not None
    assert x_shot.dim() == 5 and x_query.dim() == 5
    assert x_shot.size(0) == x_query.size(0)

    # a dictionary of parameters that will be updated in the inner loop
    #初始参数
    params = OrderedDict(self.named_parameters())
    for name in list(params.keys()):
      if not params[name].requires_grad or \
        any(s in name for s in inner_args['frozen'] + ['temp']):
        params.pop(name)

    logits = []
    batch_size = x_shot.size()[0]
    index = torch.randperm(batch_size).cuda()
    if retrain:
      batch_size=1
      index=torch.randperm(batch_size).cuda()
      index[0]=1
      #print(batch_size,index)
    for ep in range(batch_size):
      # inner-loop training
      self.train()
      if not meta_train:#测试的话执行这个
        for m in self.modules():
          if isinstance(m, BatchNorm2d) and not m.is_episodic():
            m.eval()
      #内层循环更新后的参数
      updated_params = self.hmix_adapt(
        x_shot, y_shot,params,ep,index[ep], inner_args, meta_train)

      # inner-loop validation
      with torch.set_grad_enabled(meta_train):
        self.eval()
        logits_ep,lam= self.hmix_inner_forward(x_query, updated_params, ep,index[ep])
      logits.append(logits_ep)

    self.train(meta_train)
    logits = torch.stack(logits)

    return logits,lam,y_query[index]

  def _mixinner_iter(self, x, y_a,y_b,lam, params, mom_buffer, episode, inner_args, detach):
    """ 
    Performs one inner-loop iteration of MAML including the forward and 
    backward passes and the parameter update.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): the model parameters BEFORE the update.
      mom_buffer (dict): the momentum buffer BEFORE the update.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      detach (bool): if True, detachs the graph for the current iteration.

    Returns:
      updated_params (dict): the model parameters AFTER the update.
      mom_buffer (dict): the momentum buffer AFTER the update.
    """
    with torch.enable_grad():
      # forward pass
      logits = self._inner_forward(x, params, episode)
      loss1 = F.cross_entropy(logits, y_a)
      loss2 = F.cross_entropy(logits, y_b)
      loss=lam*loss1+(1-lam)*loss2

      # backward pass
      grads = autograd.grad(loss, params.values(), 
        create_graph=(not detach and not inner_args['first_order']),
        only_inputs=True, allow_unused=True)
      # parameter update
      updated_params = OrderedDict()
      for (name, param), grad in zip(params.items(), grads):
        if grad is None:
          updated_param = param
        else:
          if inner_args['weight_decay'] > 0:
            grad = grad + inner_args['weight_decay'] * param
          if inner_args['momentum'] > 0:
            grad = grad + inner_args['momentum'] * mom_buffer[name]
            mom_buffer[name] = grad
          if 'encoder' in name:
            lr = inner_args['encoder_lr']
          elif 'classifier' in name:
            lr = inner_args['classifier_lr']
          else:
            raise ValueError('invalid parameter name')
          updated_param = param - lr * grad
        if detach:
          updated_param = updated_param.detach().requires_grad_(True)
        updated_params[name] = updated_param

    return updated_params, mom_buffer

  def _mix_adapt(self, x, y_a,y_b,lam, params, episode, inner_args, meta_train):
    """
    Performs inner-loop adaptation in MAML.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
        (T: transforms, C: channels, H: height, W: width)
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): a dictionary of parameters at meta-initialization.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      params (dict): model paramters AFTER inner-loop adaptation.
    """
    assert x.dim() == 4 and y_a.dim() == 1
    assert x.size(0) == y_a.size(0)

    # Initializes a dictionary of momentum buffer for gradient descent in the 
    # inner loop. It has the same set of keys as the parameter dictionary.
    mom_buffer = OrderedDict()
    if inner_args['momentum'] > 0:
      for name, param in params.items():
        mom_buffer[name] = torch.zeros_like(param)
    params_keys = tuple(params.keys())
    mom_buffer_keys = tuple(mom_buffer.keys())

    for m in self.modules():
      if isinstance(m, BatchNorm2d) and m.is_episodic():
        m.reset_episodic_running_stats(episode)

    def _inner_iter_cp(episode, *state):
      """ 
      Performs one inner-loop iteration when checkpointing is enabled. 
      The code is executed twice:
        - 1st time with torch.no_grad() for creating checkpoints.
        - 2nd time with torch.enable_grad() for computing gradients.
      """
      params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
      mom_buffer = OrderedDict(
        zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))

      detach = not torch.is_grad_enabled()  # detach graph in the first pass
      self.is_first_pass(detach)
      params, mom_buffer = self._mixinner_iter(
        x, y_a,y_b,lam, params, mom_buffer, int(episode), inner_args, detach)
      state = tuple(t if t.requires_grad else t.clone().requires_grad_(True)
        for t in tuple(params.values()) + tuple(mom_buffer.values()))
      return state

    for step in range(inner_args['n_step']):
      if self.efficient:  # checkpointing
        state = tuple(params.values()) + tuple(mom_buffer.values())
        state = cp.checkpoint(_inner_iter_cp, torch.as_tensor(episode), *state)
        params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
        mom_buffer = OrderedDict(
          zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))
      else:
        params, mom_buffer = self._mixinner_iter(
          x, y_a,y_b,lam, params, mom_buffer, episode, inner_args, not meta_train)
        
    return params
  
  def mix_forward(self, x_shot, x_query, y_shot_a,y_shot_b, lam,inner_args, meta_train):
    """
    Args:
      x_shot (float tensor, [n_episode, n_way * n_shot, C, H, W]): support sets.
      x_query (float tensor, [n_episode, n_way * n_query, C, H, W]): query sets.
        (T: transforms, C: channels, H: height, W: width)
      y_shot (int tensor, [n_episode, n_way * n_shot]): support set labels.
      inner_args (dict, optional): inner-loop hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      logits (float tensor, [n_episode, n_way * n_shot, n_way]): predicted logits.
    """
    assert self.encoder is not None
    assert self.classifier is not None
    assert x_shot.dim() == 5 and x_query.dim() == 5
    assert x_shot.size(0) == x_query.size(0)

    # a dictionary of parameters that will be updated in the inner loop
    params = OrderedDict(self.named_parameters())
    for name in list(params.keys()):
      if not params[name].requires_grad or \
        any(s in name for s in inner_args['frozen'] + ['temp']):
        params.pop(name)

    logits = []
    for ep in range(x_shot.size(0)):
      # inner-loop training
      self.train()
      if not meta_train:
        for m in self.modules():
          if isinstance(m, BatchNorm2d) and not m.is_episodic():
            m.eval()
      updated_params = self._mix_adapt(
        x_shot[ep], y_shot_a[ep],y_shot_b[ep],lam, params, ep, inner_args, meta_train)
      # inner-loop validation
      with torch.set_grad_enabled(meta_train):
        self.eval()
        logits_ep = self._inner_forward(x_query[ep], updated_params, ep)
      logits.append(logits_ep)

    self.train(meta_train)
    logits = torch.stack(logits)
    return logits




  def get_params_change(self, x_shot, y_shot, inner_args):
    """
    Args:
      x_shot (float tensor, [n_episode, n_way * n_shot, C, H, W]): support sets.
      x_query (float tensor, [n_episode, n_way * n_query, C, H, W]): query sets.
        (T: transforms, C: channels, H: height, W: width)
      y_shot (int tensor, [n_episode, n_way * n_shot]): support set labels.
      inner_args (dict, optional): inner-loop hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      logits (float tensor, [n_episode, n_way * n_shot, n_way]): predicted logits.
    """
    assert self.encoder is not None
    assert self.classifier is not None
    assert x_shot.dim() == 5

    # a dictionary of parameters that will be updated in the inner loop
    params = OrderedDict(self.named_parameters())
    for name in list(params.keys()):
      if not params[name].requires_grad or \
        any(s in name for s in inner_args['frozen'] + ['temp']):
        params.pop(name)
    

    mom_buffer = OrderedDict()
    if inner_args['momentum'] > 0:
      for name, param in params.items():
        mom_buffer[name] = torch.zeros_like(param)
    
    
    for step in range(inner_args['n_step']):
      
      ##########################
      logits = self._inner_forward(x_shot[0], params, 0)
      #print(logits.shape,y_shot[0].shape)
      loss = F.cross_entropy(logits, y_shot[0])
      
      # backward pass
      grads = autograd.grad(loss, params.values(), 
        create_graph=False,
        only_inputs=True, allow_unused=True)
      #手动更新参数
      tmp_result=[]

      updated_params = OrderedDict()
      for (name, param), grad in zip(params.items(), grads):
        if grad is None:
          updated_param = param
        else:
          if inner_args['weight_decay'] > 0:
            grad = grad + inner_args['weight_decay'] * param
          if inner_args['momentum'] > 0:
            grad = grad + inner_args['momentum'] * mom_buffer[name]
            mom_buffer[name] = grad
          if 'encoder' in name:
            lr = inner_args['encoder_lr']
          elif 'classifier' in name:
            lr = inner_args['classifier_lr']
          else:
            raise ValueError('invalid parameter name')
          updated_param = param - lr * grad
          tmp_result.append(torch.flatten(grad*pow(inner_args['alpha'],step)).detach())
        updated_param = updated_param.detach().requires_grad_(True)
        updated_params[name] = updated_param
      params=updated_params
        #tmp_result=torch.hstack(tmp_result)
      with torch.no_grad():
        tmp_result=torch.hstack(tmp_result)
        #print(tmp_result[:10])
        if step==0:
          result = tmp_result
        else:
          result += tmp_result
      #print(result[:10],'***')
    return result.detach()
  
  def sim_emb(self, x, inner_args):
    """ Forward pass for the inner loop. """
    params = OrderedDict(self.named_parameters())
    for name in list(params.keys()):
      if not params[name].requires_grad or \
        any(s in name for s in inner_args['frozen'] + ['temp']):
        params.pop(name)
    with torch.no_grad():
      feat = self.encoder(x[0], get_child_dict(params, 'encoder'), 0)
    #print(feat.shape)
    return feat





