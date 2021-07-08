import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

import time
from functools import partial

from data_utils import create_factorized_dataset
from data_utils import make_env
from data_utils import SpriteMaker, StateActionStateDataset
from modules import *
from utils import *
from coda import get_true_flat_mask
from tqdm import tqdm
from sklearn import metrics

from structured_transitions import gen_samples_dynamic, TransitionsData, MixtureOfMaskedNetworks, SimpleStackedAttn, MaskedNetwork
from dynamic_scm_discovery import compute_metrics

def mask_transform(mask):
  batch_size = mask.shape[0]
  transformed_mask = torch.zeros(batch_size, 20, 2)
  m = 0
  while m != batch_size:
    j = 0
    k = 0
    while j != 16:
      i = 0
      while i != 20:
        if mask[m, i, j] == 1:
          transformed_mask[m, k, 0] = 0
          transformed_mask[m, k, 1] = 1
        else:
          transformed_mask[m, k, 0] = 1
          transformed_mask[m, k, 1] = 0
        k += 1
        i += 4
      j += 4
    m += 1
  return transformed_mask

def mask_transform_back(transformed_mask, rel_rec, rel_send):
  batch_size = transformed_mask.shape[0]
  mask = torch.zeros(batch_size, 5, 5)
  m = 0
  while m != batch_size:
    index_1 = torch.argmax(rel_rec, dim = 1)
    index_2 = torch.argmax(rel_send, dim = 1)
    _edge = torch.argmax(transformed_mask[m], dim=1)
    for i in range(20):
      mask[m, index_1[i], index_2[i]] = _edge[i]
    m += 1
  return mask

SEED = 1
np.random.seed(SEED)
BATCH_SIZE = 1000
DATASET_SIZE = 50000
MASK_REGULARIZATION_COEFFICIENT = 0.
WEIGHT_LOSS_COEFFICIENT = 0.
ATTENTION_LOSS_COEFFICIENT = 0.
WEIGHT_DECAY = 0.
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

off_diag = np.ones([5, 5])
off_diag[4] = np.zeros([1, 5])
zeros = np.zeros((20, 1))
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_rec = np.concatenate((rel_rec, zeros), axis=1)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec).to(dev)
rel_send = torch.FloatTensor(rel_send).to(dev)

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)

ground_truth_kwargs = dict(num_sprites=4, seed=SEED, max_episode_length=5000, imagedim=16)
config, env = make_env(**ground_truth_kwargs)
env.action_space.seed(SEED)  # reproduce randomness in action space
sprite_maker = SpriteMaker(partial(make_env, **ground_truth_kwargs))
# data, sprites = create_factorized_dataset(env, DATASET_SIZE)
import pickle
# with open('results/cached_data.pickle', 'wb') as f:
#  pickle.dump((data, sprites), f)
with open('results/cached_data.pickle', 'rb') as f:
  (data, sprites) = pickle.load(f)

s, a, r, s2 = list(zip(*data))
s = np.array(s)
a = np.array(a)
s2 = np.array(s2)
ground_truth_masks = []
for s_, a_ in tqdm(zip(s, a)): # 50000 iters
 mask = get_true_flat_mask(sprite_maker(s_), config, a_)
 mask = mask[:, :-2]
 ground_truth_masks.append(mask)

zeros = np.zeros((DATASET_SIZE, 2))
a = np.concatenate((zeros, np.array(a)), axis=1).reshape((DATASET_SIZE, 1, -1))
sa = np.concatenate((s, a), axis=1)
ground_truth_masks = torch.FloatTensor(np.array(ground_truth_masks))
ground_truth_masks = mask_transform(ground_truth_masks)

samples = (
  torch.FloatTensor(sa),
  torch.FloatTensor(s2), 
  ground_truth_masks
)

dataset = TransitionsData(samples)
tr = TransitionsData(dataset[:int(len(dataset)*5/6)])
te = TransitionsData(dataset[int(len(dataset)*5/6):])
train_loader = torch.utils.data.DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
test_loader  = torch.utils.data.DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)

encoder = MLPEncoder(4, 256, 2, 0.0, True).to(dev)
decoder = SingleStepDecoder(4, 2, 256, 256, 256, 0.0, True).to(dev)
# decoder = MultiStepDecoder(4, 2, 256, 256, 256, 0.0, True)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=0.0005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=200,
                                gamma=0.5)

prior = np.array([0.90, 0.10])  # TODO: hard coded for now
log_prior = torch.FloatTensor(np.log(prior))
log_prior = torch.unsqueeze(log_prior, 0)
log_prior = torch.unsqueeze(log_prior, 0)
log_prior = Variable(log_prior)
log_prior = log_prior.to(dev)
_prior = True

def edge_accuracy(edges, m):
  total = m.size(0) * m.size(1) * 2
  edges_0 = edges[:, :, 0]
  m_0 = m[:, :, 0]
  diff = edges_0 - m_0
  false_positive = (diff == -1.).sum()
  false_negative = (diff == 1.).sum()
  correct = edges.float().data.eq(m.float().data.view_as(edges)).sum()
  return np.float(correct) / total, np.float(false_positive) / (total / 2), np.float(false_negative) / (total / 2)

def train(epoch):
  t = time.time()
  nll_train = []
  acc_train = []
  kl_train = []
  mse_train = []
  accuracy_train = []
  fp_train = []
  fn_train = []
  encoder.train()
  decoder.train()
  for batch_idx, (x, y, m) in enumerate(train_loader):
    x = Variable(x.to(dev))
    y = Variable(y.to(dev))
    m = Variable(m.to(dev))
    optimizer.zero_grad()

    # pred = decoder(x, m, rel_rec, rel_send)
    # prob = my_softmax(m, -1)

    logits = encoder(x, rel_rec, rel_send)
    edges = gumbel_softmax(logits, tau=0.5, hard=True)
    prob = my_softmax(logits, -1)
    pred = decoder(x, edges, rel_rec, rel_send)

    loss_nll = nll_gaussian(pred, y, 5e-5)

    if _prior:
      loss_kl = kl_categorical(prob, log_prior, 5)
    else:
      loss_kl = kl_categorical_uniform(prob, 5, 2)

    loss = loss_nll + 5 * loss_kl
    loss.backward()
    optimizer.step()

    accuracy, false_positive, false_negative = edge_accuracy(edges, m)
    mse_train.append(F.mse_loss(pred, y).item())
    nll_train.append(loss_nll.item())
    kl_train.append(loss_kl.item())
    accuracy_train.append(accuracy)
    fp_train.append(false_positive)
    fn_train.append(false_negative)
  if epoch % 25 == 0:
    edge = edges[0]
    mask = np.zeros((5, 5))
    index_1 = torch.argmax(rel_rec, dim = 1)
    index_2 = torch.argmax(rel_send, dim = 1)
    _edge = torch.argmax(edge, dim=1)
    for i in range(20):
      mask[index_1[i], index_2[i]] = _edge[i]
    print(mask)
    print(y[0])
    print(pred[0])

  print('Epoch: {:04d}'.format(epoch),
        'nll_train: {:.10f}'.format(np.mean(nll_train)),
        'kl_train: {:.10f}'.format(np.mean(kl_train)),
        'mse_train: {:.10f}'.format(np.mean(mse_train)),
        'accuracy_train: {:.10f}'.format(np.mean(accuracy_train)),
        'False_positive: {:.10f}'.format(np.mean(fp_train)),
        'False_negative: {:.10f}'.format(np.mean(fn_train)),
        'time: {:.4f}s'.format(time.time() - t))
  scheduler.step()
  return(np.mean(nll_train))


def test():
    acc_test = []
    nll_test = []
    kl_test = []
    mse_test = []
    tot_mse = 0
    counter = 0

    encoder.eval()
    decoder.eval()
    i=0
    for batch_idx, (data) in enumerate(test_loader):
        data.cuda()
        data = Variable(data, volatile=True)

        data_encoder = data[:, :, :49, :].contiguous().cuda()
        data_decoder = data[:, :, -49:, :].contiguous().cuda()

        logits = encoder(data_encoder, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=0.5, hard=True)
        i += 1
        if i % 100 == 1:
          edge = edges[0]
          mask = np.zeros((5, 5))
          index_1 = torch.argmax(rel_rec, dim = 1)
          index_2 = torch.argmax(rel_send, dim = 1)
          _edge = torch.argmax(edge, dim=1)
          for i in range(20):
            mask[index_1[i], index_2[i]] = _edge[i]
          print(mask)

        

        prob = my_softmax(logits, -1)

        output = decoder(data_decoder, edges, rel_rec, rel_send, 1)

        target = data_decoder[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, 5e-5)
        loss_kl = kl_categorical_uniform(prob, 5, 2)

        mse_test.append(F.mse_loss(output, target).item())
        nll_test.append(loss_nll.item())
        kl_test.append(loss_kl.item())
    print(len(mse_test))
    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(nll_test)),
          'kl_test: {:.10f}'.format(np.mean(kl_test)),
          'mse_test: {:.10f}'.format(np.mean(mse_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)))
for epoch in range(600):
  train(epoch)
# test()