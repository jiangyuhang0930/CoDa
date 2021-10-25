import numpy as np
import torch
import os
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_random_mask(n):
    one_side = torch.randint(0, 2, (n, 6, 1))
    other_side = torch.abs(1 - one_side)
    output = torch.cat((one_side, other_side), 2)
    return output

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def sparsity_loss(edges):
    all_zero = torch.ones_like(edges[:, :, 0])
    loss = (edges[:, :, 0] - all_zero) / (edges.size(0) * edges.size(1))
    return -1 * torch.sum(loss)

def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return F.sigmoid(y / tau)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, thresh=0):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    tau: non-negative scalar temperature
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        y_soft_copy = y_soft.detach().clone()
        y_soft_copy[:,:,0] -= thresh
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        _, k_copy = y_soft_copy.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        y_hard_copy = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
            y_hard_copy = y_hard_copy.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y_hard_copy = y_hard_copy.zero_().scatter_(-1, k_copy.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y, y_hard_copy


def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                            dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def get_minimum_distance(data):
    data = data[:, :, :, :2].transpose(1, 2)
    data_norm = (data ** 2).sum(-1, keepdim=True)
    dist = data_norm + \
        data_norm.transpose(2, 3) - \
        2 * torch.matmul(data, data.transpose(2, 3))
    min_dist, _ = dist.min(1)
    return min_dist.view(min_dist.size(0), -1)


def get_buckets(dist, num_buckets):
    dist = dist.cpu().data.numpy()

    min_dist = np.min(dist)
    max_dist = np.max(dist)
    bucket_size = (max_dist - min_dist) / num_buckets
    thresholds = bucket_size * np.arange(num_buckets)

    bucket_idx = []
    for i in range(num_buckets):
        if i < num_buckets - 1:
            idx = np.where(np.all(np.vstack((dist > thresholds[i],
                                            dist <= thresholds[i + 1])), 0))[0]
        else:
            idx = np.where(dist > thresholds[i])[0]
        bucket_idx.append(idx)

    return bucket_idx, thresholds


def get_correct_per_bucket(bucket_idx, pred, target):
    pred = pred.cpu().numpy()[:, 0]
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def get_correct_per_bucket_(bucket_idx, pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                        eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def mask_transform(mask, num_sprites):
    batch_size = mask.shape[0]
    transformed_mask = torch.zeros(batch_size, num_sprites * (num_sprites + 1), 2)
    m = 0
    while m != batch_size:
        j = 0
        k = 0
        while j != 4 * num_sprites:
            i = 0
            while i != 4 * (num_sprites + 1):
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
    
def mask_viz(transformed_mask, rel_rec, rel_send, num_sprites):
    batch_size = transformed_mask.shape[0]
    mask = torch.zeros(batch_size, num_sprites + 1, num_sprites + 1)
    m = 0
    while m != batch_size:
        index_1 = torch.argmax(rel_rec, dim = 1)
        index_2 = torch.argmax(rel_send, dim = 1)
        _edge = torch.argmax(transformed_mask[m], dim=1)
        for i in range(num_sprites * (num_sprites + 1)):
            mask[m, index_1[i], index_2[i]] = _edge[i]
        m += 1
    return mask
    
def mask_transform_back(mask, num_sprites):
    transformed_mask = mask[:, :, 1]
    transformed_mask = transformed_mask.reshape(transformed_mask.size(0), num_sprites, num_sprites + 1).transpose(-2, -1)
    transformed_mask = transformed_mask.repeat_interleave(4, 2)
    transformed_mask = transformed_mask.repeat_interleave(4, 1)
    return transformed_mask[:, :-2, :]
    
def edge_accuracy(edges, m, num_sprites):
    if edges.shape != m.shape:
	    edges = mask_transform_fetch(edges, num_sprites)
    total = m.size(0) * m.size(1) * m.size(2)
    diff = edges.to("cuda") - m
    total_pos = (edges == 1.).sum()
    total_neg = (edges == 0.).sum()
    false_positive = np.float((diff == 1.).sum())
    true_positive = np.float(total_pos - false_positive)
    false_negative = np.float((diff == -1.).sum())
    true_negative = np.float(total_neg - false_negative)
    correct = np.float(true_positive + true_negative)
    
    return correct / total, false_positive / (false_positive + true_negative), true_positive / (false_negative + true_positive)
    
def compute_roc(dataloader, encoder, rel_rec, rel_send, threshold_list, num_sprites):
        fpr_list = []
        tpr_list = []
        for thresh in threshold_list:
            fpr = []
            tpr = []
            for batch_idx, (x, y,  m) in enumerate(dataloader):
                x = Variable(x.to('cuda'))
                m = Variable(m.to('cuda'))
                logits = encoder(x, rel_rec, rel_send)
                _, copy = gumbel_softmax(logits, tau=0.5, hard=True, thresh=thresh)
                _, fpr_b, tpr_b = edge_accuracy(copy, m, num_sprites)
                fpr.append(fpr_b)
                tpr.append(tpr_b)
            fpr_list.append(np.mean(fpr))
            tpr_list.append(np.mean(tpr))
        auc = metrics.auc(fpr_list, tpr_list)
        return fpr_list, tpr_list, auc

def compute_roc_tj(dataloader, encoder, rel_rec, rel_send, threshold_list, num_sprites):
        fpr_list = []
        tpr_list = []
        for thresh in threshold_list:
            fpr = []
            tpr = []
            for batch_idx, (x, m) in enumerate(dataloader):
                x = Variable(x.to('cuda'))
                m = Variable(m.to('cuda'))
                x = x[:, :-1, :, :]
                for i in range(x.size(1)):
                    logits = encoder(x[:, i, :, :], rel_rec, rel_send)
                    _, copy = gumbel_softmax(logits, tau=0.5, hard=True, thresh=thresh)
                    _, fpr_b, tpr_b = edge_accuracy(copy, m[:, i, :, :], num_sprites)
                    fpr.append(fpr_b)
                    tpr.append(tpr_b)
            fpr_list.append(np.mean(fpr))
            tpr_list.append(np.mean(tpr))
        auc = metrics.auc(fpr_list, tpr_list)
        return fpr_list, tpr_list, auc
        
def plot_roc(fpr, tpr, auc, results_dir):

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC For Ground Truth Sparsity Recovery')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, 'roc.pdf'))
    
def setup_aux_data(num_sprites):
	off_diag = np.ones([num_sprites + 1, num_sprites + 1])
	off_diag[num_sprites] = np.zeros([1, num_sprites + 1])
	zeros = np.zeros(((num_sprites + 1) * num_sprites, 1))
	rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
	rel_rec = np.concatenate((rel_rec, zeros), axis=1)
	rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
	rel_rec = torch.tensor(rel_rec, dtype = torch.float32, requires_grad = False)
	rel_send = torch.tensor(rel_send, dtype = torch.float32, requires_grad = False)
	
	indices = [[], []]
	for i in range((num_sprites + 1) * num_sprites):
	    if (i - num_sprites) % (num_sprites + 1) == 0:
	        indices[1].append(i)
	    else:
	        indices[0].append(i)
	
	return rel_rec, rel_send, indices

def mask_transform_fetch(masks, num_component):
    batch = masks.size(0)
    transformed = torch.argmax(masks, dim=2)
    transformed = transformed.reshape(batch, num_component, num_component + 1)
    transformed = transformed.transpose(1, 2)
    zeros = torch.zeros(batch, num_component + 1, 1).to(DEVICE)
    transformed = torch.cat((transformed, zeros), 2)
    return transformed

def heuristic_transform_fetch(masks):
    masks = masks[:, :, :-1]
    masks = masks.transpose(1, 2).flatten(1)
    masks = masks.unsqueeze(1)
    masks = torch.transpose(masks, 1, 2)
    masks_1 = torch.abs(1. - masks)
    masks = torch.cat((masks_1, masks), 2)
    return masks
