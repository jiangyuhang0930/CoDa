import numpy as np
import torch.nn.functional as F
np.random.seed(2)
import torch
from structured_transitions import gen_samples_dynamic, TransitionsData, MixtureOfMaskedNetworks
from nri_modules import *
from nri_utils import *

BATCH_SIZE = 1000
SPLITS = [4,3,2] # this is the factorization
WEIGHT_DECAY = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

num_components = 9
rel_rec, rel_send, indices = setup_aux_data(num_components)
rel_rec, rel_send = rel_rec.to(DEVICE), rel_send.to(DEVICE)

def main():
    global_interactions, _, samples = gen_samples_dynamic(num_seqs=1500, seq_len=10, 
                                                            splits=SPLITS, epsilon=1.5) # 15000 samples
    print('Total global interactions: {}/{}'.format(global_interactions, len(samples[0])))

    zeros = torch.zeros(15000, 1, 1)
    samples_ = (torch.cat((samples[0].reshape((15000, 9, 1)), zeros), dim=1), samples[1].reshape((15000, 9, 1)), samples[2])
    dataset = TransitionsData(samples_)

    tr = TransitionsData(dataset[:int(len(dataset)*5/6)])
    te = TransitionsData(dataset[int(len(dataset)*5/6):])

    train_loader = torch.utils.data.DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)

    encoder_kwargs = dict(n_in=1, n_hid=256, n_out=2, do_prob=0.0, factor=True)
    decoder_kwargs = dict(n_in_node=1, edge_types=2, msg_hid=256, msg_out=256, n_hid=256, do_prob=0.0, skip_first=True)

    encoder = MLPEncoder(**encoder_kwargs).to(DEVICE)
    decoder = SingleStepActionDecoder(**decoder_kwargs).to(DEVICE)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001, weight_decay=WEIGHT_DECAY)

    # prior = np.array([0.5, 0.5])  # TODO: hard coded for now
    # log_prior = torch.FloatTensor(np.log(prior))
    # log_prior = torch.unsqueeze(log_prior, 0)
    # log_prior = torch.unsqueeze(log_prior, 0)
    # log_prior = Variable(log_prior)
    # log_prior = log_prior.to(DEVICE)
    # _prior = True


    for i in range(200):
        nll_train = []
        mse_train = []
        sparsity_train = []
        for _, (x, y, m) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            m = m.to(DEVICE)
            logits = encoder(x, rel_rec, rel_send)
            edges, _ = gumbel_softmax(logits, tau=0.5, hard=True)
            # prob = my_softmax(logits, -1)
            pred = decoder(x, edges, rel_rec, rel_send, indices)
            loss_nll = nll_gaussian(pred, y, 5e-5)
            loss_sparsity = sparsity_loss(edges)
            # if _prior:
            #     loss_kl = kl_categorical(prob, log_prior, 3)
            # else:
            #     loss_kl = kl_categorical_uniform(prob, 3, 2)
            loss = loss_nll + loss_sparsity
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            nll_train.append(loss_nll.item())
            sparsity_train.append(loss_sparsity.item())
            mse_train.append(F.mse_loss(pred, y).item())

        if i % 10 == 0:
            print(mask_transform_back(edges, 9)[0, :-1, :-1])
            print('Epoch: {:03d}'.format(i),
                'nll_train: {:.5f}'.format(np.mean(nll_train)),
                'kl_train: {:.5f}'.format(np.mean(sparsity_train)),
                'mse_train: {:.6f}'.format(np.mean(mse_train)))
        
    



if __name__ == "__main__":
    main()