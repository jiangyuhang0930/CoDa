import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torch.autograd import Variable
from nri_utils import my_softmax, gumbel_softmax, mask_transform_back

_EPS = 1e-10

class NRI(nn.Module):
    def __init__(self, encoder, decoder):
        super(NRI, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, rel_rec, rel_send, indices, thresh):
        logits = self.encoder(x, rel_rec, rel_send)
        edges, copy = gumbel_softmax(logits, tau=0.5, hard=True, thresh=thresh)
        pred = self.decoder(x, edges, rel_rec, rel_send, indices)
        return pred
    
    def forward_with_mask(self, x, rel_rec, rel_send, indices, thresh=0):
        logits = self.encoder(x, rel_rec, rel_send)
        edges, copy = gumbel_softmax(logits, tau=0.5, hard=True, thresh=thresh)
        pred = self.decoder(x, edges, rel_rec, rel_send, indices)
        return pred, mask_transform_back(copy, 2)

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)
    
    def get_mask(self, inputs, rel_rec, rel_send):
        logits = self.forward(inputs, rel_rec, rel_send)
        edges, copy = gumbel_softmax(logits, tau=0.5, hard=True)
        edges = mask_transform_back(edges)
        return edges

class MultiStepEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MultiStepEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.mlp_rec = MLP(n_out, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid * 2, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, prev, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        prev_hid = self.mlp_rec(prev)
        x = torch.cat((x, prev_hid), dim=2)
        return prev + self.fc_out(x)
    
    def get_mask(self, inputs, rel_rec, rel_send):
        logits = self.forward(inputs, rel_rec, rel_send)
        edges, copy = gumbel_softmax(logits, tau=0.5, hard=True)
        edges = mask_transform_back(edges)
        return edges

class SingleStepActionDecoder(nn.Module):
    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                    do_prob=0., skip_first=False):
        super(SingleStepActionDecoder, self).__init__()
        # state component layers
        self.msg_fc1 = nn.Linear(n_in_node, msg_hid)
        self.msg_fc2 = nn.Linear(msg_hid, msg_out)
        
        # action component layers
        self.msg_fc3 = nn.Linear(n_in_node, msg_hid)
        self.msg_fc4 = nn.Linear(msg_hid, msg_out)

        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type, indices):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        # receivers = torch.matmul(rel_rec, single_timestep_inputs)
        # senders = torch.matmul(rel_send, single_timestep_inputs)
        # pre_msg = torch.cat([senders, receivers], dim=-1)
        pre_msg = torch.matmul(rel_send, single_timestep_inputs)
        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1), 
                    self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1

        pre_msg_state = pre_msg[:, indices[0], :]

        msg_state = F.relu(self.msg_fc1(pre_msg_state))
        msg_state = F.dropout(msg_state, p=self.dropout_prob)
        msg_state = F.relu(self.msg_fc2(msg_state))
        msg_state = msg_state * single_timestep_rel_type[:, indices[0], 1:2]
        all_msgs[:, indices[0], :] += msg_state

        pre_msg_action = pre_msg[:, indices[1], :]
        msg_action = F.relu(self.msg_fc3(pre_msg_action))
        msg_action = F.dropout(msg_action, p=self.dropout_prob)
        msg_action = F.relu(self.msg_fc4(msg_action))
        msg_action = msg_action * single_timestep_rel_type[:, indices[1], 1:2]
        all_msgs[:, indices[1], :] += msg_action
        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(agg_msgs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, indices):
        pred = self.single_step_forward(inputs, rel_rec, rel_send, rel_type, indices)
        # return pred[:, :-1, :-4].contiguous()
        return pred[:, :-1, :].contiguous()



class MultiStepDecoder(nn.Module):
    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                    do_prob=0., skip_first=False):
        super(MultiStepDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_fc3 = nn.ModuleList(
            [nn.Linear(n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc4 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob
    
    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type, indices):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        # receivers = torch.matmul(rel_rec, single_timestep_inputs)
        # senders = torch.matmul(rel_send, single_timestep_inputs)
        # pre_msg = torch.cat([senders, receivers], dim=-1)
        pre_msg = torch.matmul(rel_send, single_timestep_inputs)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1), 
                    self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            pre_msg_state = pre_msg[:, indices[0], :]
            msg_state = F.relu(self.msg_fc1[i](pre_msg_state))
            msg_state = F.dropout(msg_state, p=self.dropout_prob)
            msg_state = F.relu(self.msg_fc2[i](msg_state))
            msg_state = msg_state * single_timestep_rel_type[:, indices[0], i:i + 1]
            all_msgs[:, indices[0], :] += msg_state
            pre_msg_action = pre_msg[:, indices[1], :]
            msg_action = F.relu(self.msg_fc3[i](pre_msg_action))
            msg_action = F.dropout(msg_action, p=self.dropout_prob)
            msg_action = F.relu(self.msg_fc4[i](msg_action))
            msg_action = msg_action * single_timestep_rel_type[:, indices[1], i:i + 1]
            all_msgs[:, indices[1], :] += msg_action
        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()
        # Skip connection
        #   aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(agg_msgs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred
    
    def forward(self, encoder, inputs, rel_rec, rel_send, indices):
        pred_steps = inputs.size(1)
        num_sprites = inputs.size(2) - 1

        output_pred = torch.zeros(inputs.size(0), pred_steps, inputs.size(2), inputs.size(3))
        output_prob = torch.zeros(inputs.size(0), pred_steps, num_sprites * (num_sprites + 1), 2)
        output_edges = torch.zeros(inputs.size(0), pred_steps, num_sprites * (num_sprites + 1), 2)
        
        for i in range(pred_steps):
            single_timestep_inputs = inputs[:, i, :, :]
            logits = encoder(single_timestep_inputs, rel_rec, rel_send)
            edges, copy = gumbel_softmax(logits, tau=0.5, hard=True)
            prob = my_softmax(logits, -1)
            pred = self.single_step_forward(single_timestep_inputs, rel_rec, rel_send, edges, indices)
            output_pred[:, i, :, :] = pred
            output_prob[:, i, :, :] = prob
            output_edges[:, i, :, :] = edges
        return output_pred, output_prob, output_edges
