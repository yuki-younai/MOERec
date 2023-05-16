import torch.nn as nn
import torch as th
import pdb
from tqdm import tqdm
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn as dglnn
import numpy as np
import torch

class BaseLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = args.k
        self.sigma = args.sigma
        self.gamma = args.gamma

    def similarity_matrix(self, X, sigma = 1.0, gamma = 2.0):
        dists = th.cdist(X, X)
        sims = th.exp(-dists / (sigma * dists.mean(dim = -1).mean(dim = -1).reshape(-1, 1, 1)))
        return sims

    def submodular_selection_feature(self, nodes):
        device = nodes.mailbox['m'].device
        feature = nodes.mailbox['m']
        sims = self.similarity_matrix(feature, self.sigma, self.gamma)

        batch_num, neighbor_num, feature_size = feature.shape
        nodes_selected = []
        cache = th.zeros((batch_num, 1, neighbor_num), device = device)

        for i in range(self.k):
            gain = th.sum(th.maximum(sims, cache) - cache, dim = -1)

            selected = th.argmax(gain, dim = 1)
            cache = th.maximum(sims[th.arange(batch_num, device = device), selected].unsqueeze(1), cache)

            nodes_selected.append(selected)

        return th.stack(nodes_selected).t()


    def sub_reduction(self, nodes):
        # -1 indicate user-> node, which does not include category information
        mail = nodes.mailbox['m']
        batch_size, neighbor_size, feature_size = mail.shape

        if (-1 in nodes.mailbox['c']) or nodes.mailbox['m'].shape[1] <= self.k:
            mail = mail.sum(dim = 1)
        else:
            neighbors = self.submodular_selection_feature(nodes)
            mail = mail[:,0:20,:]
            mail = mail.sum(dim = 1)
        return {'h': mail}

    def category_aggregation(self, edges):
        return {'c': edges.src['category'], 'm': edges.src['h']}

    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _ , dst = etype
            feat_src = h[src]
            feat_dst = h[dst]

            degs = graph.out_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(self.category_aggregation, self.sub_reduction, etype = etype)

            rst = graph.nodes[dst].data['h']
            degs = graph.in_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
            return rst

class DGRecLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = args.k
        self.sigma = args.sigma
        self.gamma = args.gamma

    def similarity_matrix(self, X, sigma = 1.0, gamma = 2.0):
        dists = th.cdist(X, X)
        sims = th.exp(-dists / (sigma * dists.mean(dim = -1).mean(dim = -1).reshape(-1, 1, 1)))
        return sims

    def submodular_selection_feature(self, nodes):
        device = nodes.mailbox['m'].device
        feature = nodes.mailbox['m']
        sims = self.similarity_matrix(feature, self.sigma, self.gamma)

        batch_num, neighbor_num, feature_size = feature.shape
        nodes_selected = []
        cache = th.zeros((batch_num, 1, neighbor_num), device = device)

        for i in range(self.k):
            gain = th.sum(th.maximum(sims, cache) - cache, dim = -1)

            selected = th.argmax(gain, dim = 1)
            cache = th.maximum(sims[th.arange(batch_num, device = device), selected].unsqueeze(1), cache)

            nodes_selected.append(selected)

        return th.stack(nodes_selected).t()


    def sub_reduction(self, nodes):
        # -1 indicate user-> node, which does not include category information
        mail = nodes.mailbox['m']
        batch_size, neighbor_size, feature_size = mail.shape

        if (-1 in nodes.mailbox['c']) or nodes.mailbox['m'].shape[1] <= self.k:
            mail = mail.sum(dim = 1)
        else:
            neighbors = self.submodular_selection_feature(nodes)
            mail = mail[th.arange(batch_size, dtype = th.long, device = mail.device).unsqueeze(-1), neighbors]
            mail = mail.sum(dim = 1)
        return {'h': mail}

    def category_aggregation(self, edges):
        return {'c': edges.src['category'], 'm': edges.src['h']}

    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _ , dst = etype
            feat_src = h[src]
            feat_dst = h[dst]

            degs = graph.out_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(self.category_aggregation, self.sub_reduction, etype = etype)

            rst = graph.nodes[dst].data['h']
            degs = graph.in_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
            return rst
class MOERecLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = args.k
        self.sigma = args.sigma
        self.gamma = args.gamma
        self.poly_attn=PolyAttention(args)

    def similarity_matrix(self, X, sigma = 1.0, gamma = 2.0):
        dists = th.cdist(X, X)
        sims = th.exp(-dists / (sigma * dists.mean(dim = -1).mean(dim = -1).reshape(-1, 1, 1)))
        return sims

    def submodular_selection_feature(self, nodes):
        device = nodes.mailbox['m'].device
        feature = nodes.mailbox['m']
        sims = self.similarity_matrix(feature, self.sigma, self.gamma)

        batch_num, neighbor_num, feature_size = feature.shape
        nodes_selected = []
        cache = th.zeros((batch_num, 1, neighbor_num), device = device)

        for i in range(self.k):
            gain = th.sum(th.maximum(sims, cache) - cache, dim = -1)

            selected = th.argmax(gain, dim = 1)
            cache = th.maximum(sims[th.arange(batch_num, device = device), selected].unsqueeze(1), cache)

            nodes_selected.append(selected)

        return th.stack(nodes_selected).t()


    def sub_reduction_item_user(self, nodes):
        # -1 indicate user-> node, which does not include category information
        mail = nodes.mailbox['m']
        batch_size, neighbor_size, feature_size = mail.shape

        if (-1 in nodes.mailbox['c']) or nodes.mailbox['m'].shape[1] <= self.k:
            
            muti_int=self.poly_attn(embeddings=mail, attn_mask=0, bias=None)
            mail=mail.sum(dim=1)
        else:
            neighbors = self.submodular_selection_feature(nodes)
            mail = mail[th.arange(batch_size, dtype = th.long, device = mail.device).unsqueeze(-1), neighbors]
            muti_int=self.poly_attn(embeddings=mail, attn_mask=0, bias=None)#12 20 32
            mail = mail.sum(dim = 1)
      
        return {'h': mail,'i':muti_int}
    
    def sub_reduction_user_item(self, nodes):
        # -1 indicate user-> node, which does not include category information
        mail = nodes.mailbox['m']
        batch_size, neighbor_size, feature_size = mail.shape

        if (-1 in nodes.mailbox['c']) or nodes.mailbox['m'].shape[1] <= self.k:
            mail = mail.sum(dim = 1)
        else:
            neighbors = self.submodular_selection_feature(nodes)
            mail = mail[th.arange(batch_size, dtype = th.long, device = mail.device).unsqueeze(-1), neighbors]
            mail = mail.sum(dim = 1)
        return {'h': mail}
    
    def category_aggregation(self, edges):
        return {'c': edges.src['category'], 'm': edges.src['h']}
    
    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _ , dst = etype
            feat_src = h[src]
            feat_dst = h[dst]

            degs = graph.out_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            if src=='item':
                graph.update_all(self.category_aggregation, self.sub_reduction_item_user, etype = etype)
                muti_int=graph.nodes[dst].data['i']
            else:
                graph.update_all(self.category_aggregation, self.sub_reduction_user_item, etype = etype)
            
            rst = graph.nodes[dst].data['h']
            degs = graph.in_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
            if src=='item':
                return muti_int,rst
            else:
                return rst
            
class BasetestLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = args.k
        self.sigma = args.sigma
        self.gamma = args.gamma
        self.poly_attn=PolyAttention(args)

    def similarity_matrix(self, X, sigma = 1.0, gamma = 2.0):
        dists = th.cdist(X, X)
        sims = th.exp(-dists / (sigma * dists.mean(dim = -1).mean(dim = -1).reshape(-1, 1, 1)))
        return sims

    def submodular_selection_feature(self, nodes):
        device = nodes.mailbox['m'].device
        feature = nodes.mailbox['m']
        sims = self.similarity_matrix(feature, self.sigma, self.gamma)

        batch_num, neighbor_num, feature_size = feature.shape
        nodes_selected = []
        cache = th.zeros((batch_num, 1, neighbor_num), device = device)

        for i in range(self.k):
            gain = th.sum(th.maximum(sims, cache) - cache, dim = -1)

            selected = th.argmax(gain, dim = 1)
            cache = th.maximum(sims[th.arange(batch_num, device = device), selected].unsqueeze(1), cache)

            nodes_selected.append(selected)

        return th.stack(nodes_selected).t()


    def sub_reduction_item_user(self, nodes):
        # -1 indicate user-> node, which does not include category information
        mail = nodes.mailbox['m']
        batch_size, neighbor_size, feature_size = mail.shape

        if (-1 in nodes.mailbox['c']) or nodes.mailbox['m'].shape[1] <= self.k:
            
            muti_int=self.poly_attn(embeddings=mail, attn_mask=0, bias=None)
            muti_int=muti_int.sum(dim=1)
            mail=mail.sum(dim=1)
        else:
            neighbors = self.submodular_selection_feature(nodes)
            mail = mail[th.arange(batch_size, dtype = th.long, device = mail.device).unsqueeze(-1), neighbors]
            muti_int=self.poly_attn(embeddings=mail, attn_mask=0, bias=None)#12 20 32
            muti_int=muti_int.sum(dim=1)
            mail = mail.sum(dim = 1)
      
        return {'h': muti_int,'i':muti_int}
    
    def sub_reduction_user_item(self, nodes):
        # -1 indicate user-> node, which does not include category information
        mail = nodes.mailbox['m']
        batch_size, neighbor_size, feature_size = mail.shape

        if (-1 in nodes.mailbox['c']) or nodes.mailbox['m'].shape[1] <= self.k:
            mail = mail.sum(dim = 1)
        else:
            neighbors = self.submodular_selection_feature(nodes)
            mail = mail[th.arange(batch_size, dtype = th.long, device = mail.device).unsqueeze(-1), neighbors]
            mail = mail.sum(dim = 1)
        return {'h': mail}
    
    def category_aggregation(self, edges):
        return {'c': edges.src['category'], 'm': edges.src['h']}
    
    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _ , dst = etype
            feat_src = h[src]
            feat_dst = h[dst]

            degs = graph.out_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            if src=='item':
                graph.update_all(self.category_aggregation, self.sub_reduction_item_user, etype = etype)
                muti_int=graph.nodes[dst].data['i']
            else:
                graph.update_all(self.category_aggregation, self.sub_reduction_user_item, etype = etype)
            
            rst = graph.nodes[dst].data['h']
            degs = graph.in_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
            
            return rst        
class PolyAttention(nn.Module):
    r"""
    Implementation of Poly attention scheme that extracts `K` attention vectors through `K` additive attentions
    """
    def __init__(self,args):
        r"""
        Initialization

        Args:
            in_embed_dim: The number of expected features in the input ``embeddings``#200
            num_context_codes: The number of attention vectors ``K``#32
            context_code_dim: The number of features in a context code#200
        """
        super().__init__()
        self.linear = nn.Linear(in_features=args.embed_size, out_features=args.context_code_dim, bias=False)#32 32
        self.context_codes = nn.Parameter(nn.init.xavier_uniform_(th.empty(args.num_context_codes, args.context_code_dim),
                                                                  gain=nn.init.calculate_gain('tanh')))

    def forward(self, embeddings, attn_mask, bias):
        r"""
        Forward propagation

        Args:
            embeddings: tensor of shape ``(batch_size, his_length, embed_dim)``#12 20 32
            attn_mask: tensor of shape ``(batch_size, his_length)``
            bias: tensor of shape ``(batch_size, his_length, num_candidates)``

        Returns:
            A tensor of shape ``(batch_size, num_context_codes, embed_dim)``#12 32 32
        """
        proj = th.tanh(self.linear(embeddings))
        if bias is None:
            weights = th.matmul(proj, self.context_codes.T)
        else:
            bias = bias.mean(dim=2).unsqueeze(dim=2)
            weights = th.matmul(proj, self.context_codes.T) + bias
        weights = weights.permute(0, 2, 1)
        
        weights = F.softmax(weights, dim=2)
        poly_repr = th.matmul(weights, embeddings)

        return poly_repr

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
