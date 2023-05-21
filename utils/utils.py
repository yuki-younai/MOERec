import logging
from utils.EarlyStop import EarlyStoppingCriterion
import torch
import numpy as np
import random
from tqdm import tqdm
import dgl
from models.models import DGRec
import os
from torch import nn
from models.models import MOERec
from models.models import BaseGraphModel
from models.models import BasetestRec
from torch.utils.data import Dataset, DataLoader

def load_mf_model(args, dataloader):
    model = MF(args, dataloader)
    path = './datasets/' + args.dataset + '/mf.pt'
    model.load_state_dict(torch.load(path))
    return model

def choose_model(args, dataloader):
    if args.model == 'dgrec':
        return DGRec(args, dataloader)
    elif args.model=='moe':
        return MOERec(args,dataloader)
    elif args.model=='test':
        return BasetestRec(args,dataloader)
    else:    
        return BaseGraphModel(args, dataloader)
class NegativeGraph(object):
    def __init__(self, dic):
        self.historical_dic = dic

    def __call__(self, graph, etype):
        utype, _, vtype = etype
        src, _ = graph.edges(etype = etype)

        dst = []
        for i in tqdm(range(src.shape[0])):
            s = int(src[i])
            while True:
                negitem = np.random.randint(0, graph.num_nodes(vtype))
                if negitem in self.historical_dic[s]:
                    continue
                else:
                    break
            dst.append(negitem)
        dst = torch.tensor(dst, device = src.device)
        return dgl.heterograph({etype: (src, dst)}, num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes}).to(graph.device)

def construct_negative_graph(graph, etype):
    utype, _ , vtype = etype
    src, _ = graph.edges(etype = etype)
    dst = torch.randint(graph.num_nodes(vtype), size = src.shape, device = src.device)
    graph_neg=dgl.heterograph({etype: (src, dst)}, num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes}).to(graph.device)
    
    
    return graph_neg

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def config(args):
    setup_seed(args.seed)

    path = f"{args.dataset}_model_{args.model}_lr_{args.lr}_embed_size_{args.embed_size}_batch_size_{args.batch_size}_weight_decay_{args.weight_decay}_layers_{args.layers}_neg_number_{args.neg_number}_seed_{args.seed}_k_{args.k}_sigma_{args.sigma}_gamma_{args.gamma}_beta_class_{args.beta_class}"
    if os.path.exists('./logs/' + path + '.log'):
        os.remove('./logs/' + path + '.log')

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s  %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='./logs/' + path + '.log')
    logger = logging.getLogger()
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    early_stop = EarlyStoppingCriterion(patience = args.patience, save_path = './best_models/' + path + '.pt')
    return early_stop

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



