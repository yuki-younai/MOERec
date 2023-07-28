import torch.nn as nn
from tqdm import tqdm
import torch as th
import pdb
import torch.nn.functional as F
import torch
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn import GraphConv
from models.layers import DGRecLayer
from models.layers import BaseLayer
from models.layers import MOERecLayer
from models.layers import BasetestLayer
from torch.distributions.normal import Normal
from models.layers import *

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype = etype)
            return graph.edges[etype].data['score']

class BaseGraphModel(nn.Module):
    def __init__(self, args, dataloader):
        super().__init__()
        self.args = args
        self.hid_dim = args.embed_size
        self.layer_num = args.layers
        self.graph = dataloader.train_graph
        self.user_number = dataloader.user_number
        self.item_number = dataloader.item_number

        self.user_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('user').shape[0], self.hid_dim))
        self.item_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('item').shape[0], self.hid_dim))
        self.predictor = HeteroDotProductPredictor()
        self.user_num=self.graph.nodes('user').shape[0]
        self.build_model()

        self.node_features = {'user': self.user_embedding, 'item': self.item_embedding}
        
        self.data=[]
        for i in range(self.user_num):
            self.data.append(i)
        self.data_split=self.data[0:self.user_num:32]

    def build_layer(self, idx):
        return BasetestLayer(self.args)

    def build_model(self):
        self.layers = nn.ModuleList()
        for idx in range(self.layer_num):
            h2h = self.build_layer(idx)
            self.layers.append(h2h)

    def get_embedding(self):
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        h = self.node_features
        
        for layer in self.layers:
            h_item = layer(self.graph, h, ('user', 'rate', 'item'))
            h_user = layer(self.graph, h, ('item', 'rated by', 'user'))
            h = {'user': h_user, 'item': h_item}
        return h

    def forward(self, graph_pos, graph_neg):
        if self.args.moe:
             h,loss = self.get_embedding()
        else:
             h = self.get_embedding()

        score_pos = self.predictor(graph_pos, h, 'rate')
        score_neg = self.predictor(graph_neg, h, 'rate')

        if self.args.moe:
             return score_pos, score_neg,loss
        else:
             return score_pos, score_neg

    def get_score(self, h, users):
            user_embed = h['user'][users]
            item_embed = h['item']
            scores = torch.mm(user_embed, item_embed.t())
            return scores
    def get_score_part(self,h,users,items):
            user_embed = h['user'][users]
            item_embed = h['item'][items]
            scores = torch.matmul(user_embed, item_embed.t())
            return scores

class DGRec(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(DGRec, self).__init__(args, dataloader)
        self.W = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size))#32 32
        self.a = torch.nn.Parameter(torch.randn(self.args.embed_size))#32

    def build_layer(self, idx):
        return DGRecLayer(self.args)

    def layer_attention(self, ls, W, a):
        tensor_layers = torch.stack(ls, dim = 0)#2 76000 32
        weight = torch.matmul(tensor_layers, W)#2 76000 32
        weight = F.softmax(torch.matmul(weight, a), dim = 0).unsqueeze(-1)#2 76000 1
        tensor_layers = torch.sum(tensor_layers * weight, dim = 0)
        return tensor_layers

    def get_embedding(self):
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        h = self.node_features

        for layer in self.layers:

            h_item = layer(self.graph, h, ('user', 'rate', 'item'))
            h_user = layer(self.graph, h, ('item', 'rated by', 'user'))
            h = {'user': h_user, 'item': h_item}
            user_embed.append(h_user)
            item_embed.append(h_item)
        user_embed = self.layer_attention(user_embed, self.W, self.a)
        item_embed = self.layer_attention(item_embed, self.W, self.a)
        h = {'user': user_embed, 'item': item_embed}
        return h

class BasetestRec(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(BasetestRec, self).__init__(args, dataloader)

        self.attention_experts=TargetAttention(args)  
    def build_layer(self, idx):
        return BasetestLayer(self.args)

    def get_embedding(self):
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        h = self.node_features

        for layer in self.layers:

            h_item = layer(self.graph, h, ('user', 'rate', 'item'))
            h_user,muti_int = layer(self.graph, h, ('item', 'rated by', 'user'))
            h_user=self.attention_experts(muti_int)
            #h_user=muti_int.sum(dim=1)
            h = {'user': h_user, 'item': h_item}

        
        h = {'user': h_user, 'item': h_item}

        return h




class MOERec(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(MOERec, self).__init__(args, dataloader)
        self.k=args.k_experts
        self.n_experts=args.n_experts
        self.noisy_gating=True
        self.w_gate = nn.Parameter(torch.zeros(args.embed_size*args.num_context_codes, self.n_experts), requires_grad=True)#20000 5
        self.w_noise = nn.Parameter(torch.zeros(args.embed_size*args.num_context_codes, self.n_experts), requires_grad=True)#20000 5
        self.softplus = nn.Softplus()
        self.softmax=nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.attention_experts=nn.ModuleList([TargetAttention(args) for i in range(self.n_experts)])
        #self.attention_experts=TargetAttention(args) 
    def _gates_to_load(self, gates):
     
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):#32 5,32 5,32 5,32 3
     
        batch = clean_values.size(0)#32
        m = noisy_top_values.size(1)#3
        top_values_flat = noisy_top_values.flatten()#32*3

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        
        
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):#32 20000,true
       
        clean_logits = x @ self.w_gate##32 20000,20000 5-->32 5
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise##32 20000,20000 5-->32 5
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))#32 5
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)#32 5+(_rand(32 5)*32 5)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.n_experts), dim=1)#32 3
        top_k_logits = top_logits[:, :self.k]#32 2
        top_k_indices = top_indices[:, :self.k]#32 2
        top_k_gates = self.softmax(top_k_logits)#32 2

        zeros = torch.zeros_like(logits, requires_grad=True)#32 5
        gates = zeros.scatter(1, top_k_indices, top_k_gates)#32 5，每一行只有两个不为0的数，下标为排序后前2的下标，值为k_gates

        if self.noisy_gating and self.k < self.n_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)#32 5,32 5,32 5,32 3
        else:
            load = self._gates_to_load(gates)
        return gates, load
    def cv_squared(self, x):#1 5
      
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)    
    def build_layer(self, idx):
        return BasetestLayer(self.args)

    def get_embedding(self):
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        h = self.node_features

        for layer in self.layers:

            h_item = layer(self.graph, h, ('user', 'rate', 'item'))
            h_user,muti_int = layer(self.graph, h, ('item', 'rated by', 'user'))
            #h_user=self.attention_experts(muti_int)
            #h_user=muti_int.sum(dim=1)
            h = {'user': h_user, 'item': h_item}

        gate,load=self.noisy_top_k_gating(muti_int.view(muti_int.shape[0],-1),True)
        importance = gate.sum(0)
        dispatcher = SparseDispatcher(self.n_experts, gate)
        expert_inputs_user = dispatcher.dispatch(muti_int)
        expert_outputs=[]
        for i in range(self.n_experts):
            if len(expert_inputs_user[i])>0:
                expert_outputs.append(self.attention_experts[i](expert_inputs_user[i]))
            else:
                kk=1    
        expert_outputs=dispatcher.combine(expert_outputs) 
        loss = self.cv_squared(importance) + self.cv_squared(load)
        
        h = {'user': expert_outputs, 'item': h_item}

        return h,loss
                
class TargetAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.W = torch.nn.Parameter(torch.randn(args.embed_size, args.embed_size))
        self.a = torch.nn.Parameter(torch.randn(args.embed_size,1))
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.5) 
        
    def forward(self,muti_int): 
        
        #weight = torch.matmul(muti_int, self.W)#32 32 32 32
        #weight = self.prelu(torch.matmul(weight, self.a)).unsqueeze(-1)#32 32 *32-->32 1
        weight = self.prelu(self.a)
        muti_int=self.dropout(muti_int)
        muti_int = torch.sum(muti_int *weight, dim = 1)#65000 32 32 32 1
        return muti_int
