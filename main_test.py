import sys
import dgl
import dgl.function as fn
import os
import multiprocessing as mp
from tqdm import tqdm
import pdb
import numpy as np
import torch
import torch.nn as nn
import logging
from utils.parser import parse_args
from utils.dataloader import Dataloader
from utils.utils import config, construct_negative_graph, choose_model, load_mf_model, NegativeGraph
from utils.tester import Tester
from models.sampler import NegativeSampler
import wandb

if __name__ == '__main__':
    args = parse_args()
    early_stop = config(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.wandb_enable:
       wandb.init(project="DGRec",name="DGRec_test",notes=str(args))
       wandb.config.update(args)
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    device = torch.device(device)
    args.device = device

    data = args.dataset
    dataloader = Dataloader(args, data, device)
    # NegativeGraphConstructor = NegativeGraph(dataloader.historical_dict)
    sample_weight = dataloader.sample_weight.to(device)

    model = choose_model(args, dataloader)
    model = model.to(device)
    print("model already setting")

    logging.info('loading best model for test')
    model.load_state_dict(torch.load('./best_models/'))
    # args.model_mf = load_mf_model(args, dataloader)
    tester = Tester(args, model, dataloader)
    logging.info('begin testing')
    res,ndcg5_std,ndcg10_std,mrr_std,auc_std=tester.test()
    print(ndcg5_std,ndcg10_std,mrr_std,auc_std)
    if args.wandb_enable:
        wandb.log({"recall_100":res[0],"hit_ratio_100":res[1],"coverage_100":res[2],"racall_300":res[3],"hit_ratio_300":res[4],"coverage_300":res[5],"loss":loss_val,"ndcg10_std":ndcg10_std,"ndcg5_std":ndcg5_std,"mrr_std":mrr_std,"auc_std":auc_std})
        
