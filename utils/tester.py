import pdb
import logging
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def compute_amn(y_true, y_score):
    auc = roc_auc_score(y_true,y_score)
    mrr = mrr_score(y_true,y_score)
    ndcg5 = ndcg_score(y_true,y_score,5)
    ndcg10 = ndcg_score(y_true,y_score,10)
    return auc, mrr, ndcg5, ndcg10


class Tester(object):
    def __init__(self, args, model, dataloader):
        self.args = args
        self.model = model
        # self.model_mf = args.model_mf
        self.history_dic = dataloader.historical_dict
        self.history_csr = dataloader.train_csr
        self.dataloader = dataloader.dataloader_test
        self.test_dic = dataloader.test_dic
        self.train_dic=dataloader.train_dic
        self.cate = np.array(list(dataloader.category_dic.values()))
        self.metrics = args.metrics
        self.item_number=dataloader.item_number

    def judge(self, users, items):

        results = {metric: 0.0 for metric in self.metrics}
        # for ground truth test
        # items = self.ground_truth_filter(users, items)
        stat = self.stat(items)
        for metric in self.metrics:
            f = Metrics.get_metrics(metric)
            for i in range(len(items)):
                results[metric] += f(items[i], test_pos = self.test_dic[users[i]], num_test_pos = len(self.test_dic[users[i]]), count = stat[i], model = self.model)
        return results

    def ground_truth_filter(self, users, items):
        batch_size, k = items.shape
        res = []
        for i in range(len(users)):
            gt_number = len(self.test_dic[users[i]])
            if gt_number >= k:
                res.append(items[i])
            else:
                res.append(items[i][:gt_number])
        return res

    def test(self):
        results = {}
        if self.args.moe:
              h,loss= self.model.get_embedding()
        elif self.args.model=='test':
              h= self.model.get_embedding()
        else:
              h= self.model.get_embedding()
        count = 0
        all_ndcg5={}
        all_ndcg10={}
        all_mrr={}
        all_auc={}
        for k in self.args.k_list:
            results[k] = {metric: 0.0 for metric in self.metrics}

        for batch in tqdm(self.dataloader):

            users = batch[0]
            count += users.shape[0]
            # count += len(users)
            scores = self.model.get_score(h, users)
        
            users = users.tolist()
            mask = torch.tensor(self.history_csr[users].todense(), device = scores.device).bool()
            scores[mask] = -float('inf')

            _, recommended_items = torch.topk(scores, k = max(self.args.k_list))
            recommended_items = recommended_items.cpu()
            for k in self.args.k_list:

                results_batch = self.judge(users, recommended_items[:, :k])

                for metric in self.metrics:
                    results[k][metric] += results_batch[metric]
            for i in range(len(users)):
                his=self.test_dic[users[i]]
                uid=users[i]
                for j in range(len(his)):
                    items=[]
                    items.append(his[j])
                    y_true = [1] + [0] * 20
                    label=self.cate[his[j]]
                    for j in range(20):
                        item=int(torch.randint(self.item_number,size=(1,1)))
                        while item in his:
                            item=int(torch.randint(self.item_number,size=(1,1))) 
                        items.append(item)
                    y_score=self.model.get_score_part(h,users[i],items)
                    y_score=y_score.tolist()
                    auc,mrr,ndcg5,ndcg10=compute_amn(y_true, y_score)  
                    #ndcg_5
                    if uid in all_ndcg5.keys():
                        all_ndcg5[uid]
                        if label in all_ndcg5[uid].keys():
                            all_ndcg5[uid][label].append(ndcg5)
                        else:
                            all_ndcg5[uid][label]=[ndcg5]  
                    else:
                        all_ndcg5[uid]={}
                        all_ndcg5[uid][label]=[ndcg5]
                    #ndcg_10
                    if uid in all_ndcg10.keys():
                        all_ndcg10[uid]
                        if label in all_ndcg10[uid].keys():
                            all_ndcg10[uid][label].append(ndcg10)
                        else:
                            all_ndcg10[uid][label]=[ndcg10]  
                    else:
                        all_ndcg10[uid]={}
                        all_ndcg10[uid][label]=[ndcg10]
                    #mrr
                    if uid in all_mrr.keys():
                        all_mrr[uid]
                        if label in all_mrr[uid].keys():
                            all_mrr[uid][label].append(mrr)
                        else:
                            all_mrr[uid][label]=[mrr]  
                    else:
                        all_mrr[uid]={}
                        all_mrr[uid][label]=[mrr]
                    #auc
                    if uid in all_auc.keys():
                        all_auc[uid]
                        if label in all_auc[uid].keys():
                            all_auc[uid][label].append(auc)
                        else:
                            all_auc[uid][label]=[auc]  
                    else:
                        all_auc[uid]={}
                        all_auc[uid][label]=[auc]    
                    
                    
        #ndcg_5        
        ndcg5_std=[]      
        for key1,value1 in all_ndcg5.items():
            temp_std=[]
            for key0,value0 in value1.items():
                temp_std.append(sum(value0)/len(value0))
            if len(temp_std)>=2:    
                std=np.std(temp_std)  
                ndcg5_std.append(std)
        if len(ndcg5_std)==0:        
            ndcg5_std=0
        else:  
            ndcg5_std=sum(ndcg5_std)/len(ndcg5_std) 
        #ndcg_10
        ndcg10_std=[]      
        for key1,value1 in all_ndcg10.items():
            temp_std=[]
            for key0,value0 in value1.items():
                temp_std.append(sum(value0)/len(value0))
            if len(temp_std)>=2:     
                std=np.std(temp_std)  
                ndcg10_std.append(std)
        if len(ndcg10_std)==0:
            ndcg10_std=0
        else:
            ndcg10_std=sum(ndcg10_std)/len(ndcg10_std)

        #mrr
        mrr_std=[]      
        for key1,value1 in all_mrr.items():
            temp_std=[]
            for key0,value0 in value1.items():
                temp_std.append(sum(value0)/len(value0))
            if len(temp_std)>=2:     
                std=np.std(temp_std)  
                mrr_std.append(std)
        if len(mrr_std)==0:
            mrr_std=0
        else:
            mrr_std=sum(mrr_std)/len(mrr_std)
        #auc
        auc_std=[]      
        for key1,value1 in all_auc.items():
            temp_std=[]
            for key0,value0 in value1.items():
                temp_std.append(sum(value0)/len(value0))
            if len(temp_std)>=2:    
                std=np.std(temp_std)  
                auc_std.append(std)
        if len(auc_std)==0:
            auc_std=0
        else:
            auc_std=sum(auc_std)/len(auc_std)            
                                                                                                 
        R=[]
        for k in self.args.k_list:
            for metric in self.metrics:
                results[k][metric] = results[k][metric] / count
                R.append(results[k][metric])
        self.show_results(results)
        return R,ndcg5_std,ndcg10_std,mrr_std,auc_std

    def show_results(self, results):
        for metric in self.metrics:
            for k in self.args.k_list:
                logging.info('For top{}, metric {} = {}'.format(k, metric, results[k][metric]))

    def stat(self, items):
        stat = [np.unique(self.cate[item], return_counts=True)[1] for item in items]
        return stat


class Metrics(object):

    def __init__(self):
        pass

    @staticmethod
    def get_metrics(metric):

        metrics_map = {
            'recall': Metrics.recall,
            'hit_ratio': Metrics.hr,
            'coverage': Metrics.coverage
        }

        return metrics_map[metric]

    @staticmethod
    def recall(items, **kwargs):

        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return hit_count/num_test_pos

    @staticmethod
    def hr(items, **kwargs):

        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def coverage(items, **kwargs):

        count = kwargs['count']

        return count.size

