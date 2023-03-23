import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scipy
import os

from six.moves import zip_longest
from six import iterkeys
from scipy.io import loadmat
from scipy.sparse import issparse
from scipy.special import softmax

from collections import defaultdict
from time import time
import random

from .utils import constants as Constants
from .utils.logger import DummyLogger
from .utils.eval_utils import AverageMeter
from .model import Model
from scipy import sparse

class ModelHandler(object):

    def __init__(self, config, matrix, n_hid):
        
        #graph, adj = load_matfile(config['graph_dir'])
        #graph, adj = load_npz(config['graph_dir'])
        #print(matrix)
        graph, adj = from_numpy(sparse.csr_matrix(matrix))
        config['hidden_unit'] = n_hid

        self.graph = graph
        self.adj = adj

        self._train_loss = AverageMeter()
        self._dev_loss = AverageMeter()
        self._train_metrics = {'acc': AverageMeter()}
        self._dev_metrics = {'acc': AverageMeter()}

    
        #self.logger = DummyLogger(config, dirname=config['out_dir'], pretrained=config['pretrained'])
        #self.dirname = self.logger.dirname

        config['graph'] = self.graph

        if not config['no_cuda'] and torch.cuda.is_available():
            #print('[Using CUDA]')
            self.device = torch.device('cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
        else:
            self.device = torch.device('cpu')

        config['device'] = self.device

        self.model = Model(config)
        self.model.network = self.model.network.to(self.device)
        self.adj = self.adj.to(self.device)

        ### data preprocess
        sequence_index = np.arange(self.graph.vocab_size)
        np.random.shuffle(sequence_index)


        self.trainset = DataIndex(sequence_index, self.graph, False)
        self.trainloader = DataLoader(self.trainset, batch_size=200000, shuffle=True, num_workers=0)

        ##note here
        #self.trainloader = DataLoader(self.trainset, batch_size=2000000, shuffle=False, num_workers=0)
        
        self.config = self.model.config

    def train(self):
        self._epoch = 0
        self._best_epoch = 0
        
        self._best_metrics = {}
        for k in self._dev_metrics:
            self._best_metrics[k] = self._dev_metrics[k].mean()
        self._reset_metrics()



        while self._stop_condition(self._epoch):
            self._epoch += 1
            #if self._epoch > 3:
            #    raise Exception
            #print("\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            #self.logger.write_to_file("\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self._run_epoch(self.trainloader, self.adj, training=True)
            format_str = "Training Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._train_loss.mean())
            format_str += self.metric_to_str(self._train_metrics)
            #self.logger.write_to_file(format_str)
            #print(format_str)
    

            self.model.scheduler.step(self._train_metrics['acc'].mean())

            if self._best_metrics['acc'] < self._train_metrics['acc'].mean():
                self._best_epoch = self._epoch
                self._best_metrics['acc'] = self._train_metrics['acc'].mean()
        


            self._reset_metrics()
    
    def _run_epoch(self, data_loader, adj, training=True):
        mode = "train" if training else "test"

        if training:
            self.model.optimizer.zero_grad()

        for i, (context, label, label_index) in enumerate(data_loader):
            batch_size = context.size(0)

                        
            ##v1
            context = context.to(self.config['device'])
            label = label.to(self.config['device'])
            label_index = label_index.to(self.config['device'])
            
            

            res = self.model.predict(context, label, label_index, adj, update=training, mode=mode)

            loss = res['loss']
            metrics = res['metrics']
            
            self._update_metrics(loss, metrics, batch_size, training=training)

    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k].mean())
        return format_str

    def best_metric_to_str(self, metrics):
        format_str = '\n'
        for k in metrics:
            format_str += '{} = {:0.5f}\n'.format(k.upper(), metrics[k])
        return format_str


    def _update_metrics(self, loss, metrics, batch_size, training=True):
        if training:
            if loss:
                self._train_loss.update(loss)
            for k in self._train_metrics:
                if not k in metrics:
                    continue
                self._train_metrics[k].update(metrics[k] * 100, batch_size)
        else:
            if loss:
                self._dev_loss.update(loss)
            for k in self._dev_metrics:
                if not k in metrics:
                    continue
                self._dev_metrics[k].update(metrics[k] * 100, batch_size)

    def _reset_metrics(self):
        self._train_loss.reset()
        self._dev_loss.reset()

        for k in self._train_metrics:
            self._train_metrics[k].reset()
        for k in self._dev_metrics:
            self._dev_metrics[k].reset()

    def _stop_condition(self, epoch, patience=8):
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.config['max_epochs']
        return False if exceeded_max_epochs or no_improvement else True

            

    def save_word2vec_format(self, fname, binary=False):
        vocab = np.arange(self.graph.vocab_size)


        vocab_len = len(vocab)

        memory = self.model.network.context_weight.cpu().detach().numpy()
        memory = memory/(np.linalg.norm(memory, axis=1, keepdims=True))
        #W = self.model.network.W.cpu().detach().numpy()
        vector_size = memory.shape[0]
        #vector_size = W.shape[0]

        count = 0
        with open(fname, 'w') as fout:
            fout.write("%s %s\n" % (vocab_len, vector_size))
            for i in range(len(vocab)):
                #row = vectors[i]
                #if flag == True and i not in idx:
                #    continue
                row_pre = np.array(self.graph.binary[i])
                #row_pre = row_pre*self.graph.degree_vector

                if np.linalg.norm(row_pre) > 0:
                    row_pre = row_pre/(np.linalg.norm(row_pre))
                    row = row_pre.dot(memory.T)
                    row_final = row
                else:
                    row_final = row_pre.dot(memory.T)
                
                
                

                fout.write(("%s %s\n" % (i, ' '.join(repr(val) for val in     row_final))))
                count += 1

            #print(count)
            #print(vocab_len)



class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)
        self.binary = defaultdict(list)
        self.node_embedding = defaultdict(list)
        self.vocab_size = 0
        self.oneblock = defaultdict(list)
        self.degree_vector = None
        self.adj_matrix = None
        self.target_label = defaultdict(int)
        self.target_onehot = defaultdict(list)

    def nodes(self):
        return self.keys()

    
    def subgraph(self, nodes={}):
        subgraph = Graph()

        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]

        return subgraph


    def make_undirected(self):
        t0 = time()

        for v in list(self):
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        t1 = time()
        
        self.make_consistent()
        return self


    def make_consistent(self):
        t0 = time()
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        t1 = time()

        self.remove_self_loops()

        return self


    def remove_self_loops(self):
        
        removed = 0
        t0 = time()

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1


        t1 = time()

        return self


    def get_one_hot_represent(self):
        vocab_size = len(self)
        self.vocab_size = vocab_size

        #self.degree_vector = np.zeros(vocab_size, dtype=np.float32)
         
        
        for key, value in self.items():
            self.binary[key] = [0]*vocab_size
            self.target_onehot[key] = [0]*vocab_size

            ## for target
            self.target_label[key] = key
            self.target_onehot[key][key] = 1.0

            ##note here
            
            ## for neighbor
            for item in value:
                self.binary[key][item] = 1.0

            ##normalize the input data
            if np.linalg.norm(self.binary[key]) > 0:
                self.binary[key] = self.binary[key]/(np.linalg.norm(self.binary[key]))
            
            ## two hop neighbor



    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        G = self
        
        if start:
            path = [start]
        else:
            path = [rand.choice(list(G.keys()))]
        
        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break

        walk_val = [0] * self.vocab_size
        for item in path:
            walk_val[item] += 1
        return walk_val

def enumerate_conflict(data_list):
    rst = []
    for i in range(len(data_list)):
        tmp = data_list[:]
        del tmp[i] 
        rst.append(tuple(tmp))
    return rst


def load_matfile(file_, variable_name="network", undirected=True):
    mat_varables = loadmat(file_)
    print(mat_varables.keys())
    mat_matrix = mat_varables[variable_name]
    return from_numpy(mat_matrix, undirected)

def load_npz(file_, undirected=True):
    mat_matrix = sparse.load_npz(file_)
    return from_numpy(mat_matrix, undirected)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
        raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    G.get_one_hot_represent()

    #np.save('one_hop_neighbor.npy', G.onehop_count)
    #np.save('two_hop_neighbor.npy', G.twohop_count)
    adj = sparse_mx_to_torch_sparse_tensor(x)
    
    return G, adj

def build_deepwalk_corpus(G, num_paths=80, path_length=40, alpha=0, rand=random.Random(0)):
    walks = []
    
    nodes = list(G.nodes())
    
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))

    return walks

def sparse_dense_mul(s, d):
    i = s._indices()
    v = s._values()
    dv = d[i[0, :], i[1,:]]
    return torch.sparse.FloatTensor(i, v*dv, s.size())


class DataIndex(Dataset):
    def __init__(self, index, G, twohop):
        self.index = index
        self.graph = G

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        M = torch.from_numpy(np.array(self.graph.binary[self.index[idx]]))


        label = torch.from_numpy(np.array(self.graph.target_onehot[self.index[idx]]))
        label_index = torch.from_numpy(np.array(self.graph.target_label[self.index[idx]]))

        #return M.float(), label.float(), label_index.long(), masked.float(), masked_value.float(), idx_mask, masked_good.float(), masked_good_value.float(), idx_good
        return M.float(), label.float(), label_index.long()





def get_binary_encoding(matrix, hash_length):
    '''
    input matrix dimension = number of hidden * number of input
    '''  
    hid = matrix.shape[0]
    voc = matrix.shape[1]
    matrix_sort = -np.sort(-matrix, axis=0)
    thr = matrix_sort[hash_length-1, :]
    matrix_binary = matrix > np.tile(thr.reshape(1, voc), (hid, 1))
    matrix_binary = matrix_binary.astype(int)
    return matrix_binary.T
    


