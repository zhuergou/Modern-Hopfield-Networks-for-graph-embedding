import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class GraphConvolution(nn.Module):
    def __init__(self, config):
        super(GraphConvolution, self).__init__()
        self.config = config
        self.vocab_size = self.config['graph'].vocab_size
        self.hid = self.config['hidden_unit']
        self.context_weight = nn.Parameter(torch.Tensor(self.hid, self.vocab_size))
        self.out = nn.Linear(self.hid, self.vocab_size)

    def forward(self, context, adj):
        support = torch.mm(context, self.context_weight.T)
        score = F.relu(torch.spmm(adj, support))

        output = self.out(score)
        rst = []
        rst.append(output)
        return rst
        





class Hopfield(nn.Module):
    def __init__(self, config):
        super(Hopfield, self).__init__()
        self.dim = 300
        self.config = config
        self.hid = self.config['hidden_unit']
        self.vocab_size = self.config['graph'].vocab_size
        self.context_weight = nn.Parameter(nn.init.normal_(torch.Tensor(self.hid, self.vocab_size), mean=0.0, std=0.01))
        self.W = nn.Parameter(nn.init.normal_(torch.Tensor(self.hid, self.dim), mean=0.0, std=0.1))
        self.target_weight = nn.Parameter(nn.init.normal_(torch.Tensor(self.hid, self.vocab_size), mean=0.0, std=0.01))
        self.beta1 = nn.Parameter(torch.FloatTensor([self.config['init_beta1']]), requires_grad=False)
        self.beta2 = nn.Parameter(torch.FloatTensor([self.config['init_beta2']]), requires_grad=False)

        
        self.bn_layer = nn.ModuleList([])

        
        for i in range(self.config['layer']):
            x = nn.BatchNorm1d(num_features=self.hid, track_running_stats=False)
            y = nn.BatchNorm1d(num_features=self.vocab_size, track_running_stats=False)
            z = nn.BatchNorm1d(num_features=self.hid, track_running_stats=False)
            h = nn.BatchNorm1d(num_features=self.dim, track_running_stats=False)
            
            self.bn_layer.append(nn.ModuleList([x, y, z, h]))
        

        self.set_count = set()

    def forward(self, context, adj):
        
        layer_num = self.config['layer']
        batch_size = context.size(0)
        target_block = torch.zeros(batch_size, self.vocab_size).to(self.config['device'])

        context_normal = F.normalize(self.context_weight)
        target_normal = context_normal



        context_input_normal = F.normalize(context)
        rst = []

        for i in range(layer_num):
            #print('##############' + str(i) + "##############")
            target_block_normal = F.normalize(target_block)
            v1 = torch.matmul(context_input_normal, context_normal.t())

    
            if i != 0:
                v1_plus  = torch.matmul(target_block_normal, target_normal.t())

                part2 = v1_plus * self.beta2

            else:
                v1_plus = 0
                part2 = 0

            
            part1 = v1 *self.beta1
            
            v1_final = part1 + part2

                
    
            v1_final = F.softmax(v1_final, dim=-1)

            x, y = torch.max(v1_final, dim=-1)
    


            target_block_tmp = torch.matmul(v1_final, target_normal)
            target_block = (1 - self.config['alpha'])*target_block + self.config['alpha']*(target_block_tmp - target_block)

            target_block = self.bn_layer[i][1](target_block)

        rst.append(target_block)        
        return rst


