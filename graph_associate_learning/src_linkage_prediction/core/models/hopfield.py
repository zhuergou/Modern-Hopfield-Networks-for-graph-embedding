import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

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
        #self.target_weight = self.context_weight
        #self.embedding = nn.Parameter(nn.init.normal_(torch.Tensor(self.vocab_size, self.dim), mean=0.0, std=0.1))
        #a = np.load('context_block.npy').astype(np.float32)
        #b = np.load('target_block.npy').astype(np.float32)
        #self.context_weight = nn.Parameter(torch.from_numpy(a))
        #self.target_weight = nn.Parameter(torch.from_numpy(b))
        #self.beta = self.config['beta']
        self.beta1 = nn.Parameter(torch.FloatTensor([self.config['init_beta1']]), requires_grad=False)
        self.beta2 = nn.Parameter(torch.FloatTensor([self.config['init_beta2']]), requires_grad=False)

        
        self.bn_layer = nn.ModuleList([])

        
        for i in range(self.config['layer']):
            x = nn.BatchNorm1d(num_features=self.hid, track_running_stats=False)
            y = nn.BatchNorm1d(num_features=self.vocab_size, track_running_stats=False)
            z = nn.BatchNorm1d(num_features=self.hid, track_running_stats=False)
            h = nn.BatchNorm1d(num_features=self.dim, track_running_stats=False)
            
            self.bn_layer.append(nn.ModuleList([x, y, z, h]))
        
        #self.bn_layer_v2 = nn.BatchNorm1d(num_features=self.vocab_size, track_running_stats=False)

        self.set_count = set()

    def forward(self, context, adj):
        
        layer_num = self.config['layer']
        batch_size = context.size(0)
        target_block = torch.zeros(batch_size, self.vocab_size).to(self.config['device'])

        #context_normal = F.normalize(self.context_weight)
        #target_normal = F.normalize(self.target_weight)

        #W = F.normalize(self.W)
        context_normal = (self.context_weight)
        target_normal = (self.target_weight)
        #context_input_normal = F.normalize(context)
        context_input_normal = context
        rst = []

        for i in range(layer_num):
            print('##############' + str(i) + "##############")
            #target_block_normal = F.normalize(target_block)
            target_block_normal = target_block
            v1 = torch.matmul(context_input_normal, context_normal.t())

            #v1 = torch.spmm(adj, v1)
            #v1 = F.relu(v1)

            #v1 = F.normalize(v1)
            #v1 = torch.matmul(v1, W.t())
            #v1 = self.bn_layer[i][0](v1)
            #v1 = F.normalize(v1)



            #value, idx = torch.topk(v1, 10, dim=1)
            #print('after v1')
            #print(value)
            #print(idx) 
            #print('v1') 
            #print(v1)   
        
            '''
            ## topk version
            v1_bak = torch.zeros_like(v1).to(self.config['device'])
    
            value, idx = torch.topk(v1, 5, dim=1)
            v1_bak[torch.arange(v1_bak.shape[0]).unsqueeze(1), idx] = 1

            temple_set = set([item for y1 in idx.cpu().numpy() for item in y1])
            self.set_count = self.set_count.union(temple_set)

            print('v1_value')
            print(value)
            print(idx)
            '''


            #print('v1_bak')
            #print(v1_bak)
    
            if i != 0:
                v1_plus  = torch.matmul(target_block_normal, target_normal.t())
                #v1_plus = self.bn_layer[i][2](v1_plus)

                part2 = v1_plus * self.beta2
                #value, idx = torch.topk(part2, 10, dim=1)
                #print(value)
                #print(idx)

            else:
                v1_plus = 0
                part2 = 0

            
            part1 = v1 *self.beta1
            
            #value, idx = torch.topk(part1, 10, dim=1)
            #print('topk before softmax')
            #print(value)
            #print(idx)
            v1_final = part1 + part2

            #v1_final = F.normalize(v1_final)
            #print('sum')
            #print(value)
            #print(idx)
                
    
            v1_final = F.softmax(v1_final, dim=-1)
            '''
            value, idx = torch.topk(v1_final, 10, dim=1)
            print('softmax')
            print(value)
            print(idx)
            '''

            x, y = torch.max(v1_final, dim=-1)
            print('softmax value')
            print(x)
            print(y)
    
            '''
            if i == (layer_num - 1):
                temple_set = set([item for item in y.cpu().numpy()])
                self.set_count = self.set_count.union(temple_set)
            '''


            #v2 = F.relu(v1)
            target_block_tmp = torch.matmul(v1_final, target_normal)
            #print("target_block_tmp")
            #print(target_block_tmp)
            #target_block = target_block_normal + target_block_tmp
            #target_block = 0.8*target_block + 0.2*(target_block_tmp - target_block)
            target_block = target_block_tmp
            #target_block = F.normalize(target_block)
            #if i >= 4:
            #rst.append(target_block)

        target_block = self.bn_layer[i][1](target_block)

        rst.append(target_block)        
        #print('##set')
        #print(len(self.set_count))
        return rst


