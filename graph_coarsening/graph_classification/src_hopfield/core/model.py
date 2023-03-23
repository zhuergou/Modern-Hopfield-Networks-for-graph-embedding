import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scipy
import os

from .models.hopfield import Hopfield, GraphConvolution
from .utils import constants as Constants

class Model(object):
    def __init__(self, config):
        self.config = config
        if self.config['pretrained']:
            self.init_saved_network(self.config['pretrained'])
        else:
            self.network = Hopfield(self.config)
        self.criterion = nn.MSELoss()
        self.criterion_v2 = nn.CrossEntropyLoss()
        self._init_optimizer()

    def _init_optimizer(self):
        #parameters = [p for p in self.network.parameters() if p.requires_grad]
        my_list = ['beta1', 'beta2']
        params_beta = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, self.network.named_parameters()))))
        params_other = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, self.network.named_parameters()))))


        self.optimizer = optim.Adam([{'params':params_other}, {'params':params_beta, 'lr': (self.config['learning_rate'])}], lr=self.config['learning_rate'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3, verbose=False)

    def init_saved_network(self, saved_dir):
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        print(fname)
        print('okok')
        state_dict = saved_params['state_dict']
        self.saved_epoch = saved_params.get('epoch', 0)

        self.network = Hopfield(self.config)

        if state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)


    def save(self, dirname, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
            'dir': dirname,
            'epoch': epoch
        }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')

    def predict(self, context, label, label_index, adj, update=True, mode='train'):
        batch_size = context.size(0)

        if mode == 'train':
            with torch.set_grad_enabled(True):
                self.network.train(update)
                out = self.network(context, adj)
                #loss = self.criterion(out, label)
                #loss = self.criterion_v2(out, label_index)
                loss = 0
                '''
                for item in out:
                    loss += self.criterion_v2(item, label_index)
                '''
                ## note here
                loss += self.criterion_v2(out[-1], label_index)

                #modified
                pred = torch.max(out[-1], 1)[1].view(label_index.size()).data
                correct = (pred == label_index.data).sum()
                metrics = correct*1.0/batch_size

                loss.backward()

                if self.config['grad_clipping']:
                    parameters = [p for p in self.network.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])

                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            with torch.no_grad():
                self.network.train(update)
                out = self.network(context)
                loss = self.criterion_v2(out, label_index)
                #loss = None

                pred = torch.max(out, 1)[1].view(label_index.size()).data
                correct = (pred == label_index.data).sum()
                metrics = correct*1.0/batch_size

        output = {
            'loss': loss,
            'metrics': {'acc': metrics}
        }

        return output

    def predict_ver2(self, masked, masked_value, idx, masked_good, masked_good_value, idx_good, update=True, mode='train'):
        batch_size = masked.size(0)

        if mode == 'train':
            with torch.set_grad_enabled(True):
                self.network.train(update)
                '''
                out = self.network(masked)
                out_val = torch.gather(out, 1, torch.unsqueeze(idx, 0)).squeeze()
                loss = self.criterion(out_val, masked_value)

                pred = (out_val > 0.5).long().data
                correct = (pred == masked_value.long().data).sum()
                '''

                

                out_good = self.network(masked_good)
                out_val_good = torch.gather(out_good, 1, torch.unsqueeze(idx_good, 0)).squeeze()
                #loss += self.criterion(out_val_good, masked_good_value)
                loss = self.criterion(out_val_good, masked_good_value)

                pred_good = (out_val_good > 0.5).long().data
                #correct += (pred_good == masked_good_value.long().data).sum()
                correct = (pred_good == masked_good_value.long().data).sum()
                metrics = correct*1.0/(batch_size)


                loss.backward()

                if self.config['grad_clipping']:
                    parameters = [p for p in self.network.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])

                self.optimizer.step()
                self.optimizer.zero_grad()

        else:
            with torch.no_grad():
                self.network.train(update)
                out = self.network(masked)
                out_val = torch.gather(out, 1, torch.unsqueeze(idx, 0)).squeeze()
                loss = self.criterion(out_val, masked_value)

                pred = (out_val > 0.5).long().data
                correct = (pred == masked_value.long().data).sum()


                out_good = self.network(masked_good)
                out_val_good = torch.gather(out_good, 1, torch.unsqueeze(idx_good, 0)).squeeze()
                loss += self.criterion(out_val_good, masked_good_value)

                pred_good = (out_val_good > 0.5).long().data
                correct += (pred_good == masked_good_value.long().data).sum()
                metrics = correct*1.0/(2*batch_size)


        output = {
            'loss': loss,
            'metrics': {'acc': metrics}
        }

        return output
