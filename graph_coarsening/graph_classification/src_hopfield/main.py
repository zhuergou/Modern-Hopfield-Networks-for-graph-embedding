import argparse
import yaml
import torch
import numpy as np
import sklearn.metrics as skm

from .core.model_handler import ModelHandler

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def map_graph(Kin, A, context):
    
    nA = A/np.linalg.norm(A, axis=1,keepdims=True)
    nC = context/np.linalg.norm(context, axis=1,keepdims=True)
    prod = (nC @ nA).T
    
   
    degree = np.sum(nA, axis=1) 
    ##v1
    v2K = np.ravel(np.argmax(prod, axis=1))
    W = np.zeros((Kin,Kin))
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            if A[i,j]:
                Ki = v2K[i]
                Kj = v2K[j]
                W[Ki,Kj]+=1.0
                W[Kj,Ki]+=1.0

    return W


def map_graph_new(Kin, A, context):

    ##note
    A = A/np.linalg.norm(A, axis=1,keepdims=True)
    context_return = context

    ## find projection
    idx_max = np.ravel(np.argmax(context.T, axis=1))

    ##
    #context = context/(np.linalg.norm(context, axis=1,keepdims=True))

    context = context/(np.linalg.norm(context, axis=1,keepdims=True) + 0.1)


    embedding = (context.dot(A)).T
    embedding = embedding/np.linalg.norm(embedding, axis=1,keepdims=True)


    sim = (embedding).dot(embedding.T)

    sim = sim *(sim > 0)

    np.fill_diagonal(sim, 0)

    return context, sim,  idx_max

def main_hopfield(config, matrix, n_hid, topk, learning_rate, alpha, layer):
    set_random_seed(config['random_seed'])

    model = ModelHandler(config, matrix, n_hid)
    model.train()


    context = model.model.network.context_weight.cpu().detach().numpy()

     
    ###v3
    _, E, idx_max = map_graph_new(n_hid, matrix, context)
     

    return _, E, idx_max

def get_config(dataset, config_path="./src_hopfield/config/config_"):
    config_path = config_path + dataset + '.yml'
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    args = vars(parser.parse_args())
    return args

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")

if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    main(config)
