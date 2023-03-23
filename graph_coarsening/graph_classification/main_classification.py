import random_graph
import laplacian
import argparse
import numpy as np
import measure
import coarsening
import util
import networkx as nx
import netlsd
import parse
import classification
import os
import sys
from src_hopfield.main import *


def contract(matrix, p, q):
    n = matrix.shape[0]
    new = np.zeros((n-1, n-1))
    i_idx = 0
    for i in range(n):
        if i == p or i == q:
            continue
        j_idx = 0
        for j in range(n):
            if j == p or j == q:
                continue
            new[i_idx][j_idx] = matrix[i][j]
            j_idx += 1

        i_idx += 1

    i_idx = 0
    for i in range(n):
        if i == p or i == q:
            continue
        new[n-2][i_idx] = matrix[p][i] + matrix[q][i]
        i_idx += 1
    
    j_idx = 0
    for j in range(n):
        if j == p or j == q:
            continue
        new[j_idx][n-2] = matrix[j][p] + matrix[j][q]
        j_idx += 1
    new[n-2][n-2] = matrix[p][q]+ matrix[q][p] + matrix[p][p] + matrix[q][q]
    #new[n-2][n-2] = 0
    return new

def find_max(matrix, cap, map_dict):
    n = matrix.shape[0]

    
    ## max pick
    max_value = float('-inf')
    p = -1
    q = -1
    
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > max_value:
                max_value = matrix[i][j]
                p = i
                q = j
    
    ''' 
    ## random pick
    idx = np.arange(n*n)
    if np.sum(matrix) == 0:
        draw = np.random.choice(idx, 1)
    else:
        draw = np.random.choice(idx, 1, p=matrix.flatten()/np.sum(matrix))
    p = int(draw // n)
    q = int(draw % n)
    '''


    new = np.zeros((n-1, n-1))
    i_idx = 0
    for i in range(n):
        if i == p or i == q:
            continue
        j_idx = 0
        for j in range(n):
            if j == p or j == q:
                continue
            new[i_idx][j_idx] = matrix[i][j]
            j_idx += 1
        i_idx += 1

    i_idx = 0
    for i in range(n):
        if i == p or i == q:
            continue
        new[n-2][i_idx] = max(matrix[p][i], matrix[q][i])
        i_idx += 1
    
    j_idx = 0
    for j in range(n):
        if j == p or j == q:
            continue
        new[j_idx][n-2] = max(matrix[j][p], matrix[j][q])
        j_idx += 1

    return p, q, new


def main():
    parser = argparse.ArgumentParser(description='Experiment for graph classification with coarse graphs')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                            help='name of dataset (default: MUTAG)')
    parser.add_argument('--method', type=str, default="mgc",
                            help='name of the coarsening method')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='the ratio between coarse and original graphs n/N')
    parser.add_argument('--topk', type=int, default=3,
                        help='topk for hopfield')
    args = parser.parse_args()
    if args.dataset not in ["MUTAG", "ENZYMES",  "NCI109", "PROTEINS", "PTC_MR", 'OHSU', 'tumblr_ct1', 'REDDIT-BINARY']:
        print("Incorrect input dataset")
        sys.exit()
    if args.method not in ['mgc', 'sgc', 'hopfield', 'original']:
        print("Incorrect input coarsening method")
        sys.exit()
    if args.ratio < 0 or args.ratio > 1:
        print("Incorrect input ratio")
        sys.exit()
    dir = 'dataset'
    am, labels = parse.parse_dataset(dir, args.dataset)
    num_samples = len(am)
    X = np.zeros((num_samples, 250))
    Y = labels
    for i in range(num_samples):
        print(i)
        
        N = am[i].shape[0]
        n = int(np.ceil(args.ratio*N))
        if args.method == "mgc":
            coarse_method = coarsening.multilevel_graph_coarsening
        elif args.method == 'hopfield':

            Gc = (am[i])
            _, similarity, idx_map = main_hopfield(get_config(args.dataset), am[i], n, args.topk,  0, 0, 0)

            
            old = {}
            for j in range(N):
                old[j] = {j}

            while n < N:
                p, q, similarity = find_max(similarity, N//n, old)
            
            
                Gc = contract(Gc, p, q)
                N = Gc.shape[0]
            G = nx.from_numpy_matrix(Gc)
            

            ##fast version
            #G = _.dot(_.T)

            try:
                X[i] = netlsd.heat(G)
            except:
                pass
            continue
        elif args.method == 'original':
            G = nx.from_numpy_matrix(am[i])
            X[i] = netlsd.heat(G)
            continue
        else:
            coarse_method = coarsening.spectral_graph_coarsening
        if n > 1:
            Gc, Q, idx = coarse_method(am[i], n)
        else:
            Gc, Q, idx = coarse_method(am[i], 1)
        G = nx.from_numpy_matrix(Gc)
        X[i] = netlsd.heat(G)


    #X[X>200] = 0
    #X[X<-200] = 0
    acc, std = classification.KNN_classifier_kfold(X, Y, 10)
    print('ACC, ', acc)



if __name__ == '__main__':
    main()
