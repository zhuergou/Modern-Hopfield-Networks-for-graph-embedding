import random_graph
import laplacian
import argparse
import numpy as np
import measure
import coarsening
import util
import networkx as nx
import netlsd
import os
import sys
from src_hopfield.main import *
from sklearn.cluster import AgglomerativeClustering

import time

def find_max(matrix, cap, map_dict):
    n = matrix.shape[0]

    time_start = time.time()

    ##
    max_value = float('-inf')
    p = -1
    q = -1
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > max_value and len(map_dict[i]) < cap and len(map_dict[j]) < cap:
                max_value = matrix[i][j]
                p = i
                q = j

    time_end = time.time()
    print('time cost', time_end - time_start)

    time_start = time.time()

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

    time_end =  time.time()
    print('time cost', time_end -time_start)

    return p, q, new





def main():
    parser = argparse.ArgumentParser(description='Experiment for block recovery of stochastic block model')
    parser.add_argument('--sbm_type', type=str, default="associative",
                            help='type of stochastic block model (default: associative)')
    parser.add_argument('--method', type=str, default="hopfield",
                            help='name of the coarsening method')
    parser.add_argument('--N', type=int, default=200,
                        help='node size of original graphs')
    parser.add_argument('--n', type=int, default=10,
                        help='node size of coarse graphs')
    parser.add_argument('--p', type=float, default=0.5,
                        help='edge probability between nodes within the same blocks')
    parser.add_argument('--q', type=float, default=0.1,
                        help='edge probability between nodes in different blocks')
    parser.add_argument('--max_trials', type=int, default=10,
                        help='number of repeated trials')
    args = parser.parse_args()
    if args.sbm_type not in ['associative']:
        print("Incorrect input stochastic method")
        sys.exit()
    if args.method not in ['hopfield']:
        print("Incorrect input coarsening method")
        sys.exit()
    if args.N < 0 or args.n < 0:
        print("Incorrect node size")
        sys.exit()
    if args.p < 0 or args.p > 1:
        print("Incorrect edge probability")
        sys.exit()
    if args.q < 0 or args.q > 1:
        print("Incorrect edge probability")
        sys.exit()


    ground_truth = coarsening.regular_partition(args.N, args.n)
    #nmi_result = np.zeros(args.max_trials)
    nmi_result = [0]*args.max_trials
    for p in range(args.max_trials):
        G = random_graph.sbm_pq(args.N, args.n, args.p, args.q)
        if args.method == 'hopfield':
            time_start = time.time()
            print(time_start)
            _, similarity, idx = main_hopfield(get_config('SBM'), G, args.n, 1, 0.001, 1, 1)
            

            ##v3
            #print(similarity)
            n_start = args.N
            distance = []
            for i in range(n_start):
                for j in range(n_start):
                    distance.append((similarity[i][j], i, j))

            distance.sort()
            n = n_start

            root = [ x for x in range(args.N)]
            root_size = [1]*args.N

            cap = args.N // args.n 
            while n > args.N//cap  and distance:
                _,i,j = distance.pop()

                root_i = root[i]
                root_j = root[j]

                if root_i != root_j and root_size[root_i] < cap and root_size[root_j] < cap and root_size[root_i] + root_size[root_j] < 1.2*cap:
                    if root_i < root_j:
                        for q in range(args.N):
                            if root[q] == root_j:
                                root[q] = root_i
                        root_size[root_i] = root_size[root_i] + root_size[root_j]
                    else:
                        for q in range(args.N):
                            if root[q] == root_i:
                                root[q] = root_j
                        root_size[root_j] = root_size[root_i] + root_size[root_j]
                        
                    n -= 1


            map_dict = {}
            ptr = 0
            final = []
            for item in root:
                if item not in map_dict:
                    final.append(ptr)
                    map_dict[item] = ptr
                    ptr += 1
                else:
                    final.append(map_dict[item])
            
            idx = np.array(final)
            time_end = time.time()
            print(time_end)
            print('time cost', time_end - time_start)

    
        nmi_result[p] = measure.NMI(idx, ground_truth, args.n)
        print('NMI ', nmi_result[p])

    print(nmi_result)
    print("Average NMI result is %.4f "%(np.mean(nmi_result)))





if __name__ == '__main__':
    main()
