from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors

from six import iteritems
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np
import sys

np.random.seed(20)

def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        #assert i != j
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in iteritems(G)}, G

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def main():
    parser = ArgumentParser("linkage_predict", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    #parser.add_argument("--emb", required=True, help='Embeddings file')
    parser.add_argument("--network", required=True, help='A .mat file containing the adjacency matrix and node labels of the input network.')
    parser.add_argument("--adj-matrix-name", default='network', help='Variable name of the adjacency matrix inside the .mat file.')
    parser.add_argument("--label-matrix-name", default='group', help='Variable name of the labels matrix inside the .mat file.')
    parser.add_argument("--num-samples", default=500, type=int, help='Number of samples.')


    parser.add_argument("--embedding", required=True, help='which embedding you want to use.')
    args = parser.parse_args()

    #model_deepwalk = KeyedVectors.load_word2vec_format('blogcatalog.embeddings', binary=False)
    #model_node2vec = KeyedVectors.load_word2vec_format('blogcatalog_node2vec.emd', binary=False)
    #model_bio = KeyedVectors.load_word2vec_format('bio_embedding_2000_blogcatalog.txt', binary=False)
    model_bio = KeyedVectors.load_word2vec_format(args.embedding, binary=False)
    #model_line = KeyedVectors.load_word2vec_format('blogcatalog_line.emd', binary=False)

    matfile = args.network
    mat = loadmat(matfile)
    A = mat[args.adj_matrix_name]

    graph, graph_adjlist = sparse2graph(A)
    node_num = len(graph_adjlist)


    num_sample = args.num_samples


    ## pos and neg samples
    pos = 0
    neg = 0

    sample_list = []
    label = []
    preferential_attach = []
    jaccard = []
    while True:
        i = np.random.randint(node_num)
        j = np.random.randint(node_num)
        
        if (i, j) in sample_list:
            continue

        if j in graph_adjlist[i] and pos < num_sample:
            pos += 1
            sample_list.append((i, j))
            label.append(1)

            ## common_neighbor
            inter_sect = (graph_adjlist[i]).intersection(graph_adjlist[j])
            if i in inter_sect:
                inter_sect.remove(i)
            if j in inter_sect:
                inter_sect.remove(j)

            ## jaccard
            union = (graph_adjlist[i]).union(graph_adjlist[j])
            if i in union:
                union.remove(i)
            if j in union:
                union.remove(j)
            jaccard.append(len(inter_sect)*1.0/(len(union)))

            #preferential_attach
            preferential_attach.append(len(graph_adjlist[i])* len(graph_adjlist[j]))

        elif j not in graph_adjlist[i] and neg < num_sample:
            neg += 1
            sample_list.append((i, j))
            label.append(0)
        
            ## common_neighbor
            inter_sect = (graph_adjlist[i]).intersection(graph_adjlist[j])
            if i in inter_sect:
                inter_sect.remove(i)
            if j in inter_sect:
                inter_sect.remove(j)

            ## jaccard
            union = (graph_adjlist[i]).union(graph_adjlist[j])
            if i in union:
                union.remove(i)
            if j in union:
                union.remove(j)
            jaccard.append(len(inter_sect)*1.0/(len(union)))
        
            #preferential_attach
            preferential_attach.append(len(graph_adjlist[i])* len(graph_adjlist[j]))

        if pos == num_sample and neg == num_sample:
            break

    
    #score_deepwalk = []
    #score_node2vec = []
    score_bio = []
    #score_line = []
    for i, item in enumerate(sample_list):
        #dot_product = model_deepwalk[str(item[0])].dot(model_deepwalk[str(item[1])])
        #score_deepwalk.append(sigmoid(dot_product))
        #dot_product_2 = model_node2vec[str(item[0])].dot(model_node2vec[str(item[1])])
        #score_node2vec.append(sigmoid(dot_product_2))
        dot_product_3 = model_bio[str(item[0])].dot(model_bio[str(item[1])])
        score_bio.append(sigmoid(dot_product_3))
        #dot_product_4 = model_line[str(item[0])].dot(model_line[str(item[1])])
        #score_line.append(sigmoid(dot_product_4))

    #fpr_deepwalk, tpr_deepwalk, thre = roc_curve(np.array(label), np.array(score_deepwalk))
    #roc_auc_deepwalk = auc(fpr_deepwalk, tpr_deepwalk)
    
    #fpr_node2vec, tpr_node2vec, thre = roc_curve(np.array(label), np.array(score_node2vec))
    #roc_auc_node2vec = auc(fpr_node2vec, tpr_node2vec)

    fpr_bio, tpr_bio, thre = roc_curve(np.array(label), np.array(score_bio))
    roc_auc_bio = auc(fpr_bio, tpr_bio)

    #fpr_line, tpr_line, thre = roc_curve(np.array(label), np.array(score_line))
    #roc_auc_line = auc(fpr_line, tpr_line)

    #fpr_prefer, tpr_prefer, thre = roc_curve(np.array(label), np.array(preferential_attach))
    #roc_auc_prefer = auc(fpr_prefer, tpr_prefer)

    #fpr_jaccard, tpr_jaccard, thre = roc_curve(np.array(label), np.array(jaccard))
    #roc_auc_jaccard = auc(fpr_jaccard, tpr_jaccard)

    plt.figure
    #plt.plot(fpr_deepwalk, tpr_deepwalk, lw=2, color='b', label='Deepwalk (area = %0.2f)' % roc_auc_deepwalk)
    #plt.plot(fpr_node2vec, tpr_node2vec, lw=2, color='r', label='Node2vec (area = %0.2f)' % roc_auc_node2vec)
    plt.plot(fpr_bio, tpr_bio, lw=2, color='c', label='Modern Hopfield Net (area = %0.2f)' % roc_auc_bio)
    #plt.plot(fpr_line, tpr_line, lw=2, color='m', label='LINE (area = %0.2f)' % roc_auc_line)
    #plt.plot(fpr_jaccard, tpr_jaccard, lw=2, color='g', label='Jaccard Coefficient (area = %0.2f)' % roc_auc_jaccard)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("Receiver operating characteristic (BlogCatalog)", fontsize=16)
    plt.legend(loc="lower right", prop={'size': 10})
    plt.savefig('BlogCatalog.pdf')

if __name__ == "__main__":
    sys.exit(main())
