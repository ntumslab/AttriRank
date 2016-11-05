"""
Implementation of AttriRank.

Author: Yi-An Lai

For more details, refer to the paper:
Unsupervised Ranking using Graph Structures and Node Attributes
Chin-Chi Hsu, Yi-An Lai, Wen-Hao Chen, Ming-Han Feng, and Shou-De Lin
Web Search and Data Mining (WSDM), 2017
"""

import argparse
import numpy as np
import pandas as pd

from AttriRank import AttriRank


def parse_args():
    '''
    Parses AttriRank arguments.
    '''
    parser = argparse.ArgumentParser(description="Run AttriRank.")

    parser.add_argument('--inputgraph', nargs='?',
                        default='sample/graph.edgelist',
                        help='Input graph path')

    parser.add_argument('--inputfeature', nargs='?',
                        default='sample/graph.feature',
                        help='Input feature path')

    parser.add_argument('--output', nargs='?', default='graph.rankscore',
                        help='Output rankscore path')

    parser.add_argument('--kernel', default='rbf_ap',
                        help='Kernel: rbf_ap, rbf, cos, euc, sigmoid')

    parser.add_argument('--damp', nargs='*', default=[0.5], type=float,
                        help='damping parameters')

    parser.add_argument('--totalrank', dest='totalrank', action='store_true',
                        help='Use TotalRank or not. Default is False.')
    parser.set_defaults(totalrank=False)

    parser.add_argument('--alpha', type=float, default=1.0,
                        help='alpha of beta distribution. Default is 1.0.')

    parser.add_argument('--beta', type=float, default=1.0,
                        help='beta of beta distribution. Default is 1.0.')

    parser.add_argument('--matrix', dest='matrix', action='store_true',
                        help='Using original Q matrix. Default is False.')
    parser.set_defaults(matrix=False)

    parser.add_argument('--print_every', type=int, default=1000,
                        help='Print TotalRank process. Default is 1000.')

    parser.add_argument('--itermax', type=int, default=100000,
                        help='Number of max iterations. Default is 100000.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Specifying (un)weighted. Default is unweighted.')
    parser.set_defaults(weighted=False)

    parser.add_argument('--undirected', dest='directed', action='store_false',
                        help='Graph is (un)directed. Default is directed.')
    parser.set_defaults(directed=True)

    return parser.parse_args()


def load_graph(filename):
    """Read the graph into numpy array"""
    return pd.read_csv(filename, sep=' ', header=None).values


def load_features(filename):
    """Read the features into numpy array, first column as index"""
    return pd.read_csv(filename, header=None).set_index(0).values


def main(args):
    """
    Pipeline for unsupervised ranking using graph and node features
    """
    graph = load_graph(args.inputgraph)
    feat = load_features(args.inputfeature)
    N = len(feat)

    if not args.directed:
        graph = np.concatenate((graph, graph[:, [1, 0]]))

    AR = AttriRank(graph, feat, itermax=args.itermax, weighted=args.weighted,
                   nodeCount=N)

    scores = AR.runModel(factors=args.damp, kernel=args.kernel,
                         Matrix=args.matrix, TotalRank=args.totalrank,
                         alpha=args.alpha, beta=args.beta,
                         print_every=args.print_every)

    df = pd.DataFrame(data=scores)
    df.to_csv(args.output, float_format='%.16f', index_label='node_id')


args = parse_args()
main(args)
