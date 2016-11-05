# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.metrics.pairwise import (rbf_kernel, cosine_similarity,
                                      sigmoid_kernel, euclidean_distances)
from collections import defaultdict


class AttriRank(object):
    convergenceThreshold = 1e-10
    Matrix = False
    track = False
    scores = {}
    print_every = 1000
    track_scores = {}

    def __init__(self, graph, featureMatrix, itermax=100000,
                 weighted=True, nodeCount=None):
        """
        Standardize input features and set the basic parameters
        graph: [[node_from, node_to], ...]
        featureMatrix: N * d matrix; i-th node's feature is the i-th row
        itermax: maximum iterations
        weighted: transition Matrix weighted by number of links
        """

        self.graph = np.array(graph)
        self.featMat = preprocessing.scale(np.array(featureMatrix) / 100.0)
        self.featCount = self.featMat.shape[1]

        if nodeCount is None:
            self.nodeCount = graph.max() + 1
        else:
            self.nodeCount = nodeCount

        self.iterationMax = itermax
        self.weighted = weighted

    def ResetProbVec(self, kernel='rbf_ap'):
        """
        Calculate the reset probability vector with assigned kernel
        rbf: Radial basis function
        cos: (cosine similarity + 1) / 2.0
        euc: 1.0 / (1 + euclidean distances)
        sigmoid: (tanh(gamma <X_i, X_j>) + 1) / 2.0
        rbf_ap: Taylor-expansion approximated Radial basis function
        """

        if kernel == 'rbf':
            RBF = rbf_kernel(self.featMat, gamma=1.0 / self.featCount)
            RBF = RBF.sum(axis=0)
            resetProbVec = RBF / np.sum(RBF)

        elif kernel == 'cos':
            Cos = (cosine_similarity(self.featMat) + 1) / 2.0
            Cos = Cos.sum(axis=0)
            resetProbVec = Cos / np.sum(Cos)

        elif kernel == 'euc':
            Euc = 1.0 / (euclidean_distances(self.featMat) + 1)
            Euc = Euc.sum(axis=0)
            resetProbVec = Euc / np.sum(Euc)

        elif kernel == 'sigmoid':
            gamma = 1.0 / self.featCount
            Sig = sigmoid_kernel(self.featMat, coef0=0, gamma=gamma)
            Sig = (Sig + 1.0) / 2.0
            Sig = Sig.sum(axis=0)
            resetProbVec = Sig / np.sum(Sig)

        elif kernel == 'rbf_ap':
            parameter = 1.0 / self.featCount
            # w
            lengths = np.einsum("ij, ij -> i", self.featMat, self.featMat)
            expNormVector = np.exp(- parameter * lengths)
            # y
            f_normVec = np.einsum("i, ij -> j", expNormVector, self.featMat)
            featureNormVector = f_normVec * (2.0 * parameter)
            # Z
            outerMat = np.einsum("i, ij, ik -> jk", expNormVector,
                                 self.featMat, self.featMat)
            featureOuterNorm = outerMat * (2.0 * parameter ** 2)
            # r'
            first = expNormVector * np.sum(expNormVector)
            second = np.einsum("i, j, ij -> i", expNormVector,
                               featureNormVector, self.featMat)
            third = np.einsum("i, jk, ij, ik -> i", expNormVector,
                              featureOuterNorm, self.featMat, self.featMat)
            resetProbVec = first + second + third
            # r
            resetProbVec /= np.sum(resetProbVec)

        self.resetProbVec = resetProbVec

    def ResetProbMat(self):
        """Calculate the Q transition Matrix with RBF kernel"""
        parameter = 1.0 / self.featCount
        RBF = rbf_kernel(self.featMat, gamma=parameter)
        self.resetProbMat = RBF / RBF.sum(axis=0)

    def TransMat(self):
        """Construct transition matrix"""
        links = defaultdict(int)

        for nodefrom, nodeto in self.graph:

            if self.weighted:
                links[(nodefrom, nodeto)] += 1.0

            else:
                links[(nodefrom, nodeto)] = 1.0

        entryList = list()
        rowList = list()
        columnList = list()

        for key, val in links.items():
            entryList.append(val)
            columnList.append(key[0])
            rowList.append(key[1])

        # transition matrix
        traMat = csr_matrix((entryList, (rowList, columnList)),
                            shape=(self.nodeCount, self.nodeCount))
        self.transMat = traMat.multiply(csr_matrix(1.0 / traMat.sum(axis=0)))

        # find dangling nodes
        col_sum = np.array(traMat.sum(axis=0))[0]
        self.dangVec = np.arange(col_sum.shape[0])[col_sum == 0]

    def runPageRank(self, damp=0.85, do=True, doTrans=True, kernel='rbf_ap'):
        """
        do: whether to compute the reset probability vector
        doTrans: whether to compute the transition matrix
        """
        if doTrans:
            self.TransMat()
            print("\tGenerate transition matrix")

        if do:
            if self.Matrix:
                self.ResetProbMat()
                print("\tGenerate matrix Q")
            else:
                print("\tGenerate reset probability vector")
            self.ResetProbVec(kernel=kernel)

        if damp == 0:
            scoreVector = self.resetProbVec
            return scoreVector

        # record the scores of each update
        self.track_scores[damp] = []
        scoreVector = np.ones(self.nodeCount) / self.nodeCount

        for iteration in range(self.iterationMax):
            leak_scores = np.sum(scoreVector[self.dangVec])
            dangScore = leak_scores * self.resetProbVec

            if self.Matrix:
                teleport_prob = self.resetProbMat.dot(scoreVector)
            else:
                teleport_prob = self.resetProbVec

            newScoreVector = (1.0 - damp) * teleport_prob + \
                damp * (self.transMat.dot(scoreVector) + dangScore)
            error = np.linalg.norm(newScoreVector - scoreVector)

            if error < self.convergenceThreshold:
                break

            scoreVector = newScoreVector
            if self.track:
                self.track_scores[damp].append(scoreVector)

        return scoreVector

    def TotalRank(self, alpha=1, beta=1, kernel='rbf_ap'):
        """
        Implementation of TotalRank with beta distribution as the prior
        (alpha, beta): parameters for the beta distribution
        """
        print("\tGenerate transition matrix and reset probability vector")
        self.TransMat()
        self.ResetProbVec(kernel=kernel)

        rho_t = self.resetProbVec * beta / (alpha + beta)
        pi_t = self.resetProbVec * beta / (alpha + beta)

        for iteration in range(self.iterationMax):
            dangScore = np.sum(rho_t[self.dangVec]) * self.resetProbVec
            P_rho = (self.transMat.dot(rho_t) + dangScore)
            rho_next = P_rho * (iteration + alpha) / (iteration+1+alpha+beta)
            pi_t += rho_next
            error = np.linalg.norm(rho_next)

            if iteration % self.print_every == (self.print_every - 1):
                print("\tIteration %d:\t%.10f" % (iteration + 1, error))

            if error < self.convergenceThreshold:
                break

            rho_t = rho_next
            if self.track:
                self.track_scores['total'].append(pi_t)

        return pi_t

    def runModel(self, factors=[0.85], Matrix=False, track=False,
                 TotalRank=False, alpha=1, beta=1, print_every=1000,
                 kernel='rbf_ap'):
        """
        Give a list of damping factors to work with
        return a dict: key=(damp factor); value=(scores of each node)
        Matrix: use the exact Q or approximated r (True for Q)
        track: record the score vector at each iteration during updating
        """
        self.Matrix = Matrix
        self.track = track
        self.print_every = print_every
        scores = {}

        if TotalRank:
            print("Run AttriRank with prior...")
            scores['total'] = list(self.TotalRank(alpha=alpha, beta=beta))
        else:
            do = True
            doTrans = True
            for dampFac in factors:
                print("Run AttriRank, damp:", dampFac)
                score_vec = self.runPageRank(dampFac, do=do, doTrans=doTrans,
                                             kernel=kernel)

                # already have reset vector and transition matrix
                do = False
                doTrans = False
                scores[str(dampFac)] = list(score_vec)
                print("\tDone.")

        self.scores = scores

        return scores
