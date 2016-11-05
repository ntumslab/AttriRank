# -*- coding: utf-8 -*-

from unittest import TestCase

from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.metrics.pairwise import (rbf_kernel, cosine_similarity,
                                      sigmoid_kernel, euclidean_distances)
from collections import defaultdict

import numpy as np
import sys
import os

sys.path.append(os.getcwd() + '/src')

from AttriRank import AttriRank


class TestAttriRank(TestCase):

    def setUp(self):
        self.node = 1000
        nodefrom = np.random.choice(self.node, 20 * self.node)
        nodeto = np.random.choice(self.node, 20 * self.node)
        self.fake_graph = np.array(list(zip(nodefrom, nodeto)))
        self.fake_features = np.random.randn(self.node, 20)

    def reset_vec(self, kernel='rbf_ap'):
        feat = preprocessing.scale(self.fake_features)
        count = feat.shape[1]
        if kernel == 'rbf':
            temp = rbf_kernel(feat, gamma=1.0 / count).sum(axis=0)
        elif kernel == 'cos':
            temp = ((cosine_similarity(feat) + 1) / 2.0).sum(axis=0)
        elif kernel == 'euc':
            temp = (1.0 / (euclidean_distances(feat) + 1)).sum(axis=0)
        elif kernel == 'sigmoid':
            Sig = sigmoid_kernel(feat, coef0=0, gamma=1.0 / count)
            temp = ((Sig + 1.0) / 2.0).sum(axis=0)
        elif kernel == 'rbf_ap':
            gamma = 1.0 / count
            expVec = np.exp(- gamma * np.einsum("ij, ij -> i", feat, feat))
            feaVec = np.einsum("i, ij -> j", expVec, feat) * (2.0 * gamma)
            outMat = np.einsum("i,ij,ik->jk", expVec, feat, feat)
            outMat *= (2.0 * gamma ** 2)

            first = expVec * np.sum(expVec)
            second = np.einsum("i, j, ij -> i", expVec, feaVec, feat)
            third = np.einsum("i, jk, ij, ik -> i", expVec, outMat, feat, feat)
            temp = first + second + third

        return (temp / np.sum(temp))

    def reset_mat(self):
        feat = preprocessing.scale(self.fake_features)
        RBF = rbf_kernel(feat, gamma=1.0 / feat.shape[1])
        return RBF / RBF.sum(axis=0)

    def trans_mat(self, weighted=True):
        links = defaultdict(int)
        for nodefrom, nodeto in self.fake_graph:
            links[(nodefrom, nodeto)] += 1.0

        en_col_row = [[], [], []]
        for key, val in links.items():
            val = val if weighted else 1
            en_col_row[0].append(val)
            en_col_row[1].append(key[0])
            en_col_row[2].append(key[1])

        traMat = csr_matrix((en_col_row[0], (en_col_row[2], en_col_row[1])),
                            shape=(self.node, self.node))
        traMat = traMat.multiply(csr_matrix(1.0 / traMat.sum(axis=0)))
        col_sum = np.array(traMat.sum(axis=0))[0]
        dangVec = np.arange(col_sum.shape[0])[col_sum == 0]

        return traMat, dangVec

    def PageRank(self, damp, Matrix=False, kernel='rbf_ap'):
        traMat, dang = self.trans_mat()
        if Matrix:
            reMat = self.reset_mat()
        reVec = self.reset_vec(kernel=kernel)

        if damp == 0:
            return reVec, {}

        track = {}
        track[damp] = []
        result = np.ones(self.node) / self.node

        for i in range(1000000):
            dangScore = np.sum(result[dang]) * reVec
            tele = reMat.dot(result) if Matrix else reVec
            new = (1.0 - damp) * tele + damp * (traMat.dot(result) + dangScore)
            if np.linalg.norm(new - result) < 1e-10:
                break

            result = new
            track[damp].append(result)

        return result, track

    def totalrank(self, alpha=1, beta=1):
        traMat, dang = self.trans_mat()
        reVec = self.reset_vec()

        rho_t = reVec * beta / (alpha + beta)
        pi_t = reVec * beta / (alpha + beta)

        for iterat in range(100000):
            P_rho = (traMat.dot(rho_t) + np.sum(rho_t[dang]) * reVec)
            rho_next = P_rho * (iterat+alpha) / (iterat+1+alpha+beta)
            pi_t += rho_next
            if np.linalg.norm(rho_next) < 1e-10:
                break

            rho_t = rho_next

        return pi_t

    def run_model(self, damps, TotalRank=False, alpha=1, beta=1,
                  Matrix=False, kernel='rbf_ap'):
        scores = {}
        if TotalRank:
            scores['total'] = list(self.totalrank(alpha=alpha, beta=beta))
        else:
            for damp in damps:
                score, _ = self.PageRank(damp, kernel=kernel, Matrix=Matrix)
                scores[str(damp)] = list(score)

        return scores

    def test_ResetProbVec(self):
        for kernel in ['rbf', 'cos', 'euc', 'sigmoid', 'rbf_ap']:
            AR = AttriRank(self.fake_graph, self.fake_features,
                           nodeCount=self.node)
            AR.ResetProbVec(kernel=kernel)
            scores = AR.resetProbVec.ravel()
            answers = self.reset_vec(kernel)
            assert np.linalg.norm(answers - scores) < 1e-10

    def test_ResetProbMat(self):
        AR = AttriRank(self.fake_graph, self.fake_features,
                       nodeCount=self.node)
        AR.ResetProbMat()
        scores = AR.resetProbMat.ravel()
        answers = self.reset_mat().ravel()
        assert np.linalg.norm(answers - scores) < 1e-10

    def test_TransMat(self):
        AR = AttriRank(self.fake_graph, self.fake_features,
                       nodeCount=self.node)
        AR.TransMat()
        scores = AR.transMat.toarray().ravel()
        answers_mat, answers_dang = self.trans_mat()
        assert np.linalg.norm(answers_mat.toarray().ravel() - scores) < 1e-10
        assert np.linalg.norm(answers_dang - AR.dangVec) < 1e-10

        AR = AttriRank(self.fake_graph, self.fake_features,
                       nodeCount=self.node, weighted=False)
        AR.TransMat()
        scores = AR.transMat.toarray().ravel()
        answers_mat, answers_dang = self.trans_mat(weighted=False)
        assert np.linalg.norm(answers_mat.toarray().ravel() - scores) < 1e-10
        assert np.linalg.norm(answers_dang - AR.dangVec) < 1e-10

    def test_runPageRank(self):
        AR = AttriRank(self.fake_graph, self.fake_features,
                       nodeCount=self.node)
        AR.track = True
        scores = AR.runPageRank(damp=0.85)
        track = np.array(AR.track_scores[0.85])
        answers, ans_track = self.PageRank(damp=0.85)
        ans_track = np.array(ans_track[0.85])
        assert np.linalg.norm(answers - scores) < 1e-10
        assert np.linalg.norm(ans_track - track) < 1e-10

        AR = AttriRank(self.fake_graph, self.fake_features,
                       nodeCount=self.node)
        AR.track = True
        AR.Matrix = True
        scores = AR.runPageRank(damp=0.85)
        track = np.array(AR.track_scores[0.85])
        answers, ans_track = self.PageRank(damp=0.85, Matrix=True)
        ans_track = np.array(ans_track[0.85])
        assert np.linalg.norm(answers - scores) < 1e-10
        assert np.linalg.norm(ans_track - track) < 1e-10

    def test_TotalRank(self):
        AR = AttriRank(self.fake_graph, self.fake_features,
                       nodeCount=self.node)
        TR_scores = AR.TotalRank()
        answers = self.totalrank()
        assert np.linalg.norm(answers - TR_scores) < 1e-10

        TR_scores = AR.TotalRank(alpha=2, beta=4)
        answers = self.totalrank(alpha=2, beta=4)
        assert np.linalg.norm(answers - TR_scores) < 1e-10

        TR_scores = AR.TotalRank(alpha=0.9, beta=0.8)
        answers = self.totalrank(alpha=0.9, beta=0.8)
        assert np.linalg.norm(answers - TR_scores) < 1e-10

    def test_runModel(self):
        AR = AttriRank(self.fake_graph, self.fake_features,
                       nodeCount=self.node)
        damps = [i/10.0 for i in range(10)]
        scores = AR.runModel(damps, kernel='cos')
        scores = np.array([scores[str(d)] for d in damps])
        answers = self.run_model(damps, kernel='cos')
        answers = np.array([answers[str(d)] for d in damps])
        assert np.linalg.norm(answers - scores) < 1e-10

        scores = AR.runModel(damps, TotalRank=True, alpha=3, beta=4)
        scores = np.array(scores['total'])
        answers = self.run_model(damps, TotalRank=True, alpha=3, beta=4)
        answers = np.array(answers['total'])
        assert np.linalg.norm(answers - scores) < 1e-10

        scores = AR.runModel(damps, Matrix=True)
        scores = np.array([scores[str(d)] for d in damps])
        answers = self.run_model(damps, Matrix=True)
        answers = np.array([answers[str(d)] for d in damps])
        assert np.linalg.norm(answers - scores) < 1e-10
