import pickle
import random
from pdb import set_trace
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from collections import Counter
from sklearn import svm
import matplotlib.pyplot as plt
import time
import os
from sklearn import preprocessing

import pandas
from sklearn.linear_model import LogisticRegression


class MAR(object):
    def __init__(self, dataset, enough, atleast, step, enable_est=False, stopat=.95):
        self.step = step                 # FAHID after how many steps we want to train again
        self.enough = enough            # FAHID convert to agressive undersampling
        self.atleast = atleast          # FAHID if we have unlabeled data, assume all negative (as the chances are very low)
        self.enable_est = enable_est
        self.stopat = stopat
        self.est_num = 500
        random.seed(0)


        if stopat < .8:
            self.ensemble_threshold = 1
        elif stopat < .9:
            self.ensemble_threshold = 2
        else:
            self.ensemble_threshold = 3


        # FAHID
        self.true_count = dataset.true_count
        self.false_count = dataset.false_count
        self.body = dataset.data_pd
        self.body.loc[:, 'time'] = 0
        self.body.loc[:, 'fixed'] = 0
        self.body.loc[:, 'count'] = 0
        self.csr_mat = dataset.csr_mat

        self.results = []

    # FAHID predicted results
    def get_numbers(self):
        total = len(self.body["code"])
        pos = len(self.body[self.body["code"] == 'yes'])
        neg = len(self.body[self.body["code"] == 'no'])

        self.pool = np.where(np.array(self.body['code']) == "undetermined")[0]
        self.labeled = list(set(range(len(self.body['code']))) - set(self.pool))
        return pos, neg, total


    def get_random_ids(self):
        return self.body.sample(self.step).index

    def get_ensemble_ids(self):
        return self.body.loc[self.body['no_vote'] <= self.ensemble_threshold].index

    def get_ensemble_help(self, no_vote):
        return self.body.loc[(self.body['no_vote'] <= no_vote) & (self.body['code'] == 'undetermined')].index

    def get_random_help(self):
        a = self.body.loc[self.body['code'] == 'undetermined']
        a = a.sample(self.step)
        return a.index

    ## Train model ##
    def train(self, pne=True, weighting=True):
        clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced') if weighting else svm.SVC(
            kernel='linear', probability=True)

        poses = self.body.loc[self.body['code'] == 'yes']
        negs = self.body.loc[self.body['code'] == 'no']
        labeled = self.body.loc[self.body['code'] != 'undetermined']
        unlabeled = self.body.loc[self.body['code'] == 'undetermined']

        pos_ids = list(poses.index)
        labeled_ids = list(labeled.index)

        try:
            unlabeled = unlabeled.sample(self.atleast) #np.max((len(poses), self.atleast))
        except:
            pass

        # FAHID PRESUMTIVE NON RELEVANT
        # Examples Presume all examples are false, because true examples are few
        # This reduces the biasness of not doing random sampling
        if not pne:
            unlabeled = []

        unlabeled_ids = unlabeled.index

        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = pandas.concat([negs, unlabeled])
        all_neg_ids = list(all_neg.index)

        sample = pandas.concat([poses, negs, unlabeled])
        sample_ids = list(sample.index)

        clf.fit(self.csr_mat[sample_ids], labels[sample_ids])

        ## aggressive undersampling ##
        if len(poses) >= self.enough:
            train_dist = clf.decision_function(self.csr_mat[all_neg_ids])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist = -train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(pos_ids)]
            sample_ids = list(pos_ids) + list(np.array(all_neg_ids)[negs_sel])

            clf.fit(self.csr_mat[sample_ids], labels[sample_ids])

        elif pne:
            train_dist = clf.decision_function(self.csr_mat[unlabeled_ids])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist = -train_dist
            unlabel_sel = np.argsort(train_dist)[::-1][:int(len(unlabeled_ids) / 2)]
            sample_ids = list(labeled_ids) + list(np.array(unlabeled_ids)[unlabel_sel])

            clf.fit(self.csr_mat[sample_ids], labels[sample_ids])

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)

        if self.enable_est:
            self.est_num, self.est = self.estimate_curve(clf, num_neg=len(sample) - len(poses))
            return uncertain_id, self.est[uncertain_id], certain_id, self.est[certain_id]

        return uncertain_id, uncertain_prob, certain_id, certain_prob


    ## Get certain ##
    def certain(self, clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        order = np.argsort(prob)[::-1][:self.step]
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get uncertain ##
    def uncertain(self, clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        train_dist = clf.decision_function(self.csr_mat[self.pool])
        order = np.argsort(np.abs(train_dist))[:self.step]  ## uncertainty sampling by distance to decision plane
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get random ##
    def random(self):
        return np.random.choice(self.pool, size=np.min((self.step, len(self.pool))), replace=False)

    ## Get one random ##
    def one_rand(self):
        pool_yes = [x for x in range(len(self.body['label'])) if self.body['label'][x] == 'yes']
        return np.random.choice(pool_yes, size=1, replace=False)


    ## Code candidate studies ##
    def code(self, id, label):
        self.body.loc[id, 'code'] = label
        self.body.loc[id, 'time'] = time.time()

    def code_error(self, id, error='none'):
        # FAHID: simulate a human reader
        if error == 'random':
            self.code_random(id, self.body.loc[id, 'label'])
        elif error == 'three':
            self.code_three(id, self.body['label'][id])
        else:
            self.code(id, self.body.loc[id, 'label'])

    def code_three(self, id, label):
        self.code_random(id, label)
        self.code_random(id, label)
        if self.body['fixed'][id] == 0:
            self.code_random(id, label)

    def code_random(self, id, label):
        error_rate = 0.3
        if label == 'yes':
            if random.random() < error_rate:
                new = 'no'
            else:
                new = 'yes'
        else:
            if random.random() < error_rate :
                new = 'yes'
            else:
                new = 'no'
        if new == self.body.loc[id, "code"]:
            self.body.loc[id, 'fixed'] = 1
        self.body.loc[id, "code"] = new
        self.body.loc[id, "time"] = time.time()
        self.body.loc[id, "count"] = self.body.loc[id, "count"] + 1


    def get_allpos(self):
        return len(self.body[self.body['label'] == 'yes'])

    ## BM25 ##
    def BM25(self, query):
        if query[0] == '':
            # FAHID This is not a random choice, but to generate n random numbers between [0,1)
            test = len(self.body["commenttext"])
            self.bm = np.random.rand(len(self.body["commenttext"]))
            return

        b = 0.75
        k1 = 1.5

        ### Combine title and abstract for training ###########
        content = [self.body["commenttext"][index] for index in
                   range(len(self.body["commenttext"]))]
        #######################################################

        ### Feature selection by tfidf in order to keep vocabulary ###

        tfidfer = TfidfVectorizer(lowercase=True, stop_words=None, norm=None, use_idf=True, smooth_idf=False,
                                  sublinear_tf=False, decode_error="ignore")
        tf = tfidfer.fit_transform(content)
        d_avg = np.mean(np.sum(tf, axis=1))
        score = {}
        for word in query:
            score[word] = []
            try:
                id = tfidfer.vocabulary_[word]
            except:
                score[word] = [0] * len(content)
                continue
            df = sum([1 for wc in tf[:, id] if wc > 0])
            idf = np.log((len(content) - df + 0.5) / (df + 0.5))
            for i in range(len(content)):
                score[word].append(
                    idf * tf[i, id] / (tf[i, id] + k1 * ((1 - b) + b * np.sum(tf[0], axis=1)[0, 0] / d_avg)))
        self.bm = np.sum(list(score.values()), axis=0)

    def BM25_get(self):
        # FAHID: get the indexes of bm at indexes of pool, then reverse and take the first step size of them
        return self.pool[np.argsort(self.bm[self.pool])[::-1][:self.step]]

    def estimate_curve(self, clf , num_neg=0):
        from sklearn import linear_model

        def prob_sample(probs):
            order = np.argsort(probs)[::-1]
            count = 0
            can = []
            sample = []
            for i, x in enumerate(probs[order]):
                count = count + (x * self.stopat)
                can.append(order[i])
                if count >= 1:
                    sample.append(can[0])
                    count = 0
                    can = []
            return sample

        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        poses = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[:]]

        ###############################################
        prob1 = clf.decision_function(self.csr_mat)
        prob = np.array([[x] for x in prob1])

        y = np.array([1 if x == 'yes' else 0 for x in self.body['code']])
        y0 = np.copy(y)

        all = list(set(poses) | set(negs) | set(self.pool))

        pos_num_last = Counter(y0)[1]

        lifes = 1
        life = lifes

        while (True):
            C = Counter(y[all])[1] / (num_neg)
            es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=C)

            es.fit(prob[all], y[all])
            pos_at = list(es.classes_).index(1)

            pre = es.predict_proba(prob[self.pool])[:, pos_at]

            y = np.copy(y0)

            sample = prob_sample(pre)
            for x in self.pool[sample]:
                y[x] = 1

            pos_num = Counter(y)[1]

            if pos_num == pos_num_last:
                life = life - 1
                if life == 0:
                    break
            else:
                life = lifes
            pos_num_last = pos_num
        esty = pos_num
        pre = es.predict_proba(prob)[:, pos_at]

        return esty, pre


