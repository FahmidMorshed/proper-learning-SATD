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


class MAR(object):
    def __init__(self, dataset):
        self.step = 10              # FAHID after how many steps we want to train again
        self.enough = 30            # FAHID convert to agressive undersampling
        self.atleast = 100          # FAHID if we have unlabeled data, assume all negative (as the chances are very low)
        random.seed(0)

        # FAHID
        self.true_count = dataset.true_count
        self.false_count = dataset.false_count
        self.body = dataset.data_pd
        self.body.loc[:, 'time'] = 0
        self.body.loc[:, 'fixed'] = 0
        self.body.loc[:, 'count'] = 0
        self.csr_mat = dataset.csr_mat

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
            unlabeled = unlabeled.sample(self.atleast)
        except:
            pass

        # TODO FAHID PRESUMTIVE NON RELEVANT AFTER APPLYING BM25
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


