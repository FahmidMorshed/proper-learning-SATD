import csv
import math
import os
import string

import gensim
import pandas
import random
import numpy as np
from nltk import sent_tokenize, wordpunct_tokenize, pos_tag, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.porter import *
from random import shuffle

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split


class SATDD:
    """
    Class to load the whole dataset initially. Convert all WITHOUT_CLASSIFICAITON into no, and
    rest of the data into yes. Have methods to divide the dataset into test train as needed.
    """
    def __init__(self):
        self.all_dataset_pd = None
        self.total_true = 0
        self.total_false = 0
        self.filename = None

    def load_data(self, filename):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

        self.filename = filename
        with open("../data/" + self.filename, "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]

        columns = content[0]
        content = content[1:]
        random.seed(0)
        shuffle(content)

        self.all_dataset_pd = pandas.DataFrame(content[1:], columns=columns)

        self.all_dataset_pd['label'] = np.where((self.all_dataset_pd['classification'] != "WITHOUT_CLASSIFICATION"), 'yes', 'no')
        self.all_dataset_pd.loc[:, 'code'] = 'undetermined'

        self.total_true = len(self.all_dataset_pd[(self.all_dataset_pd['label'] == 'yes')])
        self.total_false = len(self.all_dataset_pd[(self.all_dataset_pd['label'] == 'no')])

        return self


    def create_and_process_dataset(self, dataset_names=[], doInclude=True):
        """
        Preprocess by tokenizing and TFIDF vectorizing and creates datasets.
        :param dataset_names: list of dataset names to merge. if given none, all dataset merges into one. will be used
        for cross project validation.
        :param doInclude: should we include the dataset_names or exclude them. Irrelevent if dataset_names is empty
        :return: DATASET class with a csr_mat produced by TFIDF
        """
        if dataset_names:
            if doInclude:
                return DATASET(self.all_dataset_pd.loc[self.all_dataset_pd['projectname'].isin(dataset_names)])
            else:
                return DATASET(self.all_dataset_pd.loc[~self.all_dataset_pd['projectname'].isin(dataset_names)])
        return DATASET(self.all_dataset_pd)

class DATASET:
    def __init__(self, data_pd):
        data_pd.index = range(len(data_pd))
        self.data_pd = data_pd
        self.true_count = len(data_pd[(data_pd['label'] == 'yes')])
        self.false_count = len(data_pd[(data_pd['label'] == 'no')])
        self.tfer = None
        self.csr_mat = None

    def set_csr_mat(self, tfer=None ):
        """
        :param tfer: if training set, give nothing, it will learn and fit_transform. But for Test, it should use
        the tfidf from the training set
        :return:
        """
        if tfer:
            self.tfer = tfer
            self.csr_mat = tfer.transform(self.data_pd['commenttext'])
        else:
            self.tfer = TfidfVectorizer(lowercase=True, stop_words=None, use_idf=True, smooth_idf=False,
                                  sublinear_tf=False, max_features=None)
            self.csr_mat = self.tfer.fit_transform(self.data_pd['commenttext'])

    def make_test_train_on_same_dataset(self, ratio=.5):

        y = self.data_pd.loc[:, 'label'].to_frame()
        X = self.data_pd

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=ratio)

        return  DATASET(X_train), DATASET(X_test)