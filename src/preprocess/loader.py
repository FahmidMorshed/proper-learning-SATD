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

WORD2VEC_SIZE = 511

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

    def set_csr_mat(self, tfer=None, MAX_FEATURE=None):
        """
        :param tfer: if training set, give nothing, it will learn and fit_transform. But for Test, it should use
        the tfidf from the training set
        :return:
        """
        if tfer:
            self.tfer = tfer
            self.csr_mat = tfer.transform(self.data_pd['commenttext'])
        else:
            #self.tfer = CountVectorizer()
            self.tfer = TfidfVectorizer(lowercase=False, stop_words=None, use_idf=True, smooth_idf=False,
                                   sublinear_tf=False, max_features=None, token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\'")#CountVectorizer(tokenizer=tokenize)#


            self.csr_mat = self.tfer.fit_transform(self.data_pd['commenttext'])

            # if MAX_FEATURE != 1 or MAX_FEATURE != None:
            #     print("Feature Selection using IG")
            #     y_train = self.data_pd.loc[:, 'label']
            #     temp = dict(zip(self.tfer.get_feature_names(),
            #                     mutual_info_classif(self.csr_mat, y_train, discrete_features=True,
            #                                         random_state=0)))
            #     temp = sorted(temp, key=temp.get, reverse=True)
            #     max_fea = int(MAX_FEATURE * len(temp))
            #
            #     self.tfer = CountVectorizer(vocabulary=temp[:max_fea])
            #     # TfidfVectorizer(lowercase=True, stop_words=None, use_idf=True, smooth_idf=False,
            #     #                    sublinear_tf=False, max_features=None,
            #     #                             vocabulary=temp[:max_fea])
            #
            #     self.csr_mat = self.tfer.fit_transform(self.data_pd['commenttext'])
            #     print("Feature Selection using IG DONE")



    def make_test_train_on_same_dataset(self, ratio=.5):

        y = self.data_pd.loc[:, 'label'].to_frame()
        X = self.data_pd

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=ratio)

        return  DATASET(X_train), DATASET(X_test)


    # WORD2VEC
    def make_word2vec(self, dataset_name):
        if os.path.isfile('../word2vecs/' + dataset_name + ".word2vec"):
            my_model = gensim.models.Word2Vec.load('../word2vecs/' + dataset_name + ".word2vec")
        else:
            print("Creating new word2vec model for " + dataset_name)
            training_tokens = []
            for x in self.data_pd['commenttext']:
                training_tokens.append([y for y in gensim.utils.simple_preprocess(x)])  # tokenize(x)]

            my_model = build_model(training_tokens)
            my_model.save('../word2vecs/' + dataset_name + ".word2vec")
            print("Finished creating word2vec model for " + dataset_name)


        return make_output_vec(self.data_pd, my_model)

    def get_word2vec_model(self, dataset_name):
        return gensim.models.Word2Vec.load('../word2vecs/' + dataset_name + ".word2vec")

# Build Word2vec

def make_output_vec(data_pd, word2vec):
    data_pd["features"] = ""
    itr = 0
    for index, row in data_pd.iterrows():

        comment_text = [y for y in gensim.utils.simple_preprocess(row['commenttext'])]

        x = np.array(
            [word2vec[i] for i in comment_text if
             i in word2vec.wv.vocab])
        word_count_p = len(x)
        word_vecs_p = np.sum(x, axis=0)

        temp = word_vecs_p / word_count_p
        if word_count_p == 0:
            temp = np.full((WORD2VEC_SIZE), 0)
        data_pd.set_value(index, "features", temp)

    print("Feature extraction complete")
    return data_pd

def build_model(documents):
    model = gensim.models.Word2Vec(
        documents,
        size=WORD2VEC_SIZE,
        window=10,
        min_count=1,
        workers=5)
    model.train(documents, total_examples=len(documents), epochs=10)
    return model

# Preprocessing stuff for Prev Work
def tokenize(document):
    lemmatizer = WordNetLemmatizer()

    #ADDING STEMMER
    stemmer = PorterStemmer()

    "Break the document into sentences"
    for sent in sent_tokenize(document):

        "Break the sentence into part of speech tagged tokens"
        for token, tag in pos_tag(wordpunct_tokenize(sent)):

            "Apply preprocessing to the token"
            token = token.lower()  # Convert to lower case
            token = token.strip()  # Strip whitespace and other punctuations
            token = token.strip('_')  # remove _ if any
            token = token.strip('*')  # remove * if any


            "If punctuation, ignore."
            # if all(char in string.punctuation for char in token):
            #     continue

            "If number, ignore."
            if token.isdigit():
                continue

            # Lemmatize the token and yield
            # Note: Lemmatization is the process of looking up a single word form
            # from the variety of morphologic affixes that can be applied to
            # indicate tense, plurality, gender, etc.
            lemma = lemmatizer.lemmatize(token)

            # No longer using lemma, using Porter Stemmer as Huang did
            #stemmed = stemmer.stem(token)
            yield lemma


