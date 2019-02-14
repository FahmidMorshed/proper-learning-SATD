import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import scipy.sparse as sp
from sklearn.utils import resample

from model.fastread import read, print_summary
from optimizer.flash import tune_dt


def dt_ensemble(satdd, dataset_name, ignore_dataset='', clf_name="DT", bellwether_weights=None):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    test_data = satdd.create_and_process_dataset([dataset_name], doInclude=True)
    print(str(test_data.true_count) + " | " + str(test_data.false_count))
    all_dataset_names = satdd.all_dataset_pd.projectname.unique()

    result_pd = pd.DataFrame()
    for train_dataset_name in all_dataset_names:
        if train_dataset_name in dataset_name or train_dataset_name in ignore_dataset:
            continue
        train_data = satdd.create_and_process_dataset([train_dataset_name], doInclude=True)
        train_data.set_csr_mat()
        test_data.set_csr_mat(train_data.tfer)

        if clf_name in "DT":
            clf = DecisionTreeClassifier(random_state=0)
        elif clf_name in "NBM":
            clf = MultinomialNB(alpha=1.0)
        elif clf_name in "SVM":
            clf = SVC(random_state=0)

        x_train = train_data.csr_mat
        y_train = train_data.data_pd['label'].tolist()
        x_test = test_data.csr_mat
        y_test = test_data.data_pd['label'].tolist()

        predict(clf, x_test, y_test, x_train, y_train, result_pd, train_dataset_name, bellwether_weights)


    result_pd['code_ensemble'] = np.where(result_pd['yes_vote'] > result_pd['no_vote'], 'yes', 'no')

    #result_pd2['code_ensemble'] = np.where(result_pd2['yes_vote'] > result_pd2['no_vote'], 'yes', 'no')

    y_test = result_pd['label'].tolist()
    y_pred = result_pd['code_ensemble'].tolist()
    print(dataset_name)
    print(classification_report(y_test, y_pred))


    test_data.data_pd['yes_vote'] = result_pd['yes_vote']
    test_data.data_pd['no_vote'] = result_pd['no_vote']

    # test_data.data_pd['yes_vote2'] = result_pd2['yes_vote']
    # test_data.data_pd['no_vote2'] = result_pd2['no_vote']

    # double_learn(result_pd, satdd, dataset_name)
    #
    # test_data.data_pd['double_learn'] = result_pd['double_learn']
    # test_data.data_pd['double_learn_proba'] = result_pd['double_learn_proba']


    test_data.data_pd.to_csv("../temp/" + dataset_name + ".csv")

    return test_data


def predict(clf, x_test, y_test, x_train, y_train, result_pd, col_name, bellwether_weights):

    if 'label' not in result_pd.columns:
        result_pd['label'] = y_test
        result_pd['yes_vote'] = 0
        result_pd['no_vote'] = 0

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    result_pd[col_name] = y_pred.tolist()

    weight = 1
    if bellwether_weights:
        weight = bellwether_weights.get(col_name)

    yes_ids = result_pd[result_pd.loc[:, col_name] == 'yes'].index
    result_pd.loc[yes_ids, 'yes_vote'] += (1 * weight * weight)

    no_ids = result_pd[result_pd.loc[:, col_name] == 'no'].index
    result_pd.loc[no_ids, 'no_vote'] += (1 * weight * weight)


# FINAL VOTING ON ACTIVE LEARNING
def double_learn(result_pd, satdd, dataset_name):
    test_data = satdd.create_and_process_dataset([dataset_name], doInclude=True)
    train_data = satdd.create_and_process_dataset([dataset_name], doInclude=False)
    train_data.set_csr_mat()
    test_data.set_csr_mat(train_data.tfer)

    neg_ids = np.where(result_pd['yes_vote'] == 0)[0]
    pos_ids = np.where(result_pd['no_vote'] == 0)[0]

    new_pos_mat = test_data.csr_mat[pos_ids]
    new_neg_mat = test_data.csr_mat[neg_ids]
    new_pos_y = result_pd.loc[pos_ids, 'code_ensemble']
    new_neg_y = result_pd.loc[neg_ids, 'code_ensemble']

    x = sp.vstack([new_neg_mat, new_pos_mat])
    y = new_pos_y.append(new_neg_y)

    x = sp.vstack([x, train_data.csr_mat])
    y = y.append(train_data.data_pd.loc[:, 'label'])

    #clf = RandomForestClassifier(random_state=0)
    clf = AdaBoostClassifier(random_state=0)

    clf.fit(train_data.csr_mat, train_data.data_pd.loc[:, 'label'])
    y_pred = clf.predict(test_data.csr_mat)
    #y_pred_proba = clf.predict_proba(test_data.csr_mat)[:, 1]
    result_pd['double_learn'] = y_pred.tolist()
    #result_pd['double_learn_proba'] = y_pred_proba.tolist()
    print("Double Learn")
    y_test = result_pd['label'].tolist()
    print(classification_report(y_test, y_pred))


def read_ensemble(test_pd, dataset_name, stopat=1, error=None, step=10):
    import logging
    logger = logging.getLogger(dataset_name)

    total_pos = len(test_pd[test_pd['label'] == 'yes'])
    target = int(total_pos * stopat)
    print("Target: " + str(target))

    result_pd = test_pd.sort_values(by=['yes_vote'], ascending=False)

    result_pd['code'] = 'undetermined'
    count = 1
    loop = 1
    for index, row in result_pd.iterrows():
        result_pd.at[index, 'code'] = row['label']          #NO error

        if count%step==0:
            pos = len(result_pd[result_pd["code"] == 'yes'])
            neg = len(result_pd[result_pd["code"] == 'no'])

            loop += 1
            logger.info("%d, %d" % (pos, pos + neg))
            # if loop % 20 == 0:
            #     print("%d, %d" % (pos, pos + neg))
            if pos > target:
                break;
        count += 1

    return result_pd
