import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import scipy.sparse as sp

from optimizer.flash import tune_dt


def dt_ensemble(satdd, dataset_name):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    test_data = satdd.create_and_process_dataset([dataset_name], doInclude=True)
    print(str(test_data.true_count) + " | " + str(test_data.false_count))
    all_dataset_names = satdd.all_dataset_pd.projectname.unique()

    result_pd = pd.DataFrame()
    for train_dataset_name in all_dataset_names:
        if train_dataset_name in dataset_name:
            continue
        train_data = satdd.create_and_process_dataset([train_dataset_name], doInclude=True)
        train_data.set_csr_mat()
        test_data.set_csr_mat(train_data.tfer)

        #best_config = tune_dt(train_data.csr_mat.todense(), train_data.data_pd['label'], dataset_name)

        clf = DecisionTreeClassifier()#(criterion=best_config[0], splitter=best_config[1], max_depth=best_config[2],
                                     #max_features=best_config[3], class_weight=best_config[4])

        predict(clf, test_data, train_data, result_pd, train_dataset_name)

    result_pd['code_ensemble'] = np.where(result_pd['yes_vote'] > result_pd['no_vote'], 'yes', 'no')
    result_pd.to_csv("../temp/"+dataset_name + ".csv")

    y_test = result_pd['label'].tolist()
    y_pred = result_pd['code_ensemble'].tolist()
    print(dataset_name)
    print(classification_report(y_test, y_pred))

def predict(clf, test_data, train_data, result_pd, col_name):

    x_train = train_data.csr_mat
    y_train = train_data.data_pd['label'].tolist()

    x_test = test_data.csr_mat
    y_test = test_data.data_pd['label'].tolist()

    if 'label' not in result_pd.columns:
        result_pd['label'] = test_data.data_pd['label']
        result_pd['yes_vote'] = 0
        result_pd['no_vote'] = 0

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # y_pred_proba = clf.predict_proba(x_test)[:, 1]
    result_pd[col_name] = y_pred.tolist()
    # result_pd[col_name + '_proba'] = y_pred_proba.tolist()

    yes_ids = result_pd[result_pd.loc[:, col_name] == 'yes'].index
    result_pd.loc[yes_ids, 'yes_vote'] += 1

    no_ids = result_pd[result_pd.loc[:, col_name] == 'no'].index
    result_pd.loc[no_ids, 'no_vote'] += 1


# FINAL VOTING ON ACTIVE LEARNING
def double_learn(result_pd, x_test, y_test):
    neg_ids = np.where(result_pd['yes_vote'] == 0)[0]
    pos_ids = np.where(result_pd['no_vote'] == 0)[0]

    new_pos_mat = x_test.csr_mat[pos_ids]
    new_neg_mat = x_test.csr_mat[neg_ids]
    new_pos_y = result_pd.loc[pos_ids, 'code_ensemble']
    new_neg_y = result_pd.loc[neg_ids, 'code_ensemble']

    x = sp.vstack([new_neg_mat, new_pos_mat])
    y = new_pos_y.append(new_neg_y)


    clf = SVC(random_state=0)

    clf.fit(x, y)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
