from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model.mar import MAR
import numpy as np
import pandas as pd
import logging

def active_learning(dataset, dataset_name, stopat=.95, error=None, uncertain_limit=20, seed=0):
    logger = logging.getLogger(dataset_name)
    print("True: " + str(dataset.true_count) + " | False: " + str(dataset.false_count))
    logger.info("Total Yes: " + str(dataset.true_count) + " | Total No: " + str(dataset.false_count))


    stopat = float(stopat)                  # FAHID stop at recall
    starting = 5                            # FAHID Do random sampling until

    np.random.seed(seed)

    read = MAR(dataset)

    print("Stop: " + str(stopat) + " | Error: " + str(error) + " | UncertainLim: " + str(uncertain_limit)
          + " | Step: " + str(read.step) + " | Agressive Undersampling: " + str(read.enough))

    num_of_total_pos = read.get_allpos()
    target = int(num_of_total_pos * stopat)

    print ("Target: " + str(target))
    loop_count = 1
    while True:
        pos, neg, total = read.get_numbers()

        # if pos > target:
        #     read.readjust()

        loop_count += 1
        logger.info("%d, %d" % (pos, pos + neg))

        # if (loop_count % 20 == 0):
        #     print("%d, %d" % (pos, pos + neg))
        #     print('20 loops done...')

        if pos + neg >= total:
            break

        if pos < starting:
            for id in read.get_random_ids():
                read.code_error(id, error=error)
        else:
            a, b, c, d = read.train(weighting=True, pne=True)
            if pos >= target:
                break

            # QUERY
            if pos < uncertain_limit:
                # Uncertainity Sampling
                for id in a:
                    read.code_error(id, error=error)
            else:
                # Certainity Sampling
                for id in c:
                    read.code_error(id, error=error)


    pos, neg, total = read.get_numbers()
    print("Positive %d, Total Looked %d" % (pos, pos + neg))

    print_summary(read.body, dataset_name)
    # #set_trace()
    # return read


# Baselines
# Use SVM, predict the probability, then use Human reader to label, find the retrival curve
def baseline_fastread(train_data, test_data, dataset_name, clf, stopat=.95, error=None, step=10):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    logger = logging.getLogger(dataset_name)

    logger.info("Total Yes: " + str(test_data.true_count) + " | Total No: " + str(test_data.false_count))

    result_pd = pd.DataFrame()

    pred_proba(clf, test_data, train_data, result_pd, "svm")

    result_pd = read(result_pd, "svm", stopat, error, step, dataset_name)

    print_summary(result_pd, dataset_name)

# Predict probability using a clf
def pred_proba(clf, test_data, train_data, result_pd, col_name):
    x_train = train_data.csr_mat
    y_train = train_data.data_pd['label'].tolist()

    x_test = test_data.csr_mat
    y_test = test_data.data_pd['label'].tolist()

    if 'label' not in result_pd.columns:
        result_pd['label'] = test_data.data_pd['label']
        result_pd['code'] = 'undetermined'

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(test_data.csr_mat)[:, 1]
    result_pd[col_name] = y_pred.tolist()
    result_pd[col_name + '_proba'] = y_pred_proba.tolist()

    print(classification_report(y_test, y_pred))


def read(result_pd, col_name, stopat, error, step, dataset_name):
    logger = logging.getLogger(dataset_name)

    total_pos = len(result_pd[result_pd['label'] == 'yes'])
    target = int(total_pos * stopat)
    print("Target: " + str(target))

    result_pd = result_pd.sort_values(by=[col_name + "_proba"], ascending=False)

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

def print_summary(df, dataset_name):
    logger = logging.getLogger(dataset_name)
    total = len(df)
    checked_data = df.loc[df["code"] != 'undetermined']
    total_checked = len(checked_data)
    yes_data = df.loc[df['label'] == 'yes']
    total_yes = len(yes_data)
    yes_checked = yes_data.loc[yes_data['code'] != 'undetermined']
    total_yes_checked = len(yes_checked)


    test = checked_data.loc[:, "label"].tolist()
    predicted = checked_data.loc[:, "code"].tolist()
    confusion_mat = confusion_matrix(test, predicted, labels=["no", "yes"])

    print(dataset_name + " | Total: " + str(total) + " | Total Checked: " + str(total_checked) + " | Total Yes Data: "
          + str(total_yes) + " | Total Yes Checked: " + str(total_yes_checked))

    # logger.info(dataset_name + " | Total: " + str(total) + " | Total Checked: " + str(total_checked) + " | Total Yes Data: "
    #       + str(total_yes) + " | Total Yes Checked: " + str(total_yes_checked))

def divide_test_to_train(x, y, dataset_name):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=.2,
                                 random_state=0)
    for train_index, tune_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[tune_index]
        y_train, y_test = y.iloc[train_index], y.iloc[tune_index]

    return x_train, x_test, y_train, y_test