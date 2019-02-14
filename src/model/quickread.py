import time
from collections import Counter

from pandas.core.common import SettingWithCopyWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd
import logging

from model.quickmar import QUICKMAR


def quick_learning(dataset, dataset_name, train_dataset, stopat=.95, error=None, uncertain_limit=100, seed=0, enough=30, atleast=100
                    , step=20, enable_est=True):
    logger = logging.getLogger(dataset_name)
    print("True: " + str(dataset.true_count) + " | False: " + str(dataset.false_count))
    logger.info("Total Yes: " + str(dataset.true_count) + " | Total No: " + str(dataset.false_count))

    clf = SGDClassifier(random_state=1, loss='log')

    #clf.fit(train_dataset.csr_mat, train_dataset.data_pd['label'].tolist())

    stopat = float(stopat)
    starting = 5

    np.random.seed(seed)

    read = QUICKMAR(dataset, enough, atleast, clf, step, enable_est=enable_est, stopat=stopat)

    print("Stop: " + str(stopat) + " | Error: " + str(error) + " | UncertainLim: " + str(uncertain_limit)
          + " | Step: " + str(read.step) + " | Agressive Undersampling: " + str(read.enough))

    num_of_total_pos = read.get_allpos()
    total = dataset.true_count + dataset.false_count
    target = int(total * stopat)

    loop_count = 1
    no_improvement_count = 0
    old_pos = 0
    est_found = 0

    read.BM25(['TODO'])
    while True:
        no_improvement_count = 0
        pos, neg, total = read.get_numbers()
        if pos > old_pos:
            #old_pos = pos
            no_improvement_count = pos - old_pos
            read.scale = min(read.scale + no_improvement_count, 5)
        else:
            no_improvement_count += 1
            read.scale = max(read.scale - 1, 1)

        # if no_improvement_count >= 9:
        #     logger.info("%d, %d  %d" % (pos, pos + neg, int(read.est_num * stopat)))
        #     print(dataset_name + " NO IMPROVEMENT after 3 random loops. EXITING. Current estimate: " + str(read.est_num))
        #     logger.info("IGNORE " + dataset_name + " NO IMPROVEMENT after 3 random loops. EXITING. Current estimate: " + str(read.est_num))
        #     break

        loop_count += 1
        logger.info("%d, %d  %d" % (pos, pos + neg, int(read.est_num * stopat)))
        read.results.append([pos + neg, pos])

        if pos + neg >= total:
            break

        if pos < starting:
            for id in read.BM25_get():
                read.code_error(id, error=error)

        # for id in read.get_random_pos():
            #     read.code_error(id, error=error)
            # for id in read.get_random_negs():
            #     read.code_error(id, error=error)
        else:
            a, b, c, d = read.train(weighting=True, pne=True)
            if enable_est:
                if int(stopat * read.est_num) <= pos:
                    break

            elif pos >= target:
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

            # target2 = read.estimate_knee()
            #print("TARGET " + str(read.est_num))


    pos, neg, total = read.get_numbers()
    print("Positive %d, Total Looked %d, Should have found %d" % (pos, pos + neg, int(read.get_allpos() * stopat)))

    print_summary(read.body, dataset_name)

    return round(pos / dataset.true_count, 2)
    # #set_trace()
    # return read


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

    print(dataset_name + " | Total: " + str(total) + " | Percent Checked: " + str(round(total_checked/total, 2)) + " | Total Yes Data: "
          + str(total_yes) + " | Percent Yes Checked: " + str(round(total_yes_checked/total_yes, 2)))

    logger.info("IGNORE " + dataset_name + " | Total: " + str(total) + " | Percent Checked: " + str(
        round(total_checked / total, 2)) + " | Total Yes Data: "
          + str(total_yes) + " | Percent Yes Checked: " + str(round(total_yes_checked / total_yes, 2)))




# Only for testing purpose of stopping with estimation on plain SVM or SGDClfs to check how they perform
def baseline_qread(train_data, test_data, dataset_name, clf, stopat=1, error=None, step=10, atleast=40, random_stop=None):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    logger = logging.getLogger(dataset_name)

    logger.info("Total Yes: " + str(test_data.true_count) + " | Total No: " + str(test_data.false_count))

    qread(clf, test_data, train_data, dataset_name, step, stopat, atleast, random_stop)

    print_summary(test_data.data_pd, dataset_name)

def qread(clf, test_data, train_data, dataset_name, step, stopat, atleast, random_stop):
    logger = logging.getLogger(dataset_name)
    test_data.data_pd.loc[:, 'time'] = 0
    test_data.data_pd.loc[:, 'fixed'] = 0
    test_data.data_pd.loc[:, 'count'] = 0
    test_data.data_pd.loc[:, 'code'] = 'undetermined'
    test_data.data_pd.loc[:, 'proba'] = 0


    x_train = train_data.csr_mat
    y_train = train_data.data_pd['label'].tolist()

    clf.fit(x_train, y_train)

    y_pred_proba = clf.predict_proba(test_data.csr_mat)
    y_pred_proba = y_pred_proba[:, 1]
    test_data.data_pd['proba'] = y_pred_proba.tolist()

    test_data.data_pd = test_data.data_pd.sort_values(by=["proba"], ascending=False)

    count = 1
    old_pos = 0
    no_improvement_count = 1
    scale = 1
    target = 100
    for index, row in test_data.data_pd.iterrows():
        test_data.data_pd.at[index, 'code'] = row['label']  # NO error
        test_data.data_pd.at[index, 'time'] = time.time()
        if count % step == 0:
            pos = len(test_data.data_pd[test_data.data_pd["code"] == 'yes'])
            neg = len(test_data.data_pd[test_data.data_pd["code"] == 'no'])
            if not random_stop:
                est_num, proba = estimate_curve(test_data, clf, num_neg=((atleast+neg)*scale))
                target = int(est_num * stopat)
                if pos >= target:
                    break;
                if pos > old_pos:
                    # old_pos = pos
                    no_improvement_count = pos - old_pos
                    scale = min(scale + no_improvement_count, 5)
                else:
                    no_improvement_count += 1
                    scale = max(scale-1, 1)
            # Random Stop
            else:
                if pos > old_pos:
                    old_pos = pos
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count > random_stop:
                        break
            logger.info("%d, %d  %d" % (pos, pos + neg, target))
        count += 1


def estimate_curve(dataset, clf, num_neg=0, scale=1):
    from sklearn import linear_model

    def prob_sample(probs):
        order = np.argsort(probs)[::-1]
        count = 0
        can = []
        sample = []
        for i, x in enumerate(probs[order]):
            count = count + x
            can.append(order[i])
            if count >= 1:
                sample.append(can[0])
                count = 0
                can = []
        return sample

    poses = np.where(np.array(dataset.data_pd['code']) == "yes")[0]
    negs = np.where(np.array(dataset.data_pd['code']) == "no")[0]

    poses = np.array(poses)[np.argsort(np.array(dataset.data_pd['time'])[poses])[:]]
    negs = np.array(negs)[np.argsort(np.array(dataset.data_pd['time'])[negs])[:]]

    ###############################################
    prob1 = clf.decision_function(dataset.csr_mat)
    prob = np.array([[x] for x in prob1])

    y = np.array([1 if x == 'yes' else 0 for x in dataset.data_pd['code']])
    y0 = np.copy(y)

    pool = np.where(np.array(dataset.data_pd['code']) == "undetermined")[0]

    all = list(set(poses) | set(negs) | set(pool))

    pos_num_last = Counter(y0)[1]

    lifes = 1
    life = lifes

    while (True):
        C = Counter(y[all])[1] / (num_neg)
        es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=C)

        es.fit(prob[all], y[all])
        pos_at = list(es.classes_).index(1)

        pre = es.predict_proba(prob[pool])[:, pos_at]

        y = np.copy(y0)

        sample = prob_sample(pre)
        for x in pool[sample]:
            y[x] = 1

        pos_num = Counter(y)[1]

        if pos_num-pos_num_last < 2:
            life = life - 1
            if life == 0:
                break
        else:
            life = lifes
        pos_num_last = pos_num
    esty = pos_num
    pre = es.predict_proba(prob)[:, pos_at]

    return esty, pre


def todo_read(test_data, dataset_name, stopat, step):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    logger = logging.getLogger(dataset_name)

    logger.info("Total Yes: " + str(test_data.true_count) + " | Total No: " + str(test_data.false_count))
    test_data.data_pd.loc[:, 'code'] = 'undetermined'

    count = 0
    total_pos = len(test_data.data_pd.loc[test_data.data_pd['label'] == 'yes'])
    target = total_pos * stopat

    ids = []
    count = 0
    for index, row in test_data.data_pd.iterrows():
        if 'TODO' in row['commenttext']:
            ids.append(count)
        else:
            ids.append(99999)
    test_data.data_pd.loc[:, 'todo'] = ids

    test_data.data_pd = test_data.data_pd.sort_values(by=["todo"], ascending=True)

    for index, row in test_data.data_pd.iterrows():
        if 'TODO' in row['commenttext']:
            test_data.data_pd.at[index, 'code'] = 'yes'
        else:
            test_data.data_pd.at[index, 'code'] = 'no'

        if count % step == 0:
            pos = len(test_data.data_pd[test_data.data_pd["code"] == 'yes'])
            neg = len(test_data.data_pd[test_data.data_pd["code"] == 'no'])
            logger.info("%d, %d  %d" % (pos, pos + neg, target))
            if pos >= target:
                break;
        count += 1

    print_summary(test_data.data_pd, dataset_name)