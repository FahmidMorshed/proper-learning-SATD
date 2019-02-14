import logging
import operator

from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def bellwether_test(satdd, dataset_name, clf, col_name, result_pd):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    logger = logging.getLogger(dataset_name)
    test_data = satdd.create_and_process_dataset([dataset_name], doInclude=True)
    print(str(test_data.true_count) + " | " + str(test_data.false_count))
    all_dataset_names = satdd.all_dataset_pd.projectname.unique()

    if len(result_pd) == 0:
        result_pd['dataset'] = all_dataset_names
    my_dict = {}
    for train_dataset_name in all_dataset_names:
        if train_dataset_name in dataset_name:
            continue
        train_data = satdd.create_and_process_dataset([train_dataset_name], doInclude=True)
        train_data.set_csr_mat()
        test_data.set_csr_mat(train_data.tfer)

        x_train = train_data.csr_mat
        y_train = train_data.data_pd['label'].tolist()
        x_test = test_data.csr_mat
        y_test = test_data.data_pd['label'].tolist()
        predict(clf, x_test, y_test, x_train, y_train, result_pd, train_dataset_name, my_dict)

    result_pd[col_name] = result_pd['dataset'].map(my_dict)
    result_pd.to_csv("../temp/" + dataset_name + ".csv")
    return result_pd

def predict(clf, x_test, y_test, x_train, y_train):

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=["no", "yes"])
    report = classification_report(y_test, y_pred)
    return float(report.strip().split()[12])


def bellwether(satdd, dataset_name, clf):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    logger = logging.getLogger(dataset_name)
    all_dataset_names = satdd.all_dataset_pd.projectname.unique()

    score_dict = {}
    # FOR EACH DATASET THAT IS NOT THE TEST SET
    for train_dataset_name in all_dataset_names:
        if train_dataset_name in dataset_name:
            continue
        train_data = satdd.create_and_process_dataset([train_dataset_name], doInclude=True)
        train_data.set_csr_mat()
        score = 0
        count = 0
        # predict for each dataset without the test dataset
        for tune_dataset_name in all_dataset_names:
            if tune_dataset_name in dataset_name or tune_dataset_name in train_dataset_name:
                continue
            count += 1
            tune_data = satdd.create_and_process_dataset([tune_dataset_name], doInclude=True)
            tune_data.set_csr_mat(train_data.tfer)

            x_train = train_data.csr_mat
            y_train = train_data.data_pd['label'].tolist()
            x_tune = tune_data.csr_mat
            y_tune = tune_data.data_pd['label'].tolist()

            score += predict(clf, x_tune, y_tune, x_train, y_train)
        score = round(score/count*100, 2)
        score_dict.update({train_dataset_name: score})

    return normalize(score_dict)

    #return score_dict


def normalize(score_dict):
    max_val = max(score_dict.items(), key=operator.itemgetter(1))[1]
    min_val = min(score_dict.items(), key=operator.itemgetter(1))[1]

    for key, val in score_dict.items():
        normalized_val = ((val - min_val) / (max_val - min_val))
        score_dict.update({key: normalized_val})

    return score_dict