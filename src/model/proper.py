from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVC

from model.ensemble import dt_ensemble
import pandas as pd
import numpy as np

from model.fastread import baseline_fastread
from preprocess.loader import DATASET


def proper(satdd, dataset_name):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    all_dataset_names = satdd.all_dataset_pd.projectname.unique()

    data_pd = pd.DataFrame(columns=['projectname', 'commenttext', 'label'])

    base_data_pd = pd.DataFrame(columns=['projectname', 'commenttext', 'label'])

    for tune_dataset_name in all_dataset_names:
        if tune_dataset_name in dataset_name:
            continue
        predicted_data = dt_ensemble(satdd, tune_dataset_name, ignore_dataset=dataset_name)

        a = predicted_data.data_pd.loc[((predicted_data.data_pd['yes_vote'] == 0) &
                                   (predicted_data.data_pd['label'] == 'no')) |
                                       ((predicted_data.data_pd['no_vote'] == 0) &
                                        (predicted_data.data_pd['label'] == 'yes'))]


        b = predicted_data.data_pd.loc[
                                   (predicted_data.data_pd['label'] == 'no') |
                                        (predicted_data.data_pd['label'] == 'yes')]

        if data_pd.empty:
            data_pd = a
            base_data_pd = b
        else:
            data_pd = pd.concat([data_pd, a])
            base_data_pd = pd.concat([base_data_pd, b])

    data_pd.reset_index(drop=True)

    train_data = DATASET(data_pd)
    train_data.set_csr_mat()
    test_data = satdd.create_and_process_dataset([dataset_name], doInclude=True)
    print("TRAIN: " + str(train_data.true_count) + " | " + str(train_data.false_count))
    print("TEST: " + str(test_data.true_count) + " | " + str(test_data.false_count))

    test_data.set_csr_mat(train_data.tfer)

    clf = SVC(probability=True, random_state=0)
    baseline_fastread(train_data, test_data, dataset_name, clf)
    #
    # x_train = train_data.csr_mat
    # y_train = train_data.data_pd['label'].tolist()
    # x_test = test_data.csr_mat
    # y_test = test_data.data_pd['label'].tolist()
    #
    #
    # clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)

    print("PROPER " + dataset_name)
    print(classification_report(y_test, y_pred))



    base_data_pd.reset_index(drop=True)
    train_data = DATASET(base_data_pd)
    train_data.set_csr_mat()
    test_data = satdd.create_and_process_dataset([dataset_name], doInclude=True)
    print("TRAIN: " + str(train_data.true_count) + " | " + str(train_data.false_count))
    print("TEST: " + str(test_data.true_count) + " | " + str(test_data.false_count))

    test_data.set_csr_mat(train_data.tfer)

    x_train = train_data.csr_mat
    y_train = train_data.data_pd['label'].tolist()
    x_test = test_data.csr_mat
    y_test = test_data.data_pd['label'].tolist()

    clf = LinearSVC(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print("PROPER BASE " + dataset_name)
    print(classification_report(y_test, y_pred))

    print('End Tuning')