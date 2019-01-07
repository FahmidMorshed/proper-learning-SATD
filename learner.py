from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
import numpy as np
import scipy.sparse as sp

import config

import logging

from optimizer import tune_with_flash, calc_f


def classify(training_data, test_data, project_name, all_datasets):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    logger = logging.getLogger(project_name)

    y_test = test_data.data_pd.loc[:, 'label']
    test_data.data_pd.loc[:, 'yes_vote'] = 0
    test_data.data_pd.loc[:, 'no_vote'] = 0

    x_train = training_data.csr_mat
    y_train = training_data.data_pd.loc[:, 'label']

    # SVM Test
    logger.info(project_name + " | TUNING STARTS...")
    svm_config = tune_svm(x_train, y_train, project_name)
    logger.info(project_name + " | TUNING ENDS")

    clf = SVC(C=svm_config[0], kernel=svm_config[1], gamma=svm_config[2], coef0=svm_config[3],
              probability=True, random_state=0)

    predict_active(clf, x_train, y_train, test_data, 'svm_pred', project_name)
    #
    # # NBM Test
    # clf = MultinomialNB(alpha=1.0)
    # predict(clf, x_train, y_train, test_data, 'nbm_pred', project_name)
    #
    # # KNN Test
    # for train_index, tune_index in sss.split(x_train, y_train):
    #     x_train_knn, x_tune_knn = x_train[train_index], x_train[tune_index]
    #     y_train_knn, y_tune_knn = y_train.iloc[train_index], y_train.iloc[tune_index]
    #
    #     best_accu = 0
    #     best_k = 1
    #     for i, k in enumerate(np.arange(1,9)):
    #         # Setup a knn classifier with k neighbors
    #         clf = KNeighborsClassifier(n_neighbors=k)
    #
    #         # Fit the model
    #         clf.fit(x_train_knn, y_train_knn)
    #
    #         accu = clf.score(x_tune_knn, y_tune_knn)
    #         if best_accu < accu:
    #             best_k = k
    #             best_accu = accu
    #
    # clf = KNeighborsClassifier(n_neighbors=best_k, weights='uniform', algorithm='auto', leaf_size=30, p=2)
    # predict(clf, x_train, y_train, test_data, 'knn_pred', project_name)
    #
    # # This part is just for understanding the prev work
    # # trying sub classifier of NBM
    # test_data.data_pd.loc[:, 'yes_vote'] = 0
    # test_data.data_pd.loc[:, 'no_vote'] = 0
    # for train_dataset_name in all_datasets:
    #     if train_dataset_name in project_name:
    #         continue
    #     training_ids = training_data.data_pd.loc[training_data.data_pd['projectname'] == train_dataset_name]
    #     training_ids = list(training_ids.index)
    #
    #     x_train = training_data.csr_mat[training_ids]
    #     y_train = training_data.data_pd.loc[training_ids, 'label']
    #
    #     clf = MultinomialNB(alpha=1.0)
    #     clf.fit(x_train, y_train)
    #     y_pred = clf.predict(test_data.csr_mat)
    #
    #     test_data.data_pd.loc[:, 'nbm_sub_pred'] = y_pred.tolist()
    #
    #     yes_ids = test_data.data_pd[test_data.data_pd.loc[:, "nbm_sub_pred"] == 'yes'].index
    #     test_data.data_pd.loc[yes_ids, 'yes_vote'] += 1
    #
    #     no_ids = test_data.data_pd[test_data.data_pd.loc[:, "nbm_sub_pred"] == 'no'].index
    #     test_data.data_pd.loc[no_ids, 'no_vote'] += 1
    #
    # for i, row in test_data.data_pd.iterrows():
    #     if row['yes_vote']>row['no_vote']:
    #         test_data.data_pd.at[i, 'nbm_sub_pred'] = 'yes'
    #     else:
    #         test_data.data_pd.at[i, 'nbm_sub_pred'] = 'no'
    #
    # print(project_name + " | nbm_sub_pred" + "\n" + get_report(test_data, 'nbm_sub_pred'))
    # # END OF SUB CLF

    test_data.data_pd.to_csv(config.LOG_FOLDER + '/temp/' + project_name + '_data_pd.csv')



def predict_active(clf, x_train, y_train, test_data, col_name, project_name,
                   target_fscore=config.ACTIVE_PREDICTION_TARGET):
    logger = logging.getLogger(project_name)

    current_fscore = 0
    eval = 0

    while current_fscore < target_fscore:
        loop_col_name = col_name + '_' + str(eval)
        old_score = current_fscore

        clf.fit(x_train, y_train)
        y_pred = clf.predict(test_data.csr_mat)
        y_pred_proba = clf.predict_proba(test_data.csr_mat)[:, 1]
        test_data.data_pd.loc[:, loop_col_name] = y_pred.tolist()
        test_data.data_pd.loc[:, loop_col_name + '_proba'] = y_pred_proba.tolist()

        report = get_report(test_data, loop_col_name)
        current_fscore = get_f_score(test_data, loop_col_name)
        if current_fscore == old_score:
            print(project_name + "| NO IMPROVEMENT | F Score: " + str(current_fscore) + " | eval: " + str(eval))
            logger.info("| NO IMPROVEMENT | F Score: " + str(current_fscore) + " | eval: " + str(eval))
            break

        print(project_name + " | " + loop_col_name + "\n" + report + "\nF Score: " + str(current_fscore))
        logger.info(loop_col_name + "\n" + report + "\nF Score: " + str(current_fscore))

        if config.ACTIVE_PREDICTION is False:
            break

        neg_ids = np.where(test_data.data_pd[loop_col_name + '_proba'] < config.ACTIVE_PREDICTION_DELTA)[0]
        pos_ids = np.where(test_data.data_pd[loop_col_name + '_proba'] > (1-config.ACTIVE_PREDICTION_DELTA))[0]

        new_pos_mat = test_data.csr_mat[pos_ids]
        new_neg_mat = test_data.csr_mat[neg_ids]
        new_pos_y = test_data.data_pd.loc[pos_ids, 'label']
        new_neg_y = test_data.data_pd.loc[neg_ids, 'label']

        x = sp.vstack([x_train, new_pos_mat])
        x = sp.vstack([x, new_neg_mat])

        y = y_train.append(new_pos_y)
        y = y.append(new_neg_y)

        x_train = x
        y_train = y

        logger.info("Active Delta: " + str(config.ACTIVE_PREDICTION_DELTA))
        logger.info("Total Certain Pos: " + str(len(pos_ids)) + " | Total Certain Neg: " + str(len(neg_ids)))

        eval += 1

def get_report(test_data, field):
    y_pred = test_data.data_pd.loc[:, field]
    y_test = test_data.data_pd.loc[:, 'label']
    return classification_report(y_test,y_pred)

def get_f_score(test_data, field):
    y_pred = test_data.data_pd.loc[:, field]
    y_test = test_data.data_pd.loc[:, 'label']
    mat = confusion_matrix(y_test, y_pred)
    return calc_f(mat)


def tune_svm(x_train, y_train, project_name):
    if config.TUNE_SVM is False:
        return [1, 'rbf', 'auto', 0]

    best_conf = [12.23, 'rbf', 0.82, 0.28]

    sss = StratifiedShuffleSplit(n_splits=config.STRATIFIED_SPLIT, test_size=config.STRATIFIED_TEST_SIZE_RATIO,
                                 random_state=0)
    for train_index, tune_index in sss.split(x_train, y_train):
        x_train_flash, x_tune_flash = x_train[train_index], x_train[tune_index]
        y_train_flash, y_tune_flash = y_train.iloc[train_index], y_train.iloc[tune_index]
        best_conf = tune_with_flash(x_train_flash, y_train_flash, x_tune_flash, y_tune_flash, project_name, random_seed=1)

    return best_conf