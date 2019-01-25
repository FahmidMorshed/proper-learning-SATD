from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import util.log as log
from model.ensemble import dt_ensemble
from model.fastread import active_learning, baseline_fastread
from model.total import dt_total
from preprocess.loader import SATDD


import multiprocessing as mp
from joblib import Parallel, delayed


def run_rig(filename):
    satdd = SATDD()
    satdd = satdd.load_data(filename)

    cross_project(satdd)


def cross_project(satdd):
    all_datasets = satdd.all_dataset_pd.projectname.unique()

    num_cpu = mp.cpu_count()

    #fast_read_baselines(satdd) # TOTAL DATASET, 50% test
    Parallel(n_jobs=num_cpu-7)(delayed(ensemble)(satdd, dataset) for dataset in all_datasets)


def ensemble(satdd, dataset_name):
    print(dataset_name + " STARTS...")

    dt_ensemble(satdd, dataset_name)

def total(satdd, dataset_name):
    print(dataset_name + " STARTS...")
    dt_total(satdd, dataset_name)

def fast_read_baselines(satdd, dataset_name=None):
    if dataset_name:
        dataset = satdd.create_and_process_dataset([dataset_name], doInclude=True)
    else:
        dataset_name = "TOTAL"
        dataset = satdd.create_and_process_dataset()
    print(dataset_name + " STARTS...")
    logger = log.setup_custom_logger(dataset_name)
    logger.info(dataset_name)


    # IF WE ARE USING SINGLE DATASET, and x% test train on that
    # train_data = satdd.create_and_process_dataset([dataset_name], doInclude=False)
    test_data, train_data = dataset.make_test_train_on_same_dataset()

    train_data.set_csr_mat()
    test_data.set_csr_mat(train_data.tfer)

    logger.info("++++NEW CLF++++ FASTREAD")
    active_learning(test_data, dataset_name)

    clf = MultinomialNB(alpha=1.0)  # # #
    logger.info("++++NEW CLF++++ NBM")
    baseline_fastread(train_data, test_data, dataset_name, clf)

    clf = KNeighborsClassifier()
    logger.info("++++NEW CLF++++ NBM")
    baseline_fastread(train_data, test_data, dataset_name, clf)

    clf = SVC(probability=True, random_state=0)
    logger.info("++++NEW CLF++++ SVM")
    baseline_fastread(train_data, test_data, dataset_name, clf)

    clf = DecisionTreeClassifier(random_state=0)
    logger.info("++++NEW CLF++++ DT")
    baseline_fastread(train_data, test_data, dataset_name, clf)