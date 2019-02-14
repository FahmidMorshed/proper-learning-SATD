from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import util.log as log
from model.bellwether import bellwether_test, bellwether
from model.ensemble import dt_ensemble, read_ensemble
from model.fastread import active_learning, baseline_fastread, random_read
from model.proper import proper
from model.quickread import quick_learning, baseline_qread, todo_read
from model.total import dt_total
from preprocess.loader import SATDD
import pandas as pd

import multiprocessing as mp
from joblib import Parallel, delayed


def run_rig(filename):
    satdd = SATDD()
    satdd = satdd.load_data(filename)

    cross_project(satdd)


def cross_project(satdd):
    all_datasets = satdd.all_dataset_pd.projectname.unique()

    num_cpu = mp.cpu_count()

    Parallel(n_jobs=num_cpu-3)(delayed(fast_read)(satdd, dataset) for dataset in all_datasets)


def ensemble(satdd, dataset_name):
    print(dataset_name + " ENSEMBLE STARTS...")

    dt_ensemble(satdd, dataset_name)

def total(satdd, dataset_name):
    print(dataset_name + " STARTS...")
    dt_total(satdd, dataset_name)

def fast_read(satdd, dataset_name):
    print(dataset_name + " STARTS...")
    target = .95
    logger = log.setup_custom_logger(dataset_name)
    logger.info(dataset_name)

    print("CALLING ENSEMBLE... ")
    test_data = satdd.create_and_process_dataset([dataset_name], doInclude=True)#dt_ensemble(satdd, dataset_name, clf_name="DT")

    train_data = satdd.create_and_process_dataset([dataset_name], doInclude=False)

    train_data.set_csr_mat()
    test_data.set_csr_mat(train_data.tfer)




    # logger.info("++++NEW CLF++++ FASTREAD")
    enough = int(test_data.true_count / 3)
    atleast = int(test_data.true_count + test_data.false_count / 500)
    step = int((test_data.false_count + test_data.true_count)/200)
    atleast = step * 20
    uncertain_limit = step * 3
    # target_found = active_learning(test_data, dataset_name, enough=enough, atleast=atleast, stopat=target,
    #                 uncertain_limit=uncertain_limit, step=step, enable_est=True)

    # print(dataset_name + " | TARGET RECALL FOUND: " + str(target_found))



    logger.info("++++NEW CLF++++ QUICKREAD_95")
    target_found = quick_learning(test_data, dataset_name, train_data, enough=enough, atleast=atleast, stopat=target,
                                     uncertain_limit=uncertain_limit, step=step, enable_est=True)

    # logger.info("++++NEW CLF++++ TODOREAD")
    # todo_read(test_data, dataset_name, stopat=target, step=step)

    # logger.info("++++NEW CLF++++ ENSEMBLE_DT")
    # read_ensemble(test_data.data_pd, dataset_name, stopat=target)
    # nbm_ensemble_data = dt_ensemble(satdd, dataset_name)
    #
    # logger.info("++++NEW CLF++++ ENSEMBLE_NBM")
    # read_ensemble(nbm_ensemble_data.data_pd, dataset_name, stopat=target)

    test_data = satdd.create_and_process_dataset([dataset_name],
                                                 doInclude=True)  # dt_ensemble(satdd, dataset_name, clf_name="DT")

    train_data = satdd.create_and_process_dataset([dataset_name], doInclude=False)

    train_data.set_csr_mat()
    test_data.set_csr_mat(train_data.tfer)

    clf = SGDClassifier(random_state=1, loss='log')  # # #
    logger.info("++++NEW CLF++++ SGD")
    baseline_qread(train_data, test_data, dataset_name, clf, stopat=target, step=step, atleast=atleast)

    test_data = satdd.create_and_process_dataset([dataset_name],
                                                 doInclude=True)  # dt_ensemble(satdd, dataset_name, clf_name="DT")

    train_data = satdd.create_and_process_dataset([dataset_name], doInclude=False)

    train_data.set_csr_mat()
    test_data.set_csr_mat(train_data.tfer)

    clf = SGDClassifier(random_state=1, loss='log')  # # #
    logger.info("++++NEW CLF++++ SGD_RAND05")
    baseline_qread(train_data, test_data, dataset_name, clf, stopat=target, step=step, atleast=atleast, random_stop=5)

    test_data = satdd.create_and_process_dataset([dataset_name],
                                                 doInclude=True)  # dt_ensemble(satdd, dataset_name, clf_name="DT")

    train_data = satdd.create_and_process_dataset([dataset_name], doInclude=False)

    train_data.set_csr_mat()
    test_data.set_csr_mat(train_data.tfer)

    clf = SGDClassifier(random_state=1, loss='log')  # # #
    logger.info("++++NEW CLF++++ SGD_RAND10")
    baseline_qread(train_data, test_data, dataset_name, clf, stopat=target, step=step, atleast=atleast, random_stop=10)


    # print("++++NEW CLF++++ RANDOM")
    # logger.info("++++NEW CLF++++ RANDOM")
    # random_read(test_data, dataset_name, stopat=target)
    #
    # clf = MultinomialNB(alpha=1.0)  # # #
    # logger.info("++++NEW CLF++++ NBM")
    # baseline_fastread(train_data, test_data, dataset_name, clf, stopat=target)
    #
    # clf = KNeighborsClassifier()
    # logger.info("++++NEW CLF++++ KNN")
    # baseline_fastread(train_data, test_data, dataset_name, clf, stopat=target)
    #
    # clf = DecisionTreeClassifier(random_state=0)
    # logger.info("++++NEW CLF++++ DT")
    # baseline_fastread(train_data, test_data, dataset_name, clf, stopat=target)
    #
    # clf = SVC(probability=True, random_state=0)
    # logger.info("++++NEW CLF++++ SVM")
    # baseline_qread(train_data, test_data, dataset_name, clf, stopat=target, step=step)

    # clf = SVC(kernel='linear', probability=True, random_state=0)
    # logger.info("++++NEW CLF++++ SVM_LINER")
    # baseline_fastread(train_data, test_data, dataset_name, clf, stopat=target)





def proper_learning(satdd, dataset_name):
    print(dataset_name + " PROPER LEARNING STARTS...")
    logger = log.setup_custom_logger(dataset_name)
    logger.info(dataset_name)

    proper(satdd, dataset_name)



def bellwether_rig(satdd, dataset_name):
    print(dataset_name + " STARTS BELLWETHER RIG...")
    logger = log.setup_custom_logger(dataset_name)
    logger.info(dataset_name)



    clf = DecisionTreeClassifier(random_state=0)
    score_dict = bellwether(satdd, dataset_name, clf)
    dt_ensemble(satdd, dataset_name, bellwether_weights=score_dict)
    dt_ensemble(satdd, dataset_name, bellwether_weights=None)



    # clf = SGDClassifier(random_state=0, loss='log', class_weight='balanced')
    # score_dict = bellwether(satdd, dataset_name, clf)
    # print(score_dict)
    # clf = SVC(kernel='linear', random_state=0)
    # score_dict = bellwether(satdd, dataset_name, clf)
    # print(score_dict)
    #
    # print("++++NEW CLF++++ DT")
    # clf = SVC(kernel='linear', random_state=0)
    # result_pd = pd.DataFrame()
    # result_pd = bellwether_test(satdd, dataset_name, clf, "SVC_L", result_pd)
    # clf = DecisionTreeClassifier(random_state=0)
    # result_pd = bellwether_test(satdd, dataset_name, clf, "DT", result_pd)
    # clf = SGDClassifier(random_state=0, loss='log', class_weight='balanced')
    # result_pd = bellwether_test(satdd, dataset_name, clf, "SGD_LOG_B", result_pd)