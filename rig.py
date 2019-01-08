import log
from learner import classify
from loader import SATDD

import config

import multiprocessing as mp
from joblib import Parallel, delayed

import logging
logger = logging.getLogger(__name__)

def run_rig(filename):
    logger.info("++++++++++NEW RIG++++++++++")

    satdd = SATDD()
    satdd = satdd.load_data(filename)

    cross_project(satdd)


def cross_project(satdd):
    all_datasets = satdd.all_dataset_pd.projectname.unique()
    logger.info("=====CROSS PROJECT=====")

    num_cpu = mp.cpu_count()
    logger.info("Number of CPU: " + str(num_cpu))

    Parallel(n_jobs=num_cpu)(delayed(run_rig_on_project)(satdd, dataset) for dataset in all_datasets)

    logger.info("======END CROSS PROJECT======")

def run_rig_on_project(satdd, project_name):
    logger = log.setup_custom_logger(project_name)
    print(project_name + " STARTS...")


    training_data = satdd.create_and_process_dataset([project_name],
                                                     doInclude=False)
    # no need to give a tfidf, will calculate itself
    training_data.set_csr_mat(max_f=config.MAX_FEATURE, stop_w=config.STOP_WORDS)
    test_data = satdd.create_and_process_dataset([project_name], doInclude=True)
    # need to give the tfidf from training set, will just use transform to create csr_matrix
    test_data.set_csr_mat(max_f=config.MAX_FEATURE, stop_w=config.STOP_WORDS, tfer=training_data.tfer)


    # Logging rig descriptions
    config.print_config(project_name)

    # Logging Ground Truth
    logger.info("TRAINING DATA: TRUE: " + str(training_data.true_count) + " | FALSE: "
                + str(training_data.false_count))
    logger.info("TEST DATA: TRUE: " + str(test_data.true_count) + " | FALSE: "
                + str(test_data.false_count))

    classify(training_data, test_data, project_name, satdd.all_dataset_pd.projectname.unique())

    print(project_name + " is DONE")

    logger.info(project_name + " | END PROJECT=====")



