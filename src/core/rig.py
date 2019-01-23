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

    Parallel(n_jobs=num_cpu-3)(delayed(fast_read_baselines)(satdd, dataset) for dataset in all_datasets)


def ensemble(satdd, dataset_name):
    print(dataset_name + " STARTS...")
    dt_ensemble(satdd, dataset_name)

def total(satdd, dataset_name):
    print(dataset_name + " STARTS...")
    dt_total(satdd, dataset_name)

def fast_read(satdd, dataset_name):
    print(dataset_name + " STARTS...")
    logger = log.setup_custom_logger(dataset_name)
    logger.info(dataset_name)
    dataset = satdd.create_and_process_dataset([dataset_name], doInclude=True)
    dataset.set_csr_mat()
    print("True: " + str(dataset.true_count) + " | False: " + str(dataset.false_count))
    logger.info("Total Yes: " + str(dataset.true_count) + " | Total No: " + str(dataset.false_count))

    active_learning(dataset, dataset_name)

def fast_read_baselines(satdd, dataset_name):
    print(dataset_name + " STARTS...")
    logger = log.setup_custom_logger(dataset_name)
    logger.info(dataset_name)
    baseline_fastread(satdd, dataset_name)