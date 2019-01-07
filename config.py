import logging


MAX_FEATURE = .1
STOP_WORDS = 'english'

POOL_SIZE = 10000
INIT_POOL_SIZE = 10
BUDGET = 10

LOG_FOLDER = 'logs/1_5_19_fea10_eng'

STRATIFIED_TEST_SIZE_RATIO = .2
STRATIFIED_SPLIT = 1

TUNE_SVM = True

ACTIVE_PREDICTION_DELTA = .1
ACTIVE_PREDICTION_TARGET = 1
ACTIVE_PREDICTION = True

def print_config(project_name):
    logger = logging.getLogger(project_name)

    logger.info("DATASET: " + project_name + " | MAX_FEATURE: " + str(MAX_FEATURE) + " | STOP_WORDS: " + str(STOP_WORDS)
                + " | POOL_SIZE: " + str(POOL_SIZE) + " | INIT_POOL: "
                + str(INIT_POOL_SIZE) + " | BUDGET: " + str(BUDGET) + "\n"
                + "STRATIFIED_TEST_SIZE_RATIO: " + str(STRATIFIED_TEST_SIZE_RATIO) + " | TUNE SVM: " + str(TUNE_SVM)
                + " | ACTIVE PREDICTION DELTA: " + str(ACTIVE_PREDICTION_DELTA) + " | ACTIVE PREDICTION TARGET: "
                + str(ACTIVE_PREDICTION_TARGET))


def set_experiment(max_feature, stop_words):
    global MAX_FEATURE
    global STOP_WORDS
    global LOG_FOLDER

    MAX_FEATURE = max_feature
    STOP_WORDS = stop_words

    LOG_FOLDER = 'logs/1_5_19_fea' + str(int(max_feature * 100)) + '_stop' + str(stop_words)
    print(LOG_FOLDER)

