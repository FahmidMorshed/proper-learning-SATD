import logging

import os

LOG_FOLDER = '../logs/1_22_fastread_base_knn'
def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)
        #os.makedirs(LOG_FOLDER + '/temp')

    handler = logging.FileHandler(LOG_FOLDER + "/" + name + '.log')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger