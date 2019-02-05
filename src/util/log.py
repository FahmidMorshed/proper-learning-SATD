import logging

import os

LOG_FOLDER = '../logs/2_5_fastread_95_test'
def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    if not os.path.isdir(LOG_FOLDER):
        try:
            os.makedirs(LOG_FOLDER)
        except FileExistsError:
            print("Log folder already exists!")

    handler = logging.FileHandler(LOG_FOLDER + "/" + name + '.log')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger