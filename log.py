import logging

import config
import os

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    if not os.path.exists(config.LOG_FOLDER):
        os.makedirs(config.LOG_FOLDER)
        os.makedirs(config.LOG_FOLDER + '/temp')

    handler = logging.FileHandler(config.LOG_FOLDER + "/" + name + '.log')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger