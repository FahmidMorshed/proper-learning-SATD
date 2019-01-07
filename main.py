import config
from result import calculate_results
from rig import run_rig

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    #calculate_results('feaNone_none.csv')
    run_rig('td_2016.csv')

    # for i in [.3, .2, .1, .05]:
    #     config.set_experiment(max_feature=i, stop_words='english')
    #     logging.basicConfig(filename=config.LOG_FOLDER + '.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
    #     logger.info("STARTING EXPERIMENT")
    #     logger.info("LOG FOLDER: " + config.LOG_FOLDER)
    #     run_rig('td_2016.csv')
    #     print("RIG DONE ENGLISH")
    #
    #     config.set_experiment(max_feature=i, stop_words=None)
    #     logger.info("STARTING EXPERIMENT")
    #     logger.info("LOG FOLDER: " + config.LOG_FOLDER)
    #     run_rig('td_2016.csv')
    #     print("RIG DONE NONE")
    #
    #
    #
    # #