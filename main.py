import config
from result import calculate_results
from rig import run_rig

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # calculate_results('fea20_eng.csv')
    run_rig('td_2016.csv')
