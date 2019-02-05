from sklearn.metrics import classification_report

from core.rig import run_rig
from util.result import calculate_results, process_fastread, process_fastread_recall, process_fastread_new
import pandas as pd
SHUFFLE_DATA = False
import numpy as np


if __name__ == "__main__":
    #calculate_results('../logs/1_18_19_dt_ensemble_gini_max50.txt', '../results/1_18_9_dt_ensemble_gini_max50.csv')
    run_rig('td_2016.csv')
    #process_fastread_new("../logs/2_5_fastread", '2_5_fastread')
    #process_fastread_recall("../logs/2_4_fastread_test" + '/log.txt', "../logs/2_4_fastread_test" + '/log_sum95.txt')

    print('End')






