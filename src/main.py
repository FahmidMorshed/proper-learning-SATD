
from core.rig import run_rig
from util.result import calculate_results, process_fastread

SHUFFLE_DATA = False



if __name__ == "__main__":
    #calculate_results('../logs/1_18_19_dt_ensemble_gini_max50.txt', '../results/1_18_9_dt_ensemble_gini_max50.csv')
    run_rig('td_2016.csv')
    #process_fastread("../logs/1_22_fastread_base_knn", '1_22_fastread_base_knn')






