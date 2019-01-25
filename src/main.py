
from core.rig import run_rig
from util.result import calculate_results, process_fastread
import pandas as pd
SHUFFLE_DATA = False



if __name__ == "__main__":
    #calculate_results('../logs/1_18_19_dt_ensemble_gini_max50.txt', '../results/1_18_9_dt_ensemble_gini_max50.csv')
    run_rig('td_2016.csv')
    #process_fastread("../logs/1_24_fastread", '1_24_fastread')

    df = pd.DataFrame()
    perc = [i for i in range(0, 101)]
    df['Percent Read'] = perc
    print('End')






