import pandas as pd
import os
import config


def calculate_results(output_filename):
    result_pd = pd.DataFrame.from_csv('results/huang.csv')


    precision = []
    recall = []
    f1 = []

    if not os.path.exists('results/temp'):
        os.makedirs('results/temp')

    for filename in os.listdir(config.LOG_FOLDER):
        if filename in 'temp':
            continue

        found_results = False
        count = 0

        last_prec = 0
        last_recall = 0
        last_f1 = 0

        output = 'precision,recall,f1\n'

        with open(config.LOG_FOLDER + '/' + filename) as file:
            for line in file.readlines():
                line = line.strip()
                line = " ".join(line.split())

                if found_results:
                    if count == 2:
                        tokens = line.split(" ")

                        last_prec = tokens[1]
                        last_recall = tokens[2]
                        last_f1 = tokens[3]

                        output += last_prec + ',' + last_recall + ',' + last_f1 + '\n'

                        count = 0
                        found_results = False

                        continue
                    else:
                        count += 1
                        continue

                if 'precision' in line and 'recall' in line and 'f1-score' in line:
                    found_results = True



            with open('results/temp/' + filename[:-4] + '.csv', 'w+') as f:
                f.write(output)

        precision.append(last_prec)
        recall.append(last_recall)
        f1.append(last_f1)

    result_pd['proper_prec'] = precision
    result_pd['proper_recall'] = recall
    result_pd['proper_f1'] = f1

    result_pd = result_pd.convert_objects(convert_numeric=True)

    result_pd['f1_improvement'] = result_pd['proper_f1'] - result_pd['huang_f1']

    # Rearranging columns and rows
    result_pd = result_pd[['proper_prec', 'huang_prec', 'proper_recall', 'huang_recall', 'proper_f1', 'huang_f1', 'f1_improvement']]
    result_pd = result_pd.reindex(["apache-jmeter-2.10", "argouml", "columba-1.4-src",
                                   'hibernate-distribution-3.3.2.GA', 'jEdit-4.2', 'jfreechart-1.0.19', 'jruby-1.4.0',
                                   'sql12', 'apache-ant-1.7.0', 'emf-2.4.1'])


    result_pd.to_csv('results/' + output_filename)

    print('Result processed')