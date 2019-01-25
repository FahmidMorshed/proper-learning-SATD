import pandas as pd
import os


def calculate_results(input_filename, output_filename):
    found_results = False
    count = 0
    last_prec = 0
    last_recall = 0
    last_f1 = 0

    prev_line = ''
    first_line = True
    output = 'projectname,precision,recall,f1\n'

    with open(input_filename) as file:
        for line in file.readlines():
            line = line.strip()
            line = " ".join(line.split())

            if found_results:
                if count == 2:
                    tokens = line.split(" ")

                    last_prec = tokens[1]
                    last_recall = tokens[2]
                    last_f1 = tokens[3]

                    output += prev_line + ',' + last_prec + ',' + last_recall + ',' + last_f1 + '\n'

                    count = 0
                    found_results = False

                    continue
                else:
                    count += 1
                    continue

            if 'precision' in line and 'recall' in line and 'f1-score' in line:
                found_results = True
            else:
                prev_line = line



        with open(output_filename, 'w+') as f:
            f.write(output)
    print('Result processed')


def process_fastread(directory, output_file):
    output = ''
    perc = [i for i in range(0, 101)]
    os.makedirs('../results/' + output_file)

    for filename in os.listdir(directory):
        df = pd.DataFrame()
        df['perc_read'] = perc

        current_col = ''
        last_read_perc = 0
        col_val = []
        with open(directory + '/' + filename) as file:
            for line in file.readlines():
                if 'rig - ++' in line:
                    if len(col_val) > 0:
                        df[current_col] = pd.Series(col_val)
                    tokens = line.split()
                    current_col = tokens[9]
                    df[current_col] = 0
                    last_read_perc = 0
                    col_val = []
                    continue
                elif 'rig' in line:
                    continue
                line = line.strip()
                if 'Total Yes:' in line and 'Total No:' in line:
                    tokens = line.split()
                    total_yes = int(tokens[9])
                    total_no = int(tokens[13])
                    continue

                tokens = line.split()
                found = float(tokens[7].split(',')[0])
                read = float(tokens[8])
                read_prec = round(read / (total_yes + total_no) * 100, 0)
                if read_prec > last_read_perc:
                    last_read_perc = read_prec
                    col_val.append(round(found/ total_yes * 100, 2))
        if len(col_val) > 0:
            df[current_col] = pd.Series(col_val)
        df.to_csv('../results/' + output_file + '/' + filename[:-4] + '.csv')




