import csv


def write_erros(error_log, filename='default_output.csv'):

    header = ['it', 'error']

    with open('logs/' + filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(error_log)