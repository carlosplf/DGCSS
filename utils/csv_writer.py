import csv


def write_erros(error_log, filename="logs/default_output.csv"):
    header = ["epoch", "loss", "c_loss", "gae_loss"]

    with open(filename, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(error_log)
