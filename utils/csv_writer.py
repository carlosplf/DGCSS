import csv


def write_loss(loss_log, filename="logs/default_output.csv"):
    header = ["epoch", "loss", "c_loss", "gae_loss"]

    with open(filename, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(loss_log)
