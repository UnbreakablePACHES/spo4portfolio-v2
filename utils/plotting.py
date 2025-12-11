import matplotlib.pyplot as plt
import csv

def plot_curve(csv_path, save_path, title="Loss Curve"):
    epochs = []
    values = []

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            epochs.append(int(row[0]))
            values.append(float(row[1]))

    plt.figure(figsize=(6,4))
    plt.plot(epochs, values, marker="o")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
