import csv
import os


class LossRecorder:
    def __init__(self, save_dir):
        self.path = os.path.join(save_dir, "losses.csv")
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss"])

    def add(self, epoch, loss):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss])


class WeightRecorder:
    def __init__(self, save_dir, num_assets):
        self.path = os.path.join(save_dir, "weights.csv")
        self.num_assets = num_assets

        # 写表头
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + [f"asset_{i}" for i in range(num_assets)])

    def add(self, epoch, w):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch] + list(w))


class RegretRecorder:
    def __init__(self, save_dir):
        self.path = os.path.join(save_dir, "regrets.csv")
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_regret"])

    def add(self, epoch, regret):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, regret])
