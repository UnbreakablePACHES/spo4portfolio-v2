import os
from datetime import datetime


class Logger:
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.log_path = os.path.join(save_dir, "train.log")

    def log(self, msg):
        time_str = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{time_str} {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
