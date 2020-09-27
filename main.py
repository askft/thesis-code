# coding=utf-8

import json
import os
from scripts import downloader

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.loads(f.read())

    os.makedirs("data", exist_ok=True)

    dl = config["downloader"]
    downloader.run(dl["input_path"], dl["output_path"], dl["batch_size"])
