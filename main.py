import json
from scripts import downloader

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.loads(f.read())

    dl = config["downloader"]
    downloader.run(dl["input_path"], dl["output_path"], dl["batch_size"])
