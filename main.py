# coding=utf-8

import json
import os
from scripts import downloader, tokenization
from scripts.ner_inference import NERInferenceSession

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.loads(f.read())

    os.makedirs("data", exist_ok=True)

    dl_config = config["downloader"]
    downloader.run(dl_config["input_path"], dl_config["output_path"], dl_config["batch_size"])

    ner_config = config["ner"]

    with open(dl_config["output_path"], "r") as f, open(ner_config["input_path"], "w") as g:
        full_articles = json.loads(f.read())
        sentences = {}
        for id in full_articles:
            sentences[id] = full_articles[id]["abstract"]

        g.write(json.dumps(sentences))

    ner_session = NERInferenceSession(ner_config["model_dir"], ner_config["model_name"], ner_config["model_vocab"],
                                    ner_config["labels"], ner_config["input_path"], ner_config["output_path"])

    
