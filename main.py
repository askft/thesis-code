# coding=utf-8

import json
import os
from scripts import downloader
from scripts.ner_inference import NERInferenceSession
from scripts.splitter import split_into_sentences
from scripts.entity_parser import co_occurrence_extractor, detokenize


def download(dl_config: dict):
    downloader.run(
        input_file=dl_config["input_path"],
        output_file=dl_config["output_path"],
        batch_size=dl_config["batch_size"],
    )


def sentences_from_text(input_path: str, output_path: str) -> dict:
    with open(input_path, "r") as f:
        full_articles = json.loads(f.read())

    articles = {}

    for id, article in full_articles.items():
        articles[id] = {
            # **articles[id], # include other fields
            "title": article["title"],
            "sentences": list(map(
                lambda sentence: {"text": sentence},
                split_into_sentences(article["abstract"])
            ))
        }

    with open(output_path, "w") as f:
        f.write(json.dumps(articles, indent=2, ensure_ascii=False))

    return articles


def ner(ner_config: dict):

    with open(ner_config["input_path"], "r") as f:
        articles = json.loads(f.read())

    print("Creating NER session...")
    ner_session = NERInferenceSession(
        model_dir=ner_config["model_dir"],
        model_name=ner_config["model_name"],
        model_vocab=ner_config["model_vocab"],
        labels=ner_config["labels"],
    )
    print("Created NER session.")

    # Temporary: shorten number of articles to n ————————
    a = {}
    i = 0
    n = 25
    for pmid in articles:
        if i >= n:
            break
        a[pmid] = articles[pmid]
        i += 1
    articles = a
    # ———————————————————————————————————————————————————

    for pmid in articles:

        sentences = articles[pmid]["sentences"]

        for i, sentence in enumerate(sentences):
            token_label_pairs = ner_session.predict(sentence["text"])

            x = co_occurrence_extractor(detokenize(token_label_pairs))

            articles[pmid]["sentences"][i]["entities"] = x["entities"]
            articles[pmid]["sentences"][i]["text_new"] = x["text"]

    with open(ner_config["output_path"], "w") as f:
        f.write(json.dumps(articles, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.loads(f.read())

    os.makedirs("data", exist_ok=True)

    dl_config = config["downloader"]
    ner_config = config["ner"]

    # ——————————————————————————————————————————————————————————————————————————

    if dl_config["ignore"]:
        print("Ignoring downloader script.")
    else:
        print("Running downloader script.")
        download(dl_config)
        print("Finished running downloader script.")

    # ——————————————————————————————————————————————————————————————————————————

    print("Converting sentences.")
    _ = sentences_from_text(
        input_path="data/pmid-covid-set.json",  # dl_config["output_path"],
        output_path="data/sentences_done.json",  # actually doesn't matter now
    )
    print("Finished converting sentences.")

    # ——————————————————————————————————————————————————————————————————————————

    if ner_config["ignore"]:
        print("Ignoring NER script.")
    else:
        print("Running NER script.")
        ner(ner_config)
        print("Finished running NER script.")

    # ——————————————————————————————————————————————————————————————————————————

    print("Running analysis script.")
    from scripts import analysis
    with open(ner_config["output_path"], "r") as f:
        articles = json.loads(f.read())
    analysis.run(articles)
    print("Finished running analysis script.")

    # with open("data/main_out.json", "w") as f:
    #     f.write(json.dumps(articles, indent=2, ensure_ascii=False))

    print("Program finished successfully.")
