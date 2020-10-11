# coding=utf-8

import json
import os

from scripts import downloader
from scripts import splitter
from scripts import analysis
from scripts import util
from scripts.ner_inference import NERInferenceSession
from scripts.entity_parser import co_occurrence_extractor, detokenize


def run_download(dl_config: dict, ignore: bool):
    if ignore:
        print("Ignoring script: downloader.")
        return

    print("Running downloader script.")
    downloader.run(
        input_file=dl_config["input_path"],
        output_file=dl_config["output_path"],
        batch_size=dl_config["batch_size"],
    )
    print("Finished running downloader script.")


def sentencer(sentencer_config: dict, ignore: bool) -> dict:
    if ignore:
        print("Ignoring script: sentencer.")
        return {}

    print("Running sentencer script.")

    with open(sentencer_config["input_path"], "r") as f:
        full_articles = json.loads(f.read())

    articles = {}

    for id, article in full_articles.items():
        articles[id] = {
            # **articles[id], # include other fields
            "title": article["title"],
            "sentences": list(map(
                lambda sentence: {"text": sentence},
                splitter.split_into_sentences(article["abstract"])
            ))
        }

    with open(sentencer_config["output_path"], "w") as f:
        f.write(json.dumps(articles, indent=2, ensure_ascii=False))

    print("Finished running sentencer script.")

    return articles


def run_ner(ner_config: dict, ignore: bool):

    if ignore:
        print("Ignoring script: NER.")
        return

    print("Running NER script.")

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

    # For experimentation: limit number of articles to process (and to output)
    limit = ner_config["article_limit"]
    if limit > 0:
        print(f"Limiting NER to {limit} articles.")
        a = {}
        i = 0
        for pmid in articles:
            if i >= limit:
                break
            a[pmid] = articles[pmid]
            i += 1
        articles = a

    if ner_config.get("clear_old_results", True):
        os.remove(ner_config["output_path"])

    # Becuase we want to save the result periodically.
    batch_index = 0
    batch_size = 10

    # Run prediction on each sentence in each article.
    for pmid in articles:
        if batch_index > batch_size:
            util.append_to_json_file(ner_config["output_path"], articles)
            batch_index = 0
        sentences = articles[pmid]["sentences"]
        for i, sentence in enumerate(sentences):
            token_label_pairs = ner_session.predict(sentence["text"])
            x = co_occurrence_extractor(detokenize(token_label_pairs))
            articles[pmid]["sentences"][i]["entities"] = x["entities"]
            articles[pmid]["sentences"][i]["text_new"] = x["text"]
        batch_index += 1

    util.append_to_json_file(ner_config["output_path"], articles)

    print("Finished running NER script.")


def run_re(re_config: dict, ignore: bool):
    if ignore:
        print("Ignoring script: RE.")
        return

    print("Running RE script.")
    # TODO
    print("Finished running RE script.")


def run_analysis(analysis_config: dict, ignore: bool):
    if ignore:
        print("Ignoring script: analysis.")
        return

    print("Running analysis script.")

    with open(analysis_config["input_path"], "r") as f:
        articles = json.loads(f.read())

    analysis.run(articles)

    print("Finished running analysis script.")


if __name__ == "__main__":
    print("Please see config.json for configuration!")

    with open("config.json", "r") as f:
        config = json.loads(f.read())

    print("Loaded config:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print()

    os.makedirs("data", exist_ok=True)

    ignore = config["ignore"]

    # Download articles from the PubMed API.
    run_download(config["downloader"], ignore=ignore["downloader"])
    print()

    # Extract sentences from each article.
    sentencer(config["sentencer"], ignore=ignore["sentencer"])
    print()

    # Run NER inference on each sentence for each article.
    run_ner(config["ner"], ignore=ignore["ner"])
    print()

    # Run relationship extraction
    run_re(config["re"], ignore=ignore["re"])
    print()

    # Run analysis on the entities that were found by NER.
    run_analysis(config["analysis"], ignore=ignore["analysis"])
    print()

    print("Program finished successfully.")
