# coding=utf-8

import json
import os
from scripts import downloader
from scripts.ner_inference import NERInferenceSession
from scripts.splitter import split_into_sentences
from scripts.entity_parser import co_occurrence_extractor, detokenize


def sentences_from_text(input_path: str, output_path: str):

    # TODO: Move somewhere else

    with open(input_path, "r") as f:
        articles = json.loads(f.read())

    for id in articles:
        articles[id] = {
            **articles[id],  # include other fields
            "sentences": list(map(
                lambda sentence: {"text": sentence},
                split_into_sentences(articles[id]["abstract"])
            ))
        }

    with open(output_path, "w") as f:
        f.write(json.dumps(articles, indent=2, ensure_ascii=False))

    return articles


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.loads(f.read())

    os.makedirs("data", exist_ok=True)

    dl_config = config["downloader"]

    downloader.run(
        input_file=dl_config["input_path"],
        output_file=dl_config["output_path"],
        batch_size=dl_config["batch_size"],
    )

    ner_config = config["ner"]

    articles = sentences_from_text(
        input_path=dl_config["output_path"],
        output_path=ner_config["input_path"],
    )

    ner_session = NERInferenceSession(
        model_dir=ner_config["model_dir"],
        model_name=ner_config["model_name"],
        model_vocab=ner_config["model_vocab"],
        labels=ner_config["labels"],
    )

    for pmid in articles:
        sentences = articles[pmid]["sentences"]

        for i, sentence in enumerate(sentences):
            token_label_pairs = ner_session.predict(sentence["text"])

            x = co_occurrence_extractor(detokenize(token_label_pairs))

            if not x["hasCoOccurrence"]:
                x["entities"] = []

            articles[pmid]["sentences"][i]["entities"] = x["entities"]
            articles[pmid]["sentences"][i]["text_new"] = x["text"]

    # TODO: It seems like it only predicts entites for the first sentence
    #       of the first article...?

    with open("data/main_out.json", "w") as f:
        f.write(json.dumps(articles, indent=2, ensure_ascii=False))
