# coding=utf-8

import json

# TODO: Remove parts we don't use


def main(input_path, output_path):
    sentences = list()

    with open(input_path, "r") as f:
        lines = f.readlines()

    xs = map(lambda line: line.split(" "), lines)

    sentence = list()

    for token, label in xs:
        label = label.strip()
        token = token.strip()

        sentence.append((token, label))

        if '[SEP]' in token:
            sentence = sentence[1:len(sentence)-1]
            sentences.append(sentence)
            sentence = list()

    data = []
    for sentence in sentences:
        x = co_occurrence_extractor(detokenize(sentence))
        if not x["hasCoOccurrence"]:
            continue
        data.append(dict(
            entities=x["entities"],
            text=x["text"])
        )

    with open(output_path, "w") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False))


def detokenize(token_label_pairs):
    labels = []
    words = []

    for token, label in token_label_pairs:
        if 'X' not in label:
            words.append(token)
            labels.append(label)
        else:
            word = words.pop(len(words)-1) + token[2:]
            words.append(word)

    return list(zip(labels, words))


def co_occurrence_extractor(label_word_pairs):
    entities = []
    entity = ""
    in_entity = False

    for label, word in label_word_pairs:

        if 'B' in label:
            entity = entity + word
            in_entity = True

        elif in_entity:
            if 'I' in label:
                entity = entity + " " + word
            elif 'O' in label:
                in_entity = False
                # TODO: format inside of entity e.g. " , ", " - ", etc.
                entity = entity.replace(' - ', '-')
                entity = entity.replace(' , ', ',')
                entities.append(entity)
                entity = ''

    return {
        "hasCoOccurrence": len(entities) >= 2,
        "entities": entities,
        "text": " ".join(list(map(lambda t: t[1], label_word_pairs))).
        replace(" .", ".").
        replace(" ,", ",").
        replace(" - ", "-").
        replace("( ", "(").
        replace(" )", ")").
        replace(" :", ":").
        replace(" ;", ";").
        replace(" !", "!").
        replace(" ?", "?")
    }


if __name__ == '__main__':
    input_path = "predicted_labels.txt"
    output_path = "data.json"

    main(input_path, output_path)
