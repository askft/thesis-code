# coding=utf-8

import json


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
        x = co_occurrence_extractor(de_tokenize(sentence))
        if not x["hasCoOccurrence"]:
            continue
        data.append(dict(
            entities=x["entities"],
            text=x["text"])
        )

    with open(output_path, 'w', encoding='utf-8') as label:
        json.dump(data, label, ensure_ascii=False, indent=4)


def de_tokenize(sentence):
    labels = list()
    words = list()

    for token, label in sentence:

        if 'X' not in label:
            words.append(token)
            labels.append(label)
        else:
            word = words.pop(len(words)-1) + token[2:]
            words.append(word)

    return list(zip(labels, words))


def co_occurrence_extractor(sentence):

    words = list()
    entity = ''
    in_entity = False

    for label, word in sentence:

        if 'B' in label:
            entity = entity + word
            in_entity = True

        elif in_entity:
            if 'I' in label:
                entity = entity + " " + word
            elif 'O' in label:
                in_entity = False
                # TODO: format inside of entity e.g. " , ", " - ", etc.
                entity = entity.replace(' - ', '-', -1)
                entity = entity.replace(' , ', ',', -1)
                words.append(entity)
                entity = ''

    return dict(
        hasCoOccurrence=len(words) >= 2,
        entities=words,
        text=" ".join(list(map(lambda t: t[1], sentence))).
        replace(" .", ".").
        replace(" ,", ",").
        replace(" - ", "-").
        replace("( ", "(").
        replace(" )", ")").
        replace(" :", ":").
        replace(" ;", ";").
        replace(" !", "!").
        replace(" ?", "?")
    )


if __name__ == '__main__':
    input_path = "predicted_labels.txt"
    output_path = "data.json"

    main(input_path, output_path)
