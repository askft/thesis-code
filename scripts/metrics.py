from collections import defaultdict, Counter
from typing import DefaultDict, List, Counter as CounterT
from scripts.ner_inference import NERInferenceSession


def gs_metrics(input_path: str):
    with open(input_path, "r") as f:
        data = f.readlines()

    label_count = defaultdict(int)
    occurrence_count = defaultdict(int)
    occurrences = 0

    for line in data:
        line = line.strip()

        if line:
            line = line.split()[1]
            label_count[line] += 1

            if line == "B":
                occurrences += 1

        else:
            occurrence_count[occurrences] += 1
            occurrences = 0

    print(" - - - Gold standard metrics - - -")

    print("Label count:")
    for key in label_count:
        print("\t" + key + " label count: " + str(label_count[key]))

    print("\nOccurrence count:")
    for key in sorted(occurrence_count):
        print("\t" + str(key) + "_occurrence count: " + str(occurrence_count[key]))

    print(" - - - - - - - - - - - - - - - - - \n")

# Gets the index-range-tuples from a list of labels
def  get_indices(labels):
    indices = list()
    start = 0
    counter = 0
    in_entity = False

    for label in labels:
        counter += 1

        if in_entity:
            if label == "O":
                indices.append((start, counter - 1))
                in_entity = False

            elif label == "B":
                indices.append((start, start))
                start = counter

        elif label == "B":
            start = counter
            in_entity = True

    return indices

def sentence_metrics(pred_labels: List[str], gs_labels: List[str]):

    # Treating B = I
    confusion_matrix = defaultdict(int)
    for pred, gs in zip(pred_labels, gs_labels):

        if pred == "B" or pred == "I":
            if gs == "B" or gs == "I":
                confusion_matrix["true_positive"] += 1
            elif gs == "O":
                confusion_matrix["false_positive"] += 1
        elif pred == "O":
            if gs == "O":
                confusion_matrix["true_negative"] += 1
            elif gs == "B" or gs == "I":
                confusion_matrix["false_negative"] += 1

    # Treating B=/=I
    token_matrix = defaultdict(lambda: defaultdict(int))

    for pred, gs in zip(pred_labels, gs_labels):
        token_matrix[gs][pred] += 1

    # Entity Level Perfect. Naive way of taking the metrics
    entity_matrix = defaultdict(int)
    pred_indices = get_indices(pred_labels)
    gs_indices = get_indices(gs_labels)

    while pred_indices and gs_indices:
        pred = pred_indices.pop(0)
        gs = gs_indices.pop(0)

        pred_set = set(range(pred[0], pred[1] + 1 ))
        gs_set = set(range(gs[0], gs[1] + 1 ))

        if pred_set & gs_set:
            if not pred_set.difference(gs_set):
                entity_matrix["true_positive"] += 1

            # there is some overlap so the entity has been mispredicted
            # there are no strict rules for this, but it should make some sense
            elif not pred[0] in gs_set or not pred[1] in gs_set:
                entity_matrix["false_positive"] += 1

            else:
                entity_matrix["false_negative"] += 1

        # one tuple will have to be returned to its list
        else:
            if pred[0] > gs[0]:
                entity_matrix["false_negative"] += 1
                pred_indices.insert(0, pred)

            else:
                entity_matrix["false_positive"] += 1
                gs_indices.insert(0, gs)

    entity_matrix["false_positive"] += len(pred_indices)
    entity_matrix["false_negative"] += len(gs_indices)
    entity_matrix["true_negative"] = confusion_matrix["true_negative"]

    return confusion_matrix, token_matrix, entity_matrix


def biobert_metrics(model: NERInferenceSession, input_path: str):
    with open(input_path, "r") as f:
        data = f.readlines()

    counter = 0
    for i in data:
        if i == "\n":
            counter += 1

    print("Running over " + str(counter) + " sentences")

    confusion_matrix: CounterT[str] = Counter()
    token_matrix: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
    entity_matrix: CounterT[str] = Counter()

    gs_labels: List[str] = []
    sequence = ""

    counter_2 = 0

    for line in data:

        if line == "\n":
            counter_2 += 1
            if counter_2 % 20 == 0:
                print(str(counter_2) + " / " + str(counter))

            pred_pairs = model.predict(sequence.strip())

            # The tokenization label X and special labels hold no more value
            pred_labels = [label[1] for label in pred_pairs if label[1]
                           != 'X' and label[0] != '[CLS]' and label[0] != '[SEP]']

            cm, tm, em = sentence_metrics(pred_labels, gs_labels)

            confusion_matrix.update(cm)

            for gs_label in tm:
                for pred_label in tm[gs_label]:
                    token_matrix[gs_label][pred_label] += tm[gs_label][pred_label]

            entity_matrix.update(em)

            gs_labels = []
            sequence = ""
            continue

        columns = line.split("\t")
        sequence += columns[0] + " "
        gs_labels.append(columns[1].strip())

        #if counter_2 == 100:
            #break

    print("Confusion matrix:")
    print({**confusion_matrix})
    print("Recall: " + str(confusion_matrix["true_positive"]/(confusion_matrix["true_positive"] + confusion_matrix["false_negative"])))
    print("Precision: " + str(confusion_matrix["true_positive"]/(confusion_matrix["true_positive"] + confusion_matrix["false_positive"])))
    print()

    print("Token matrix:")
    print({**token_matrix})
    print()

    print("Entity matrix:")
    print({**entity_matrix})
    print("Recall: " + str(entity_matrix["true_positive"]/(entity_matrix["true_positive"] + entity_matrix["false_negative"])))
    print("Precision: " + str(entity_matrix["true_positive"]/(entity_matrix["true_positive"] + entity_matrix["false_positive"])))
    print()
