from collections import defaultdict
from collections import Counter
from scripts.ner_inference import NERInferenceSession

def gs_metrics(file):
    with open(file, "r") as f:
        data = f.readlines()

        count = defaultdict(int)
        occurrences = 0

        for line in data:
            entity_occurrence = False
            line = line.strip()

            if line:
                line = line.split()[1]
                count[line] +=1
                if line == 'B':
                    occurrences +=1
                    if occurrences > 1:
                        count[occurrences] += 1

                elif line == 'O':
                    occurrences = 0

        print(count)

def sentence_metrics(pred_labels, gs_labels):

    # Treating B = I
    confusion_matrix = defaultdict(int)
    for pred, gs in zip(pred_labels, gs_labels):
        if pred == "B" or pred == "I":
            if gs == "B" or gs == "I":
                confusion_matrix["true_positive"] += 1
            else:
                confusion_matrix["false_negative"] += 1
        else:
            if gs == "O":
                confusion_matrix["true_negative"] += 1
            else:
                confusion_matrix["false_positive"] += 1



    # Treating B=/=I
    token_matrix = defaultdict(lambda: defaultdict(int))

    for pred, gs in zip(pred_labels, gs_labels):
        token_matrix[gs][pred] += 1



    # Entity Level Perfect. Naive way of taking the metrics
    in_entity = False
    entity_matrix = defaultdict(int)
    num_entities = 0

    for pred, gs in zip(pred_labels, gs_labels):

        if gs == "B":
            num_entities += 1

        if pred == "O" and gs == "O":
            entity_matrix["true_negative"] += 1

        if in_entity:
            if pred == "I" and gs == "I":
                continue
            elif pred == "O" and gs == "O":
                entity_matrix["true_positive"] += 1
            elif pred == "O" and gs != "O":
                entity_matrix["false_negative"] += 1
            elif pred != "O" and gs == "O":
                entity_matrix["false_positive"] += 1

            in_entity = False

        if pred == "B" and gs == "B":
            in_entity = True
        elif pred == "B" and gs != "B":
            entity_matrix["false_positive"] += 1
        elif pred != "B" and gs == "B":
            entity_matrix["false_negative"] += 1
        elif pred == "O" and gs != "O":
            entity_matrix["false_negative"] += 1


    return confusion_matrix, token_matrix, entity_matrix

    # Entity Level Relaxed. The way it will be used in the program
    # TODO
    

def biobert_metrics(model: NERInferenceSession, file):
    with open(file, "r") as f:
        data = f.readlines()

        counter = 0
        for i in data:
            if i == "\n":
                counter += 1

        print("Running over " + str(counter) + " sentences")

        confusion_matrix = defaultdict(int)
        token_matrix = defaultdict(lambda: defaultdict(int))
        entity_matrix = defaultdict(int)

        gs_labels = list()
        sequence = ""

        counter_2 = 0

        for line in data:

            if line == "\n":
                counter_2 += 1
                if counter_2 % 200 == 0:
                    print(str(counter_2) + " / " + str(counter))

                pred_pairs = model.predict(sequence.strip())

                # The tokenization label X and special labels hold no more value
                pred_labels = [label[1] for label in pred_pairs if label[1] != 'X' and label[0] != '[CLS]' and label[0] != '[SEP]']
                cm, tm, em = sentence_metrics(pred_labels, gs_labels)

                confusion_matrix = Counter(confusion_matrix) + Counter(cm)
                for gs_label in tm:
                    for pred_label in tm[gs_label]:
                        token_matrix[gs_label][pred_label] += tm[gs_label][pred_label]

                entity_matrix = Counter(entity_matrix) + Counter(em)

                gs_labels = list()
                sequence = ""
                continue

            columns = line.split("\t")
            sequence += columns[0] + " "
            gs_labels.append(columns[1].strip())


        print("Confusion matrix:")
        print({**confusion_matrix})
        print()

        print("Token matrix:")
        print({**token_matrix})
        print()

        print("Entity matrix:")
        print({**entity_matrix})
        print()
