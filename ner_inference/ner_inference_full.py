# coding=utf-8

import os
import sys
import numpy as np
import onnxruntime
import tokenization


def create_session(model_path: str):
    # Allow caller to use symlink to model
    if os.path.islink(model_path):
        model_path = os.readlink(model_path)

    print("Loading model:\n  {}\n".format(model_path))
    return onnxruntime.InferenceSession(model_path)


def load_sequences(path: str):
    with open(path, "r") as f:
        for line in f:
            yield line


def parse_sequence(sequence):
    tokenized_sequence = tokenizer.tokenize(sequence)
    tokenized_sequence.insert(0, '[CLS]')
    tokenized_sequence.append('[SEP]')

    # Could be 0 or 1, not sure which index is *supposed* to represent a first segment
    token_type_ids = [0]*len(tokenized_sequence)
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sequence)
    # Not sure if label_ids should be padded to sequence length or not
    label_ids = ["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"]
    attention_mask = [1]*len(tokenized_sequence)
    attention_mask[0] = 0

    # Default for our model
    max_seq_length = 128

    # Pad arrays
    while len(input_ids) < max_seq_length:

        # Not sure if padding belongs to the sequence or not
        token_type_ids.append(0)

        # Zero is the [PAD]-token for the BERT-vocab
        input_ids.append(0)

        # We probably should exclude the sequence padding from the attention-mask
        attention_mask.append(0)

    return token_type_ids, attention_mask, input_ids, label_ids


tokenizer = tokenization.FullTokenizer(
    vocab_file="biobert_vocab.txt",
    do_lower_case=True
)

# Might want to re-check this if issues arise
onnxruntime.set_default_logger_severity(3)

session = create_session("biobert_ner.onnx")

with open("pred_labels.txt", "w") as out:
    i = 0
    for sequence in load_sequences("processed_data.txt"):
        if i % 10 == 0:
            sys.stdout.write("\rHandled %d sequences" % i)
            sys.stdout.flush()
        i += 1

        token_type_ids, attention_mask, input_ids, label_ids = parse_sequence(
            sequence)

        _, out_2, _ = session.run([], {
            "segment_ids_1:0": np.array([token_type_ids], dtype=np.int32),
            "input_mask_1_raw_output___9:0": np.array([attention_mask], dtype=np.int32),
            "input_ids_1:0": np.array([input_ids], dtype=np.int32),
            "label_ids_1:0": np.array([0], dtype=np.int32)}
        )

        pred_labels = out_2[0]
        for index in pred_labels:
            out.write(label_ids[index] + " ")
        out.write("\n")
