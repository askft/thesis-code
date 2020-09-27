# coding=utf-8

import os
import sys
import numpy as np
import onnxruntime
import tokenization as t10n

from typing import List, NamedTuple


def create_session(model_path: str) -> onnxruntime.InferenceSession:
    # Allow caller to use symlink to model
    if os.path.islink(model_path):
        model_path = os.readlink(model_path)
    print("Loading model:\n  {}".format(model_path))
    session = onnxruntime.InferenceSession(model_path)
    print("Done.\n")
    return session


class SequenceParseResult(NamedTuple):
    tokens: List[str]
    token_type_ids: List[int]
    attention_mask: List[int]
    input_ids: List[int]


def parse_sequence(tokenizer: t10n.FullTokenizer, sequence: str) -> SequenceParseResult:
    tokens = tokenizer.tokenize(sequence)
    tokens.insert(0, '[CLS]')
    tokens.append('[SEP]')

    # Could be 0 or 1, not sure which index is *supposed* to represent a first segment
    token_type_ids = [0]*len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1]*len(tokens)
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

    return SequenceParseResult(
        tokens=tokens,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        input_ids=input_ids,
    )


def main():
    tokenizer = t10n.FullTokenizer(
        vocab_file="biobert_vocab.txt",
        do_lower_case=True
    )

    onnxruntime.set_default_logger_severity(3)
    session = create_session("biobert_ner.onnx")

    label_names = ["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"]

    print("Writing results to ./predicted_labels.txt ...")
    with open("predicted_labels.txt", "w") as out:
        i = 0
        for sequence in open("data_proc.txt", "r"):
            if i > 50:
                break

            sys.stdout.write("\rHandled %d sequences ... " % i)
            sys.stdout.flush()
            i += 1

            r = parse_sequence(tokenizer, sequence)

            _, out_2, _ = session.run([], {
                "segment_ids_1:0": np.array([r.token_type_ids], dtype=np.int32),
                "input_mask_1_raw_output___9:0": np.array([r.attention_mask], dtype=np.int32),
                "input_ids_1:0": np.array([r.input_ids], dtype=np.int32),
                "label_ids_1:0": np.array([0], dtype=np.int32)}
            )

            labels = []

            for index in out_2[0]:
                labels.append(label_names[index])

            for token, label in zip(r.tokens, labels):
                out.write("{} {}\n".format(token, label))

    print("Done.")


if __name__ == "__main__":
    main()
