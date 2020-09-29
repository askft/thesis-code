# coding=utf-8

import os
import sys
import json
import numpy as np
import onnxruntime
from scripts import tokenization as t10n
from transformers import BertTokenizer
import torch
from typing import List, NamedTuple

class NERInferenceSession:

    def __init__(self, model_dir, model_name, labels, input_path, out_path):
        self.model_path = os.path.join(model_dir, model_name)
        self.vocab_path = os.path.join(model_dir, "vocab.txt")
        self.labels = labels
        self.input_path = input_path
        self.out_path = out_path

        self.tokenizer = tokenizer = BertTokenizer.from_pretrained(self.vocab_path)

    def create_session(self) -> onnxruntime.InferenceSession:
        # Allow caller to use symlink to model
        if os.path.islink(self.model_path):
            self.model_path = os.readlink(self.model_path)
        print("Loading model:\n  {}".format(self.model_path))
        session = onnxruntime.InferenceSession(self.model_path)
        print("Model loaded succesfully\n")
        return session

    def encode_sequence(self, sequence: str):

        encoded = self.tokenizer.encode_plus(
            sequence,
            max_length=128,
            add_special_tokens=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Cleanest way I could think of
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        for i, token in enumerate(tokens):
            if token == self.tokenizer.convert_ids_to_tokens(0):
                tokens = tokens[:i]
                break

        token_type_ids = np.array(encoded["token_type_ids"].numpy(), dtype=np.int32)
        attention_mask = np.array(encoded["attention_mask"].numpy(), dtype=np.int32)
        input_ids = np.array(encoded["input_ids"].numpy(), dtype=np.int32)
        label_ids = np.array([0], dtype=np.int32)


        # Default for our model
        max_seq_length = 128

        return {
            "tokens": tokens,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "label_ids": label_ids
        }

    def predict(self):
        print("\nInit NER-inference")
        onnxruntime.set_default_logger_severity(3)
        session = self.create_session()

        print("\nRunning predictions")

        print("Predicted labels will be written to " + self.out_path)
        with open(self.out_path, "w") as out, open(self.input_path, "r") as in_data:
            input = json.loads(in_data.read())
            i = 0
            for sequence in input:
                if i > 50:
                    break

                encodings = self.encode_sequence(input[sequence])

                _, logits, _ = session.run([], {
                    "segment_ids_1:0": encodings["token_type_ids"],
                    "input_mask_1_raw_output___9:0": encodings["attention_mask"],
                    "input_ids_1:0": encodings["input_ids"],
                    "label_ids_1:0": encodings["label_ids"]}
                                          )

                pred_labels = []

                for index in logits[0]:
                    pred_labels.append(self.labels[index])

                for token, label in zip(encodings["tokens"], pred_labels):
                    out.write("{} {}\n".format(token, label))

                i += 1
                sys.stdout.write("\rHandled %d sequences ... " % i)
                sys.stdout.flush()

        print("Prediction done")
