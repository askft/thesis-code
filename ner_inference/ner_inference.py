import os
import numpy as np
import tokenization
import tensorflow as tf
import onnxruntime
from onnxruntime import ExecutionMode, InferenceSession, SessionOptions


# import BertTokenizer.from_pretrained("bert-base-uncased") # Return to this if everything goes to shit
tokenizer = tokenization.FullTokenizer(
    vocab_file="biobert_vocab.txt", do_lower_case=True)

# Might want to re-check this if issues arise
# 4 (FATAL)
# 3 (ERROR)
# 2 (WARNING)
onnxruntime.set_default_logger_severity(3)

sequence = ("The adverse events during combined therapy with cyclosporin A and "
            + "nifedipine included an increase in blood urea nitrogen levels "
            + "in 9 of the 13 patients and development of gingival hyperplasia "
            + "in 2 of the 13 patients.")

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

while len(input_ids) < max_seq_length:

    # Not sure if padding belongs to the sequence or not
    token_type_ids.append(0)

    # Zero is the [PAD]-token for the BERT-vocab
    input_ids.append(0)

    # We probably should exclude the sequence padding from the attention-mask
    attention_mask.append(0)

model_path = "biobert_ner.onnx"

# Allow caller to use symlink to model
if os.path.islink(model_path):
    model_path = os.readlink(model_path)

print("Loading model:\n  {}\n".format(model_path))
session = onnxruntime.InferenceSession(model_path)

out_1, out_2, out_3 = session.run([], {
    "segment_ids_1:0": np.array([token_type_ids], dtype=np.int32),
    "input_mask_1_raw_output___9:0": np.array([attention_mask], dtype=np.int32),
    "input_ids_1:0": np.array([input_ids], dtype=np.int32),
    "label_ids_1:0": np.array([0], dtype=np.int32)}
)

#print("\n\n\n------------------------- INPUT NAMES -------------------------")
for input in session.get_inputs():
    print(input)

pred_labels = out_2[0]
with open("pred_labels.txt", "w") as out:
    for index in pred_labels:
        out.write(label_ids[index] + "\n")
