import os
import torch
from transformers import BertTokenizer, BertModel
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd   # Not needed, can we remove?
# from tqdm import tqdm # Needed but not imported

# MODEL_PATH = './hf/'
# MODEL_NAME = 'pytorch_model.bin'


def get_real_link(model_dir: str) -> str:
    if os.path.islink(model_dir):
        model_dir = os.readlink(model_dir)
    return model_dir


# model_dir = "./scibert_model/model"
model_dir = get_real_link("./model_dir")  # Arg is either a path or a symlink

device = torch.device('cpu')

# Can't get any model-type to load it correctly. Tried the following:
#  - BertModel
#  - BertForTokenClassification
#  - AutoModel
#  - AutoModelForTokenClassification
# According to SciBERT themselves, AutoModel should be used

vocab_path = os.path.join("/", model_dir, "vocab.txt")
tokenizer = BertTokenizer.from_pretrained(vocab_path)
model = BertModel.from_pretrained(model_dir)

model.to(device)
model.eval()

text = "Down-regulation of << prostate-specific antigen >> (PSA) expression, an AR-target gene, by [[ estramustine ]] and bicalutamide was accompanied by the blockade of the mutated androgen receptor."
tokenized_non_padded = tokenizer.tokenize(text)
encoded_non_padded = tokenizer.encode(tokenized_non_padded)

while len(encoded_non_padded) < 128:
    encoded_non_padded.append(0)

encoded_padded = encoded_non_padded.copy()

att_mask = []
for i in range(128):
    if encoded_non_padded[i] != 0:
        att_mask.append(1)
    else:
        att_mask.append(0)


# for reading
print("\nParameter names with the string 'embeddings' in them:")
for name, _ in model.named_parameters():
    if "embeddings" in name:
        print("  " + name)

prediction_layer = torch.nn.Linear(768, 13)

we = [0]*128
pe = [0]*128
tt = [0]*128
ii = [108]*128
am = [1]*128

input_ids = torch.tensor([encoded_padded])
attention_mask = torch.tensor([att_mask])
word_embeddings = torch.tensor([we])
position_embeddings = torch.tensor([pe])
token_type_ids = torch.tensor([tt])

# Seem to be the accepted keyword arguments for the forward-function
with torch.no_grad():
    out_1, out_2 = model.forward(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

print("\n**** OUTPUT ****")
logits = prediction_layer(out_1)
preds = torch.softmax(logits, dim=1)
preds_max = torch.argmax(preds, dim=1)
print(preds_max)
