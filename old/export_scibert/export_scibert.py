import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification


def get_real_link(model_dir: str) -> str:
    if os.path.islink(model_dir):
        model_dir = os.readlink(model_dir)
    return model_dir


# model_dir = "./scibert_model/model"
model_dir = get_real_link("./model_dir")  # Arg is either a path or a symlink

device = torch.device('cpu')

vocab_path = os.path.join("/", model_dir, "vocab.txt")
tokenizer = BertTokenizer.from_pretrained(vocab_path)

model = BertForSequenceClassification.from_pretrained(
    model_dir,
    output_hidden_states=True,
    output_attentions=True
)

model.to(device)
model.eval()

with torch.no_grad():

    text = "The current study evaluates the effects of a selective << COX-2 >> inhibitor ([[ SC-236 ]]) on renal function in cirrhotic rats with ascites."
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


    input_ids = torch.tensor([encoded_padded])
    attention_mask = torch.tensor([att_mask])
    position_embeddings = torch.tensor([[0]*128])
    token_type_ids = torch.tensor([[0]*128])

    # logits, hidden_states, attentions
    _, hidden_states, attentions = model(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    print("\n", "length of hidden_states: ", len(hidden_states), "\n")
    for lbl in hidden_states:
        print(lbl[0][0][0])

    #torch.onnx.export(model, input_ids, "scibert.onnx")
