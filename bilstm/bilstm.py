import time
import torch
import json
import conlleval
import sys
from collections import defaultdict, Counter
from typing import DefaultDict, List, Counter as CounterT
from torch import nn
from torch.optim import Adam
from torchtext.data import Field, BucketIterator
from torchtext.datasets import SequenceTaggingDataset
from spacy.lang.en import English
from collections import defaultdict
from collections import Counter


class Corpus(object):

    def __init__(self, input_folder, min_word_freq, batch_size):
        # list all the fields
        self.word_field = Field(lower=True)
        self.tag_field = Field(unk_token=None)
        # create dataset using built-in parser from torchtext
        self.train_dataset, self.val_dataset, self.test_dataset = SequenceTaggingDataset.splits(
            path=input_folder,
            train="train.tsv",
            validation="devel.tsv",
            test="test.tsv",
            fields=(("word", self.word_field), ("tag", self.tag_field))
        )
        # convert fields to vocabulary list
        self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
        self.tag_field.build_vocab(self.train_dataset.tag)
        # create iterator for batch input
        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            datasets=(self.train_dataset, self.val_dataset, self.test_dataset),
            batch_size=batch_size
        )
        # prepare padding index to be ignored during model training/evaluation
        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]

class BiLSTM(nn.Module):

  def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, lstm_layers,
               emb_dropout, lstm_dropout, fc_dropout, word_pad_idx):
    super().__init__()
    self.embedding_dim = embedding_dim
    # LAYER 1: Embedding
    self.embedding = nn.Embedding(
        num_embeddings=input_dim,
        embedding_dim=embedding_dim,
        padding_idx=word_pad_idx
    )
    self.emb_dropout = nn.Dropout(emb_dropout)
    # LAYER 2: BiLSTM
    self.lstm = nn.LSTM(
        input_size=embedding_dim,
        hidden_size=hidden_dim,
        num_layers=lstm_layers,
        bidirectional=True,
        dropout=lstm_dropout if lstm_layers > 1 else 0
    )
    # LAYER 3: Fully-connected
    self.fc_dropout = nn.Dropout(fc_dropout)
    self.fc = nn.Linear(hidden_dim * 2, output_dim)  # times 2 for bidirectional

  def forward(self, sentence):
    # sentence = [sentence length, batch size]
    # embedding_out = [sentence length, batch size, embedding dim]
    embedding_out = self.emb_dropout(self.embedding(sentence))
    # lstm_out = [sentence length, batch size, hidden dim * 2]
    lstm_out, _ = self.lstm(embedding_out)
    # ner_out = [sentence length, batch size, output dim]
    ner_out = self.fc(self.fc_dropout(lstm_out))
    return ner_out

  def init_weights(self):
    # to initialize all parameters from normal distribution
    # helps with converging during training
    for name, param in self.named_parameters():
      nn.init.normal_(param.data, mean=0, std=0.1)

  def init_embeddings(self, word_pad_idx):
    # initialize embedding for padding as zero
    self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

class NER(object):

  def __init__(self, model, data, optimizer_cls, loss_fn_cls):
    self.model = model
    self.data = data
    self.optimizer = optimizer_cls(model.parameters())
    self.loss_fn = loss_fn_cls(ignore_index=self.data.tag_pad_idx)

  @staticmethod
  def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

  def accuracy(self, preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != self.data.tag_pad_idx).nonzero()  # prepare masking for paddings
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

  def epoch(self):
      epoch_loss = 0
      epoch_acc = 0
      self.model.train()
      for batch in self.data.train_iter:
        # text = [sent len, batch size]
        text = batch.word
        # tags = [sent len, batch size]
        true_tags = batch.tag
        self.optimizer.zero_grad()
        pred_tags = self.model(text)
        # to calculate the loss and accuracy, we flatten both prediction and true tags
        # flatten pred_tags to [sent len, batch size, output dim]
        pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
        # flatten true_tags to [sent len * batch size]
        true_tags = true_tags.view(-1)
        batch_loss = self.loss_fn(pred_tags, true_tags)
        batch_acc = self.accuracy(pred_tags, true_tags)
        batch_loss.backward()
        self.optimizer.step()
        epoch_loss += batch_loss.item()
        epoch_acc += batch_acc.item()
      return epoch_loss / len(self.data.train_iter), epoch_acc / len(self.data.train_iter)

  def evaluate(self, iterator):
      epoch_loss = 0
      epoch_acc = 0
      self.model.eval()
      with torch.no_grad():
          # similar to epoch() but model is in evaluation mode and no backprop
          for batch in iterator:
              text = batch.word
              true_tags = batch.tag
              pred_tags = self.model(text)
              pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
              true_tags = true_tags.view(-1)
              batch_loss = self.loss_fn(pred_tags, true_tags)
              batch_acc = self.accuracy(pred_tags, true_tags)
              epoch_loss += batch_loss.item()
              epoch_acc += batch_acc.item()
      return epoch_loss / len(iterator), epoch_acc / len(iterator)

  # main training sequence
  def train(self, n_epochs):
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc = self.epoch()
        end_time = time.time()
        epoch_mins, epoch_secs = NER.epoch_time(start_time, end_time)
        print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrn Loss: {train_loss:.3f} | Trn Acc: {train_acc * 100:.2f}%")
        val_loss, val_acc = self.evaluate(self.data.val_iter)
        print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}%")
    test_loss, test_acc = self.evaluate(self.data.test_iter)
    print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%")

  def infer(self, sentence, true_tags=None):
    self.model.eval()
    # tokenize sentence
    nlp = English()
    tokens = [token.text.lower() for token in nlp(sentence)]
    # transform to indices based on corpus vocab
    numericalized_tokens = [self.data.word_field.vocab.stoi[t] for t in tokens]
    # find unknown words
    unk_idx = self.data.word_field.vocab.stoi[self.data.word_field.unk_token]
    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
    # begin prediction
    token_tensor = torch.LongTensor(numericalized_tokens)
    token_tensor = token_tensor.unsqueeze(-1)
    predictions = self.model(token_tensor)
    # convert results to tags
    top_predictions = predictions.argmax(-1)
    predicted_tags = [self.data.tag_field.vocab.itos[t.item()] for t in top_predictions]
    # print inferred tags
    max_len_token = max([len(token) for token in tokens] + [len("word")])
    max_len_tag = max([len(tag) for tag in predicted_tags] + [len("pred")])
    '''print(
        f"{'word'.ljust(max_len_token)}\t{'unk'.ljust(max_len_token)}\t{'pred tag'.ljust(max_len_tag)}"
        + ("\ttrue tag" if true_tags else "")
        )
    for i, token in enumerate(tokens):
      is_unk = "✓" if token in unks else ""
      print(
          f"{token.ljust(max_len_token)}\t{is_unk.ljust(max_len_token)}\t{predicted_tags[i].ljust(max_len_tag)}"
          + (f"\t{true_tags[i]}" if true_tags else "")
          )'''
    return tokens, predicted_tags, unks

corpus = Corpus(
    input_folder="./BC5CDR-chem/temp/",
    min_word_freq = 3, # any words occurring less than 3 times wil be ignored from vocab
    batch_size=64
)

print(f"Train set: {len(corpus.train_dataset)} sentences")
print(f"Val set: {len(corpus.val_dataset)} sentences")
print(f"Test set: {len(corpus.test_dataset)} sentences")

bilstm = BiLSTM(
    input_dim=len(corpus.word_field.vocab),
    embedding_dim=300,
    hidden_dim=64,
    output_dim=len(corpus.tag_field.vocab),
    lstm_layers=2,
    emb_dropout=0.5,
    lstm_dropout=0.1,
    fc_dropout=0.25,
    word_pad_idx=corpus.word_pad_idx
)
bilstm.init_weights()
bilstm.init_embeddings(word_pad_idx=corpus.word_pad_idx)
print(f"The model has {bilstm.count_parameters():,} trainable parameters.")
print(bilstm)

ner = NER(
    model=bilstm,
    data=corpus,
    optimizer_cls=Adam,
    loss_fn_cls=nn.CrossEntropyLoss
)

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
            if not pred_set.symmetric_difference(gs_set):
                entity_matrix["true_positive"] += 1

            # there is some overlap so the entity has been mispredicted
            # there are no strict rules for this, but it should make some sense
            elif not pred[0] in gs_set:
                entity_matrix["false_positive"] += 1

            #else:
                #entity_matrix["false_negative"] += 1

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


ner.train(10)

with open("./BC5CDR-chem/parsed_data.txt", "r") as f:
    data = list(json.load(f))

confusion_matrix: CounterT[str] = Counter()
token_matrix: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
entity_matrix: CounterT[str] = Counter()

line_list = list()

total = len(data)
counter = 0

for item in data:
    counter += 1
    sys.stdout.write("Predicted {}/{} sentences so far.\r".format(counter, total))
    sys.stdout.flush()

    words, infer_tags, unknown_tokens = ner.infer(sentence=item["sentence"], true_tags=item["labels"])
    cm, tm, em = sentence_metrics(infer_tags, item["labels"])

    confusion_matrix.update(cm)
    entity_matrix.update(em)

    for gs_label in tm:
        for pred_label in tm[gs_label]:
            token_matrix[gs_label][pred_label] += tm[gs_label][pred_label]

    line_list = line_list + list(map(lambda token, gs, pred: token + " TK " + gs + " " + pred, item["sentence"].split(), item["labels"], infer_tags))

conlleval_res = conlleval.report(conlleval.evaluate(line_list))
print(conlleval_res)

# CM
cm_r = confusion_matrix["true_positive"]/(confusion_matrix["true_positive"] + confusion_matrix["false_negative"])
cm_p = confusion_matrix["true_positive"]/(confusion_matrix["true_positive"] + confusion_matrix["false_positive"])
cm_f1 = 2*cm_r*cm_p / (cm_r + cm_p)

# EM
em_r = entity_matrix["true_positive"]/(entity_matrix["true_positive"] + entity_matrix["false_negative"])
em_p = entity_matrix["true_positive"]/(entity_matrix["true_positive"] + entity_matrix["false_positive"])
em_f1 = 2*em_r*em_p / (em_r + em_p)

# TM
b_r = token_matrix["B"]["B"] / (token_matrix["B"]["B"] + token_matrix["B"]["I"] + token_matrix["B"]["O"])
b_p = token_matrix["B"]["B"] / (token_matrix["B"]["B"] + token_matrix["I"]["B"] + token_matrix["O"]["B"])
b_f1 = 2*b_r*b_p / (b_r + b_p)

i_r = token_matrix["I"]["I"] / (token_matrix["I"]["B"] + token_matrix["I"]["I"] + token_matrix["I"]["O"])
i_p = token_matrix["I"]["I"] / (token_matrix["B"]["I"] + token_matrix["I"]["I"] + token_matrix["O"]["I"])
i_f1 = 2*i_r*i_p / (i_r + i_p)

o_r = token_matrix["O"]["O"] / (token_matrix["O"]["B"] + token_matrix["O"]["I"] + token_matrix["O"]["O"])
o_p = token_matrix["O"]["O"] / (token_matrix["B"]["O"] + token_matrix["I"]["O"] + token_matrix["O"]["O"])
o_f1 = 2*o_r*o_p / (o_r + o_p)

with open("./bilstm_metrics.json", "a+") as out_f:
    out_f.write("\nConlleval results:\n" + conlleval_res)
    out_f.write("\nToken-Level Confusion Matrix:\n"
                + "True Positive:\t" + str(confusion_matrix["true_positive"])
                + "\nTrue Negative:\t" + str(confusion_matrix["true_negative"])
                + "\nFalse Positive:\t" + str(confusion_matrix["false_positive"])
                + "\nFalse Negative:\t" + str(confusion_matrix["false_negative"])
                + "\nRecall:\t\t" + str(cm_r)
                + "\nPrecision:\t" + str(cm_p)
                + "\nF1-score:\t" + str(cm_f1))

    out_f.write("\n\nEntity-Level Confusion Matrix:\n"
                + "True Positive:\t" + str(entity_matrix["true_positive"])
                + "\nTrue Negative:\t" + str(entity_matrix["true_negative"])
                + "\nFalse Positive:\t" + str(entity_matrix["false_positive"])
                + "\nFalse Negative:\t" + str(entity_matrix["false_negative"])
                + "\nRecall:\t\t" + str(em_r)
                + "\nPrecision:\t" + str(em_p)
                + "\nF1-score:\t" + str(em_f1))

    out_f.write("\n\nToken Matrix (true\predicted):\n\tB\tI\tO\n"
                + "B\t" + str(token_matrix["B"]["B"]) + "\t" + str(token_matrix["B"]["I"]) + "\t" + str(token_matrix["B"]["O"])
                + "\nI\t" + str(token_matrix["I"]["B"]) + "\t" + str(token_matrix["I"]["I"]) + "\t" + str(token_matrix["I"]["O"])
                + "\nO\t" + str(token_matrix["O"]["B"]) + "\t" + str(token_matrix["O"]["I"]) + "\t" + str(token_matrix["O"]["O"])
                + "\nB_Recall:\t" + str(b_r)
                + "\nB_Precision:\t" + str(b_p)
                + "\nB_F1:\t\t" + str(b_f1)
                + "\nI_Recall:\t" + str(i_r)
                + "\nI_Precision:\t" + str(i_p)
                + "\nI_F1:\t\t" + str(i_f1)
                + "\nO_Recall:\t" + str(o_r)
                + "\nO_Precision:\t" + str(o_p)
                + "\nO_F1:\t\t" + str(o_f1) + "\n")



print("Confusion matrix:")
print({**confusion_matrix})
print("Recall: " + str(cm_r))
print("Precision: " + str(cm_p))
print()

print("Token matrix:")
print({**token_matrix})
print()

print("Entity matrix:")
print({**entity_matrix})
print("Recall: " + str(em_r))
print("Precision: " + str(em_p))
print()


