from ner_inference import NERInferenceSession

ner = NERInferenceSession("", ["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"], "in", "./out")
ner.predict()

