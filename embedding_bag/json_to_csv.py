import json
import os

'''
Short script for converting the dataset available from the SciBERT repository to fit the EmbeddingBag-classifier
Removes entity markers and converts the JSON-format into CSV
'''


path = ".data/chemprot/"

labels = ["INHIBITOR", "SUBSTRATE", "INDIRECT-DOWNREGULATOR",
          "INDIRECT-UPREGULATOR", "ACTIVATOR","ANTAGONIST",
          "PRODUCT-OF", "AGONIST", "DOWNREGULATOR",
          "UPREGULATOR", "AGONIST-ACTIVATOR", "SUBSTRATE_PRODUCT-OF",
          "AGONIST-INHIBITOR"]


for file in os.listdir(path):
    out = file.split(".")[0]
    with open(path + file, "r") as f, open(path + out + ".csv", "w") as out_f:
        for line in f.readlines():
            data = json.loads(line)
            # removes entity markers
            text = data["text"].replace("<< ", "").replace(" >>", "").replace("[[ ", "").replace(" ]]", "")
            index = labels.index(data["label"])

            out_f.write(f'"{index}", "{text}"\n')