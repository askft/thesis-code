import json

with open("./temp/test.tsv", "r") as f:
    lines = f.readlines()
listan = list()
sentence = ""
labels = []
for line in lines:
    if not line == "\n":
        l, r = line.split("\t")
        sentence += " " + l
        labels.append(r.strip())
    else:
        print(labels)
        listan.append({"sentence": sentence[1:], "labels": labels})
        labels = []
        sentence = ""

with open("./parsed_data.txt", "w") as f:
    f.write(json.dumps(listan))