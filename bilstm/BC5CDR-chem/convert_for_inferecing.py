import json

with open("./temp/devel.tsv", "r") as f:
    lines = f.readlines()

parsed = list()
sentence = ""
labels = []
for line in lines:
    if not line == "\n":
        l, r = line.split("\t")
        sentence += " " + l
        labels.append(r.strip())
    else:
        print(labels)
        parsed.append({"sentence": sentence[1:], "labels": labels})
        labels = []
        sentence = ""

with open("./parsed_data.txt", "w") as f:
    f.write(json.dumps(parsed))
