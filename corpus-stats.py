import json
import os
from collections import defaultdict
from typing import DefaultDict


def chemprot(corpus_dir: str, s: str):
    counter: DefaultDict[str, int] = defaultdict(int)
    with open(os.path.join(corpus_dir, s + ".txt"), "r") as f:
        lines = f.readlines()

    for line in lines:
        data = json.loads(line)
        label = data["label"]
        counter[label] += 1

    lines = []

    # Create list of comma-separated values
    for label, count in sorted(counter.items(), key=lambda item: item[0]):
        n = sum(count for count in counter.values())
        lines.append(f"{label},{count},{round(100*count/n, 3)}")

    # Write to CSV file
    with open(os.path.join(corpus_dir, f"stats-{s}.csv"), "w") as f:
        f.write("label,count,percentage\n")
        for line in lines:
            f.write(line+"\n")

    def create_latex_table(rows):
        for row in rows:
            cols = row.split(",")
            entry = f"{cols[0]:<24}  &  {cols[1]:>8}  &  {cols[2]:>8} \\%" + "  \\\\"
            print(entry)

    # Print out LaTeX table rows, just for convenience
    print(s)
    create_latex_table(lines)
    print()


# work in progress
def bc5cdr(input_dir: str, s: str):
    with open(input_dir, "r") as f:
        lines = f.readlines()

    results = []
    sentence = ""
    labels = []
    for line in lines:
        if not line == "\n":
            l, r = line.split("\t")
            sentence += " " + l
            labels.append(r.strip())
        else:
            results.append({"sentence": sentence[1:], "labels": labels})
            labels = []
            sentence = ""

    with open("./parsed_data.txt", "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    chemprot("data/chemprot/", "dev")
    chemprot("data/chemprot/", "test")
    chemprot("data/chemprot/", "train")

    # bc5cdr("bilstm/BC5CDR-chem/temp/", "devel")
    # bc5cdr("bilstm/BC5CDR-chem/temp/", "test")
    # bc5cdr("bilstm/BC5CDR-chem/temp/", "train_dev")
    # bc5cdr("bilstm/BC5CDR-chem/temp/", "train")
