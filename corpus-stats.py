import json
import os
from collections import defaultdict
from typing import DefaultDict, NamedTuple


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
            entry = f"{cols[0]:<24}  &  {cols[1]:>8}  &  {cols[2]:>8} \\\\"
            print(entry)

    # Print out LaTeX table rows, just for convenience
    print()
    print("~~~~~~", s, "~~~~~~")
    print()
    create_latex_table(lines)
    print()


def bc5cdr(input_dir: str, s: str):
    with open(os.path.join(input_dir, s + ".tsv"), "r") as f:
        lines = f.readlines()

    tokens: DefaultDict[str, int] = defaultdict(int)
    labels: DefaultDict[str, int] = defaultdict(int)

    for line in lines:
        if line == "\n":
            continue

        token, label = line.split("\t")
        token = token.strip()
        label = label.strip()

        tokens[token] += 1
        labels[label] += 1

    def perc(count, n): return round(100*count/n, 3)

    # ntokens = sum(item[1] for item in tokens.items())
    # ts = sorted(tokens.items(), key=lambda item: item[1], reverse=True)
    # tf = filter(lambda item: item[1] >= 20, ts)
    # print(f"{'Token':<24} & {'Occurrences':>12} & {'Percentage':>12} \\\\")
    # for token, count in ts[:25]:
    #     print(f"{token:<24} & {count:>12} & {perc(count, nlabels):>12} \\\\")
    # print("Total:", ntokens)

    print()
    print("~~~~~~", s, "~~~~~~")
    print()

    nlabels = sum(item[1] for item in labels.items())
    print(f"{'Label':<8} & {'Occurrences':>12} & {'Percentage':>12} \\\\")
    for label, count in sorted(labels.items()):
        print(f"{label:<8} & {count:>12} & {perc(count, nlabels):>12} \\\\")

    print()
    print("N =", nlabels)
    print()

    # Create list of comma-separated values
    lines = []
    for label, count in sorted(labels.items(), key=lambda item: item[0]):
        lines.append(f"{label},{count},{perc(count, nlabels)}")

    # Write to CSV file
    with open(os.path.join(input_dir, f"stats-{s}.csv"), "w") as f:
        f.write("label,count,percentage\n")
        for line in lines:
            f.write(line+"\n")


if __name__ == "__main__":
    print("——————————————————————————————————————————————————————")
    print("Loading from data/chemprot/")
    print("——————————————————————————————————————————————————————")

    chemprot("data/chemprot/", "dev")
    chemprot("data/chemprot/", "test")
    chemprot("data/chemprot/", "train")

    print()
    print("——————————————————————————————————————————————————————")
    print("Loading from bilstm/BC5CDR-chem/temp/")
    print("——————————————————————————————————————————————————————")

    bc5cdr("bilstm/BC5CDR-chem/temp/", "devel")
    bc5cdr("bilstm/BC5CDR-chem/temp/", "test")
    bc5cdr("bilstm/BC5CDR-chem/temp/", "train_dev")
    bc5cdr("bilstm/BC5CDR-chem/temp/", "train")
