# coding=utf-8

import sys


def main(fin, fout):
    with open(fin, "r") as f:
        lines = f.readlines()

    tokens = list(map(lambda line: line.split("\t")[0], lines))
    sentences = " ".join(tokens).split("\n")

    with open(fout, "w") as f:
        for sentence in sentences:
            f.write(sentence.strip() + "\n")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: python process_data.py input output")
        sys.exit(1)
    fin, fout = sys.argv[1], sys.argv[2]
    main(fin, fout)
