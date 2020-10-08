# coding=utf-8

import math
import itertools as it
import random
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Tuple


def run(articles: dict):
    pass
#     xs = set()

#     a: Dict[str, int] = {}
#     b: Dict[str, int] = {}
#     ab: Dict[Tuple[str, str], int] = {}

#     for pmid, article in articles.items():

#         sentences = article["sentences"]

#         for i, sentence in enumerate(sentences):
#             text = sentence["text"]
#             entities = sentence["entities"]

#             for entity in entities:
#                 xs.add(entity)

#             combs = it.combinations(entities, 2)
#             for x, y in combs:
#                 a[x] += 1
#                 b[y] += 1
#                 ab[(x, y)] += 1

#             # articles[pmid]["sentences"][i]["entities"] = None
#             # articles[pmid]["sentences"][i]["text_new"] = None

#     print(a)


def create_fake_set_list(n: int) -> List[int]:
    return [x for x in range(1, n+1)]


class TableEntry(NamedTuple):
    x: str
    y: str
    nx: int
    ny: int
    nxy: int
    pmixy: float

    def str(self, sep):
        return sep.join(
            map(str, [self.x, self.y, self.nx, self.ny, self.nxy, self.pmixy])
        )

    @staticmethod
    def html_header():
        return ("<tr>\n" +
                "  <th>word 1</th>\n" +
                "  <th>word 2</th>\n" +
                "  <th>count word 1</th>\n" +
                "  <th>count word 2</th>\n" +
                "  <th>count co-occurrences</th>\n" +
                "  <th>PMI</th>\n" +
                "</tr>\n"
                )

    def html_row(self):
        return ("<tr>\n" +
                "  <td>" + str(self.x) + "</td>\n" +
                "  <td>" + str(self.y) + "</td>\n" +
                "  <td>" + str(self.nx) + "</td>\n" +
                "  <td>" + str(self.ny) + "</td>\n" +
                "  <td>" + str(self.nxy) + "</td>\n" +
                "  <td>" + "{0:.3f}".format(self.pmixy) + "</td>\n" +
                "</tr>\n"
                )


def pmi(x: str, y: str, co: Dict[Tuple[Any, Any], int], wc: Dict[Any, int], n: int) -> float:
    pxy = co[(x, y)]
    px = wc[x]
    py = wc[y]
    return math.log(n * pxy / (px * py))


if __name__ == "__main__":

    with open("data/analysis_sentences.txt", "r") as f:
        lines = [sentence.strip()[:-1] for sentence in f.readlines()]
        word_lists = [[word.lower() for word in sentence.split(" ")] for sentence in lines]

    N = sum(len(word_list) for word_list in word_lists)

    ############################################################################
    # samples = []
    # for word_list in word_lists:
    #     sample = random.sample(word_list, random.randint(2, 3))
    #     sample = sorted(sample)
    #     samples.append(sample)
    # samples = sorted(samples)
    # entity_lists = samples
    ############################################################################

    entity_lists = word_lists

    wc: Dict[str, int] = defaultdict(int)
    co: Dict[Tuple[str, str], int] = defaultdict(int)

    for entity_list in entity_lists:
        for x, y in it.combinations(entity_list, 2):
            co[(x, y)] += 1
        for w in entity_list:
            wc[w] += 1

    entries: List[TableEntry] = []
    for x, y in co.keys():
        entry = TableEntry(
            x=x,
            y=y,
            nx=wc[x],
            ny=wc[y],
            nxy=co[(x, y)],
            pmixy=pmi(x, y, co, wc, N),
        )
        entries.append(entry)

    entries = sorted(entries, key=lambda entry: entry.pmixy, reverse=True)
    # entries = list(filter(lambda entry: entry.nxy > 50, entries))

    with open("data/analysis_results.html", "w") as f:
        f.write("<html>\n")
        f.write(
            "<head>\n<style>\n{}</style>\n</head>\n".format(
                # "th,tr {border:1px solid;}\n"
                "* { margin: 0; padding: 0; }\n"
                "td, th { padding: 0.1em 1em; text-align: left; }\n"
                "table tr:nth-child(odd) td { background:#eaeaea; }\n"
            ))
        f.write(
            "<body>\n<table>\n{0}\n{1}\n</table>\n</body>\n".format(
                TableEntry.html_header(),
                "\n".join(entry.html_row() for entry in entries),
            ))
        f.write("</html>")
