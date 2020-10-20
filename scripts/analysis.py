# coding=utf-8

import numpy as np
import pandas as pd
import math
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Tuple
from sklearn.manifold import TSNE

"""
meeting notes (unrelated to this file)
RESULT
- which corpus, look at corpus, this is corpus statistics
- NER with BioBERT, we initialized with weight, fine tuned with this dataset
- token level evaluation results was this and that, say which set (test)
"""

"""
    NOTES

Visualization of recognized entities:
- PMI / PPMI
  - Heatmap from co-occurrence matrix
  - PCA - but our data is probably too high dimensional?
  - "Locking" one word, i.e. looking at how "glutamate" co-occurs with the rest?

Perhaps tables are enough?

https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/calculate-tweet-word-bigrams/
https://www.kaggle.com/itoeiji/simple-co-occurrence-network (same but on Kaggle)
"""


def create_entity_lists(articles: dict) -> List[List[str]]:
    entity_lists: List[List[str]] = []

    for article in articles.values():
        sentences = article["sentences"]
        for i, sentence in enumerate(sentences):
            entities = sentence["entities"]
            entity_lists.append(entities)

    return entity_lists


def create_fake_set_list(n: int) -> List[int]:
    return [x for x in range(1, n+1)]


class TableEntry(NamedTuple):
    x: str
    y: str
    nx: int
    ny: int
    nxy: int
    pmixy: float
    ppmixy: float

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
                "  <th>PPMI</th>\n" +
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
                "  <td>" + "{0:.3f}".format(self.ppmixy) + "</td>\n" +
                "</tr>\n"
                )

    def latex_row(self):
        return (f"{self.x} & {self.y} & {self.nx} & {self.ny} & {self.nxy} & "
                f"{self.pmixy: .3f} & {self.ppmixy: .3f}")


def write_to_html(path, entries):
    with open(path, "w") as f:
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


def make_table_entries(cocount, wcount, n):
    entries: List[TableEntry] = []
    for x, y in cocount.keys():
        pmixy = pmi(x, y, cocount, wcount, n)
        entry = TableEntry(
            x=x,
            y=y,
            nx=wcount[x],
            ny=wcount[y],
            nxy=cocount[(x, y)],
            pmixy=pmixy,
            ppmixy=max(pmixy, 0)
        )
        entries.append(entry)
    return entries


def pmi(x: str, y: str, co: Dict[Tuple[Any, Any], int], wc: Dict[Any, int], n: int) -> float:
    pxy = co[(x, y)]
    px = wc[x]
    py = wc[y]
    return math.log(n * pxy / (px * py))


def create_matrix(ppmis: Dict[Tuple[Any, Any], float], labels: List[Any]):
    n = len(labels)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = labels[i]
            y = labels[j]
            matrix[i][j] = round(ppmis.get((x, y), 0.0), 2)

    df = pd.DataFrame(matrix, index=labels, columns=labels)
    # u, s, v = np.linalg.svd(df, full_matrices=True)
    # svd = np.dot(u, np.diag(s))
    return df


def run(articles: dict, output_path: str):
    entity_lists = create_entity_lists(articles)
    num_words = sum(len(entity_list) for entity_list in entity_lists)

    wcount: Dict[str, int] = defaultdict(int)               # Word count
    cocount: Dict[Tuple[str, str], int] = defaultdict(int)  # Co-occurrence count

    for entity_list in entity_lists:
        entity_list.sort()
        for x, y in it.combinations(entity_list, 2):
            cocount[(x, y)] += 1
        for w in entity_list:
            wcount[w] += 1

    entries = make_table_entries(cocount, wcount, num_words)
    entries.sort(key=lambda entry: entry.pmixy, reverse=True)
    # entries = list(filter(lambda entry: entry.nxy > 50, entries))

    write_to_html(output_path, entries)


if __name__ == "__main__":

    with open("data/analysis_sentences.txt", "r") as f:
        lines = [sentence.strip()[:-1] for sentence in f.readlines()]
        word_lists = [[word.lower() for word in sentence.split(" ")] for sentence in lines]

    num_words = sum(len(word_list) for word_list in word_lists)

    entity_lists = word_lists

    wcount: Dict[str, int] = defaultdict(int)               # Word count
    cocount: Dict[Tuple[str, str], int] = defaultdict(int)  # Co-occurrence count

    for entity_list in entity_lists:
        for x, y in it.combinations(entity_list, 2):
            cocount[(x, y)] += 1
        for w in entity_list:
            wcount[w] += 1

    entries = make_table_entries(cocount, wcount, num_words)
    entries.sort(key=lambda entry: entry.pmixy, reverse=True)
    # entries = list(filter(lambda entry: entry.nxy > 50, entries))

    write_to_html("data/analysis_results.html", entries)

    ppmis = {}
    for x, y in cocount.keys():
        pmixy = pmi(x, y, cocount, wcount, num_words)
        ppmis[(x, y)] = max(pmixy, 0)

    matrix = create_matrix(ppmis, [w for w in wcount.keys()])

    # matrix.style.background_gradient(cmap='viridis')\
    #     .set_properties(**{'font-size': '20px'})

    # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    ############################################################################
    # # https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
    # # http://cmdlinetips.com/2019/05/singular-value-decomposition-svd-in-python/
    # tsne = TSNE(n_components=2).fit_transform(matrix)
    # print(tsne)
    # for x, y in tsne:
    #     plt.scatter(x, y)
    # plt.show()
    ############################################################################

    ############################################################################
    # Displaying dataframe as an heatmap with diverging colourmap as RdYlBu
    # plt.imshow(matrix, cmap="RdYlBu")
    # plt.colorbar()
    # plt.xticks(range(len(matrix)), matrix.columns)
    # plt.yticks(range(len(matrix)), matrix.index)
    # plt.show()
    ############################################################################

    # var_explained = np.round(s**2/np.sum(s**2), decimals=3)

    # sns.barplot(
    #     x=list(range(1, len(var_explained)+1)),
    #     y=var_explained, color="limegreen",
    # )
    # plt.xlabel('SVs', fontsize=16)
    # plt.ylabel('Percent Variance Explained', fontsize=16)
    # plt.savefig('svd_scree_plot.png', dpi=500)
