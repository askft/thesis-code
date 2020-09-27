# coding=utf-8

import json
import os
import pubmed_parser as pp
import requests
import shutil
import sys

from typing import Any, List


def make_batches(xs: List[Any], size: int):
    for i in range(0, len(xs), size):
        yield xs[i:i+size]


def article_list(input_file: str, output_file: str, batch_size: int):
    lines = []
    for line in open(input_file, "r"):
        lines.append(line.strip())

    pmid_batches = []
    for batch in make_batches(lines, batch_size):
        pmid_batches.append(batch)

    i = 0
    n = 0
    for pmid_batch in pmid_batches:
        i += 1
        n += len(pmid_batch)
        print("Downloading and saving batch {}...".format(i))

        api_url = build_api_url(pmid_batch, retmode="xml")
        new_data = download_data(api_url)
        append_json(output_file, new_data)

        print("Saved {}/{} articles so far.\n".format(n, len(lines)))


def build_api_url(pmid_list: List[str], retmode="xml"):
    return ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            "?db=pubmed&id={}&retmode={}&rettype=abstract"
            ).format(",".join(pmid_list), retmode)


def download_data(api_url: str):
    res = requests.get(api_url)
    if res.status_code != 200:
        raise requests.HTTPError(res.reason)

    with open("tmp/medline.xml", "w") as f:
        f.write(res.text)

    medline_json_list = pp.parse_medline_xml("tmp/medline.xml")

    # Map PMID to article
    new_data = {}
    for article in medline_json_list:
        new_data[article["pmid"]] = article

    return new_data


def append_json(path: str, new_data: dict):
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("{}")

    with open(path, "r") as f:
        old_data = json.loads(f.read())

    data = {**old_data, **new_data}  # Merge dicts (new overwrites old)

    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False))


def run(input_file: str, output_file: str, batch_size: int):
    os.makedirs("json", exist_ok=True)
    os.makedirs("tmp", exist_ok=True)

    try:
        article_list(input_file, output_file, batch_size)
    except KeyboardInterrupt:
        pass

    shutil.rmtree("tmp")


"""
———————————————————————————————————————————————————————————————————————————————
Get research paper abstracts from list of PMIDs.

Arguments:
    input_file - path to .txt file with list of newline-separated PMIDs.
    batch_size - how many articles to download each API call (default: 10).

TODO: Specify output file when calling arguments.
———————————————————————————————————————————————————————————————————————————————
"""
if __name__ == "__main__":

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.exit("usage: {} input_path [batch_size]".format(sys.argv[0]))

    input_file = sys.argv[1]
    batch_size = 400

    if len(sys.argv) == 3:
        try:
            batch_size = int(sys.argv[2])
        except ValueError:
            sys.exit("error: batch_size must be an integer")

    output_file = "outpuuuuut.json"

    print("input_file  = {}".format(input_file))
    print("output_file = {}".format(output_file))
    print("batch_size  = {}".format(batch_size))
    print()

    run(
        input_file=input_file,
        output_file=output_file,
        batch_size=batch_size
    )
