import signal
import json
import os
import pubmed_parser as pp
import requests
import shutil
import sys


def build_api_url(pmid, retmode="xml"):
    return ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            "?db=pubmed&id={}&retmode={}&rettype=abstract"
            ).format(pmid, retmode)


def make_batches(xs, size):
    for i in range(0, len(xs), size):
        yield xs[i:i+size]


def article_stream(path, batch_size):
    lines = []
    for line in open(path, "r"):
        lines.append(line.strip())

    pmid_batches = []
    for batch in make_batches(lines, batch_size):
        pmid_batches.append(batch)

    pmid_lists = []
    for batch in pmid_batches:
        pmid_lists.append(",".join(batch))

    for pmid_list in pmid_lists:
        print("Doing " + pmid_list.replace(",", ", ") + "...")

        api_url = build_api_url(pmid_list, retmode="xml")
        res = requests.get(api_url)

        if res.status_code != 200:
            raise requests.HTTPError(res.reason)

        with open("xml/tmp.xml", "w") as f:
            f.write(res.text)

        medline_json = pp.parse_medline_xml("xml/tmp.xml")

        for article in medline_json:
            yield article


def main(path, batch_size=1):
    os.makedirs("json", exist_ok=True)
    os.makedirs("xml", exist_ok=True)

    stream = article_stream(path, batch_size)

    with open("example.json", "w") as f:
        f.write(json.dumps(list(stream), indent=2))

    shutil.rmtree("xml")


# def exit_gracefully(signum, frame):
#     # restore the original signal handler as otherwise evil things will happen
#     # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
#     signal.signal(signal.SIGINT, original_sigint)

#     try:
#         if input("\nReally quit? (y/n) ").lower().startswith('y'):
#             sys.exit(1)

#     except KeyboardInterrupt:
#         print("\nOK, quitting.")
#         sys.exit(1)

#     # restore the exit gracefully handler here
#     signal.signal(signal.SIGINT, exit_gracefully)


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

    # original_sigint = signal.getsignal(signal.SIGINT)
    # signal.signal(signal.SIGINT, exit_gracefully)

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.exit("usage: {} input_path [batch_size]".format(sys.argv[0]))

    input_file = sys.argv[1]
    batch_size = 400

    if len(sys.argv) == 3:
        try:
            batch_size = int(sys.argv[2])
        except ValueError:
            sys.exit("error: batch_size must be an integer")

    print("input_file = {}".format(input_file))
    print("batch_size = {}".format(batch_size))

    main(path=input_file, batch_size=batch_size)
