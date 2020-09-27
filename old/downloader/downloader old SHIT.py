import json
import os
import pprint
import pubmed_parser as pp
import requests
import shutil
from requests.models import HTTPError
import xmltodict

# https://www.ncbi.nlm.nih.gov/pmc/tools/ftp/
# https://ftp.ncbi.nlm.nih.gov/pub/pmc/
# ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/
# https://titipata.github.io/pubmed_parser/api.html
# https://github.com/titipata/pubmed_parser

# From meeting:
#
# Examples of link with 10 articles about covid
# https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=32247212,32493597,32380453,32130038,32425321,32141569,32461198,32373993,32691024,32178975&retmode=text&rettype=abstract
#
# two articles but XML (see retmode)
# https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=17284678,9997&retmode=xml&rettype=abstract
#
# https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC7102662
#
# use requests library for these ^
#
# example logbook
# https://github.com/AnttonLA/BINP37/blob/master/logbook.ipynb
#
# format conversion
# https://github.com/Aitslab/BioNLP/tree/master/formatconversion


def build_api_url(pmid, retmode="xml"):
    return ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            "?db=pubmed&id={}&retmode={}&rettype=abstract"
            ).format(pmid, retmode)


def make_batches(xs, size):
    for i in range(0, len(xs), size):
        yield xs[i:i+size]


def download_from_pmid_list(path, batch_size=1):

    lines = []
    for line in open("pmids/example.txt", "r"):
        lines.append(line.strip())

    pmid_batches = []
    for batch in make_batches(lines, batch_size):
        pmid_batches.append(batch)

    pmid_lists = []
    for batch in pmid_batches:
        pmid_lists.append(",".join(batch))

    print()

    for pmid_list in pmid_lists:
        print("———————" + pmid_list + "———————")
        print()

        api_url = build_api_url(pmid_list, retmode="xml")
        res = requests.get(api_url)

        if res.status_code != 200:
            raise HTTPError(res.reason)

        d = xmltodict.parse(res.text)

        # with open("xml/tmp.xml", "w") as f:
        #     f.write(res.text)

        # medline_json = pp.parse_medline_xml("xml/tmp.xml")

        # for article in medline_json:
        #     with open("json/{}.json".format(article["pmid"]), "w") as f:
        #         f.write(json.dumps(article, indent=2))

        articles = d["PubmedArticleSet"]["PubmedArticle"]

        # When using xmltodict to convert a multiple-article request, the
        # PubmedArticle tag becomes a list instead of a dict.
        if isinstance(articles, dict):
            articles = [articles]

        for article in articles:
            out = {"PubmedArticleSet": {"PubmedArticle": article}}
            pmid = article["MedlineCitation"]["PMID"]["#text"]
            xml = xmltodict.unparse(out, pretty=True, full_document=False)

            # todo: write to /tmp/ instead?
            with open("xml/{}.xml".format(pmid), "w") as f:
                f.write(xml)

            medline_json = pp.parse_medline_xml("xml/{}.xml".format(pmid))

            with open("json/{}.json".format(pmid), "w") as f:
                f.write(json.dumps(medline_json, indent=2))

        shutil.rmtree("xml")



def stuff(t):
    with open("xml/tmp.xml", "w") as f:
        f.write(t)

    medline_json = pp.parse_medline_xml("xml/tmp.xml")

    for article in medline_json:
        with open("json/{}.json".format(article["pmid"]), "w") as f:
            f.write(json.dumps(article, indent=2))


if __name__ == "__main__":
    os.makedirs("json", exist_ok=True)
    os.makedirs("xml", exist_ok=True)
    download_from_pmid_list(path="pmids/sample.txt", batch_size=2)

# ———————————————————————————————————————————————————————————————————————————
# def download_from_pmid_list(path, batch_size=1):

#     lines = []
#     for line in open("pmids/example.txt", "r"):
#         lines.append(line.strip())

#     pmid_batches = []
#     for batch in make_batches(lines, batch_size):
#         pmid_batches.append(batch)

#     pmid_lists = []
#     for batch in pmid_batches:
#         pmid_lists.append(",".join(batch))
#     # pmid_lists = map(lambda l: ",".join(l), pmid_batches)

#     print()
#     for pmid_list in pmid_lists:
#         print("———————" + pmid_list + "———————")
#         print()

#         api_url = build_api_url(pmid_list, retmode="xml")
#         res = requests.get(api_url)

#         if res.status_code != 200:
#             raise HTTPError(res.reason)

#         d = xmltodict.parse(res.text)
#         stuff(res.text)  # ooo
#         continue         # ooo
#         articles = d["PubmedArticleSet"]["PubmedArticle"]

#         # When using xmltodict to convert a multiple-article request, the
#         # PubmedArticle tag becomes a list instead of a dict.
#         if isinstance(articles, dict):
#             articles = [articles]

#         for article in articles:
#             out = {"PubmedArticleSet": {"PubmedArticle": article}}
#             pmid = article["MedlineCitation"]["PMID"]["#text"]
#             xml = xmltodict.unparse(out, pretty=True, full_document=False)

#             # todo: write to /tmp/ instead?
#             with open("xml/{}.xml".format(pmid), "w") as f:
#                 f.write(xml)

#             medline_json = pp.parse_medline_xml("xml/{}.xml".format(pmid))

#             with open("json/{}.json".format(pmid), "w") as f:
#                 f.write(json.dumps(medline_json, indent=2))

#         shutil.rmtree("xml")
# ———————————————————————————————————————————————————————————————————————————

# ———————————————————————————————————————————————————————————————————————————

# ———————————————————————————————————————————————————————————————————————————

# ———————————————————————————————————————————————————————————————————————————


def read_one_from_web():
    pmid = "11250746"
    d = pp.parse_xml_web(pmid, save_xml=True)
    xml = d["xml"]
    d.pop("xml", None)

    pprint.pprint(d)
    print()

    os.makedirs("xml", exist_ok=True)

    with open("xml/{}.xml".format(pmid), "wb") as f:
        f.write(xml)


# def download_many():
#     with open("pmids/covid10k.txt", "r") as f:
#         lines = f.readlines()
#     pmids = ",".join(map(lambda s: s.strip(), lines[:2]))
#     print(pmids)

#     retmode = "xml"
#     api_url = build_api_url(pmids, retmode)
#     r = requests.get(api_url)
#     if r.status_code != 200:
#         print("error")
#     ## UNNECESSARY, use parse_medline_xml #####################################
#     js = json.dumps(xmltodict.parse(r.text))
#     d = json.loads(js)
#     articles = d["PubmedArticleSet"]["PubmedArticle"]
#     for article in articles:
#         mc = article["MedlineCitation"]
#         a = mc["Article"]
#         title = a["ArticleTitle"]
#         abstract = a["Abstract"]
#         pd = article["PubmedData"]
#     ###########################################################################

#     for pmid in pmids.split(","):
#         with open("json/{}.json".format(pmid), "w") as f:
#             f.write(js)

    # d = pp.parse_pubmed_xml("xml/11250746.xml")
    # pprint.pprint(d)

# def download_many_old():
#     with open("pmids/covid10k.txt", "r") as f:
#         lines = f.readlines()
#     for line in lines[:1]:
#         pmid = line.strip()
#         retmode = "xml"
#         api_url = build_api_url(pmid, retmode)
#         r = requests.get(api_url)
#         if r.status_code != 200:
#             print("error")
#         # print(r.text)
#         # with open("xml/{}.xml".format(pmid), "w") as f:
#         #     f.write(r.text)

#         j = json.dumps(xmltodict.parse(r.text))
#         os.makedirs("json", exist_ok=True)
#         with open("json/{}.json".format(pmid), "w") as f:
#             f.write(j)

#         # d = pp.parse_pubmed_xml("xml/11250746.xml")
#         # pprint.pprint(d)
