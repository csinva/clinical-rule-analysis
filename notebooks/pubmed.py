import pathlib
import re
import numpy as np

import notebooks.mdcalc as mdcalc
from notebooks.mdcalc import try_or_none

from collections import defaultdict
import fitz
import dvu
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import os.path
from bs4 import BeautifulSoup
from tqdm import tqdm
import imodelsx.llm
import json
import requests
import joblib
import os
import numpy as np
import pubmed
import openai
from notebooks.mdcalc import try_or_none

plt.style.use("default")
dvu.set_style()


@try_or_none
def parse_name(name: str):
    name_arr = name.split()

    # drop if too long
    if len(name) > 40:
        return None

    # drop special names
    for k in [
        "investigator",
        "group",
        "committee",
        "network",
    ]:
        if k in name.lower():
            return None

    # drop when first name is only one letter
    if len(name_arr[0]) == 1:
        return None

    # drop middle initial
    if len(name_arr) > 2 and len(name_arr[1]) == 1:
        name_arr = [name_arr[0], name_arr[-1]]

    # return name
    return " ".join(name_arr)


def get_metadata(paper_id: str):
    cache_file = f"../data/metadata/{paper_id}.json"
    if os.path.exists(cache_file):
        metadata = json.load(open(cache_file))
    else:
        resp = requests.get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={paper_id}&retmode=json"
        )
        metadata = json.loads(resp.text)
        with open(cache_file, "w") as f:
            json.dump(metadata, f, indent=2)
    return metadata


def get_authors_with_firstname(paper_link: str, paper_id: str):
    cache_file = f"../data/metadata/{paper_id}_full.joblib"
    if os.path.exists(cache_file):
        return joblib.load(cache_file)['author_names']
    else:
        resp = requests.get(paper_link).text
        soup = BeautifulSoup(resp)
        author_names = set()
        # print(soup.find_all("span", {"class": "authors-list-item"}))
        for s in soup.find_all("span", {"class": "authors-list-item"}):
            try:
                author_name = s.a["data-ga-label"]
                author_names.add(author_name)
                # print('author_name', author_name)
            except:
                pass
        # print('a', author_names)
        joblib.dump({"author_names": author_names, "resp": resp}, cache_file)
        return author_names


@try_or_none
def get_free_text_link(paper_id: str):
    cache_file = f"../data/metadata/{paper_id}_free_text_link.json"
    if os.path.exists(cache_file):
        free_text_link = json.load(open(cache_file))
    else:
        resp = requests.get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&id={paper_id}&cmd=prlinks&retmode=json"
        )
        free_text_link = resp.json()
        with open(cache_file, "w") as f:
            json.dump(free_text_link, f, indent=2)

    return free_text_link["linksets"][0]["idurllist"][0]["objurls"][0]["url"]["value"]


def extract_texts_from_pdf(ids, papers_dir='../papers'):
    for id in tqdm(ids):
        paper_file = join(papers_dir, str(id) + ".pdf")
        if pathlib.Path(paper_file).exists():
            with fitz.open(paper_file) as doc:  # open document
                text = chr(12).join([page.get_text() for page in doc])
                text = text.replace('-\n', '')
                pathlib.Path(join(papers_dir, str(id) + ".txt")).write_bytes(
                    text.encode()
                )

def get_paper_id(paper_link: str):
    if paper_link.endswith("/"):
        paper_link = paper_link[:-1]
    paper_id = paper_link.split("/")[-1]

    # remove leading zeros
    while paper_id.startswith("0"):
        paper_id = paper_id[1:]
    return paper_id

def get_updated_refs(df):
    refs = df['ref_href'].values
    idxs_corrected = (df["ref_href_corrected"].notna() & ~(df['ref_href_corrected'] == 'Unk'))
    refs[idxs_corrected] = df["ref_href_corrected"][idxs_corrected]
    return refs