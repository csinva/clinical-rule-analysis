import pathlib
import re
import numpy as np

import mdcalc
from mdcalc import try_or_none

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

plt.style.use("default")
dvu.set_style()


# def download_open_source_papers(df: pd.DataFrame):
# # download papers
# refs = get_updated_refs(df)
# all_ids = df.id
# ids_missing = [str(id) for id in all_ids if id not in ids_found]
# pmids_missing = {}
# for id in ids_missing:
#     ref = refs[df["id"] == int(id)][0]

#     if isinstance(ref, str) and 'pubmed' in ref:
#         paper_id = get_paper_id(ref)
#         # print(id, ref, paper_id)
#         pmids_missing[paper_id] = id
# s = ",".join(list(pmids_missing.keys()))
# # !python -m pubmed2pdf pdf --pmids="{s}"

# # rename each pdf file in pubmed2pdf to its id
# pubmed_papers_dir = pathlib.Path("../pubmed2pdf")
# papers_downloaded = os.listdir(pubmed_papers_dir)
# for paper in papers_downloaded:
#     paper_id = paper.split(".")[0]
#     paper_id = pmids_missing[paper_id]
#     os.rename(
#         join(pubmed_papers_dir, paper),
#         join(pubmed_papers_dir, f"{paper_id}.pdf"),
#     )


def check_papers(df, ids_with_paper, ids_found):
    for paper_id in ids_with_paper:
        if paper_id in ids_found:
            continue
        else:
            print("should have paper", paper_id)

    for paper_id in ids_found:
        if paper_id in ids_with_paper:
            continue
        else:
            print(paper_id, "in local pdfs but not in main.csv")
            idx = df[df.id == paper_id].index[0]
            print(df.loc[idx, "found_paper (0=no, 1=yes)"])
            df.loc[idx, "found_paper (0=no, 1=yes)"] = 1
    return df


def extract_texts_from_pdf(ids, papers_dir="../papers"):
    for id in tqdm(ids):
        paper_file = join(papers_dir, str(id) + ".pdf")
        if pathlib.Path(paper_file).exists():
            with fitz.open(paper_file) as doc:  # open document
                text = chr(12).join([page.get_text() for page in doc])
                text = text.replace("-\n", "")
                pathlib.Path(join(papers_dir, str(id) + ".txt")).write_bytes(
                    text.encode()
                )

def rename_to_none(x: str):
    if x in {"", "unknown", "N/A"}:
        return None
    else:
        return x


def call_on_subsets(x: str, subset_len_tokens=4750, max_calls=3):
    subset_len_chars = subset_len_tokens * 4

    args = None
    subset_num = 0

    while args is None and subset_num < max_calls:
        subset = x[subset_num * subset_len_chars : (subset_num + 1) * subset_len_chars]

        # if approx_tokens < 6000:
        messages[0]["content"] = content_str.format(input=subset)
        msg = llm(messages, functions=functions, return_str=False, temperature=0.0)
        if msg is not None and msg.get("function_call") is not None:
            args = json.loads(msg.get("function_call")["arguments"])
            return args

        subset_num += 1

        # next segment should have atleast 0.5 * subset_len_chars_left
        if len(x) < (subset_num + 0.5) * subset_len_chars:
            break

    return None


def check_evidence(ev: str, real_input: str):
    if ev is not None:
        # remove all whitespace
        ev = "".join(ev.split())
        real_input = "".join(real_input.split())
        return ev.lower() in real_input.lower()
    return False

def cast_int(x):
    try:
        return int(x)
    except:
        return -1