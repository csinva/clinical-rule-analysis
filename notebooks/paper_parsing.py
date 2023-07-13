import pathlib
import re
from typing import Dict, List
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


def download_and_check_gsheet():
    def remove_html_tags(text):
        clean = re.compile("<.*?>")
        return re.sub(clean, "", text).strip()

    df = pd.read_csv(
        "https://docs.google.com/spreadsheets/d/1x-epUl-KidVMI-AhMvT-M6nSULvE5JwNi7uF-vI1rbU/gviz/tq?tqx=out:csv&sheet=main",
        skiprows=0,
    )
    df.columns = list(map(remove_html_tags, df.columns))

    df["ref_href"] = pubmed.get_updated_refs(df)

    # check that found papers are present
    ids_with_paper = df[df["found_paper (0=no, 1=yes)"] > 0].id.astype(int).values
    ids_found = sorted(
        [
            int(x.replace(".pdf", ""))
            for x in os.listdir("../papers")
            if x.endswith(".pdf")
        ]
    )
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

    # check that values are integers or Unk
    for col in df.columns:
        if col.startswith('num_') and col.endswith('_corrected'):
            vals = df[col][df[col].notna()].values
            for val in vals:
                assert val in {"Unk"} or round(val) == int(
                    val
                ), f"{col} has {val} which is not an int or Unk"

    return df, ids_with_paper


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


def add_columns_based_on_properties(
    df, ids_with_paper, properties, functions, content_str, llm
):
    # initialize empty columns
    for k in properties.keys():
        if not k in df.columns:
            df.loc[:, k] = None

    # run loop
    for id in tqdm(ids_with_paper):
        i = df[df.id == id].index[0]
        row = df.iloc[i]
        paper_file = join("../papers", str(int(row.id)) + ".txt")

        try:
            real_input = pathlib.Path(paper_file).read_text()
            args = call_on_subsets(
                real_input, content_str=content_str, functions=functions, llm=llm
            )

            # print(json.dumps(args, indent=2))
            if args is not None:
                for k in properties.keys():
                    if k in args:
                        df.loc[i, k] = rename_to_none(args[k])

                        # remove spans if they are not actually contained in the text
                        if "_span" in k:
                            if not _check_evidence(args[k], real_input):
                                df.loc[i, k] = None
        except Exception as e:
            print(row.id, e)
    return df


def call_on_subsets(
    x: str,
    content_str: str,
    functions: List[Dict],
    llm,
    subset_len_tokens=4750,
    max_calls=3,
):
    messages = [
        {
            "role": "user",
            "content": content_str,
        }
    ]
    subset_len_chars = subset_len_tokens * 4

    args = None
    subset_num = 0

    while args is None and subset_num < max_calls:
        subset = x[subset_num * subset_len_chars : (subset_num + 1) * subset_len_chars]

        # if approx_tokens < 6000:
        messages[0]["content"] = content_str.format(input=subset)
        msg = llm(messages, functions=functions, return_str=False, temperature=0.0, verbose=True)
        if msg is not None and msg.get("function_call") is not None:
            args = json.loads(msg.get("function_call")["arguments"])
            return args

        subset_num += 1

        # next segment should have atleast 0.5 * subset_len_chars_left
        if len(x) < (subset_num + 0.5) * subset_len_chars:
            break

    return None


def _check_evidence(ev: str, real_input: str):
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
