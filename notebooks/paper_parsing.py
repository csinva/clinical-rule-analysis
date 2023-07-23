import pathlib
import re
from typing import Dict, List
import numpy as np


from collections import defaultdict
import fitz
import pandas as pd
from os.path import join
import os.path
from tqdm import tqdm
import json
import os
import numpy as np
import pubmed
import openai
from os.path import dirname
from paper_setup import papers_dir


def rename_to_none(x: str):
    if x in {"", "unknown", "N/A"}:
        return None
    else:
        return x


def add_columns_based_on_properties(
    df,
    ids_with_paper,
    properties,
    functions,
    content_str,
    llm,
    papers_dir=papers_dir,
):
    # initialize empty columns
    for k in properties.keys():
        if not k in df.columns:
            df.loc[:, k] = None

    # run loop
    for id in tqdm(ids_with_paper):
        i = df[df.id == id].index[0]
        row = df.iloc[i]
        paper_file = join(papers_dir, str(int(row.id)) + ".txt")

        try:
            real_input = pathlib.Path(paper_file).read_text()
            args = call_on_subsets(
                real_input, content_str=content_str, functions=functions, llm=llm
            )
            # print('args', args)
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
        msg = llm(
            messages,
            functions=functions,
            return_str=False,
            temperature=0.0,
            verbose=True,
        )
        if msg is not None and "function_call" in msg["choices"][0]["message"]:
            args = json.loads(
                msg["choices"][0]["message"]["function_call"]["arguments"]
            )
            # and msg.get("function_call") is not None:
            # args = json.loads(msg.get("function_call")["arguments"])
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


def check_race_keywords(
    df,
    ids_with_paper,
    KEYWORDS={
        "asian",
        "caucasian",
        "african",
        "latino",
        "hispanic",
    },
):
    def _check_keywords(text):
        text = text.lower()
        for k in KEYWORDS:
            if k in text:
                return True
        return False

    df["paper_contains_keywords"] = ""
    # run loop
    for id in tqdm(ids_with_paper):
        i = df[df.id == id].index[0]
        row = df.iloc[i]
        try:
            paper_file = join(papers_dir, str(int(row.id)) + ".txt")
            real_input = pathlib.Path(paper_file).read_text()
            df.loc[i, 'paper_contains_race_keywords'] = int(_check_keywords(real_input))
        except Exception as e:
            pass
    return df
