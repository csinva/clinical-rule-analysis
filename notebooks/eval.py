import pathlib
import re
from typing import Dict, List
import numpy as np


from collections import defaultdict
import fitz
import dvu
import matplotlib.pyplot as plt
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

path_to_file = dirname(__file__)
path_to_repo = dirname(path_to_file)
papers_dir = join(path_to_repo, "papers")


def compute_metrics(
    df,
    preds_col_to_gt_col_dict={
        "num_male": "num_male_corrected",
        "num_female": "num_female_corrected",
        "num_total": "num_total_corrected",
    },
) -> pd.DataFrame:
    d = defaultdict(list)
    for k in df.columns:
        # if k.startswith('num_') and k + '_corrected' in df.columns:
        if k in preds_col_to_gt_col_dict:
            gt_col = preds_col_to_gt_col_dict[k]
            idxs_with_labels = df[gt_col].notnull() & ~(df[gt_col].isin({"Unk", "-"}))
            gt = df[gt_col][idxs_with_labels].astype(int)
            pred = df[k].apply(cast_int)[idxs_with_labels].astype(int)
            recall = (np.abs(gt - pred) <= 1).sum()
            d["target"].append(k)
            d["recall"].append(recall)
            d["n_labeled"].append(len(gt))
            # d['n_predicted'].append(df[k].notnull().sum())
            # count number of values which contain a number
            d["n_predicted_num"].append(df[k].apply(str_contains_number).sum())
    return pd.DataFrame.from_dict(d).round(2)


def process_gender_counts(row):
    """Process counts (convert percentages to nums if conditions are correct)"""
    m = row["num_male"]
    f = row["num_female"]
    tot = row["num_total"]
    if tot is not None and isinstance(tot, str):
        tot = tot.replace(",", "").replace(" ", "")
    if (
        str_contains_number(m)
        and str_is_percentage(m)
        and str_contains_number(f)
        and str_is_percentage(f)
        and str_contains_number(tot)
        and not str_is_percentage(tot)
    ):
        m = percentage_to_num(m)
        f = percentage_to_num(f)
        tot = int(tot)
        # print(m, f, tot)
        m = round(m * tot / 100)
        f = tot - m
    return m, f


def cast_int(x):
    try:
        return int(x)
    except:
        return -1


def int_or_empty(x):
    try:
        return int(x)
    except:
        return ""


def int_or_neg1(x):
    try:
        return int(x)
    except:
        return -1


def str_contains_number(x):
    return (
        x is not None
        and any(char.isdigit() for char in str(x))
        and not any(char.isalpha() for char in str(x))
    )


def str_is_percentage(s):
    return "%" in s or "." in s


def percentage_to_num(s):
    if "%" in s:
        s = s.replace("%", "")
    return float(s)
