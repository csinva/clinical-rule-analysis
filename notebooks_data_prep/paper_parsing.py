import pathlib
import re
from typing import Dict, List
import numpy as np


from collections import defaultdict
import pandas as pd
from os.path import join
import os.path
from tqdm import tqdm
import json
import os
import numpy as np
import openai
from os.path import dirname
from paper_setup import papers_dir
import imodelsx


def get_paper_text(id, papers_dir=papers_dir):
    paper_file = join(papers_dir, str(int(id)) + ".txt")
    real_input = pathlib.Path(paper_file).read_text()
    return real_input


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

    df["paper_contains_race_keywords"] = np.nan

    # run loop
    for id in tqdm(ids_with_paper):
        i = df[df.id == id].index[0]
        row = df.iloc[i]
        try:
            real_input = get_paper_text(row.id, papers_dir)
            df.loc[i, "paper_contains_race_keywords"] = int(_check_keywords(real_input))
        except Exception as e:
            pass
    return df
