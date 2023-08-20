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
import imodelsx
import prompts_extraction

openai.api_key = open("/home/chansingh/.OPENAI_KEY").read().strip()
imodelsx.llm.LLM_CONFIG["LLM_REPEAT_DELAY"] = 1


def extract_nums_df(texts: List[str], repeat_delay=3) -> pd.DataFrame:
    """Return dataframe with different extracted fields as columns"""

    # get prompt
    llm = imodelsx.llm.get_llm("gpt-4-0613", repeat_delay=repeat_delay)  # gpt-3.5-turbo-0613

    # properties, functions, content_str = prompts_extraction.get_prompts_gender_and_race()
    # print('attempting to add', properties.keys())
    # add_columns_based_on_properties(df, ids_with_paper, properties, functions, content_str, llm)

    properties, functions, content_str = prompts_extraction.get_prompts_gender()
    print("attempting to add", properties.keys())
    extractions1 = extract_columns_based_on_properties(
        texts, properties, functions, content_str, llm
    )

    properties, functions, content_str = prompts_extraction.get_prompts_race()
    print("attempting to add", properties.keys())
    extractions2 = extract_columns_based_on_properties(
        texts, properties, functions, content_str, llm
    )
    return pd.DataFrame.from_dict(extractions1 | extractions2)


def rename_to_none(x: str):
    if x in {"", "unknown", "N/A"}:
        return None
    else:
        return x


def extract_columns_based_on_properties(
    texts,
    properties,
    functions,
    content_str,
    llm,
) -> Dict[str, List]:
    # initialize empty columns
    out = {}
    for k in properties.keys():
        out[k] = len(texts) * [None]

    # run loop
    for i, text in tqdm(enumerate(texts)):
        try:
            args = call_on_subsets(
                text, content_str=content_str, functions=functions, llm=llm
            )
            if args is not None:
                for k in properties.keys():
                    if k in args:
                        out[k][i] = rename_to_none(args[k])

                        # remove spans if they are not actually contained in the text
                        if "_span" in k:
                            if not _check_evidence(args[k], text):
                                out[k][i] = None
        except Exception as e:
            print(e)
    return out


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
