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


def cast_int(x):
    try:
        return int(x)
    except:
        return -1

def int_or_empty(x):
    try:
        return int(x)
    except:
        return ''


def str_contains_number(x):
    return x is not None and any(char.isdigit() for char in str(x))

def str_is_percentage(s):
    return '%' in s or '.' in s

def percentage_to_num(s):
    if '%' in s:
        s = s.replace('%', '')
    return float(s)
    