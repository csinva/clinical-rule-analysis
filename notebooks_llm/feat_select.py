from copy import deepcopy
from typing import List
import numpy as np
import imodelsx.llm
import pandas as pd
import imodels


def raw_to_abbrev(feat_name: str):
    return feat_name.split("_")[0]


def abbrevs_to_idxs_raw(feats_abbrev, feats_raw: pd.Series):
    return feats_raw.apply(lambda x: raw_to_abbrev(x) in feats_abbrev).values


def get_iai_data(outcome):
    df_full = pd.read_pickle(f"../data/pecarn/{outcome}.pkl").infer_objects()
    y = df_full["outcome"].values
    df = df_full.drop(columns=["outcome"])
    X = df.values
    feats_raw = pd.Series(df.columns)

    # remove redundant features
    idxs = feats_raw.str.endswith("_no") | feats_raw.str.endswith("_unknown")

    # remove compound features
    idxs |= feats_raw.str.contains("_or_")

    # remove ambiguous features
    idxs |= feats_raw.str.lower().str.startswith("other")

    # remove specific features
    idxs |= feats_raw.isin(["Age<2_yes"])
    for k in ["LtCostalTender", "RtCostalTender"]:
        idxs |= feats_raw.str.startswith(k)

    # apply
    X = X[:, ~idxs]
    feats_raw = feats_raw[~idxs]
    feats_abbrev_unique = set(feats_raw.apply(raw_to_abbrev))

    return X, y, feats_raw, feats_abbrev_unique


def get_tbi_data(outcome='tbi_young'):
    X, y, feats_raw = imodels.get_clean_dataset(
        "tbi_pecarn_prop", data_source="imodels"
    )
    feats_raw = pd.Series(feats_raw)
    # df = pd.DataFrame(X, columns=feats_raw)

    # remove specific features
    idxs = feats_raw.str.endswith("_nan")
    # idxs |= feats_raw.isin(['AgeTwoPlus', 'AgeInMonth'])
    # for k in ['LtCostalTender', 'RtCostalTender']:
    # idxs |= feats_raw.str.startswith(k)

    # apply
    X = X[:, ~idxs]
    feats_raw = feats_raw[~idxs]
    feats_abbrev_unique = set(feats_raw.apply(raw_to_abbrev))

    # split dataset
    idxs_age = X[:, feats_raw == "AgeTwoPlus"].astype(bool).squeeze()
    if 'young' in outcome:
        idxs_age = ~idxs_age
    print(idxs_age.shape)
    X = X[idxs_age]
    y = y[idxs_age]


    return X, y, feats_raw, feats_abbrev_unique


ABBREV_TO_CLEAN_IAI = {
    "AbdDistention": "Abdominal distention",
    "AbdTenderDegree": "Degree of abdominal tenderness",
    "AbdTrauma": "Abdominal wall trauma",
    "AbdomenPain": "Abdominal pain",
    "Age": "Age",
    "CostalTender": "Costal margin tenderness",
    "DecrBreathSound": "Decreased breath sounds",
    "DistractingPain": "Distracting pain",
    "GCSScore": "GCS score",
    "Hypotension": "Hypotension",
    "InitHeartRate": "Heart rate",
    "InitSysBPRange": "Systolic blood pressure",
    "LtCostalTender": "Left costal tenderness",
    "MOI": "Mechanism of injury",
    "Race": "Race",
    "RtCostalTender": "Right costal tenderness",
    "SeatBeltSign": "Seatbelt sign",
    "Sex": "Sex",
    "ThoracicTender": "Thoracic tenderness",
    "ThoracicTrauma": "Thoracic trauma",
    "VomitWretch": "Vomiting",
}
PECARN_FEATS_ORDERED_IAI = [
    "AbdTrauma",
    "SeatBeltSign",
    "GCSScore",
    "AbdTenderDegree",
    "ThoracicTrauma",
    "AbdomenPain",
    "DecrBreathSound",
    "VomitWretch",
]

PECARN_FEATS_ORDERED_TBI_YOUNG = {
    "AMS",
    "HemaLoc",
    "LocLen",
    "InjuryMech",
    "SFxPalp",
    "ActNorm",
}

ABBREV_TO_CLEAN_TBI_YOUNG = {
    "AMS": "Altered mental status",
    "HemaLoc": "Scalp haematoma",
    "LocLen": "Loss of consciousness",
    "InjuryMech": "Mechanism of injury",
    "SFxPalp": "Skull fracture",
    "ActNorm": "Acting normally",
}

PECARN_FEATS_ORDERED_TBI_OLD = {
    "AMS",
    "LocLen",
    "Vomit",
    "InjuryMech",
    "SFxBas",
    "HASeverity",
}

ABBREV_TO_CLEAN_TBI_OLD = {
    "AMS": "Altered mental status",
    "LocLen": "Loss of consciousness",
    "Vomit": "Vomiting",
    "InjuryMech": "Mechanism of injury",
    "SFxBas": "Basilar skull fracture",
    "HASeverity": "Sever headache",
}
FEATS = {
    "iai": {
        "abbrev_to_clean": ABBREV_TO_CLEAN_IAI,
        "clean_to_abbrev": {v.lower(): k for k, v in ABBREV_TO_CLEAN_IAI.items()},
        "pecarn_feats_ordered": PECARN_FEATS_ORDERED_IAI,
    },
    "tbi_young": {
        "abbrev_to_clean": ABBREV_TO_CLEAN_TBI_YOUNG,
        "clean_to_abbrev": {v.lower(): k for k, v in ABBREV_TO_CLEAN_TBI_YOUNG.items()},
        "pecarn_feats_ordered": PECARN_FEATS_ORDERED_TBI_YOUNG,
    },
    "tbi_old": {
        "abbrev_to_clean": ABBREV_TO_CLEAN_TBI_OLD,
        "clean_to_abbrev": {v.lower(): k for k, v in ABBREV_TO_CLEAN_TBI_OLD.items()},
        "pecarn_feats_ordered": PECARN_FEATS_ORDERED_TBI_OLD,
    },
}
FEATS["iai-i"] = FEATS["iai"]


def get_llm_feats_ordered(
    feats_abbrev_unique: List[str], strategy="random", seed: int = 42
):
    llm = imodelsx.llm.get_llm(checkpoint=strategy, repeat_delay=1, seed=seed)
    demonstration = """Return the following bulleted list in order of how each feature is for predicting body mass index. First should be the most important.
    - Age
    - Vomiting
    - Weight"""
    demonstration_answer = "- Weight\n- Age\n- Vomiting"
    MESSAGES_INIT = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": demonstration},
        {"role": "assistant", "content": demonstration_answer},
    ]

    feats_clean_unique = sorted(list(map(ABBREV_TO_CLEAN_IAI.get, feats_abbrev_unique)))
    # shuffle list
    # rng.shuffle(feats_clean_unique)
    feats_bulleted_list = "- " + "\n- ".join(feats_clean_unique)
    question = f"""Return the following bulleted list in order of how each feature is for predicting intra-abdominal injury requiring intervention. First should be the most important.
    {feats_bulleted_list}"""
    messages = deepcopy(MESSAGES_INIT)
    messages.append({"role": "user", "content": question})
    bulleted_list_ranked = llm(messages, temperature=0)

    # parse bulleted list
    feats_parsed = bulleted_list_ranked.strip("- ").split("\n- ")
    assert [
        feat for feat in feats_parsed if not feat.lower() in CLEAN_TO_ABBREV_IAI
    ] == []
    assert len(feats_parsed) == len(feats_abbrev_unique), [
        k for k in feats_clean_unique if not k in feats_parsed
    ]
    feats_ordered = [CLEAN_TO_ABBREV_IAI.get(feat.lower()) for feat in feats_parsed]
    return feats_ordered


def get_feats_ordered(
    feats_abbrev_unique: List[str], strategy="random", seed: int = 42
):
    rng = np.random.default_rng(seed)
    if strategy == "random":
        feats_ordered = rng.choice(
            list(feats_abbrev_unique),
            size=len(feats_abbrev_unique),
            replace=False,
        )
        return feats_ordered
    elif strategy == "pecarn":
        remaining_feats = [
            k for k in feats_abbrev_unique if k not in PECARN_FEATS_ORDERED_IAI
        ]
        return (
            list(PECARN_FEATS_ORDERED_IAI)
            + rng.choice(
                remaining_feats, size=len(remaining_feats), replace=False
            ).tolist()
        )
    elif strategy == "pecarn___gpt-4-0314":
        remaining_feats = [
            k for k in feats_abbrev_unique if k not in PECARN_FEATS_ORDERED_IAI
        ]
        feats_ordered = get_llm_feats_ordered(
            remaining_feats, strategy=strategy.split("___")[1], seed=seed
        )
        return list(PECARN_FEATS_ORDERED_IAI) + feats_ordered
    elif "gpt" in strategy:
        return get_llm_feats_ordered(feats_abbrev_unique, strategy=strategy, seed=seed)


# get best next feature
# demonstration = """Which one of the following features is most important for predicting body mass index?
# - Weight
# - Heart rate
# - Vomiting"""
# demonstration_answer = "Weight"
# MESSAGES_INIT = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": demonstration},
#     {"role": "assistant", "content": demonstration_answer},
# ]

# feats_clean_unique = list(map(feat_select.ABBREV_TO_CLEAN.get, feats_abbrev_unique))
# feats_bulleted_list = "- " + "\n- ".join(feats_clean_unique)
# question = f"""Which of the following features is most important for predicting intra-abominal injury requiring intervention?
# {feats_bulleted_list}"""
# messages = deepcopy(MESSAGES_INIT)
# messages.append({"role": "user", "content": question})

# llm(messages)
