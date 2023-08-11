from copy import deepcopy
from typing import List
import numpy as np
import imodelsx.llm


ABBREV_TO_CLEAN = {
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
CLEAN_TO_ABBREV = {v.lower(): k for k, v in ABBREV_TO_CLEAN.items()}
PECARN_FEATS_ORDERED = [
    "AbdTrauma",
    "SeatBeltSign",
    "GCSScore",
    "AbdTenderDegree",
    "ThoracicTrauma",
    "AbdomenPain",
    "DecrBreathSound",
    "VomitWretch",
]


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

    feats_clean_unique = sorted(list(map(ABBREV_TO_CLEAN.get, feats_abbrev_unique)))
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
    assert [feat for feat in feats_parsed if not feat.lower() in CLEAN_TO_ABBREV] == []
    assert len(feats_parsed) == len(feats_abbrev_unique), [
        k for k in feats_clean_unique if not k in feats_parsed
    ]
    feats_ordered = [CLEAN_TO_ABBREV.get(feat.lower()) for feat in feats_parsed]
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
            k for k in feats_abbrev_unique if k not in PECARN_FEATS_ORDERED
        ]
        return (
            list(PECARN_FEATS_ORDERED)
            + rng.choice(
                remaining_feats, size=len(remaining_feats), replace=False
            ).tolist()
        )
    elif strategy == "pecarn___gpt-4-0314":
        remaining_feats = [
            k for k in feats_abbrev_unique if k not in PECARN_FEATS_ORDERED
        ]
        feats_ordered = get_llm_feats_ordered(
            remaining_feats, strategy=strategy.split("___")[1], seed=seed
        )
        return list(PECARN_FEATS_ORDERED) + feats_ordered
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
