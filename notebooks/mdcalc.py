import re
from typing import List, Tuple
import numpy as np
import pandas as pd

CLEANR = re.compile("<.*?>")


def try_or_none(func):
    def wrapper(*args):
        try:
            return func(*args)
        except:
            return None

    return wrapper


def clean_list_valued_strings(df):
    LIST_VALUED_COLS = [
        "disease_en",
        "system_en",
        "purpose_en",
        "chief_complaint_en",
        "specialty_en",
    ]

    def clean_list_valued_string(s):
        if isinstance(s, list):
            return s
        elif s is None or pd.isna(s):
            return []
        elif isinstance(s, str) and s.startswith("[") and s.endswith("]"):
            return s[1:-1].replace("'", "").replace('"', "").split(", ")
        elif isinstance(s, str):
            return [s]

    for col in LIST_VALUED_COLS:
        # print(df_merged[col])
        df[col] = df[col].apply(clean_list_valued_string)
        assert np.all(
            df[col].apply(lambda x: isinstance(x, list))
        ), "all values are lists"

    return df


def clean_feature_name(feature_name: str):
    return re.sub(CLEANR, "", feature_name).strip()


KEYWORDS_CONTAIN = [
    "Ethnicity",
    "D-dimer",
    "Appetite",
    "Level of consciousness",
    "Chest x-ray",
    "Weight loss",
    "Karnofsky",
    "Dysarthria",
    "Facial palsy",
    "Acidosis",
    "Abdominal pain",
    "Altered mental status",
    "Bilirubin",
    "Eosinophilia",
    "Enthesitis",
    "White blood cell",
    "Erythrocyte sedimentation rate",
    "Estimated blood loss",
    "Female",
    "Ferritin",
    "gestational age",
    "Glasgow Coma Scale",
    "Heart rate",
    "Pulse",
    "Headache",
    "Height",
    "Hematuria",
    "Fever",
    "Blood in stool",
    "Diastolic BP",
    "Hypotension",
    "Hypertension",
    "Immobilized",
    "Immobilization",
    "Insulin",
    "Intraventricular hemorrhage",
    "Intubation",
    "Intubated",
    "Lactate",
    "Length of stay",
    "Leukocyte",
    "Marked change in tone",
    "NIH Stroke Scale",
    " Age ",
    "(Age",
    "Verbal response",
    "Diastolic blood pressure",
    "Systolic pressure",
    "Vomiting",
]
KEYWORDS_CASED_CONTAIN = [
    "ALT",
    "AST",
    "BMI",
    "CRP",
    "C-reactive protein",
    "CD4",
    "CHF",
    "ECG",
    "EKG",
    "HDL",
    "GCS",
    "WBC",
    "INR",
    "LDL",
    "LDH",
    "NIHSS",
    "NYHA",
    "PSA",
    "PaCO₂",
    "PaO₂",
    "PaO2",
    "sBP",
]
KEYWORD_PREFIXES = [
    "Age",
    "ASA",
    "Albumin",
    "Anxiety",
    "Atrial fibrillation",
    "ED visits",
    "EKG",
    "ESR",
    "Endoscopy",
    "BMI",
    "BUN",
    "Biliary",
    "Calcium",
    "Congestive heart failure",
    "Creatinine",
    "Dementia",
    "Distracting ",
    "ECOG",
    "Erythema",
    "Glucose",
    "Hematocrit",
    "Hemoglobin",
    "Race",
    "Regional lymph node",
    "Respiratory rate",
    "Scalp hematoma",
    "Sex",
    "Systolic BP",
    "Male",
    "Nausea",
    "Oxygen saturation",
    "Platelet",
    "Potassium",
    "Pregnancy",
    "Pregnant",
    "SaO₂",
    "Seizure",
    "Sodium",
    "Sp02",
    "SpO₂",
    "Temp",
    "Tremor",
    "Triglyceride",
    "Wheezing",
    "eGFR",
    "Weight",
    "Suidicid",
]
KEYWORDS_MAP = {
    "Systolic Blood Pressure": "Systolic BP",
    "Ethnicity": "Race",
    "White blood cell": "White blood cell count",
    "WBC": "White blood cell count",
    "Sex": "Gender",
    "CRP": "C-reactive protein",
    "Diminished breath sounds": "Decreased breath sounds",
    "EKG": "ECG",
    "Female": "Gender",
    "GCS": "Glasgow Coma Scale",
    "Glasgow Coma Score": "Glasgow Coma Scale",
    "Pulse": "Heart rate",
    "Immobilization": "Immobilized",
    "Intubation": "Intubated",
    "Leukocyte": "White blood cell count",
    "Vomiting": "Nausea/vomiting",
    "NIHSS": "NIH Stroke Scale",
    "Obesity": "BMI",
    "O₂ sat": "Oxygen saturation",
    "PaO₂": "PaO2",
    "Patient age": "Age",
    "Patient sex": "Sex",
    "Persistent vomiting": "Nausea/vomiting",
    "Platelet": "Platelet count",
    "Pregnant": "Pregnancy",
    "SpO₂": "Oxygen saturation",
    "Sp02": "Oxygen saturation",
    "Suicid": "Suicidality",
    "Temp": "Temperature",
    "Tremor": "Tremors",
    "Triglyceride": "Triglycerides",
    "sBP": "Systolic BP",
    "Diastolic blood pressure": "Diastolic BP",
    "Systolic pressure": "Systolic BP",
}
KEYWORD_PREFIXES_CASED_MAP = {
    "HR": "Heart rate",
}
KEYWORD_RENAME_FINAL_MAP = {
    "Race": "Race/Ethnicity",
    "Gender": "Sex/Gender",
    "Male": "Sex/Gender",
    "Nausea": "Nausea/vomiting",
    "Vomiting": "Nausea/vomiting",
    "White": "Race/Ethnicity",
}

def rename_feature_name(feature_name: str):
    # remove units from a feature
    feature_name = feature_name.replace('"', "")
    feature_name = feature_name.strip()
    feature_name = feature_name.replace(", mmHg", "")

    # if word contains the keyword_contain, rename it to keyword_contain

    for keyword in KEYWORDS_CONTAIN:
        if keyword.lower() in feature_name.lower():
            k = keyword.strip(".,:;!?()")
            feature_name = k

    for keyword in KEYWORDS_CASED_CONTAIN:
        if keyword in feature_name:
            feature_name = keyword

    # if word starts with keyword_prefix, rename it to the prefix

    for keyword in KEYWORD_PREFIXES:
        if feature_name.lower().startswith(keyword.lower()):
            feature_name = keyword

    # remap specific words to other names

    for keyword in KEYWORDS_MAP.keys():
        if feature_name.lower().startswith(keyword.lower()):
            feature_name = KEYWORDS_MAP[keyword]

    for keyword in KEYWORD_PREFIXES_CASED_MAP.keys():
        if feature_name.startswith(keyword):
            feature_name = KEYWORD_PREFIXES_CASED_MAP[keyword]

    # if word ends with keyword_suffix, rename it to the suffix
    KEYWORD_SUFFIXES = ["Saline", "Race"]
    for keyword in KEYWORD_SUFFIXES:
        if feature_name.lower().endswith(keyword.lower()):
            feature_name = keyword

    # final cleanup

    feature_name = KEYWORD_RENAME_FINAL_MAP.get(feature_name, feature_name)
    feature_name = clean_feature_name(feature_name)
    # remove leading/trailing punctuation

    return feature_name


def get_feature_score_tuples_list_from_schema(schema) -> List[Tuple[str, float]]:
    """For each feature in the schema, calculate the number of points it can contribute
    and normalize by the total number of points in the schema."""
    if isinstance(schema, list):
        feature_names_with_vals = []
        tot_points = 0
        for s in schema:
            feature_name = (
                clean_feature_name(s["label_en"]) if "label_en" in s else "unknown"
            )
            if not s == "unknown":
                feature_name = rename_feature_name(feature_name)
                if "options" in s:
                    points = [opt["value"] for opt in s["options"]]
                    point_range = max(points) - min(points)
                    tot_points += point_range
                else:  # example: age text box
                    # print('feature_name', feature_name, 'has no options')
                    # point_range = None
                    return []  # skip anything that isn't all options for now
                feature_names_with_vals.append((feature_name, point_range))

        # normalize by tot_points
        return [
            (feature_name, point_range / tot_points)
            for (feature_name, point_range) in feature_names_with_vals
        ]
    else:
        return []
