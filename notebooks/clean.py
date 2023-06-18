import re
import numpy as np

CLEANR = re.compile("<.*?>")


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
    return re.sub(CLEANR, "", feature_name)


def rename_feature_name(feature_name: str):
    # remove units from a feature
    feature_name = feature_name.replace('"', "")
    feature_name = feature_name.strip()
    feature_name = feature_name.replace(", mmHg", "")

    # if word starts with keyword_prefix, rename it to the prefix
    keyword_prefixes = [
        "Age",
        "ASA",
        "Albumin",
        "ED visits",
        "EKG",
        "ESR",
        "Endoscopy",
        "BMI",
        "BUN",
        "Biliary",
        "Bilirubin",
        "Blood in stool",
        "Congestive heart failure",
        "Creatinine",
        "D-dimer",
        "Dementia",
        "Diastolic BP",
        "Distracting ",
        "ECOG",
        "Glucose",
        "Hematocrit",
        "Hemoglobin",
        "Pulse",
        "Race",
        "Regional lymph node",
        "Respiratory rate",
        "Scalp hematoma",
        "Sex",
        "Systolic BP",
        "WBC",
        "White blood",
    ]
    for keyword in keyword_prefixes:
        if feature_name.startswith(keyword):
            feature_name = keyword

    # if word ends with keyword_suffix, rename it to the suffix
    keyword_suffixes = ["saline"]
    for keyword in keyword_suffixes:
        if feature_name.endswith(keyword):
            feature_name = keyword

    # if word contains the keyword_contain, rename it to keyword_contain
    keywords_contain = ["BMI"]
    for keyword in keywords_contain:
        if keyword in feature_name:
            feature_name = keyword
            # feature_name = feature_name.replace('Age, years', 'Age')

    # remap specific words to other names
    keywords_map = {
        "Systolic Blood Pressure": "Systolic BP",
        "Ethnicity": "Race",
        "WBC": "White blood cell count",
    }
    for keyword in keywords_map.keys():
        if feature_name.lower().startswith(keyword.lower()):
            feature_name = keywords_map[keyword]

    return feature_name


def get_renamed_unique_feature_names_from_list(all_feature_names: list):
    """Returns list of unique feature names after cleaning"""
    ans = []
    for feature_name in all_feature_names:
        feature_name = rename_feature_name(feature_name)
        ans.append(feature_name)
    return list(set(ans))
