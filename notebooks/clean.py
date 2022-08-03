import re

CLEANR = re.compile('<.*?>')


def clean_feature_name(feature_name: str):
    return re.sub(CLEANR, '', feature_name)


def rename_feature_name(feature_name: str):
    # remove units from a feature
    feature_name = feature_name.replace(', mmHg', '')

    # if word starts with keyword_prefix, rename it to the prefix
    keyword_prefixes = ['Age', 'ASA', 'Albumin', 'ED visits', 'EKG', 'ESR', 'Endoscopy', 'BMI', 'BUN', 'Biliary',
                        'Bilirubin', 'Blood in stool', 'Congestive heart failure', 'Creatinine', 'D-dimer', 'Dementia',
                        'Diastolic BP', 'Distracting ', 'ECOG', 'Glucose', 'Hematocrit', 'Hemoglobin', 'Pulse', 'Race',
                        'Regional lymph node',
                        'Respiratory rate', 'Scalp hematoma', 'Sex', 'Systolic BP', 'WBC', 'White blood']
    for keyword in keyword_prefixes:
        if feature_name.startswith(keyword):
            feature_name = keyword

    # if word contains the keyword_contain, rename it to keyword_contain
    keywords_contain = ['BMI']
    for keyword in keywords_contain:
        if keyword in feature_name:
            feature_name = keyword
            # feature_name = feature_name.replace('Age, years', 'Age')

    # remap specific words to other names
    keywords_map = {'Systolic Blood Pressure': 'Systolic BP',
                    'Ethnicity': 'Race',
                    'WBC': 'White blood cell count'}
    for keyword in keywords_map.keys():
        if feature_name.lower().startswith(keyword.lower()):
            feature_name = keywords_map[keyword]

    return feature_name


def get_clean_unique_feature_names_from_list(all_feature_names: list):
    """Returns list of unique feature names after cleaning
    """
    ans = []
    for feature_name in all_feature_names:
        # feature_name = clean_feature_name(feature_name)
        feature_name = rename_feature_name(feature_name)
        ans.append(feature_name)
    return list(set(ans))
