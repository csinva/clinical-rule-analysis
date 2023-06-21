def get_prompts_demographics():
    properties = {
        "num_male": {
            "type": "string",
            "description": "The number of male patients in the study",
        },
        "num_female": {
            "type": "string",
            "description": "The number of female patients in the study",
        },
        "num_male_evidence_span": {
            "type": "string",
            "description": "The long text span in the input that includes evidence for num_male.",
        },
        "num_female_evidence_span": {
            "type": "string",
            "description": "The long text span in the input that includes evidence for num_female.",
        },
        "num_white": {
            "type": "string",
            "description": "The number of white/caucasian patients in the study",
        },
        "num_black": {
            "type": "string",
            "description": "The number of black/african american patients in the study",
        },
        "num_latino": {
            "type": "string",
            "description": "The number of latino patients in the study",
        },
        "num_white_evidence_span": {
            "type": "string",
            "description": "The long text span in the input that includes evidence for num_white.",
        },
        "num_black_evidence_span": {
            "type": "string",
            "description": "The long text span in the input that includes evidence for num_black.",
        },
        "num_latino_evidence_span": {
            "type": "string",
            "description": "The long text span in the input that includes evidence for num_latino.",
        },
    }

    functions = [
        {
            "name": "extract_patient_nums_by_demographics",
            "description": "Get the number of patients in this study for different gender and race demographics.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [
                    "num_male",
                    "num_female",
                    "num_male_evidence_span",
                    "num_female_evidence_span",
                ],
            },
        },
    ]

    content_str = """### QUESTION: How many patients were in the study, broken down by gender and race?

###  STUDY: {input}"""
    return properties, functions, content_str


def get_prompts_gender():
    properties = {
        "num_male": {
            "type": "string",
            "description": "The number of male patients in the study",
        },
        "num_female": {
            "type": "string",
            "description": "The number of female patients in the study",
        },
        "num_male_evidence_span": {
            "type": "string",
            "description": "The long text span in the input that includes evidence for num_male.",
        },
        "num_female_evidence_span": {
            "type": "string",
            "description": "The long text span in the input that includes evidence for num_female.",
        },
    }

    functions = [
        {
            "name": "extract_patient_nums_by_gender",
            "description": "Get the number of patients in this study for different genders",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [
                    "num_male",
                    "num_female",
                    "num_male_evidence_span",
                    "num_female_evidence_span",
                ],
            },
        },
    ]

    content_str = """### QUESTION: How many male and female patients were in the study?

###  STUDY: {input}"""
    return properties, functions, content_str
