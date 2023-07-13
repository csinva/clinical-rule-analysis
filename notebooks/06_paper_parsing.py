from collections import defaultdict
import pandas as pd
import imodelsx.llm
import numpy as np
import paper_parsing
import prompts
import openai
openai.api_key = open('/home/chansingh/.OPENAI_KEY').read().strip()

# pubmed.download_open_source_papers(df) 
# need to first download papers from https://drive.google.com/drive/folders/1OUXtsddxEAOl3tKEZegBQSwArUecb-6J into ../papers
df, ids_with_paper = paper_parsing.download_gsheet()

# extract text from pdfs (create file num.txt for each file num.pdf)
paper_parsing.extract_texts_from_pdf(ids_with_paper, papers_dir='../papers')

# get prompt
llm = imodelsx.llm.get_llm("gpt-4-0613") # gpt-3.5-turbo-0613
imodelsx.llm.LLM_CONFIG['LLM_REPEAT_DELAY'] = 10

# properties, functions, content_str = prompts.get_prompts_gender_and_race()
# print('attempting to add', properties.keys())
# paper_parsing.add_columns_based_on_properties(df, ids_with_paper, properties, functions, content_str, llm)

properties, functions, content_str = prompts.get_prompts_gender()
print('attempting to add', properties.keys())
paper_parsing.add_columns_based_on_properties(df, ids_with_paper, properties, functions, content_str, llm)

properties, functions, content_str = prompts.get_prompts_demographics()
print('attempting to add', properties.keys())
paper_parsing.add_columns_based_on_properties(df, ids_with_paper, properties, functions, content_str, llm)