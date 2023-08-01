# Files
- [notebooks_data_prep](notebooks_data_prep): Contains notebooks used to clean and prepare the data
- [notebooks_llm](notebooks_llm): Contains notebooks used for LLM experiments
- [data](data): Contains the raw data, processed data, and data dictionary

# Working with the CDI Dataset
- Raw data is contained as a dataframe in the [data/data_cleaned.pkl](data/data_cleaned.pkl) file.
  - A semi-complete data dictionary is given in [data/data_cleaned_dictionary.json](data/data_cleaned_dictionary.json).
- (Optional) Paper pdfs are put into this [gdrive folder](https://drive.google.com/drive/folders/1OUXtsddxEAOl3tKEZegBQSwArUecb-6J).
  - Each paper is named using its `id`
  - If using rclone, download with `rclone ls gdrive:data/public_data/mdcalc-papers --drive-shared-with-me`
