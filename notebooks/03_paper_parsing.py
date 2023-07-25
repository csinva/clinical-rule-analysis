import paper_setup
import paper_parsing


if __name__ == "__main__":
    df, ids_with_paper = paper_setup.download_gsheet()
    paper_parsing.extract_nums_and_add_to_df(df, ids_with_paper)
