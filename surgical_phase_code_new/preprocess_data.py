import os
import pandas as pd

def load_data_from_folder(folder_path):
    """
    Read all CSV files from the specified folder and merge them, returning a DataFrame.
        Assume each CSV contains at least:
           - 'Text': conversation content
           - 'Phase_Label': manual labeling phase
    """
    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            full_path = os.path.join(folder_path, file)
            df_temp = pd.read_csv(full_path, encoding='utf-8')
            dfs.append(df_temp)
    if len(dfs) > 0:
        df_all = pd.concat(dfs, ignore_index=True)
        return df_all
    else:
        print("No CSV files found in the folder.")
        return pd.DataFrame()
    
def preprocess_text(text):
    """
    Perform simple preprocessing on the conversation text:
        - Remove extra spaces
        - (You can do desensitization, chat filtering, etc. here)
    """
    if not isinstance(text, str):
        return ""
    text_clean = " ".join(text.split())
    return text_clean


def preprocess_df(df):
    """Preprocess the text in a DataFrame."""
    df["Text"] = df["Text"].apply(preprocess_text)
    return df