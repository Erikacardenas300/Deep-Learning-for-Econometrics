
def get_csv_sentences(file_1_path, file_2_path, sent_col_name):
  import pandas as pd
  import numpy as np
  
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
  
  descriptions = []
  labels = []
  
  file1_df = pd.read_csv(file_1_path)
  file1_df = file1_df[sent_col_name].dropna()
  file1_df = file1_df.to_numpy()
  
  for desc in file1_df:
    token_list = tokenizer(desc).input_ids
    descriptions.append(token_list)
    labels.append(1)
   
  file2_df = pd.read_csv(file_2_path)
  file2_df = file2_df[sent_col_name].dropna()
  file2_df = file2_df.to_numpy()
  
  for desc in file2_df:
    token_list = tokenizer(desc).input_ids
    descriptions.append(token_list)
    labels.append(0)
    
  tokenized_descriptions = []
  for tok_desc in descriptions:
    new_tok_list = []
    for tok in tok_desc:
      if len(new_tok_list) < 512:
        new_tok_list.append(tok)
    while len(new_tok_list) < 512:
      new_tok_list.append(0)
    tokenized_descriptions.append(new_tok_list)
    
  tokenized_descriptions = np.array(tokenized_descriptions)
  labels = np.array(labels)
  return tokenized_descriptions, labels
  
