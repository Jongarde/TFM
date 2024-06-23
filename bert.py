import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

X_train = df_train["text"]
y_train = df_train["cmp_code"]
X_test = df_test["text"]
y_test = df_test["cmp_code"]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
BertHFmodel = BertModel.from_pretrained('bert-base-cased').to(device)

def embed_features(texts):
    embedding_list = []
    for text in tqdm(texts):
      encoded_input = tokenizer(text, return_tensors = 'pt', max_length = 512, truncation = True, padding = 'max_length').to(device)
      with torch.no_grad():
          output_embeddings = BertHFmodel(**encoded_input)
          embedding_list.append(output_embeddings.last_hidden_state[0, 0, :].cpu().numpy())
    return embedding_list

X_train_bert = embed_features(X_train.to_list())
X_test_bert = embed_features(X_test.to_list())

x_tr = torch.tensor(np.array(X_train_bert))
y_tr = torch.tensor(np.array(y_train))

x_ts = torch.tensor(np.array(X_test_bert))
y_ts = torch.tensor(np.array(y_test))

torch.save(TensorDataset(x_tr, y_tr), "train.pt")
torch.save(TensorDataset(x_ts, y_ts), "test.pt")
