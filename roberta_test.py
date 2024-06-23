from transformers import RobertaModel, RobertaTokenizer
import torch
from sklearn.metrics import accuracy_score, top_k_accuracy_score, recall_score, f1_score, precision_score, matthews_corrcoef
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

train = pd.read_csv("train.csv")
class PromptData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.cmp_code
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained('roberta-base')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, len(set(train['cmp_code'].to_list())))

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = RobertaTokenizer.from_pretrained("./")

model = RobertaClass()
model.to(device)

model.load_state_dict(torch.load("model.bin", map_location=torch.device(device)))
test = pd.read_csv("test.csv")

MAX_LEN=256
test_params={'batch_size': 1,
                'shuffle': True,
                'num_workers': 0
                }

test_data = test.reset_index(drop=True)
testing_set = PromptData(test_data, tokenizer, MAX_LEN)
testing_loader = DataLoader(testing_set, **test_params)

model.eval()
predictions = []
true_labels = []
y_score = []

with torch.no_grad():
    for _, data in tqdm(enumerate(testing_loader, 0)):
        val_ids = data['ids'].to(device, dtype = torch.long)
        val_mask = data['mask'].to(device, dtype = torch.long)
        val_token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        val_targets = data['targets'].to(device, dtype = torch.long)

        val_outputs = model(val_ids, val_mask, val_token_type_ids).squeeze()
        y_score.append(val_outputs.detach().cpu().numpy().tolist())

        v_big_val, v_big_idx = torch.max(val_outputs.data, dim=0)

        true_labels+=val_targets.tolist()
        predictions.append(v_big_idx.item())

print(accuracy_score(true_labels, predictions))
print(top_k_accuracy_score(true_labels, y_score, labels=range(56), k=2))
print(top_k_accuracy_score(true_labels, y_score, labels=range(56), k=3))
print(top_k_accuracy_score(true_labels, y_score, labels=range(56), k=5))
print(recall_score(true_labels, predictions, average="micro"))
print(recall_score(true_labels, predictions, average="macro"))
print(f1_score(true_labels, predictions, average="micro"))
print(f1_score(true_labels, predictions, average="macro"))
print(precision_score(true_labels, predictions, average="micro"))
print(precision_score(true_labels, predictions, average="macro"))
print(matthews_corrcoef(true_labels, predictions))
