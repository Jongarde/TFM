from transformers import AutoTokenizer, AutoModel
from torch import cuda
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_NAME = "distilbert/distilbert-base-uncased"

train = pd.read_csv("train.csv")
validation = pd.read_csv("validation.csv")

MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
LEARNING_RATE = 1e-05
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncation=True, do_lower_case=True)

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

train_data = train.reset_index(drop=True)
val_data = validation.reset_index(drop=True)

print("TRAIN Dataset: {}".format(train_data.shape))
print("TEST Dataset: {}".format(val_data.shape))

training_set = PromptData(train_data, tokenizer, MAX_LEN)
validation_set = PromptData(val_data, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **test_params)

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(MODEL_NAME)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, len(set(train['cmp_code'].to_list())))

        """
        for param in self.l1.parameters():
            param.requires_grad = False
        """

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = DistilBERTClass()
model.to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

def calculate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def train(epoch):
    tr_loss = 0; n_correct = 0; nb_tr_steps = 0; nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        #token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    model.eval()
    val_n_correct = 0; n_wrong = 0; total = 0; val_loss=0; val_nb_tr_steps=0; val_nb_tr_examples=0
    with torch.no_grad():
        for _, data in tqdm(enumerate(validation_loader, 0)):
            val_ids = data['ids'].to(device, dtype = torch.long)
            val_mask = data['mask'].to(device, dtype = torch.long)
            #val_token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            val_targets = data['targets'].to(device, dtype = torch.long)

            val_outputs = model(val_ids, val_mask).squeeze()
            v_loss = loss_function(val_outputs, val_targets)
            val_loss += v_loss.item()
            v_big_val, v_big_idx = torch.max(val_outputs.data, dim=1)
            val_n_correct += calculate_accuracy(v_big_idx, val_targets)

            val_nb_tr_examples += 1
            val_nb_tr_steps+=val_targets.size(0)

    val_epoch_loss = val_loss/val_nb_tr_examples
    val_epoch_accu = (val_n_correct*100)/val_nb_tr_steps
    print(f"Validation Loss Epoch: {val_epoch_loss}")
    print(f"Validation Accuracy Epoch: {val_epoch_accu}")

    return val_epoch_loss

EPOCHS = 500
early_stopping_factor = 5
count = 0
val_loss_pre = 1000

for epoch in range(EPOCHS):
    print("================================ Epoch:" + str(epoch) + "================================")
    val_loss_epoch = train(epoch)
    if val_loss_pre > val_loss_epoch:
      val_loss_pre = val_loss_epoch
      torch.save(model.state_dict(), "model.bin")
      count = 0
    else:
      count += 1

    if count == early_stopping_factor:
      break

tokenizer.save_vocabulary("./")
