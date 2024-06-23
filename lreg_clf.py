from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score, recall_score, f1_score, matthews_corrcoef, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train["text"]
y_train = train["cmp_code"]
X_test = test["text"]
y_test = test["cmp_code"]

text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=200000)
X_train_text = text_transformer.fit_transform(X_train)
X_test_text = text_transformer.transform(X_test)

logits = LogisticRegression(max_iter = 3000)

logits.fit(X_train_text, y_train)

y_pred = logits.predict(X_test_text)
y_score = logits.predict_proba(X_test_text)

print(accuracy_score(y_test, y_pred))
print(top_k_accuracy_score(y_test, y_score, labels=range(56), k=2))
print(top_k_accuracy_score(y_test, y_score, labels=range(56), k=3))
print(top_k_accuracy_score(y_test, y_score, labels=range(56), k=5))
print(recall_score(y_test, y_pred, average="micro"))
print(recall_score(y_test, y_pred, average="macro"))
print(f1_score(y_test, y_pred, average="micro"))
print(f1_score(y_test, y_pred, average="macro"))
print(matthews_corrcoef(y_test, y_pred))
print(precision_score(y_test, y_pred, average="micro"))
print(precision_score(y_test, y_pred, average="macro"))
