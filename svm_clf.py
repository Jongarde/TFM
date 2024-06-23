from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, top_k_accuracy_score, recall_score, f1_score, matthews_corrcoef, precision_score
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

X_train = df_train["text"]
y_train = df_train["cmp_code"]
X_test = df_test["text"]
y_test = df_test["cmp_code"]

text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=200000)

X_train_text = text_transformer.fit_transform(X_train)
X_test_text = text_transformer.transform(X_test)

SVM_clf = SVC(probability=True, kernel='rbf', verbose=True)

SVM_clf.fit(X_train_text, y_train)

y_pred = SVM_clf.predict(X_test_text)
y_score = SVM_clf.predict_proba(X_test_text)

print(accuracy_score(y_test, y_pred))
print(top_k_accuracy_score(y_test, y_score, labels=range(56), k=2))
print(top_k_accuracy_score(y_test, y_score, labels=range(56), k=3))
print(top_k_accuracy_score(y_test, y_score, labels=range(56), k=5))
print(recall_score(y_test, y_pred, average="micro"))
print(recall_score(y_test, y_pred, average="macro"))
print(f1_score(y_test, y_pred, average="micro"))
print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="micro"))
print(precision_score(y_test, y_pred, average="macro"))
print(matthews_corrcoef(y_test, y_pred))

joblib.dump(SVM_clf, "svm_model.pkl")

joblib.dump(text_transformer, "tfidf_vectorizer.joblib")
