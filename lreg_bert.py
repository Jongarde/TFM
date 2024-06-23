import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score, recall_score, f1_score, matthews_corrcoef, precision_score

train_data = torch.load("train.pt")
test_data = torch.load("test.pt")

X_train, y_train = train_data.tensors
X_test, y_test = test_data.tensors

lreg_clf = LogisticRegression(max_iter=3000)
lreg_clf.fit(X_train, y_train)

y_pred = lreg_clf.predict(X_test)
y_score = lreg_clf.predict_proba(X_test)

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
