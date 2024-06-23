import torch
from sklearn.metrics import accuracy_score, top_k_accuracy_score, recall_score, f1_score, matthews_corrcoef, precision_score
from sklearn.svm import SVC
from tqdm import tqdm
import joblib

train_data = torch.load("train.pt")
test_data = torch.load("test.pt")

X_train, y_train = train_data.tensors
X_test, y_test = test_data.tensors

SVM_clf = SVC(probability=True, kernel='rbf', verbose=True)

SVM_clf.fit(X_train, y_train)

y_pred = SVM_clf.predict(X_test)
y_score = SVM_clf.predict_proba(X_test)

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

joblib.dump(SVM_clf, "svm_bert.pkl")
