from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.svm import SVC
import pandas as pd

df_train = pd.read_csv("train.csv")

X_train = df_train["text"]
y_train = df_train["cmp_code"]

empty_dimensionality_model = BaseDimensionalityReduction()
SVM_clf = SVC(probability=True, kernel='rbf', verbose=True)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

topic_model= BERTopic(
        umap_model=empty_dimensionality_model,
        hdbscan_model=SVM_clf,
        ctfidf_model=ctfidf_model
)

print("Currently training")
topics, probs = topic_model.fit_transform(X_train, y=y_train)

topic_model.save("topic_model", serialization="pickle")
print("Topic model saved")
