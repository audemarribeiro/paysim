import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.linear_model import LogisticRegression

df = pd.read_parquet("/data/transactions_with_graph_features.parquet")
feats = ["amount","orig_pagerank","dest_pagerank","orig_deg_out","dest_deg_in"]
X = df[feats].fillna(0.0)
y = df["isFraud"].astype(int)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)
clf.fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:,1]
print("ROC-AUC:", roc_auc_score(yte, p))
print("PR-AUC :", average_precision_score(yte, p))

