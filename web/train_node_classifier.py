#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Treina um classificador para NÓS (contas) usando node_features.csv.
Produz: /app/model_node_dt_nodes.joblib e imprime métricas e importâncias.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report,
    confusion_matrix
)
from sklearn.impute import SimpleImputer
from joblib import dump

RANDOM_STATE = 42
TARGET_COL = "isFraudster"

# Evitar leakage explícito pedido pelo usuário
cols_to_remove = ["newBalanceOrig", "newBalanceDest"]

# Caminho do arquivo de nodes (ajuste se necessário)
NODE_FEATURES_PATH = "/data/node_features.csv"

print(">> Iniciando treino de classificador de NÓS")
print(f">> Lendo: {NODE_FEATURES_PATH}")

if not os.path.exists(NODE_FEATURES_PATH):
    raise FileNotFoundError(f"{NODE_FEATURES_PATH} não encontrado. Verifique caminho/volume.")

# ===== 1) Ler nodes =====
nodes = pd.read_csv(NODE_FEATURES_PATH)
print(f">> Columns in node features: {list(nodes.columns)[:50]}")
print(f">> Sample rows: {min(5, len(nodes))}")
print(nodes.head().to_string(index=False))

# ===== 2) Derivar rótulo por nó =====
# Procura automaticamente colunas de contagem de fraudes comuns
possible_fraud_cols = [c for c in nodes.columns if "fraud" in c.lower() and nodes[c].dtype.kind in "biuf"]
label_source = None
for c in ["fraud_edge_count", "fraud_cnt", "fraud_count", "fraudsters", "fraud_cnt_sum", "fraud"]:
    if c in nodes.columns:
        label_source = c
        break
# se não achou entre nomes exatos, pega qualquer coluna com 'fraud' no nome e tipo numérico
if label_source is None:
    cand = [c for c in nodes.columns if "fraud" in c.lower() and nodes[c].dtype.kind in "biuf"]
    if len(cand) > 0:
        label_source = cand[0]

if label_source is None:
    raise ValueError("Não foi possível detectar coluna de fraude no node_features.csv. "
                     "Esperei algo como 'fraud_edge_count' ou coluna com 'fraud' no nome. "
                     "Você precisa fornecer rótulo por nó (ex.: qualquer aresta fraudulenta).")

print(f">> Usando coluna '{label_source}' para criar rótulo '{TARGET_COL}' (isFraudster = >0)")

# criar label binária: conta é fraudster se teve alguma aresta fraudulenta
nodes[TARGET_COL] = (nodes[label_source].fillna(0).astype(int) > 0).astype(int)

# REMOVER a coluna usada para derivar o label (evita leakage)
nodes = nodes.drop(columns=[label_source], errors="ignore")

# ===== 3) Evitar vazamento/colunas irrelevantes =====
# Colunas que tipicamente representam ID ou vazamento
candidate_leaks = {
    "account", "acct", "acct_id",
    "node", "node_id",
    "nameOrig", "nameDest",
    "isFraud", "isFlaggedFraud",
    "transactionId", "tx_id",
    "orig", "dest", "index", "_index", "__index_level_0__"
}
existing_leaks = [c for c in candidate_leaks if c in nodes.columns]
if existing_leaks:
    print(">> Removendo colunas de ID/possível leakage:", existing_leaks)
nodes = nodes.drop(columns=existing_leaks, errors="ignore")

# Remover explicitamente colunas pedidas pelo usuário
nodes = nodes.drop(columns=cols_to_remove, errors="ignore")
print(f">> Depois da limpeza, colunas: {list(nodes.columns)[:60]}")

# ===== 4) Preparar X, y =====
if TARGET_COL not in nodes.columns:
    raise ValueError(f"Coluna alvo '{TARGET_COL}' não encontrada depois da preparação.")

y = nodes[TARGET_COL].astype(int)
X = nodes.drop(columns=[TARGET_COL], errors="ignore")

# ===== 5) Tratar categorias de alta cardinalidade =====
# Identifica colunas categóricas
cat_cols_all = X.select_dtypes(include=["object", "category"]).columns.tolist()
print(f">> Colunas categóricas detectadas: {cat_cols_all}")

# Se houver muitas categorias em uma coluna (id-like), substitui por frequency-encoding
cat_small = []
cat_large = []
CARDINALITY_THRESHOLD = 50  # limite para one-hot; acima disso usamos frequency encoding

for c in cat_cols_all:
    nun = X[c].nunique(dropna=False)
    if nun <= CARDINALITY_THRESHOLD:
        cat_small.append(c)
    else:
        cat_large.append(c)

print(f">> Categorical small (OHE): {cat_small}")
print(f">> Categorical large (freq-encoding): {cat_large}")

# Aplicar frequency encoding para cat_large
for c in cat_large:
    freq = X[c].map(X[c].value_counts()).fillna(0).astype(int)
    X[c + "_freq"] = freq
    X = X.drop(columns=[c], errors="ignore")

# Atualizar cat_cols para usar apenas as small
cat_cols = cat_small[:]  # safe copy

# ===== 6) Remover colunas não numéricas residuais que não queremos OHE (se existirem) =====
# (por segurança) se ainda houver colunas object-type não em cat_cols, convertê-las para freq encoding
residual_objs = [c for c in X.select_dtypes(include=["object"]).columns if c not in cat_cols]
if residual_objs:
    print(">> Detected leftover object cols (will freq-encode):", residual_objs)
    for c in residual_objs:
        X[c + "_freq"] = X[c].map(X[c].value_counts()).fillna(0).astype(int)
        X = X.drop(columns=[c], errors="ignore")

# ===== 7) Tipos finais de colunas =====
num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
# recalc cat_cols as those present
cat_cols = [c for c in cat_cols if c in X.columns]

print(f">> Final numeric cols ({len(num_cols)}): {num_cols[:20]}{'...' if len(num_cols)>20 else ''}")
print(f">> Final categorical cols ({len(cat_cols)}): {cat_cols}")

if len(num_cols) + len(cat_cols) == 0:
    raise ValueError("Nenhuma feature válida encontrada após filtragem.")

# ===== 8) Split - temporal se possível, senão estratificado =====
if "last_step" in nodes.columns:
    # separa por tempo: treino = nodes com last_step <= cutoff; teste = last_step > cutoff
    cutoff = nodes["last_step"].quantile(0.75)
    train_mask = nodes["last_step"] <= cutoff
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[~train_mask]
    y_test = y[~train_mask]
    print(f">> Split temporal por last_step <= {cutoff}: train={len(X_train)} test={len(X_test)}")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f">> Split estratificado: train={len(X_train)} | test={len(X_test)} | prevalência train={y_train.mean():.6f}")

# ===== 9) Pré-processamento e modelo =====
numeric_transformer = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="median"))
])

# OneHotEncoder compatível com versões
try:
    categorical_transformer = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32))
    ])
except TypeError:
    categorical_transformer = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32))
    ])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop"
)

clf = DecisionTreeClassifier(
    class_weight="balanced",
    max_depth=None,
    min_samples_leaf=10,
    random_state=RANDOM_STATE
)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", clf)
])

# ===== 10) Treinar =====
print(">> Treinando o modelo (pode demorar dependendo do tamanho)...")
pipe.fit(X_train, y_train)

# ===== 11) Avaliação =====
y_scores = pipe.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, y_scores)
ap  = average_precision_score(y_test, y_scores)
print(f"\n== Métricas globais ==")
print(f"ROC-AUC: {roc:.6f}")
print(f"PR-AUC (AP): {ap:.6f}")

# Escolha de limiares
prec, rec, thr = precision_recall_curve(y_test, y_scores)
thr_for_calc = np.r_[thr, 1.0]
f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
best_ix = np.nanargmax(f1)
best_thr = thr_for_calc[best_ix]

print(f"\n== Threshold ótimo por F1 ==")
print(f"F1*={f1[best_ix]:.4f} @ thr={best_thr:.6f} | P={prec[best_ix]:.4f} | R={rec[best_ix]:.4f}")

TARGET_P = 0.90
ok = np.where(prec >= TARGET_P)[0]
if len(ok):
    ix_p = ok[np.argmax(rec[ok])]
    thr_p = thr_for_calc[ix_p]
    print(f"\n== Threshold para P>={TARGET_P:.0%} ==")
    print(f"P={prec[ix_p]:.4f} | R={rec[ix_p]:.4f} @ thr={thr_p:.6f}")
else:
    thr_p = None
    print(f"\n[Aviso] Nenhum ponto na curva atingiu precisão >= {TARGET_P:.0%}.")

def report_at_threshold(t, label="F1*"):
    y_pred = (y_scores >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    print(f"\n== Relatório @ {label} (thr={t:.6f}) ==")
    print("Matriz de confusão [TN FP; FN TP]:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

report_at_threshold(best_thr, "F1*")
if thr_p is not None:
    report_at_threshold(thr_p, f"P>={TARGET_P:.0%}")

# ===== 12) Importância de features (agregada por coluna original) =====
print("\n>> Calculando importância de features (agregada por coluna original)...")
prep = preprocess.fit(X_train)
Xt = prep.transform(X_train)

num_names = num_cols
if len(cat_cols):
    ohe = prep.named_transformers_["cat"].named_steps["ohe"]
    try:
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    except Exception:
        # fallback para versões antigas
        cat_names = []
        # tentar construir nomes manualmente (limitado)
else:
    cat_names = []

feat_names = num_names + cat_names

tree = DecisionTreeClassifier(
    class_weight="balanced",
    max_depth=pipe.named_steps["clf"].max_depth,
    min_samples_leaf=pipe.named_steps["clf"].min_samples_leaf,
    random_state=RANDOM_STATE
).fit(Xt, y_train)

importances = tree.feature_importances_

agg = {}
for n in num_names:
    idxs = [i for i, fn in enumerate(feat_names) if fn == n]
    agg[n] = float(np.sum(importances[idxs])) if idxs else 0.0
for c in cat_cols:
    idxs = [i for i, fn in enumerate(feat_names) if fn.startswith(c + "_")]
    agg[c] = float(np.sum(importances[idxs])) if idxs else 0.0

imp_df = pd.DataFrame({"feature": list(agg.keys()), "importance": list(agg.values())})
imp_df = imp_df.sort_values("importance", ascending=False)
print("\n== Top 20 features (agregadas por coluna original) ==")
print(imp_df.head(20).to_string(index=False))

# ===== 13) Salvar modelo =====
MODEL_PATH = "/app/model_node_dt_nodes.joblib"
dump(pipe, MODEL_PATH)
print(f"\nModelo salvo em {MODEL_PATH}")

print("\n>> Fim.")
