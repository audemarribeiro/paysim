#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reprodução do N2V-GCN (PaySim) alinhada ao artigo
-------------------------------------------------
- Node2Vec -> GCN (2 camadas, 16 filtros, ReLU, Dropout 0.3)
- Classificação de ARESTAS (transações)
- Treino balanceado 1:1 (neg:pos=1.0) apenas no TREINO
- Val/Test preservam prevalência real
- Métrica-alvo: Recall@FAR<=α (relata para α em {0.2, 0.1} por padrão)
- Varredura de limiar 0.30–0.40 (passo 0.01) com Recall e FAR
- BCEWithLogits + pos_weight automático (pode desativar ou sobrescrever)
- Avaliação em CHUNKS p/ economizar memória
- (Opcional) Split temporal, se houver coluna de tempo em transações (--tx-time-col)

Uso típico:
  python n2v_gcn_repro_article.py \
    --nodes /data/node_features.csv \
    --edges /data/edge_features.csv \
    --id-col account \
    --undirected \
    --tx-file /data/transactions_with_graph_features.parquet \
    --epochs-n2v 30 --epochs-gcn 100

Se quiser split temporal:
  ... --temporal-split --tx-time-col step  --train-days 20 --val-days 5 --test-days 5
"""

import argparse
import os
import random
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import Node2Vec

# =========================
# Utils
# =========================

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def infer_pair_cols(df: pd.DataFrame, prefer_tx=False) -> Tuple[str, str]:
    cand_tx = [("nameOrig","nameDest"),("orig","dest"),("payer","payee"),("source","target")]
    cand_e  = [("src","dst"),("source","target"),("u","v"),("orig","dest"),("account_orig","account_dest"),("nameOrig","nameDest")]
    order = cand_tx + cand_e if prefer_tx else cand_e + cand_tx
    for s,t in order:
        if s in df.columns and t in df.columns: return s,t
    if df.shape[1] >= 2: return df.columns[0], df.columns[1]
    raise ValueError("Não foi possível identificar colunas de origem/destino.")

def normalize_binary(series: pd.Series) -> np.ndarray:
    if series.dtype == object:
        arr = (series.astype(str).str.strip().str.lower()
               .map({"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0,"sim":1,"nao":0,"não":0})
               .fillna(0).astype(int).values)
    else:
        arr = pd.to_numeric(series, errors="coerce").fillna(0).astype(int).values
    return (arr > 0).astype(np.int64)

def build_id_mapping(ids: List[str]) -> dict:
    uniq = pd.unique(pd.Series(ids)).tolist()
    return {k: i for i, k in enumerate(uniq)}

def map_ids_to_index(arr: np.ndarray, id_map: dict) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.int64)
    for i,a in enumerate(arr.astype(str)):
        if a not in id_map: id_map[a] = len(id_map)
        out[i] = id_map[a]
    return out

def far_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn = np.sum((y_true==0)&(y_pred==0)); fp = np.sum((y_true==0)&(y_pred==1))
    return float(fp)/(fp+tn) if (fp+tn)>0 else 0.0

# =========================
# Modelos
# =========================

class GCNNodeEncoder(nn.Module):
    def __init__(self, in_dim=128, hidden=16, p_drop=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p_drop)
    def forward(self, x, edge_index):
        h = self.act(self.conv1(x, edge_index)); h = self.drop(h)
        h = self.act(self.conv2(h, edge_index)); h = self.drop(h)
        return h

class EdgeHead(nn.Module):
    """[hu || hv || |hu-hv| || hu*hv] -> MLP -> logit"""
    def __init__(self, node_hidden=16, mlp_hidden=32, p_drop=0.3):
        super().__init__()
        in_dim = 4*node_hidden
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(mlp_hidden, 1),
        )
    def forward(self, h, edge_pairs):
        u = h[edge_pairs[0]]; v = h[edge_pairs[1]]
        feat = torch.cat([u, v, torch.abs(u-v), u*v], dim=-1)
        return self.mlp(feat).view(-1)

class N2V_GCN(nn.Module):
    def __init__(self, n2v_dim=128, hidden=16, mlp_hidden=32, p_drop=0.3):
        super().__init__()
        self.enc = GCNNodeEncoder(n2v_dim, hidden, p_drop)
        self.head = EdgeHead(hidden, mlp_hidden, p_drop)
    def forward(self, x, edge_index, edge_pairs):
        h = self.enc(x, edge_index)
        return self.head(h, edge_pairs)  # logits

# =========================
# Carregamento / Rotulagem
# =========================

def load_graph_and_edges(nodes_path, edges_path, id_col, make_undirected,
                         tx_file=None, tx_label_candidates=None, tx_time_col=None):
    # Nós → id_map
    df_nodes = pd.read_csv(nodes_path)
    if id_col is None:
        for c in ["account","id","node","node_id"]:
            if c in df_nodes.columns: id_col=c; break
        if id_col is None: id_col = df_nodes.columns[0]
    id_map = build_id_mapping(df_nodes[id_col].astype(str).tolist())

    # Arestas base
    df_edges = pd.read_csv(edges_path)
    s_col, t_col = infer_pair_cols(df_edges, prefer_tx=False)
    src = map_ids_to_index(df_edges[s_col].astype(str).values, id_map)
    dst = map_ids_to_index(df_edges[t_col].astype(str).values, id_map)
    edge_pairs = np.vstack([src, dst]).astype(np.int64)

    # edge_index para GCN
    edge_index_np = np.vstack([src, dst]).astype(np.int64)
    if make_undirected:
        edge_index_np = np.hstack([edge_index_np, edge_index_np[::-1,:]])
    edge_index = torch.from_numpy(edge_index_np)
    data = Data(x=None, edge_index=edge_index, num_nodes=len(id_map))

    # Rótulos a partir de transações (mais fiel ao artigo)
    if not tx_file or not os.path.exists(tx_file):
        raise ValueError("Forneça --tx-file com transações e rótulo (isFraud/label...).")
    dft = pd.read_parquet(tx_file) if tx_file.lower().endswith(".parquet") else pd.read_csv(tx_file)
    ts_col, tt_col = infer_pair_cols(dft, prefer_tx=True)
    tx_cands = tx_label_candidates or ["isFraud","is_fraud","label","fraud","isFlaggedFraud"]
    tx_lbl = next((c for c in tx_cands if c in dft.columns), None)
    if tx_lbl is None:
        raise ValueError(f"{tx_file} sem coluna de rótulo. Tente --tx-label-candidates isFraud is_fraud label ...")

    ts = map_ids_to_index(dft[ts_col].astype(str).values, id_map)
    tt = map_ids_to_index(dft[tt_col].astype(str).values, id_map)
    tl = normalize_binary(dft[tx_lbl])

    # agrega por par (u,v): 1 se QUALQUER transação (u,v) foi fraude
    df_tmp = pd.DataFrame({"u": ts, "v": tt, "y": tl})
    if tx_time_col and tx_time_col in dft.columns:
        df_tmp["t"] = dft[tx_time_col].values
    agg = df_tmp.groupby(["u","v"])["y"].max().reset_index()

    key_to_y = {(int(r.u), int(r.v)): int(r.y) for r in agg.itertuples(index=False)}
    labels = np.zeros(edge_pairs.shape[1], dtype=np.int64)
    u,v = edge_pairs[0], edge_pairs[1]
    for i in range(edge_pairs.shape[1]):
        labels[i] = key_to_y.get((int(u[i]),int(v[i])), 0)

    # timestamp por par (para split temporal): usa o último tempo do par
    times = None
    if tx_time_col and tx_time_col in dft.columns:
        # pega o max(t) por par
        agg_t = df_tmp.groupby(["u","v"])["t"].max().reset_index()
        key_to_t = {(int(r.u), int(r.v)): r.t for r in agg_t.itertuples(index=False)}
        times = np.array([key_to_t.get((int(u[i]),int(v[i])), None) for i in range(edge_pairs.shape[1])])

    return data, edge_pairs, labels, id_map, times

# =========================
# Splits
# =========================

def split_edges_stratified(labels: np.ndarray, test_size=0.2, val_size=0.1, seed=42):
    idx_all = np.arange(len(labels)); y = labels.astype(int)
    idx_trv, idx_te, y_trv, _ = train_test_split(idx_all, y, test_size=test_size, stratify=y, random_state=seed)
    val_rel = val_size / (1.0 - test_size)
    idx_tr, idx_va, _, _ = train_test_split(idx_trv, y_trv, test_size=val_rel, stratify=y_trv, random_state=seed)
    return idx_tr, idx_va, idx_te

def split_edges_temporal(labels: np.ndarray, times: np.ndarray,
                         train_days=20, val_days=5, test_days=5):
    if times is None:
        raise ValueError("Split temporal requisitado mas sem coluna de tempo.")
    # Ordena por tempo e corta em janelas
    order = np.argsort(pd.to_datetime(times))
    n = len(order)
    # simples: usa proporções relativas de dias (não exatos) para criar 3 blocos
    p_train = train_days/(train_days+val_days+test_days)
    p_val   = val_days/(train_days+val_days+test_days)
    cut1 = int(n*p_train); cut2 = cut1 + int(n*p_val)
    idx_tr = order[:cut1]; idx_va = order[cut1:cut2]; idx_te = order[cut2:]
    # garante que há positivos em cada split; se não, cai para estratificado
    for part in [idx_tr, idx_va, idx_te]:
        if not (np.any(labels[part]==1) and np.any(labels[part]==0)):
            return split_edges_stratified(labels)
    return idx_tr, idx_va, idx_te

def make_balanced_train(labels: np.ndarray, idx_train: np.ndarray, seed=42, neg_pos_ratio=1.0):
    rng = np.random.default_rng(seed)
    pos = idx_train[labels[idx_train]==1]; neg = idx_train[labels[idx_train]==0]
    if len(pos)==0 or len(neg)==0: raise ValueError("Sem positivos/negativos no treino.")
    n_neg = min(len(neg), int(len(pos)*neg_pos_ratio))
    chosen_neg = rng.choice(neg, size=n_neg, replace=False)
    out = np.concatenate([pos, chosen_neg]); rng.shuffle(out)
    return out

# =========================
# Node2Vec
# =========================

def train_node2vec(edge_index, num_nodes, emb_dim=128, walk_length=32, walks_per_node=15,
                   p=1.0, q=1.0, epochs=30, batch_size=128, lr=0.01, device="cpu"):
    n2v = Node2Vec(edge_index=edge_index, embedding_dim=emb_dim, walk_length=walk_length,
                   context_size=walk_length, walks_per_node=walks_per_node, p=p, q=q,
                   num_nodes=num_nodes, sparse=True).to(device)
    loader = n2v.loader(batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.SparseAdam(list(n2v.parameters()), lr=lr)
    n2v.train()
    for ep in range(1, epochs+1):
        tot=0.0
        for pos_rw, neg_rw in loader:
            opt.zero_grad(set_to_none=True)
            loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward(); opt.step(); tot += float(loss)
        if ep % 2 == 0 or ep == 1 or ep == epochs:
            print(f"[Node2Vec] Época {ep:02d}/{epochs} | Loss: {tot:.4f}")
    with torch.no_grad(): emb = n2v.embedding.weight.detach().cpu()
    return emb

# =========================
# Treino / Avaliação
# =========================

def evaluate_probs_chunks(model, data, x, edge_pairs, idx, device, batch_size=50000):
    """Inferência em chunks para economizar memória."""
    model.eval(); probs = np.zeros(idx.shape[0], dtype=np.float32)
    from math import ceil
    n = idx.shape[0]; steps = ceil(n / batch_size)
    with torch.no_grad():
        for s in range(steps):
            lo=s*batch_size; hi=min((s+1)*batch_size, n)
            sel = torch.from_numpy(edge_pairs[:, idx[lo:hi]]).long().to(device)
            logits = model(x.to(device), data.edge_index.to(device), sel)
            probs[lo:hi] = torch.sigmoid(logits).cpu().numpy()
    return probs

def pr_auc_roc_auc(y_true, prob):
    pr = average_precision_score(y_true, prob) if (np.any(y_true==1) and np.any(y_true==0)) else 0.0
    try:
        roc = roc_auc_score(y_true, prob) if (np.any(y_true==1) and np.any(y_true==0)) else 0.5
    except ValueError:
        roc = 0.5
    return pr, roc

def recall_at_far(prob: np.ndarray, y_true: np.ndarray, far_cap: float):
    """Retorna (best_recall, best_thr, far) para o menor FAR <= far_cap com maior recall."""
    best = (0.0, None, None)
    for thr in np.linspace(0,1,101):
        y_hat = (prob >= thr).astype(int)
        tp = np.sum((y_true==1)&(y_hat==1))
        fn = np.sum((y_true==1)&(y_hat==0))
        tn = np.sum((y_true==0)&(y_hat==0))
        fp = np.sum((y_true==0)&(y_hat==1))
        far = float(fp)/(fp+tn) if (fp+tn)>0 else 0.0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
        if far <= far_cap and rec >= best[0]:
            best = (rec, float(thr), float(far))
    return best

def threshold_sweep(prob, y_true, start, end, step, title):
    print(f"\n=== {title} ({start:.2f}–{end:.2f}) ===")
    print("thr\tRecall\tFAR")
    thr = start
    while thr <= end + 1e-12:
        y_hat = (prob >= thr).astype(int)
        tp = np.sum((y_true==1)&(y_hat==1))
        fn = np.sum((y_true==1)&(y_hat==0))
        tn = np.sum((y_true==0)&(y_hat==0))
        fp = np.sum((y_true==0)&(y_hat==1))
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        far = float(fp)/(fp+tn) if (fp+tn)>0 else 0.0
        print(f"{thr:.2f}\t{recall:.4f}\t{far:.4f}")
        thr = round(thr + step, 10)

def train_model(data, x, edge_pairs, labels, idx_train_bal, idx_val,
                epochs=100, lr=2e-3, wd=5e-4, device="cpu", use_pos_weight=True):
    model = N2V_GCN(n2v_dim=x.size(1), hidden=16, mlp_hidden=32, p_drop=0.3).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)

    y_all = torch.tensor(labels, dtype=torch.float32, device=device)
    # pos_weight a partir do treino NÃO balanceado (filtrado de volta)
    # aqui usamos a distribuição real dos índices do treino base (antes do balanceamento)
    # => passe idx_train_base em vez de idx_train_bal se quiser estritamente real;
    # neste reprodutor, usamos idx_val para estimar pos_weight (robusto).
    n_pos = max(1, int(np.sum(labels[idx_val]==1)))
    n_neg = max(1, int(np.sum(labels[idx_val]==0)))
    pos_weight = (n_neg / n_pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device)) if use_pos_weight \
                else nn.BCEWithLogitsLoss()

    best_state = None; best_pr = -1.0; bad=0; patience=10

    # minibatches de treino (aresta) para caber em memória
    def batches(idxs, bs):
        for i in range(0, len(idxs), bs): yield idxs[i:i+bs]

    train_bs = 8192
    eval_bs  = 50000

    for ep in range(1, epochs+1):
        model.train(); tot=0.0
        for chunk in batches(idx_train_bal, train_bs):
            opt.zero_grad(set_to_none=True)
            sel = torch.from_numpy(edge_pairs[:, chunk]).long().to(device)
            logits = model(x.to(device), data.edge_index.to(device), sel)
            loss = criterion(logits, y_all[chunk])
            loss.backward(); opt.step()
            tot += float(loss)

        # valida toda 2 épocas (mais barato)
        if ep % 2 == 0 or ep in (1, epochs):
            prob_val = evaluate_probs_chunks(model, data, x, edge_pairs, idx_val, device, batch_size=eval_bs)
            pr_val, roc_val = pr_auc_roc_auc(labels[idx_val], prob_val)
            if pr_val > best_pr:
                best_pr = pr_val
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                bad=0
            else:
                bad+=1
            print(f"[GCN] Ép {ep:03d}/{epochs} | TrainLoss(sum): {tot:.4f} | Val PR-AUC: {pr_val:.4f} | Val ROC-AUC: {roc_val:.4f} | pos_weight={pos_weight:.2f}")
            if bad>=patience:
                print(f"[EarlyStopping] Sem melhora por {patience} validações. Parando.")
                break

    if best_state is not None: model.load_state_dict(best_state)
    return model

# =========================
# Main
# =========================

def main():
    p = argparse.ArgumentParser(description="N2V-GCN (PyG) - reprodução alinhada ao artigo")
    p.add_argument("--nodes", required=True, type=str)
    p.add_argument("--edges", required=True, type=str)
    p.add_argument("--id-col", type=str, default=None)
    p.add_argument("--undirected", action="store_true")

    p.add_argument("--tx-file", type=str, required=True)
    p.add_argument("--tx-label-candidates", type=str, nargs="*", default=None)

    # Split temporal
    p.add_argument("--temporal-split", action="store_true")
    p.add_argument("--tx-time-col", type=str, default=None)
    p.add_argument("--train-days", type=int, default=20)
    p.add_argument("--val-days", type=int, default=5)
    p.add_argument("--test-days", type=int, default=5)

    # Node2Vec
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--walk-length", type=int, default=32)
    p.add_argument("--num-walks", type=int, default=15)
    p.add_argument("--p", type=float, default=1.0)
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--epochs-n2v", type=int, default=30)
    p.add_argument("--lr-n2v", type=float, default=0.01)
    p.add_argument("--batch-n2v", type=int, default=128)

    # GCN/Head
    p.add_argument("--epochs-gcn", type=int, default=100)
    p.add_argument("--lr-gcn", type=float, default=2e-3)
    p.add_argument("--wd-gcn", type=float, default=5e-4)

    # Splits
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)

    # Balanceamento 1:1 (default = artigo)
    p.add_argument("--neg-pos-ratio", type=float, default=1.0)

    # Memória (eval em chunks)
    p.add_argument("--eval-batch-size", type=int, default=50000)

    # FAR caps reportados
    p.add_argument("--far-caps", type=float, nargs="*", default=[0.2, 0.1])

    # Device
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()
    set_seed(args.seed)

    print(">> Carregando dados (nós/arestas)...")
    data, edge_pairs, labels, id_map, times = load_graph_and_edges(
        nodes_path=args.nodes, edges_path=args.edges, id_col=args.id_col,
        make_undirected=args.undirected, tx_file=args.tx_file,
        tx_label_candidates=args.tx_label_candidates, tx_time_col=args.tx_time_col
    )
    print(f"Grafo: N={data.num_nodes} | E(graph)={data.edge_index.size(1)} | E(classif)={edge_pairs.shape[1]} | Pos={int(labels.sum())}")

    # Node2Vec
    if args.epochs_n2v > 0:
        print("\n>> Treinando Node2Vec...")
        x = train_node2vec(
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            emb_dim=args.emb_dim, walk_length=args.walk_length, walks_per_node=args.num_walks,
            p=args.p, q=args.q, epochs=args.epochs_n2v, batch_size=args.batch_n2v,
            lr=args.lr_n2v, device=args.device
        )
    else:
        x = torch.randn(data.num_nodes, args.emb_dim)
        print("(!) Node2Vec pulado (--epochs-n2v 0). Usando embeddings aleatórios.")
    data.x = x; data.num_node_features = x.size(1)

    # Splits (estratificado ou temporal)
    print("\n>> Gerando splits (treino/val/test) ...")
    if args.temporal_split:
        idx_tr, idx_va, idx_te = split_edges_temporal(labels, times, args.train_days, args.val_days, args.test_days)
        print("Split: temporal")
    else:
        idx_tr, idx_va, idx_te = split_edges_stratified(labels, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
        print("Split: estratificado")
    idx_tr_bal = make_balanced_train(labels, idx_tr, seed=args.seed, neg_pos_ratio=args.neg_pos_ratio)
    print(f"Treino (1:1 ==> neg:pos={args.neg_pos_ratio:.1f}): {len(idx_tr_bal)} | Val: {len(idx_va)} | Teste: {len(idx_te)}")

    # Treino GCN + head (com BCE+pos_weight)
    print("\n>> Treinando GCN (2 camadas, 16 filtros) + head de aresta...")
    model = train_model(
        data=data, x=data.x.to(args.device), edge_pairs=edge_pairs, labels=labels,
        idx_train_bal=idx_tr_bal, idx_val=idx_va,
        epochs=args.epochs_gcn, lr=args.lr_gcn, wd=args.wd_gcn,
        device=args.device, use_pos_weight=True
    )

    # Avaliação TESTE (prevalência real) em chunks
    print("\n>> Avaliando no TESTE...")
    prob_test = evaluate_probs_chunks(model, data, data.x.to(args.device), edge_pairs, idx_te, args.device, batch_size=args.eval_batch_size)
    y_test = labels[idx_te]
    pr_test, roc_test = pr_auc_roc_auc(y_test, prob_test)
    print(f"[TEST] PR-AUC: {pr_test:.4f} | ROC-AUC: {roc_test:.4f}")

    # Recall@FAR<=α (para α da lista)
    for cap in args.far_caps:
        rec, thr, far = recall_at_far(prob_test, y_test, cap)
        if thr is None:
            print(f"[TEST] Recall@FAR<={cap:.2f}: não encontrado no sweep 0..1.")
        else:
            print(f"[TEST] Recall@FAR<={cap:.2f}: Recall={rec:.4f} @ thr={thr:.2f} (FAR={far:.4f})")

    # Sweep pedido (0.30–0.40) e sweep total 0–1
    threshold_sweep(prob_test, y_test, 0.30, 0.40, 0.01, "Threshold Sweep (0.30–0.40)")
    threshold_sweep(prob_test, y_test, 0.00, 1.00, 0.01, "Threshold Sweep (0.00–1.00)")

    print("\nConcluído.")

if __name__ == "__main__":
    main()
