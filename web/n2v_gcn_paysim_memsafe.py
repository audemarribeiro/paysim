#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
N2V-GCN para AML (PaySim) - CLASSIFICAÇÃO DE TRANSAÇÕES (ARESTAS)
Versão "mem-safe": avaliação e treino em minibatches para evitar OOM.

Recursos:
- BCEWithLogits(pos_weight) OU FocalLoss (--use-focal)
- GCN hidden=64, dropout=0.5
- Scheduler ReduceLROnPlateau (Val PR-AUC)
- Early stopping por Val PR-AUC
- Val/Test em CHUNKS (--eval-batch-size)
- Treino em CHUNKS (--train-batch-size)
- Amostragem opcional de negativos no Val/Test (--val-max-neg, --test-max-neg)
- AMP automático em CUDA
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

# ---------------- Utils ----------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def infer_pair_cols(df: pd.DataFrame, prefer_tx=False) -> Tuple[str, str]:
    cand_tx = [("nameOrig","nameDest"),("orig","dest"),("payer","payee"),("source","target")]
    cand_e  = [("src","dst"),("source","target"),("u","v"),("orig","dest"),("account_orig","account_dest"),("nameOrig","nameDest")]
    for s,t in (cand_tx+cand_e if prefer_tx else cand_e+cand_tx):
        if s in df.columns and t in df.columns: return s,t
    if df.shape[1] >= 2: return df.columns[0], df.columns[1]
    raise ValueError("Não foi possível identificar colunas de origem/destino.")

def infer_edge_label_column(df_edges: pd.DataFrame) -> Optional[str]:
    for c in ["isFraud","is_fraud","label","fraud","isFlaggedFraud"]:
        if c in df_edges.columns: return c
    return None

def infer_edge_fraudcount_column(df_edges: pd.DataFrame) -> Optional[str]:
    for c in ["fraud_count","fraud_edges","count_fraud","num_fraud","fraudedgecount"]:
        if c in df_edges.columns: return c
    return None

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

# -------------- Models --------------

class GCNNodeEncoder(nn.Module):
    def __init__(self, in_dim=128, hidden=64, p_drop=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p_drop)
    def forward(self, x, edge_index):
        h = self.act(self.conv1(x, edge_index)); h = self.drop(h)
        h = self.act(self.conv2(h, edge_index)); h = self.drop(h)
        return h

class EdgeClassifier(nn.Module):
    def __init__(self, node_hidden=64, mlp_hidden=64, p_drop=0.5):
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

class N2V_GCN_EdgeModel(nn.Module):
    def __init__(self, n2v_dim=128, gcn_hidden=64, mlp_hidden=64, p_drop=0.5):
        super().__init__()
        self.node_encoder = GCNNodeEncoder(n2v_dim, gcn_hidden, p_drop)
        self.edge_head    = EdgeClassifier(gcn_hidden, mlp_hidden, p_drop)
    def forward(self, x, graph_edge_index, edge_pairs):
        h = self.node_encoder(x, graph_edge_index)
        return self.edge_head(h, edge_pairs)

# -------------- Losses --------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        pt = prob*targets + (1-prob)*(1-targets)
        w = (self.alpha*targets + (1-self.alpha)*(1-targets)) * ((1-pt)**self.gamma)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = w * bce
        if self.reduction=="mean": return loss.mean()
        if self.reduction=="sum":  return loss.sum()
        return loss

# -------------- Data --------------

def load_graph_and_edges(nodes_path, edges_path, id_col, make_undirected, tx_file=None, tx_label_col_candidates=None):
    df_nodes = pd.read_csv(nodes_path)
    if id_col is None:
        for c in ["account","id","node","node_id"]:
            if c in df_nodes.columns: id_col=c; break
        if id_col is None: id_col = df_nodes.columns[0]
    id_map = build_id_mapping(df_nodes[id_col].astype(str).tolist())

    df_edges = pd.read_csv(edges_path)
    s_col, t_col = infer_pair_cols(df_edges, prefer_tx=False)
    src = map_ids_to_index(df_edges[s_col].astype(str).values, id_map)
    dst = map_ids_to_index(df_edges[t_col].astype(str).values, id_map)
    edge_pairs = np.vstack([src, dst]).astype(np.int64)

    edge_index_np = np.vstack([src, dst]).astype(np.int64)
    if make_undirected:
        edge_index_np = np.hstack([edge_index_np, edge_index_np[::-1,:]])
    edge_index = torch.from_numpy(edge_index_np)
    data = Data(x=None, edge_index=edge_index, num_nodes=len(id_map))

    lbl_col = infer_edge_label_column(df_edges)
    if lbl_col is not None:
        return data, edge_pairs, normalize_binary(df_edges[lbl_col]), id_map

    cnt_col = infer_edge_fraudcount_column(df_edges)
    if cnt_col is not None:
        lbl = (pd.to_numeric(df_edges[cnt_col], errors="coerce").fillna(0).astype(int).values>0).astype(np.int64)
        return data, edge_pairs, lbl, id_map

    if tx_file and os.path.exists(tx_file):
        print(f">> Derivando rótulos a partir de transações: {tx_file}")
        dft = pd.read_parquet(tx_file) if tx_file.lower().endswith(".parquet") else pd.read_csv(tx_file)
        ts_col, tt_col = infer_pair_cols(dft, prefer_tx=True)
        tx_cands = tx_label_col_candidates or ["isFraud","is_fraud","label","fraud","isFlaggedFraud"]
        tx_lbl = next((c for c in tx_cands if c in dft.columns), None)
        if tx_lbl is None: raise ValueError("Arquivo de transações sem coluna de rótulo.")

        ts = map_ids_to_index(dft[ts_col].astype(str).values, id_map)
        tt = map_ids_to_index(dft[tt_col].astype(str).values, id_map)
        tl = normalize_binary(dft[tx_lbl])
        agg = pd.DataFrame({"u":ts,"v":tt,"y":tl}).groupby(["u","v"])["y"].max().reset_index()
        key = {(int(r.u),int(r.v)): int(r.y) for r in agg.itertuples(index=False)}

        labels = np.zeros(edge_pairs.shape[1], dtype=np.int64)
        u,v = edge_pairs[0], edge_pairs[1]
        for i in range(edge_pairs.shape[1]): labels[i] = key.get((int(u[i]),int(v[i])), 0)
        return data, edge_pairs, labels, id_map

    raise ValueError("Sem rótulos/contagem nas arestas e sem --tx-file com rótulos por transação.")

def split_edges(labels: np.ndarray, test_size=0.2, val_size=0.1, seed=42):
    idx_all = np.arange(len(labels)); y = labels.astype(int)
    idx_trv, idx_te, y_trv, _ = train_test_split(idx_all, y, test_size=test_size, stratify=y, random_state=seed)
    val_rel = val_size / (1.0 - test_size)
    idx_tr, idx_va, _, _ = train_test_split(idx_trv, y_trv, test_size=val_rel, stratify=y_trv, random_state=seed)
    return idx_tr, idx_va, idx_te

def make_balanced_edge_train(labels: np.ndarray, idx_train: np.ndarray, seed=42, neg_pos_ratio=1.0):
    rng = np.random.default_rng(seed)
    pos = idx_train[labels[idx_train]==1]; neg = idx_train[labels[idx_train]==0]
    n_pos = len(pos); 
    if n_pos==0 or len(neg)==0: raise ValueError("Sem positivos/negativos no treino.")
    n_neg = min(len(neg), int(n_pos*neg_pos_ratio))
    chosen_neg = rng.choice(neg, size=n_neg, replace=False)
    out = np.concatenate([pos, chosen_neg]); rng.shuffle(out); return out

def maybe_downsample_eval(labels: np.ndarray, idx: np.ndarray, max_neg: Optional[int]) -> np.ndarray:
    if not max_neg or max_neg <= 0: return idx
    pos_idx = idx[labels[idx]==1]; neg_idx = idx[labels[idx]==0]
    if len(neg_idx) <= max_neg: return idx
    keep_neg = np.random.default_rng(42).choice(neg_idx, size=max_neg, replace=False)
    out = np.concatenate([pos_idx, keep_neg]); np.random.default_rng(42).shuffle(out)
    return out

# -------------- Node2Vec --------------

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

# -------------- Metrics & Eval in chunks --------------

def evaluate_probs_in_chunks(model, data, x, edge_pairs, idx, device, batch_size=50000, amp=False):
    """Retorna probs (numpy) para idx em chunks para economizar memória."""
    model.eval()
    probs = np.zeros(idx.shape[0], dtype=np.float32)
    use_cuda_amp = amp and (device.startswith("cuda"))
    from math import ceil
    n = idx.shape[0]; steps = ceil(n / batch_size)
    with torch.no_grad():
        for s in range(steps):
            lo = s*batch_size; hi = min((s+1)*batch_size, n)
            sel = torch.from_numpy(edge_pairs[:, idx[lo:hi]]).long().to(device)
            if use_cuda_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(x.to(device), data.edge_index.to(device), sel)
            else:
                logits = model(x.to(device), data.edge_index.to(device), sel)
            probs[lo:hi] = torch.sigmoid(logits).detach().cpu().numpy()
    return probs

def pr_auc_roc_auc(y_true, prob):
    pr = average_precision_score(y_true, prob) if (np.any(y_true==1) and np.any(y_true==0)) else 0.0
    try:
        roc = roc_auc_score(y_true, prob) if (np.any(y_true==1) and np.any(y_true==0)) else 0.5
    except ValueError:
        roc = 0.5
    return pr, roc

# -------------- Train --------------

def train_edge_model(data, node_feats, edge_pairs, labels, idx_train, idx_val,
                     epochs=100, lr=2e-3, weight_decay=5e-4, device="cpu",
                     use_focal=False, gamma=2.0, alpha=0.25, pos_weight_override=None,
                     train_batch_size=8192, eval_batch_size=50000, val_every=2, amp=False):
    model = N2V_GCN_EdgeModel(n2v_dim=node_feats.size(1), gcn_hidden=64, mlp_hidden=64, p_drop=0.5).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.startswith("cuda"))

    x = node_feats.to(device)
    y_all = torch.tensor(labels, dtype=torch.float32, device=device)

    # Loss
    if use_focal:
        criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        pos_weight_used = None
    else:
        # peso a partir do treino NÃO balanceado (distribuição real)
        y_tr = labels[idx_train]
        n_pos = max(1, int(np.sum(y_tr==1))); n_neg = max(1, int(np.sum(y_tr==0)))
        pw = (n_neg / n_pos)
        if pos_weight_override is not None: pw = float(pos_weight_override)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
        pos_weight_used = pw

    # Scheduler + Early stopping
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    best_state = None; best_val = -1.0; bad = 0; patience = 10

    # mini-batch helper
    def batches(idxs, bs):
        for i in range(0, len(idxs), bs):
            yield idxs[i:i+bs]

    for ep in range(1, epochs+1):
        model.train()
        tot_loss = 0.0
        for chunk in batches(idx_train, train_batch_size):
            opt.zero_grad(set_to_none=True)
            sel = torch.from_numpy(edge_pairs[:, chunk]).long().to(device)
            if amp and device.startswith("cuda"):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(x, data.edge_index.to(device), sel)
                    loss = criterion(logits, y_all[chunk])
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                logits = model(x, data.edge_index.to(device), sel)
                loss = criterion(logits, y_all[chunk])
                loss.backward(); opt.step()
            tot_loss += float(loss)

        # valida a cada N épocas
        if (ep % val_every == 0) or ep == 1 or ep == epochs:
            prob_val = evaluate_probs_in_chunks(model, data, x, edge_pairs, idx_val, device,
                                                batch_size=eval_batch_size, amp=amp)
            yv = labels[idx_val]
            pr_val, roc_val = pr_auc_roc_auc(yv, prob_val)
            sched.step(pr_val)
            if pr_val > best_val:
                best_val = pr_val
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            extra = f" | Focal(gamma={gamma},alpha={alpha})" if use_focal else f" | pos_weight={pos_weight_used:.2f}"
            print(f"[GCN-EDGE] Ép {ep:03d}/{epochs} | TrainLoss(sum): {tot_loss:.4f} | Val PR-AUC: {pr_val:.4f} | Val ROC-AUC: {roc_val:.4f}{extra}")

            if bad >= patience:
                print(f"[EarlyStopping] Sem melhoria por {patience} validações. Parando.")
                break

    if best_state is not None: model.load_state_dict(best_state)
    return model

# -------------- Sweeps --------------

def threshold_sweep(prob, y_true, start, end, step, title):
    print(f"\n=== {title} ({start:.2f}–{end:.2f}) ===")
    print("thr\tRecall\tFAR")
    thr = start
    while thr <= end + 1e-12:
        y_hat = (prob >= thr).astype(int)
        tp = np.sum((y_true==1)&(y_hat==1))
        fn = np.sum((y_true==1)&(y_hat==0))
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        tn = np.sum((y_true==0)&(y_hat==0))
        fp = np.sum((y_true==0)&(y_hat==1))
        far = float(fp)/(fp+tn) if (fp+tn)>0 else 0.0
        print(f"{thr:.2f}\t{recall:.4f}\t{far:.4f}")
        thr = round(thr + step, 10)

# ---------------- Main ----------------

def main():
    p = argparse.ArgumentParser(description="N2V-GCN (PyG) - transações (arestas) PaySim - mem-safe")
    p.add_argument("--nodes", type=str, required=True)
    p.add_argument("--edges", type=str, required=True)
    p.add_argument("--id-col", type=str, default=None)
    p.add_argument("--undirected", action="store_true")

    p.add_argument("--tx-file", type=str, default=None)
    p.add_argument("--tx-label-candidates", type=str, nargs="*", default=None)

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

    # Balanceamento treino
    p.add_argument("--neg-pos-ratio", type=float, default=5.0)

    # Loss options
    p.add_argument("--use-focal", action="store_true")
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--pos-weight", type=float, default=None)

    # Mem-safe knobs
    p.add_argument("--train-batch-size", type=int, default=8192)
    p.add_argument("--eval-batch-size", type=int, default=50000)
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--val-max-neg", type=int, default=0, help="0 = sem amostragem; >0 limita #negativos no Val")
    p.add_argument("--test-max-neg", type=int, default=0, help="0 = sem amostragem; >0 limita #negativos no Test")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="Usar mixed precision (CUDA)")

    args = p.parse_args()
    set_seed(args.seed)

    print(">> Carregando dados (nós/arestas)...")
    data, edge_pairs, edge_labels, id_map = load_graph_and_edges(
        nodes_path=args.nodes, edges_path=args.edges, id_col=args.id_col,
        make_undirected=args.undirected, tx_file=args.tx_file,
        tx_label_col_candidates=args.tx_label_candidates
    )
    print(f"Grafo: N={data.num_nodes} | E(graph)={data.edge_index.size(1)} | E(classif)={edge_pairs.shape[1]} | Pos={int(edge_labels.sum())}")

    # Node2Vec
    if args.epochs_n2v > 0:
        print("\n>> Treinando Node2Vec...")
        n2v_emb = train_node2vec(
            edge_index=data.edge_index, num_nodes=data.num_nodes, emb_dim=args.emb_dim,
            walk_length=args.walk_length, walks_per_node=args.num_walks,
            p=args.p, q=args.q, epochs=args.epochs_n2v, batch_size=args.batch_n2v,
            lr=args.lr_n2v, device=args.device
        )
    else:
        # fallback: aleatório (não recomendado, mas útil pra testes de pipeline/memória)
        n2v_emb = torch.randn(data.num_nodes, args.emb_dim)
        print("(!) Node2Vec pulado (--epochs-n2v 0). Usando embeddings aleatórios.")

    data.x = n2v_emb
    data.num_node_features = data.x.size(1)

    # Splits
    print("\n>> Gerando splits de arestas (treino/val/test)...")
    idx_train, idx_val, idx_test = split_edges(edge_labels, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    idx_train_bal = make_balanced_edge_train(edge_labels, idx_train, seed=args.seed, neg_pos_ratio=args.neg_pos_ratio)
    # Amostra negativos no val/test se solicitado
    idx_val_eff  = maybe_downsample_eval(edge_labels, idx_val, args.val_max_neg)
    idx_test_eff = maybe_downsample_eval(edge_labels, idx_test, args.test_max_neg)

    print(f"Treino (neg:pos={args.neg_pos_ratio:.1f}): {len(idx_train_bal)} | Val: {len(idx_val)} -> {len(idx_val_eff)} | Teste: {len(idx_test)} -> {len(idx_test_eff)}")

    # Treino GCN+Head (minibatches + AMP opcional)
    print("\n>> Treinando GCN (nós) + Head (arestas) com early stopping por Val PR-AUC...")
    model = train_edge_model(
        data=data, node_feats=data.x, edge_pairs=edge_pairs, labels=edge_labels,
        idx_train=idx_train_bal, idx_val=idx_val_eff,
        epochs=args.epochs_gcn, lr=args.lr_gcn, weight_decay=args.wd_gcn,
        device=args.device, use_focal=args.use_focal, gamma=args.gamma, alpha=args.alpha,
        pos_weight_override=args.pos_weight, train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size, val_every=args.val_every, amp=args.amp
    )

    # Avaliação (TESTE) em chunks
    print("\n>> Avaliando no TESTE (prevalência real ou amostrada)...")
    prob_test = evaluate_probs_in_chunks(
        model, data, data.x.to(args.device), edge_pairs, idx_test_eff, args.device,
        batch_size=args.eval_batch_size, amp=args.amp
    )
    y_test = edge_labels[idx_test_eff]
    pr_test, roc_test = pr_auc_roc_auc(y_test, prob_test)
    print(f"[TEST] PR-AUC: {pr_test:.4f} | ROC-AUC: {roc_test:.4f}")

    # Sweeps
    threshold_sweep(prob_test, y_test, 0.00, 1.00, 0.01, "Threshold Sweep (0.00–1.00)")
    threshold_sweep(prob_test, y_test, 0.30, 0.40, 0.01, "Threshold Sweep (0.30–0.40)")

    print("\nConcluído.")

if __name__ == "__main__":
    main()
