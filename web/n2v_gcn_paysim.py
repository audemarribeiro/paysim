#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
N2V-GCN para AML (PaySim) - CLASSIFICAÇÃO DE TRANSAÇÕES (ARESTAS)
------------------------------------------------------------------
- Node2Vec nos nós (gera X)
- GCN 2 camadas (16 filtros, ReLU + Dropout)
- Head de aresta (MLP sobre concat/absdiff/hadamard de pares (u,v))
- Treino balanceado 1:1 no TREINO (amostragem de arestas negativas)
- Val/Test com prevalência real
- Early stopping por AP (PR-AUC) em validação
- Sweep de thresholds (0.05–0.95) e (0.30–0.40) reportando Recall e FAR

Uso típico:
  python n2v_gcn_paysim.py \
      --nodes /data/node_features.csv \
      --edges /data/edge_features.csv \
      --id-col account \
      --undirected \
      --epochs-n2v 30 --epochs-gcn 50

Se seu edge_features.csv NÃO tem rótulo, informe um arquivo de transações:
  --tx-file /data/transactions_with_graph_features.parquet
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
from sklearn.metrics import average_precision_score

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import Node2Vec
from torch_geometric.utils import to_undirected

# -----------------------------
# Utilidades
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_pair_cols(df: pd.DataFrame, prefer_tx=False) -> Tuple[str, str]:
    """
    Detecta colunas (u,v) padrão. Em dados de transação (prefer_tx=True), prioriza nameOrig/nameDest.
    """
    candidates_tx = [
        ("nameOrig", "nameDest"),
        ("orig", "dest"),
        ("payer", "payee"),
        ("source", "target"),
    ]
    candidates_edges = [
        ("src", "dst"),
        ("source", "target"),
        ("u", "v"),
        ("orig", "dest"),
        ("account_orig", "account_dest"),
        ("nameOrig", "nameDest"),
    ]
    cand = candidates_tx + candidates_edges if prefer_tx else candidates_edges + candidates_tx
    for s, t in cand:
        if s in df.columns and t in df.columns:
            return s, t
    # fallback: primeiras duas colunas
    if df.shape[1] >= 2:
        return df.columns[0], df.columns[1]
    raise ValueError("Não foi possível identificar colunas de origem/destino.")


def infer_edge_label_column(df_edges: pd.DataFrame) -> Optional[str]:
    for c in ["isFraud", "is_fraud", "label", "fraud", "isFlaggedFraud"]:
        if c in df_edges.columns:
            return c
    return None


def infer_edge_fraudcount_column(df_edges: pd.DataFrame) -> Optional[str]:
    for c in ["fraud_count", "fraud_edges", "count_fraud", "num_fraud", "fraudedgecount"]:
        if c in df_edges.columns:
            return c
    return None


def normalize_binary(series: pd.Series) -> np.ndarray:
    if series.dtype == object:
        arr = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0, "sim": 1, "nao": 0, "não": 0})
            .fillna(0)
            .astype(int)
            .values
        )
    else:
        arr = pd.to_numeric(series, errors="coerce").fillna(0).astype(int).values
    return (arr > 0).astype(np.int64)


def build_id_mapping(ids: List[str]) -> dict:
    uniq = pd.unique(pd.Series(ids)).tolist()
    return {k: i for i, k in enumerate(uniq)}


def far_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    denom = fp + tn
    return float(fp) / denom if denom > 0 else 0.0


# -----------------------------
# Modelos
# -----------------------------

class GCNNodeEncoder(nn.Module):
    def __init__(self, in_dim=128, hidden=16, p_drop=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x, edge_index):
        h = self.act(self.conv1(x, edge_index))
        h = self.drop(h)
        h = self.act(self.conv2(h, edge_index))
        h = self.drop(h)
        return h  # [N, hidden]


class EdgeClassifier(nn.Module):
    """
    Classifica arestas a partir de embeddings dos nós (u,v).
    Usa concatenações: [hu || hv || |hu-hv| || hu*hv] -> MLP -> logit.
    """
    def __init__(self, node_hidden=16, mlp_hidden=32):
        super().__init__()
        in_dim = 4 * node_hidden
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, 1),  # logit
        )

    def forward(self, h, edge_pairs):
        u = h[edge_pairs[0]]
        v = h[edge_pairs[1]]
        feat = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
        logit = self.mlp(feat).view(-1)
        return logit  # logits


class N2V_GCN_EdgeModel(nn.Module):
    def __init__(self, n2v_dim=128, gcn_hidden=16, mlp_hidden=32, p_drop=0.3):
        super().__init__()
        self.node_encoder = GCNNodeEncoder(in_dim=n2v_dim, hidden=gcn_hidden, p_drop=p_drop)
        self.edge_head = EdgeClassifier(node_hidden=gcn_hidden, mlp_hidden=mlp_hidden)

    def forward(self, x, graph_edge_index, edge_pairs):
        h = self.node_encoder(x, graph_edge_index)
        logits = self.edge_head(h, edge_pairs)
        return logits  # logits para arestas em edge_pairs


# -----------------------------
# Carregamento e rotulagem
# -----------------------------

def map_ids_to_index(arr: np.ndarray, id_map: dict) -> np.ndarray:
    mapped = np.empty_like(arr, dtype=np.int64)
    for i, a in enumerate(arr.astype(str)):
        if a not in id_map:
            id_map[a] = len(id_map)
        mapped[i] = id_map[a]
    return mapped


def load_graph_and_edges(
    nodes_path: str,
    edges_path: str,
    id_col: Optional[str],
    make_undirected: bool,
    tx_file: Optional[str] = None,
    tx_label_col_candidates: Optional[List[str]] = None,
) -> Tuple[Data, np.ndarray, np.ndarray, dict]:
    """
    Retorna:
      - data: Data com edge_index do grafo e placeholder X (Node2Vec preenche depois)
      - edge_pairs: np.ndarray shape (2, E) com índices (u,v) das arestas a classificar (direcionadas como no arquivo)
      - edge_labels: np.ndarray shape (E,) com {0,1}
      - id_map: mapeamento de string->índice
    """
    # Nós (apenas para id_map)
    df_nodes = pd.read_csv(nodes_path)
    if id_col is None:
        for c in ["account", "id", "node", "node_id"]:
            if c in df_nodes.columns: id_col = c; break
        if id_col is None:
            id_col = df_nodes.columns[0]
    node_ids = df_nodes[id_col].astype(str).tolist()
    id_map = build_id_mapping(node_ids)

    # Arestas base (para grafo e para classificação)
    df_edges = pd.read_csv(edges_path)
    s_col, t_col = infer_pair_cols(df_edges, prefer_tx=False)
    src = map_ids_to_index(df_edges[s_col].astype(str).values, id_map)
    dst = map_ids_to_index(df_edges[t_col].astype(str).values, id_map)
    edge_pairs = np.vstack([src, dst]).astype(np.int64)

    # edge_index para convolução (pode ser não-direcionado se desejar)
    edge_index_np = np.vstack([src, dst]).astype(np.int64)
    if make_undirected:
        edge_index_np = np.hstack([edge_index_np, edge_index_np[::-1, :]])
    edge_index = torch.from_numpy(edge_index_np)
    data = Data(x=None, edge_index=edge_index, num_nodes=len(id_map))

    # ---------- RÓTULOS DAS ARESTAS ----------
    # 1) Na própria tabela de arestas?
    lbl_col = infer_edge_label_column(df_edges)
    if lbl_col is not None:
        edge_labels = normalize_binary(df_edges[lbl_col])
        return data, edge_pairs, edge_labels, id_map

    # 2) Contagem de fraudes na aresta?
    cnt_col = infer_edge_fraudcount_column(df_edges)
    if cnt_col is not None:
        edge_labels = (pd.to_numeric(df_edges[cnt_col], errors="coerce").fillna(0).astype(int).values > 0).astype(np.int64)
        return data, edge_pairs, edge_labels, id_map

    # 3) Derivar de um arquivo de transações (CSV/Parquet) com rótulo por transação
    if tx_file is not None and os.path.exists(tx_file):
        print(f">> Derivando rótulos a partir de transações: {tx_file}")
        if tx_file.lower().endswith(".parquet"):
            dft = pd.read_parquet(tx_file)
        else:
            dft = pd.read_csv(tx_file)

        # detectar colunas de pares em transação
        ts_col, tt_col = infer_pair_cols(dft, prefer_tx=True)

        # detectar coluna de rótulo em transação
        tx_label_candidates = tx_label_col_candidates or ["isFraud", "is_fraud", "label", "fraud", "isFlaggedFraud"]
        tx_lbl = None
        for c in tx_label_candidates:
            if c in dft.columns:
                tx_lbl = c
                break
        if tx_lbl is None:
            raise ValueError(
                f"Arquivo de transações informado ({tx_file}) não tem coluna de rótulo "
                f"(tente nomes como {tx_label_candidates})."
            )

        # mapeia IDs do arquivo de transações ao mesmo id_map
        ts = map_ids_to_index(dft[ts_col].astype(str).values, id_map)
        tt = map_ids_to_index(dft[tt_col].astype(str).values, id_map)
        tl = normalize_binary(dft[tx_lbl])

        # agrega por par (u,v): label = 1 se QUALQUER transação (u,v) for fraudulenta
        # construir dict de (u,v) -> 1/0
        df_tmp = pd.DataFrame({"u": ts, "v": tt, "y": tl})
        agg = df_tmp.groupby(["u", "v"])["y"].max().reset_index()

        # agora precisamos alinhar com edge_pairs (src,dst)
        # cria chave para pesquisa rápida
        key_to_y = {(int(r.u), int(r.v)): int(r.y) for r in agg.itertuples(index=False)}

        labels = np.zeros(edge_pairs.shape[1], dtype=np.int64)
        u = edge_pairs[0]; v = edge_pairs[1]
        for i in range(edge_pairs.shape[1]):
            labels[i] = key_to_y.get((int(u[i]), int(v[i])), 0)

        return data, edge_pairs, labels, id_map

    # 4) Nada encontrado → mensagem clara
    raise ValueError(
        "Não encontrei rótulo nas arestas nem contagem de fraude. "
        "Soluções:\n"
        "  (a) adicione uma coluna de rótulo nas arestas (ex.: isFraud), OU\n"
        "  (b) adicione uma coluna de contagem (ex.: fraud_count) e ela será mapeada para label>0, OU\n"
        "  (c) forneça um arquivo de transações com rótulos usando --tx-file (CSV/Parquet), "
        "que eu agrego por par (u,v) para rotular as arestas."
    )


# -----------------------------
# Splits e balanceamento (ARESTAS)
# -----------------------------

def split_edges(labels: np.ndarray, test_size=0.2, val_size=0.1, seed=42):
    idx_all = np.arange(len(labels))
    y_all = labels.astype(int)

    idx_trainval, idx_test, y_trainval, _ = train_test_split(
        idx_all, y_all, test_size=test_size, stratify=y_all, random_state=seed
    )
    val_rel = val_size / (1.0 - test_size)
    idx_train, idx_val, _, _ = train_test_split(
        idx_trainval, y_trainval, test_size=val_rel, stratify=y_trainval, random_state=seed
    )
    return idx_train, idx_val, idx_test


def make_balanced_edge_train(labels: np.ndarray, idx_train: np.ndarray, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos_idx = idx_train[labels[idx_train] == 1]
    neg_idx = idx_train[labels[idx_train] == 0]
    n_pos = len(pos_idx)
    if n_pos == 0: raise ValueError("Não há arestas positivas no treino.")
    if len(neg_idx) == 0: raise ValueError("Não há arestas negativas no treino.")
    chosen_neg = rng.choice(neg_idx, size=n_pos, replace=False)
    balanced = np.concatenate([pos_idx, chosen_neg])
    rng.shuffle(balanced)
    return balanced


# -----------------------------
# Node2Vec (nós)
# -----------------------------

def train_node2vec(
    edge_index: torch.Tensor,
    num_nodes: int,
    emb_dim: int = 128,
    walk_length: int = 32,
    walks_per_node: int = 15,
    p: float = 1.0,
    q: float = 1.0,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.01,
    device: str = "cpu",
) -> torch.Tensor:
    n2v = Node2Vec(
        edge_index=edge_index,
        embedding_dim=emb_dim,
        walk_length=walk_length,
        context_size=walk_length,
        walks_per_node=walks_per_node,
        p=p,
        q=q,
        num_nodes=num_nodes,
        sparse=True,
    ).to(device)

    loader = n2v.loader(batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(n2v.parameters()), lr=lr)

    n2v.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss)
        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            print(f"[Node2Vec] Época {epoch:02d}/{epochs} | Loss: {total:.4f}")

    with torch.no_grad():
        emb = n2v.embedding.weight.detach().cpu()
    return emb


# -----------------------------
# Treino/Val/Test (ARESTAS)
# -----------------------------

def evaluate_ap(model, data, x, graph_edge_index, edge_pairs, labels, idx, device="cpu"):
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device), graph_edge_index.to(device),
                       torch.from_numpy(edge_pairs[:, idx]).long().to(device))
        prob = torch.sigmoid(logits).cpu().numpy()
    y = labels[idx]
    ap = average_precision_score(y, prob) if np.any(y==1) and np.any(y==0) else 0.0
    return ap


def train_edge_model(
    data: Data,
    node_feats: torch.Tensor,
    edge_pairs: np.ndarray,
    edge_labels: np.ndarray,
    idx_train_bal: np.ndarray,
    idx_val: np.ndarray,
    epochs: int = 50,
    lr: float = 2e-3,
    weight_decay: float = 5e-4,
    device: str = "cpu",
):
    model = N2V_GCN_EdgeModel(n2v_dim=node_feats.size(1), gcn_hidden=16, mlp_hidden=32, p_drop=0.3).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()

    x = node_feats.to(device)
    y = torch.tensor(edge_labels, dtype=torch.float32, device=device)

    best_state = None
    best_ap = -1.0
    patience = 10
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(x, data.edge_index.to(device),
                       torch.from_numpy(edge_pairs[:, idx_train_bal]).long().to(device))
        loss = crit(logits, y[idx_train_bal])
        loss.backward()
        optimizer.step()

        ap_val = evaluate_ap(model, data, x, data.edge_index, edge_pairs, edge_labels, idx_val, device=device)

        if ap_val > best_ap:
            best_ap = ap_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[GCN-EDGE] Época {epoch:02d}/{epochs} | TrainLoss: {loss.item():.4f} | Val AP: {ap_val:.4f} | Best AP: {best_ap:.4f}")

        if bad >= patience:
            print(f"[EarlyStopping] Sem melhora por {patience} épocas. Parando.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_threshold_sweep_edges(
    model: nn.Module,
    data: Data,
    node_feats: torch.Tensor,
    edge_pairs: np.ndarray,
    edge_labels: np.ndarray,
    idx_eval: np.ndarray,
    thr_start: float = 0.30,
    thr_end: float = 0.40,
    thr_step: float = 0.01,
    device: str = "cpu",
    title: str = "Threshold Sweep",
):
    model.eval()
    with torch.no_grad():
        logits = model(
            node_feats.to(device),
            data.edge_index.to(device),
            torch.from_numpy(edge_pairs[:, idx_eval]).long().to(device),
        )
        prob = torch.sigmoid(logits).cpu().numpy()

    y_true = edge_labels[idx_eval].astype(int)

    print(f"\n=== {title} ({thr_start:.2f}–{thr_end:.2f}) ===")
    print("thr\tRecall\tFAR")
    thr = thr_start
    while thr <= thr_end + 1e-12:
        y_hat = (prob >= thr).astype(int)
        tp = np.sum((y_true == 1) & (y_hat == 1))
        fn = np.sum((y_true == 1) & (y_hat == 0))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # FAR = FP / (FP + TN)
        tn = np.sum((y_true == 0) & (y_hat == 0))
        fp = np.sum((y_true == 0) & (y_hat == 1))
        far = float(fp) / (fp + tn) if (fp + tn) > 0 else 0.0
        print(f"{thr:.2f}\t{recall:.4f}\t{far:.4f}")
        thr = round(thr + thr_step, 10)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="N2V-GCN (PyG) - classificação de transações (arestas) no PaySim.")
    parser.add_argument("--nodes", type=str, default="/data/node_features.csv")
    parser.add_argument("--edges", type=str, default="/data/edge_features.csv")
    parser.add_argument("--id-col", type=str, default=None, help="Coluna do ID do nó (ex.: account)")
    parser.add_argument("--undirected", action="store_true", help="Converte o grafo para não-direcionado para o GCN")

    # Fonte alternativa de rótulos (transações)
    parser.add_argument("--tx-file", type=str, default=None, help="CSV/Parquet com transações e rótulo por transação")
    parser.add_argument("--tx-label-candidates", type=str, nargs="*", default=None,
                        help="Nomes possíveis da coluna de rótulo nas transações (default: isFraud, is_fraud, label, fraud, isFlaggedFraud)")

    # Node2Vec
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--walk-length", type=int, default=32)
    parser.add_argument("--num-walks", type=int, default=15)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--epochs-n2v", type=int, default=30)
    parser.add_argument("--lr-n2v", type=float, default=0.01)
    parser.add_argument("--batch-n2v", type=int, default=128)

    # GCN/Head
    parser.add_argument("--epochs-gcn", type=int, default=50)
    parser.add_argument("--lr-gcn", type=float, default=2e-3)
    parser.add_argument("--wd-gcn", type=float, default=5e-4)

    # Splits
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)

    # Sweeps
    parser.add_argument("--thr-start", type=float, default=0.30)
    parser.add_argument("--thr-end", type=float, default=0.40)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    set_seed(args.seed)

    print(">> Carregando dados (nós/arestas)...")
    data, edge_pairs, edge_labels, id_map = load_graph_and_edges(
        nodes_path=args.nodes,
        edges_path=args.edges,
        id_col=args.id_col,
        make_undirected=args.undirected,
        tx_file=args.tx_file,
        tx_label_col_candidates=args.tx_label_candidates,
    )
    print(f"Grafo: N={data.num_nodes} | E(graph)={data.edge_index.size(1)} | E(originais p/ classificação)={edge_pairs.shape[1]} | Pos={int(edge_labels.sum())}")

    # Node2Vec (features dos nós)
    print("\n>> Treinando Node2Vec...")
    n2v_emb = train_node2vec(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        emb_dim=args.emb_dim,
        walk_length=args.walk_length,
        walks_per_node=args.num_walks,
        p=args.p,
        q=args.q,
        epochs=args.epochs_n2v,
        batch_size=args.batch_n2v,
        lr=args.lr_n2v,
        device=args.device,
    )
    data.x = n2v_emb
    data.num_node_features = data.x.size(1)

    # Splits de ARESTAS (estratificados)
    print("\n>> Gerando splits de arestas (treino/val/test)...")
    idx_train, idx_val, idx_test = split_edges(edge_labels, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    idx_train_bal = make_balanced_edge_train(edge_labels, idx_train, seed=args.seed)
    print(f"Treino (balanceado 1:1): {len(idx_train_bal)} arestas | Val: {len(idx_val)} | Teste: {len(idx_test)}")

    # Treino do GCN+Head (arestas)
    print("\n>> Treinando GCN (nós) + Head (arestas) com early stopping por AP...")
    model = train_edge_model(
        data=data,
        node_feats=data.x,
        edge_pairs=edge_pairs,
        edge_labels=edge_labels,
        idx_train_bal=idx_train_bal,
        idx_val=idx_val,
        epochs=args.epochs_gcn,
        lr=args.lr_gcn,
        weight_decay=args.wd_gcn,
        device=args.device,
    )

    # Avaliação (Test)
    print("\n>> Avaliando no TESTE (prevalência real)...")
    # Sweep amplo para localizar bons pontos (estilo AML)
    eval_threshold_sweep_edges(
        model, data, data.x, edge_pairs, edge_labels, idx_test,
        thr_start=0.05, thr_end=0.95, thr_step=0.05, device=args.device,
        title="Threshold Sweep (0.05–0.95)"
    )
    # Sweep solicitado (0.30–0.40)
    eval_threshold_sweep_edges(
        model, data, data.x, edge_pairs, edge_labels, idx_test,
        thr_start=args.thr_start, thr_end=args.thr_end, thr_step=args.thr_step, device=args.device,
        title="Threshold Sweep (0.30–0.40)"
    )

    print("\nConcluído.")

if __name__ == "__main__":
    main()
