import argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse

# ---------- Utils ----------
def set_seed(seed):
    import random; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def threshold_sweep(probs, y_true, grid=None):
    import numpy as np
    if grid is None:
        grid = np.round(np.linspace(0.05, 0.995, 20), 3)
    best = None
    for t in grid:
        pred = (probs >= t).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        rec  = recall_score(y_true, pred, zero_division=0)
        f1 = 0. if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        if best is None or f1 > best["f1"]:
            best = {"thr": t, "precision": prec, "recall": rec, "f1": f1}
    return best

def threshold_at_precision(probs, y_true, min_precision=0.90):
    grid = np.sort(np.unique(np.round(probs, 6)))
    best = {"thr":1.0, "precision":1.0, "recall":0.0, "f1":0.0}
    for t in grid[::-1]:
        pred = (probs >= t).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        if prec >= min_precision:
            rec  = recall_score(y_true, pred, zero_division=0)
            f1   = 0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
            if rec > best["recall"]:
                best = {"thr": t, "precision": prec, "recall": rec, "f1": f1}
    return best

def stratified_node_split(y, train_p=0.6, val_p=0.2, test_p=0.2, seed=42):
    set_seed(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.RandomState(seed)
    def take(arr, p):
        n = int(len(arr)*p); rng.shuffle(arr); return arr[:n], arr[n:]
    pos_tr, pos_rest = take(pos, train_p); val_portion = val_p/(val_p+test_p)
    pos_va, pos_te = take(pos_rest, val_portion)
    neg_tr, neg_rest = take(neg, train_p)
    neg_va, neg_te = take(neg_rest, val_portion)
    idx_tr = np.concatenate([pos_tr, neg_tr]); idx_va = np.concatenate([pos_va, neg_va]); idx_te = np.concatenate([pos_te, neg_te])
    rng.shuffle(idx_tr); rng.shuffle(idx_va); rng.shuffle(idx_te)
    return idx_tr, idx_va, idx_te

# ---------- Models ----------
class GCNNode(nn.Module):
    def __init__(self, in_dim, hid=32, out_dim=1, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.lin   = nn.Linear(hid, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x).squeeze(-1)

class GATNode(nn.Module):
    def __init__(self, in_dim, hid=16, heads=4, out_dim=1, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_dim, hid, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hid*heads, hid, heads=1, dropout=dropout)
        self.lin  = nn.Linear(hid, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        return self.lin(x).squeeze(-1)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes_csv", default="/data/node_features.csv")
    ap.add_argument("--edges_csv", default="/data/edge_features.csv")
    ap.add_argument("--model", choices=["gcn","gat"], default="gcn")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--num_neighbors", type=int, nargs=2, default=[15,10], help="vizinhos por camada")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # --- Nodes ---
    nf = pd.read_csv(args.nodes_csv)
    assert "account" in nf.columns and "fraud_edge_count" in nf.columns
    y = (nf["fraud_edge_count"].values.astype(int) > 0).astype(int)
    drop_cols = ["account", "fraud_edge_count"]
    feat_cols = [c for c in nf.columns if c not in drop_cols]
    X = nf[feat_cols].astype(float).values
    if args.standardize:
        X = StandardScaler().fit_transform(X)

    # map accounts
    acc_index = pd.Series(np.arange(len(nf)), index=nf["account"].values)

    # --- Edges ---
    ef = pd.read_csv(args.edges_csv)
    ok = ef["nameOrig"].isin(acc_index.index) & ef["nameDest"].isin(acc_index.index)
    ef = ef[ok]
    src = acc_index.loc[ef["nameOrig"].values].values
    dst = acc_index.loc[ef["nameDest"].values].values
    # undirected
    row = np.concatenate([src, dst]); col = np.concatenate([dst, src])
    num_nodes = len(nf)
    A = sparse.coo_matrix((np.ones_like(row, dtype=np.float32), (row, col)), shape=(num_nodes, num_nodes))
    edge_index, _ = from_scipy_sparse_matrix(A)

    x = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, y=y_t)

    # --- Splits ---
    idx_tr, idx_va, idx_te = stratified_node_split(y, 0.6, 0.2, 0.2, seed=args.seed)
    # Neighbor loaders (subgrafos por lote)
    train_loader = NeighborLoader(
        data, input_nodes=torch.tensor(idx_tr), num_neighbors=args.num_neighbors,
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = NeighborLoader(
        data, input_nodes=torch.tensor(idx_va), num_neighbors=args.num_neighbors,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    test_loader = NeighborLoader(
        data, input_nodes=torch.tensor(idx_te), num_neighbors=args.num_neighbors,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    in_dim = data.x.size(1)
    if args.model == "gcn":
        model = GCNNode(in_dim, hid=32, out_dim=1, dropout=args.dropout).to(device)
    else:
        model = GATNode(in_dim, hid=16, heads=4, out_dim=1, dropout=args.dropout).to(device)

    # Desbalanceamento usando a taxa do treino
    train_pos_ratio = y[idx_tr].mean()
    pos_weight = torch.tensor([(1 - train_pos_ratio) / max(train_pos_ratio, 1e-6)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_ap = -1.0; best_state = None; patience=10; bad=0

    for epoch in range(1, args.epochs+1):
        # ---- Train ----
        model.train()
        total_loss = 0.0; n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            # batch has a subset of nodes; its 'y' aligns with batch.n_id indexing
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item(); n_batches += 1
        avg_loss = total_loss / max(n_batches,1)

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            probs_va = []; y_va = []
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                prob = torch.sigmoid(out[:batch.batch_size]).detach().cpu().numpy()
                yb = batch.y[:batch.batch_size].detach().cpu().numpy()
                probs_va.append(prob); y_va.append(yb)
            probs_va = np.concatenate(probs_va); y_va = np.concatenate(y_va)
            try:
                roc_va = roc_auc_score(y_va, probs_va)
            except Exception:
                roc_va = float("nan")
            ap_va = average_precision_score(y_va, probs_va)

        print(f"[{epoch:02d}] loss={avg_loss:.4f} | VAL: ROC={roc_va:.4f} AP={ap_va:.4f} | best_AP={max(best_ap,0):.4f}")

        if ap_va > best_ap + 1e-6:
            best_ap = ap_va
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (paciência={patience}) no epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Test ----
    model.eval()
    with torch.no_grad():
        probs_te = []; y_te = []
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            prob = torch.sigmoid(out[:batch.batch_size]).detach().cpu().numpy()
            yb = batch.y[:batch.batch_size].detach().cpu().numpy()
            probs_te.append(prob); y_te.append(yb)
        probs_te = np.concatenate(probs_te); y_te = np.concatenate(y_te)

    roc = roc_auc_score(y_te, probs_te)
    ap  = average_precision_score(y_te, probs_te)
    print("\n== TESTE ==")
    print(f"ROC-AUC: {roc:.6f}")
    print(f"PR-AUC : {ap:.6f}")

    best_f1 = threshold_sweep(probs_te, y_te)
    print(f"\n== Threshold ótimo por F1 ==\nF1*={best_f1['f1']:.4f} @ thr={best_f1['thr']:.6f} | P={best_f1['precision']:.4f} | R={best_f1['recall']:.4f}")

    p90 = threshold_at_precision(probs_te, y_te, 0.90)
    print(f"\n== Threshold para P>=90% ==\nP={p90['precision']:.4f} | R={p90['recall']:.4f} @ thr={p90['thr']:.6f} | F1={p90['f1']:.4f}")

if __name__ == "__main__":
    main()
