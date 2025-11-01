#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_ENCODING = "utf-8"

# ---------- Parsers (regex) ----------
RE_N2V = re.compile(
    r"\[Node2Vec\]\s*Época\s*(\d+)\s*/\s*(\d+)\s*\|\s*Loss:\s*([0-9.]+)",
    flags=re.IGNORECASE,
)

RE_GCN = re.compile(
    r"\[GCN\]\s*Ép\s*(\d+)\s*/\s*(\d+)\s*\|\s*TrainLoss\(sum\):\s*([0-9.]+)\s*\|\s*Val PR-AUC:\s*([0-9.]+)\s*\|\s*Val ROC-AUC:\s*([0-9.]+)",
    flags=re.IGNORECASE,
)

RE_TEST_PRROC = re.compile(
    r"\[TEST\]\s*PR-AUC:\s*([0-9.]+)\s*\|\s*ROC-AUC:\s*([0-9.]+)",
    flags=re.IGNORECASE,
)

RE_TEST_RECALL_CAP = re.compile(
    r"\[TEST\]\s*Recall@FAR<=\s*([0-9.]+)\s*:\s*Recall=\s*([0-9.]+)\s*@\s*thr=([0-9.]+)\s*\(FAR=([0-9.]+)\)",
    flags=re.IGNORECASE,
)

RE_SWEEP_HEADER = re.compile(
    r"===\s*Threshold Sweep\s*\(([^)]+)\)\s*\(([^)]+)\)\s*===",
    flags=re.IGNORECASE,
)

RE_SWEEP_ROW = re.compile(
    r"^\s*([01]\.\d{2}|0\.\d{2}|1\.00|0\.00)\s+([0-9.]+)\s+([0-9.]+)\s*$"
)

def parse_log(text: str):
    # Node2Vec
    n2v_epochs, n2v_loss = [], []
    for ep, ep_total, loss in RE_N2V.findall(text):
        n2v_epochs.append(int(ep))
        n2v_loss.append(float(loss))
    n2v_df = pd.DataFrame({"epoch": n2v_epochs, "loss": n2v_loss}).sort_values("epoch")

    # GCN
    gcn_epochs, gcn_trainloss, gcn_pr, gcn_roc = [], [], [], []
    for ep, ep_tot, tl, pr, roc in RE_GCN.findall(text):
        gcn_epochs.append(int(ep))
        gcn_trainloss.append(float(tl))
        gcn_pr.append(float(pr))
        gcn_roc.append(float(roc))
    gcn_df = pd.DataFrame(
        {"epoch": gcn_epochs, "train_loss": gcn_trainloss, "val_pr_auc": gcn_pr, "val_roc_auc": gcn_roc}
    ).sort_values("epoch")

    # TEST metrics
    test_match = RE_TEST_PRROC.search(text)
    test_pr, test_roc = (None, None)
    if test_match:
        test_pr, test_roc = float(test_match.group(1)), float(test_match.group(2))

    # Recall@FAR<=x
    test_caps = []
    for cap, rec, thr, far in RE_TEST_RECALL_CAP.findall(text):
        test_caps.append(
            {"far_cap": float(cap), "recall": float(rec), "thr": float(thr), "far": float(far)}
        )
    caps_df = pd.DataFrame(test_caps).sort_values("far_cap") if test_caps else pd.DataFrame()

    # Sweeps
    # Identify blocks by headers, then parse subsequent numeric lines until next header/blank section.
    sweeps = {}  # name -> DataFrame(thr, recall, far)
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = RE_SWEEP_HEADER.search(lines[i])
        if m:
            # name like "0.30–0.40" and duplicate; we can compose a cleaner key
            rng1 = m.group(1).strip()
            rng2 = m.group(2).strip()
            # Expect rows after this line
            rows = []
            j = i + 1
            # skip lines until header row encountered (starts with 'thr' maybe)
            while j < len(lines) and not RE_SWEEP_ROW.search(lines[j]):
                j += 1
            # now parse rows until break
            while j < len(lines):
                mr = RE_SWEEP_ROW.search(lines[j])
                if not mr:
                    # stop if we hit an empty line or next header
                    if RE_SWEEP_HEADER.search(lines[j]):
                        break
                    # allow non-matching lines (may be "thr Recall FAR" header)
                    j += 1
                    continue
                thr, rec, far = mr.groups()
                rows.append(
                    {
                        "thr": float(thr),
                        "recall": float(rec),
                        "far": float(far),
                    }
                )
                j += 1
            key = f"{rng1}"
            if key in sweeps:
                # disambiguate
                k = 2
                while f"{key}_{k}" in sweeps:
                    k += 1
                key = f"{key}_{k}"
            sweeps[key] = pd.DataFrame(rows)
            i = j
        else:
            i += 1

    return n2v_df, gcn_df, (test_pr, test_roc), caps_df, sweeps


# ---------- Plot helpers ----------
def stylize_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title)
    if grid:   ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

def savefig(fname: Path):
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    print(f"[OK] Figura salva em: {fname}")

# ---------- Main ----------
def main():
    if len(sys.argv) < 2:
        print("Uso: python plot_treinamento.py <arquivo_de_log.txt>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Arquivo não encontrado: {log_path}")
        sys.exit(2)

    text = log_path.read_text(encoding=LOG_ENCODING, errors="ignore")

    n2v_df, gcn_df, (test_pr, test_roc), caps_df, sweeps = parse_log(text)

    out_dir = log_path.parent

    # 1) Node2Vec loss
    if not n2v_df.empty:
        plt.figure(figsize=(7,4.2))
        plt.plot(n2v_df["epoch"], n2v_df["loss"], marker="o")
        stylize_ax(plt.gca(), xlabel="Época", ylabel="Loss", title="Node2Vec — Loss por época")
        savefig(out_dir / "n2v_loss.png")
        plt.close()

    # 2) GCN Train Loss
    if not gcn_df.empty:
        plt.figure(figsize=(7,4.2))
        plt.plot(gcn_df["epoch"], gcn_df["train_loss"], marker="o")
        stylize_ax(plt.gca(), xlabel="Época", ylabel="TrainLoss (sum)", title="GCN — TrainLoss por época")
        savefig(out_dir / "gcn_trainloss.png")
        plt.close()

        # 3) GCN Val PR-AUC e ROC-AUC (em dois eixos Y para leitura clara)
        fig, ax1 = plt.subplots(figsize=(7,4.2))
        ax1.plot(gcn_df["epoch"], gcn_df["val_pr_auc"], marker="o", label="Val PR-AUC")
        ax1.set_xlabel("Época"); ax1.set_ylabel("PR-AUC")
        ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

        ax2 = ax1.twinx()
        ax2.plot(gcn_df["epoch"], gcn_df["val_roc_auc"], marker="s", linestyle="--", label="Val ROC-AUC")
        ax2.set_ylabel("ROC-AUC")

        title = "GCN — Val PR-AUC / ROC-AUC por época"
        if test_pr is not None and test_roc is not None:
            title += f"\n(Teste: PR-AUC={test_pr:.4f}, ROC-AUC={test_roc:.4f})"
        plt.title(title)

        # Legenda combinada
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

        savefig(out_dir / "gcn_val_curves.png")
        plt.close()

    # 4) Sweeps
    # Escolha os nomes que tendem a aparecer no seu log:
    # "0.30–0.40" e "0.00–1.00"
    def plot_sweep(df, title, fname):
        if df.empty:
            print(f"[WARN] Sweep vazio para: {title}")
            return
        fig, ax1 = plt.subplots(figsize=(7,4.2))
        ax1.plot(df["thr"], df["recall"], marker="o", label="Recall")
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Recall")

        ax2 = ax1.twinx()
        ax2.plot(df["thr"], df["far"], marker="s", linestyle="--", label="FAR")
        ax2.set_ylabel("FAR")

        plt.title(title)
        # Legenda combinada
        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="center right")

        savefig(out_dir / fname)
        plt.close()

    # tentar pegar a janela "0.30–0.40" (com traço unicode OU hifen normal)
    key_030_040 = None
    for k in sweeps.keys():
        if ("0.30" in k and "0.40" in k) or ("0,30" in k and "0,40" in k):
            key_030_040 = k; break
    if key_030_040:
        plot_sweep(sweeps[key_030_040], f"Threshold Sweep {key_030_040}", "sweep_030_040.png")

    # tentar pegar a janela "0.00–1.00"
    key_000_100 = None
    for k in sweeps.keys():
        if ("0.00" in k and "1.00" in k) or ("0,00" in k and "1,00" in k):
            key_000_100 = k; break
    if key_000_100:
        plot_sweep(sweeps[key_000_100], f"Threshold Sweep {key_000_100}", "sweep_000_100.png")

    # 5) Resumo Teste na tela
    print("\n===== RESUMO (TESTE) =====")
    if test_pr is not None:
        print(f"PR-AUC  : {test_pr:.4f}")
    if test_roc is not None:
        print(f"ROC-AUC : {test_roc:.4f}")
    if not caps_df.empty:
        for r in caps_df.itertuples(index=False):
            print(f"Recall@FAR<={r.far_cap:.2f}: recall={r.recall:.4f} @thr={r.thr:.2f} (FAR={r.far:.4f})")

if __name__ == "__main__":
    main()
