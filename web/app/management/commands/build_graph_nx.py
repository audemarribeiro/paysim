import os
import pandas as pd
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import time, logging
from pathlib import Path

from django.core.management.base import BaseCommand

log = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Constrói grafo de transações PaySim e gera features de nós, arestas e transações."

    def handle(self, *args, **options):
        CSV_PATH = "/data/transactions.csv"
        CHUNKSIZE = 500_000
        AGG_METHOD = "aggregate"   # ou 'stream'
        OUTPUT_GRAPH_GEXF = "/data/paysim_graph.gexf"
        CHUNK2 = 200_000
        out_features_path = "/data/transactions_with_graph_features.parquet"
        out_file = Path(out_features_path)

        def mark(msg):
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

        # -------------------------------------------------------------------
        # 1) Leitura e agregação
        # -------------------------------------------------------------------
        if AGG_METHOD == "aggregate":
            cols_variants = [
                ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"],
                ["step", "action", "amount", "nameOrig", "nameDest", "isFraud"],
            ]

            df = None
            last_err = None
            for cols in cols_variants:
                try:
                    df = pd.read_csv(CSV_PATH, usecols=cols)
                    break
                except Exception as e:
                    last_err = e
                    df = None

            if df is None:
                raise ValueError(
                    f"Não foi possível ler {CSV_PATH} com cabeçalhos esperados. Erro: {last_err}"
                )

            if "action" in df.columns and "type" not in df.columns:
                df = df.rename(columns={"action": "type"})

            df["isFraud"] = df["isFraud"].fillna(0).astype(int)

            agg = df.groupby(["nameOrig", "nameDest"]).agg(
                cnt=("amount", "size"),
                sum_amount=("amount", "sum"),
                mean_amount=("amount", "mean"),
                max_amount=("amount", "max"),
                min_amount=("amount", "min"),
                last_step=("step", "max"),
                fraud_cnt=("isFraud", "sum"),
            ).reset_index()
        else:
            edge_stats = defaultdict(
                lambda: {
                    "cnt": 0,
                    "sum": 0.0,
                    "max": 0.0,
                    "min": float("inf"),
                    "last_step": None,
                    "fraud_cnt": 0,
                }
            )
            for chunk in pd.read_csv(
                CSV_PATH,
                usecols=["step", "amount", "nameOrig", "nameDest", "isFraud"],
                chunksize=CHUNKSIZE,
            ):
                for _, r in chunk.iterrows():
                    key = (r["nameOrig"], r["nameDest"])
                    s = edge_stats[key]
                    s["cnt"] += 1
                    s["sum"] += float(r["amount"])
                    s["max"] = max(s["max"], float(r["amount"]))
                    s["min"] = min(s["min"], float(r["amount"]))
                    s["last_step"] = (
                        max(s["last_step"], int(r["step"]))
                        if s["last_step"] is not None
                        else int(r["step"])
                    )
                    s["fraud_cnt"] += int(r.get("isFraud", 0))

            rows = []
            for (o, d), s in edge_stats.items():
                rows.append(
                    (
                        o,
                        d,
                        s["cnt"],
                        s["sum"],
                        s["sum"] / s["cnt"],
                        s["max"],
                        s["min"],
                        s["last_step"],
                        s["fraud_cnt"],
                    )
                )
            agg = pd.DataFrame(
                rows,
                columns=[
                    "nameOrig",
                    "nameDest",
                    "cnt",
                    "sum_amount",
                    "mean_amount",
                    "max_amount",
                    "min_amount",
                    "last_step",
                    "fraud_cnt",
                ],
            )

        print("Edges aggregated:", len(agg))

        # -------------------------------------------------------------
        # 2) Construir grafo
        # -------------------------------------------------------------
        G = nx.DiGraph()
        for _, row in tqdm(agg.iterrows(), total=len(agg), desc="Adding edges"):
            u = str(row["nameOrig"])
            v = str(row["nameDest"])
            G.add_edge(
                u,
                v,
                weight=row["sum_amount"],
                cnt=int(row["cnt"]),
                mean_amount=float(row["mean_amount"]),
                max_amount=float(row["max_amount"]),
                min_amount=float(row["min_amount"]),
                last_step=int(row["last_step"])
                if pd.notna(row["last_step"])
                else None,
                fraud_cnt=int(row["fraud_cnt"]),
            )

        print("Graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

        # -------------------------------------------------------------
        # 3) Atributos de nó
        # -------------------------------------------------------------
        deg_out = dict(G.out_degree())
        deg_in = dict(G.in_degree())
        total_sent = defaultdict(float)
        total_recv = defaultdict(float)
        for u, v, attr in G.edges(data=True):
            total_sent[u] += attr.get("weight", 0.0)
            total_recv[v] += attr.get("weight", 0.0)

        for n in G.nodes():
            G.nodes[n]["deg_out"] = deg_out.get(n, 0)
            G.nodes[n]["deg_in"] = deg_in.get(n, 0)
            G.nodes[n]["total_sent"] = total_sent.get(n, 0.0)
            G.nodes[n]["total_recv"] = total_recv.get(n, 0.0)
            fsum = sum(a.get("fraud_cnt", 0) for _, _, a in G.out_edges(n, data=True))
            fsum += sum(a.get("fraud_cnt", 0) for _, _, a in G.in_edges(n, data=True))
            G.nodes[n]["fraud_edge_count"] = int(fsum)

        print("Node attributes assigned.")

        # -------------------------------------------------------------
        # 4) Pagerank
        # -------------------------------------------------------------
        mark("Pagerank: iniciando")
        pr = nx.pagerank(G, alpha=0.85, max_iter=200)
        nx.set_node_attributes(G, pr, "pagerank")
        print("Pagerank calculated.")
        mark("Pagerank: concluído")

        # -------------------------------------------------------------
        # 5) Exportar CSVs
        # -------------------------------------------------------------
        mark("Iniciando exportação de features")
        nodes_df = (
            pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
            .reset_index()
            .rename(columns={"index": "account"})
        )
        edges_df = pd.DataFrame(
            [(u, v, attr) for u, v, attr in G.edges(data=True)],
            columns=["nameOrig", "nameDest", "attr"],
        )
        edges_df = edges_df.join(pd.json_normalize(edges_df["attr"])).drop(columns=["attr"])
        nodes_df.to_csv("/data/node_features.csv", index=False)
        edges_df.to_csv("/data/edge_features.csv", index=False)
        print("Node/edge feature CSVs written to /data/")

        # -------------------------------------------------------------
        # 6) Enriquecer transações
        # -------------------------------------------------------------
        node_pagerank = pr
        reader = pd.read_csv(CSV_PATH, chunksize=CHUNK2)

        for i, chunk in enumerate(tqdm(reader, desc="Enrich tx chunks")):
            chunk["orig_pagerank"] = chunk["nameOrig"].map(node_pagerank).fillna(0.0)
            chunk["dest_pagerank"] = chunk["nameDest"].map(node_pagerank).fillna(0.0)
            chunk["orig_deg_out"] = chunk["nameOrig"].map(
                lambda x: G.nodes[x].get("deg_out") if x in G else 0
            )
            chunk["dest_deg_in"] = chunk["nameDest"].map(
                lambda x: G.nodes[x].get("deg_in") if x in G else 0
            )

            if i == 0 and out_file.exists():
                out_file.unlink()

            chunk.to_parquet(
                out_features_path,
                engine="fastparquet",
                index=False,
                compression="snappy",
                append=(i > 0),
            )

        print("Transactions enriched saved to", out_features_path)
