import pandas as pd
p = '/data/casexxx_with_graph_features.parquet'
df = pd.read_parquet(p)

# Ajuste os nomes para as colunas corretas do seu arquivo:
src_col = 'nameOrig'           # ou 'Número Conta'
dst_col = 'nameDest'           # ou 'Número Conta (OD)'

# Nós
nodes = pd.DataFrame({'account': pd.unique(pd.concat([df[src_col], df[dst_col]]).astype(str))})
nodes.to_csv('/data/node_casexxx_features.csv', index=False)

# Arestas
edges = pd.DataFrame({'src': df[src_col].astype(str), 'dst': df[dst_col].astype(str)})
edges.to_csv('/data/edge_casexxx_features.csv', index=False)

print('nodes:', nodes.shape, 'edges:', edges.shape)