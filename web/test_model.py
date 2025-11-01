import torch
from model import GCNModel  # ou o nome do modelo usado
import pandas as pd

# Carregar o modelo
model = GCNModel(...)
model.load_state_dict(torch.load("/data/model_best.pt"))
model.eval()

# Carregar dados de teste
tx = pd.read_parquet("/data/casexxx_with_graph_features.parquet")
X = torch.tensor(tx.drop(columns=["isFraud"]).values, dtype=torch.float32)
y = torch.tensor(tx["isFraud"].values, dtype=torch.float32)

# Fazer inferência
with torch.no_grad():
    preds = model(X)
    probs = torch.sigmoid(preds)
    predicted_labels = (probs > 0.5).int()

print("Accuracy:", (predicted_labels == y).float().mean().item())
