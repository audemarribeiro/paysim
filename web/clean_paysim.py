import pandas as pd

def parse_valor(valor):
    if pd.isna(valor):
        return 0.0
    valor = str(valor)
    valor = valor.replace("R$", "").replace(" ", "")
    valor = valor.replace(".", "").replace(",", ".")
    try:
        return float(valor)
    except ValueError:
        return 0.0
    
# Carregar o CSV real
df = pd.read_csv("data/casoxxx.csv", sep=",", dtype=str)

# Limpar e converter valores numéricos
df["Valor Lançamento"] = df["Valor Lançamento"].str.replace("R$", "", regex=False)
# df["Valor Lançamento"] = df["Valor Lançamento"].str.replace(",", ".").astype(float)
df["Valor Lançamento"] = df["Valor Lançamento"].apply(parse_valor)

df["Valor Saldo"] = df["Valor Saldo"].str.replace("R$", "", regex=False)
# df["Valor Saldo"] = df["Valor Saldo"].str.replace(",", ".").astype(float)
df["Valor Saldo"] = df["Valor Saldo"].apply(parse_valor)
# Criar campos no formato PaySim
converted = pd.DataFrame({
    "step": pd.factorize(df["Data Lançamento"])[0],
    "type": df["Tipo Lançamento"].fillna("DESCONHECIDO"),
    "amount": df["Valor Lançamento"],
    "nameOrig": df["Número Conta"].fillna("UNK_ORIG"),
    "oldbalanceOrg": df["Valor Saldo"],
    "newbalanceOrg": df["Valor Saldo"] - df["Valor Lançamento"],
    "nameDest": df["Número Conta (OD)"].fillna("UNK_DEST"),
    "oldbalanceDest": 0,
    "newbalanceDest": 0,
    "isFraud": df["Pessoa Investigada Caso?"].apply(lambda x: 1 if str(x).strip().lower() == "sim" else 0),
    "isFlaggedFraud": 0
})

# Salvar no formato Parquet
converted.to_parquet("/data/casexxx_with_graph_features.parquet", index=False)
print("Arquivo convertido salvo em /data/casexxx_with_graph_features.parquet")
