import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

st.title("Previsão de Preço de Casas")

# Ler CSV
df = pd.read_csv("Book.csv", encoding="latin1", sep=";")
#dropa linhas em branco
df = df.dropna(how="all")
df = df.rename(columns={df.columns[0]: "tamanho", df.columns[1]: "precos"})
df = df[~df["tamanho"].str.contains("Tamanho", na=False)]
df = df[~df["precos"].str.contains("preço|precos|Preços", na=False)]
df["tamanho"] = df["tamanho"].str.replace("m2", "", regex=False).astype(float)
df["precos"] = (
    df["precos"]
    .str.replace("R$", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# Treinar modelo
modelo = LinearRegression()
modelo.fit(df[["tamanho"]], df[["precos"]])

# Entrada do usuário
tamanho_input = st.number_input("Digite o tamanho da casa (m²): ", min_value=0.0, step=1.0)

# Previsão
if tamanho_input > 0:
    preco_previsto = modelo.predict([[tamanho_input]])[0][0]
    st.write(f"💰 Preço previsto: R$ {preco_previsto:,.2f}")
