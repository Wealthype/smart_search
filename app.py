import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

@st.cache_data
def load_data():
    df = pd.read_csv("gamma_funds_20250611.csv")
    return df[df['is_universo_investibile'] == 1]  # filtro qui direttamente

df = load_data()

st.title("Ricerca strumenti finanziari investibili")

query = st.text_input("Cerca uno strumento (nome o ISIN):")
if query:
    risultati = df[
        df['nomeProdotto_frontoffice'].str.contains(query, case=False, na=False) |
        df['ISIN'].astype(str).str.contains(query, case=False, na=False)
    ]
    columns_to_show = ['ISIN', 'nomeProdotto_frontoffice', 'Asset Class', 'income', 'multimanager', 'growth', 'lifestyle', 'ig_planning', 'protection', 'retirement',
                       'active_management', 'tax_optimization', 'esg', 'risk_level', 'minimum_lot', 'is_private_markets']
    st.write("Risultati ricerca:", risultati[columns_to_show])

    if not risultati.empty:
        idx = risultati.index[0]
        strumento = df.loc[idx]
        st.write("Strumento selezionato:", strumento)

        # Similitudine: su expected_return e expected_volatility
        features = df[['expected_return', 'expected_volatility']].fillna(0).values
        target = features[list(df.index).index(idx)].reshape(1, -1)
        dist = euclidean_distances(features, target).flatten()
        simili_idx = dist.argsort()[1:6]  # primi 5 simili (escludendo sé stesso)

        st.write("Strumenti simili:")
        st.write(df.iloc[simili_idx][columns_to_show])