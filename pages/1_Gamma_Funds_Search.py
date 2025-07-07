import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv("gamma_funds.csv")
    # Show only investable products by default
    if 'is_universo_investibile' in df.columns:
        df = df[df['is_universo_investibile'] == 1]
    return df

df = load_data()

st.title("Gamma Funds Search")

# Search box
search_query = st.text_input("Search by name or ISIN")

# Filter widgets
asset_classes = sorted(df['asset_class_to_report'].dropna().unique())
selected_asset_classes = st.multiselect("Asset Class", asset_classes)

product_types = sorted(df['Tipologia Prodotto'].dropna().unique())
selected_product_types = st.multiselect("Product Type", product_types)

satisfactions = ['None'] + ['income','multimanager','growth','lifestyle','ig_planning','protection','retirement','active_management','tax_optimization','esg']
selected_satisfaction = st.radio("Filter satisfaction > 70", satisfactions, horizontal=True)

# Apply filters
filtered_df = df
if search_query:
    filtered_df = filtered_df[
        filtered_df['nomeProdotto_frontoffice'].str.contains(search_query, case=False, na=False) |
        filtered_df['ISIN'].astype(str).str.contains(search_query, case=False, na=False)
    ]
if selected_asset_classes:
    filtered_df = filtered_df[filtered_df['asset_class_to_report'].isin(selected_asset_classes)]
if selected_product_types:
    filtered_df = filtered_df[filtered_df['Tipologia Prodotto'].isin(selected_product_types)]
if selected_satisfaction != 'None':
    filtered_df = filtered_df[pd.to_numeric(filtered_df[selected_satisfaction], errors='coerce') > 70]

st.write(f"{len(filtered_df)} products found")

st.dataframe(filtered_df.reset_index(drop=True))
