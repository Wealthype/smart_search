import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from config import FEATURE_WEIGHTS, VALUE_RANGES, SATISFACTION_THRESHOLD

def normalize_value(value, min_val, max_val):
    """Normalize a value to range [0, 1]"""
    try:
        return (float(value) - min_val) / (max_val - min_val)
    except (ValueError, TypeError):
        return 0.0  # Return 0 for invalid values

def jaccard_distance(x1, x2):
    """Calculate Jaccard distance between two values"""
    try:
        return 1.0 if x1 != x2 else 0.0
    except:
        return 0.0

def weighted_distance(x1, x2):
    """
    Calculate weighted distance between two products based on feature importance.
    Returns a value between 0 and 1, where 0 means identical and 1 means completely different.
    """
    total_distance = 0.0
    
    # Calculate distance for numerical features (0-100)
    numerical_features = FEATURE_WEIGHTS['satisfactions']['features']
    numerical_dist = 0.0
    for feature in numerical_features:
        try:
            val1 = normalize_value(x1[feature], *VALUE_RANGES['satisfactions'])
            val2 = normalize_value(x2[feature], *VALUE_RANGES['satisfactions'])
            numerical_dist += (val1 - val2) ** 2
        except:
            continue
    numerical_dist = np.sqrt(numerical_dist / len(numerical_features))
    total_distance += FEATURE_WEIGHTS['satisfactions']['weight'] * numerical_dist
    
    # Calculate distance for risk level
    try:
        risk_val1 = normalize_value(x1['risk_level'], *VALUE_RANGES['risk_level'])
        risk_val2 = normalize_value(x2['risk_level'], *VALUE_RANGES['risk_level'])
        risk_dist = abs(risk_val1 - risk_val2)
        total_distance += FEATURE_WEIGHTS['risk_level']['weight'] * risk_dist
    except:
        pass
    
    # Calculate Jaccard distance for private markets
    try:
        private_markets_dist = jaccard_distance(x1['is_private_markets'], x2['is_private_markets'])
        total_distance += FEATURE_WEIGHTS['private_markets']['weight'] * private_markets_dist
    except:
        pass
    
    # Calculate Jaccard distance for asset class
    try:
        asset_class_dist = jaccard_distance(x1['Asset Class'], x2['Asset Class'])
        total_distance += FEATURE_WEIGHTS['asset_class']['weight'] * asset_class_dist
    except:
        pass
    
    # Calculate Jaccard distance for product type
    try:
        product_type_dist = jaccard_distance(x1['Tipologia Prodotto'], x2['Tipologia Prodotto'])
        total_distance += FEATURE_WEIGHTS['product_type']['weight'] * product_type_dist
    except:
        pass
    
    return total_distance

@st.cache_data
def load_data():
    df = pd.read_csv("gamma_funds.csv")
    # Filter for investable products right at the start
    df = df[df['is_universo_investibile'] == 1].reset_index(drop=True)
    
    # Convert numerical columns to float
    numerical_cols = FEATURE_WEIGHTS['satisfactions']['features'] + ['risk_level']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

@st.cache_data
def load_goalbased():
    """Load goal-based portfolio data."""
    return pd.read_csv("goalbased_ptfs.csv")

# Set page config
st.set_page_config(page_title="Smart Product Search", layout="wide")

# Load data
df = load_data()
ptf_df = load_goalbased()

# Create a title
st.title("Smart Product Search")

# Top search filters section
with st.container():
    st.subheader("Search Filters")

    # Text search
    search_query = st.text_input("Search by name or ISIN:", key="search")

    # Model portfolio filter
    ptf_options = ['All', 'Any Portfolio'] + sorted(ptf_df['kpi_portfolio_id'].dropna().unique())
    selected_ptf = st.selectbox("Ptf Modello:", ptf_options, index=0)

    # Private markets and asset class filters inside a collapsed section
    with st.expander("Advanced Filters", expanded=False):
        private_market_option = st.selectbox(
            "Private Markets:", ["All", "Yes", "No"], index=0
        )

        asset_classes = sorted(df['asset_class_to_report'].dropna().unique())
        selected_asset_classes = st.multiselect("Asset Class:", asset_classes)

# Filter products based on user input
filtered_df = df
if search_query:
    filtered_df = filtered_df[
        filtered_df['nomeProdotto_frontoffice'].str.contains(search_query, case=False, na=False) |
        filtered_df['ISIN'].astype(str).str.contains(search_query, case=False, na=False)
    ]

if selected_ptf == 'Any Portfolio':
    ptf_codes = ptf_df['codiceProdotto_frontoffice'].astype(str)
    filtered_df = filtered_df[filtered_df['codiceProdotto_frontoffice'].astype(str).isin(ptf_codes)]
elif selected_ptf != 'All':
    ptf_codes = ptf_df[ptf_df['kpi_portfolio_id'] == selected_ptf]['codiceProdotto_frontoffice'].astype(str)
    filtered_df = filtered_df[filtered_df['codiceProdotto_frontoffice'].astype(str).isin(ptf_codes)]

if private_market_option == "Yes":
    filtered_df = filtered_df[filtered_df['is_private_markets'] == 1]
elif private_market_option == "No":
    filtered_df = filtered_df[filtered_df['is_private_markets'] == 0]

if selected_asset_classes:
    filtered_df = filtered_df[filtered_df['asset_class_to_report'].isin(selected_asset_classes)]

# Build a compact table of results so users can scroll through them
results_table = filtered_df[
    ["nomeProdotto_frontoffice", "ISIN", "asset_class_to_report"]
].reset_index(drop=True)

product_options = {
    f"{row['ISIN']} - {row['nomeProdotto_frontoffice']}": row["ISIN"]
    for _, row in results_table.iterrows()
}

results_col, details_col = st.columns([2, 3])

with results_col:
    st.subheader("Select Product")
    if not results_table.empty:
        option_labels = list(product_options.keys())
        selected_label = st.selectbox(
            "Choose a product:", option_labels, key="product_select"
        )
        selected_isin = product_options[selected_label]
    else:
        st.write("No results found.")
        selected_isin = None

with details_col:
    if selected_isin:
        selected_product_data = df[df['ISIN'] == selected_isin].iloc[0]

        header_col, button_col = st.columns([3,1])
        header_col.subheader("Selected Product Details")
        find_similar = button_col.button("Find Similar Products")

        # Define display categories and their columns
        display_categories = {
            "Basic Information": {
                'ISIN': 'ISIN',
                'nomeProdotto_frontoffice': 'Product Name',
                'asset_class_to_report': 'Asset Class',
                'Tipologia Prodotto': 'Product Type'
            },
            "Risk and Market Information": {
                'risk_level': 'Risk Level',
                'minimum_lot': 'Minimum Lot',
                'is_private_markets': 'Private Markets'
            },
            "Satisfactions": {
                'income': 'Income',
                'multimanager': 'Multi Manager',
                'growth': 'Growth',
                'lifestyle': 'Lifestyle',
                'ig_planning': 'IG Planning',
                'protection': 'Protection',
                'retirement': 'Retirement',
                'active_management': 'Active Management',
                'tax_optimization': 'Tax Optimization',
                'esg': 'ESG'
            }
        }
        
        # Display each category
        for category, columns in display_categories.items():
            st.write(f"**{category}**")
            # Create columns for each category
            cols = st.columns(3)  # 3 columns for each category
            for i, (col, display_name) in enumerate(columns.items()):
                value = selected_product_data[col]
                # Round numerical values to integers
                if isinstance(value, (int, float)):
                    value = int(round(value))
                # Use modulo to distribute items across columns
                cols[i % 3].write(f"**{display_name}:** {value}")
            st.write("---")

        # Find and display similar products when button is clicked
        if find_similar:
            st.subheader("Similar Products")

            # --- NEW: Filter by satisfaction if a specific Ptf Modello is selected ---
            satisfaction_threshold = SATISFACTION_THRESHOLD
            filtered_df_for_similarity = df.copy()
            if selected_ptf not in ['All', 'Any Portfolio']:
                satisfaction_feature = selected_ptf.rsplit('_', 1)[0]
                if satisfaction_feature in FEATURE_WEIGHTS['satisfactions']['features']:
                    filtered_df_for_similarity = filtered_df_for_similarity[filtered_df_for_similarity[satisfaction_feature] > satisfaction_threshold]

            # Calculate distances and similarities (from filtered_df_for_similarity)
            distances = []
            for _, row in filtered_df_for_similarity.iterrows():
                dist = weighted_distance(selected_product_data, row)
                distances.append(dist)

            distances = np.array(distances)
            similarities = 1 - distances  # Convert distances to similarities

            # Exclude the selected product itself if present
            filtered_indices = filtered_df_for_similarity.index.tolist()
            if selected_isin in filtered_df_for_similarity['ISIN'].values:
                selected_idx = filtered_df_for_similarity[filtered_df_for_similarity['ISIN'] == selected_isin].index[0]
                mask = np.ones(len(similarities), dtype=bool)
                mask[filtered_df_for_similarity.index.get_loc(selected_idx)] = False
            else:
                mask = np.ones(len(similarities), dtype=bool)

            # Get top 5 similar products from the filtered set
            top_indices = np.argsort(similarities[mask])[-5:][::-1]
            original_indices = np.array(filtered_indices)[mask][top_indices]

            # Display similar products with similarity scores
            for idx in original_indices:
                product = df.loc[idx]
                similarity_score = similarities[filtered_df_for_similarity.index.get_loc(idx)]
                similarity_score_int = int(round(similarity_score * 100))
                with st.expander(
                    f"{product['nomeProdotto_frontoffice']} ({product['ISIN']}) - Similarity: {similarity_score_int}%"
                ):
                    for category, columns in display_categories.items():
                        st.write(f"**{category}**")
                        cols = st.columns(3)
                        for i, (col, display_name) in enumerate(columns.items()):
                            value = product[col]
                            if isinstance(value, (int, float)):
                                value = int(round(value))
                            if category == "Satisfactions":
                                ref_val = selected_product_data[col]
                                if isinstance(ref_val, (int, float)):
                                    ref_val = int(round(ref_val))
                                diff = value - ref_val
                                color = "green" if diff > 0 else "red" if diff < 0 else "black"
                                diff_text = f" ({diff:+d})" if diff != 0 else ""
                                cols[i % 3].markdown(
                                    f"**{display_name}:** <span style='color:{color}'>{value}{diff_text}</span>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                cols[i % 3].write(f"**{display_name}:** {value}")
                        st.write("---")
