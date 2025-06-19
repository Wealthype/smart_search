import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from config import FEATURE_WEIGHTS, VALUE_RANGES

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

# Sidebar for all search controls
st.sidebar.header("Search Filters")

# Text search
search_query = st.sidebar.text_input("Search by name or ISIN:", key="search")

# Private markets filter
private_market_option = st.sidebar.selectbox(
    "Private Markets:", ["All", "Yes", "No"], index=0
)

# Asset class filter (uses asset_class_to_report)
asset_classes = sorted(df['asset_class_to_report'].dropna().unique())
selected_asset_classes = st.sidebar.multiselect("Asset Class:", asset_classes)

# Model portfolio filter
ptf_options = ['All'] + sorted(ptf_df['kpi_portfolio_id'].dropna().unique())
selected_ptf = st.sidebar.selectbox("Ptf Modello:", ptf_options, index=0)

# Filter products based on user input
filtered_df = df
if search_query:
    filtered_df = filtered_df[
        filtered_df['nomeProdotto_frontoffice'].str.contains(search_query, case=False, na=False) |
        filtered_df['ISIN'].astype(str).str.contains(search_query, case=False, na=False)
    ]

if private_market_option == "Yes":
    filtered_df = filtered_df[filtered_df['is_private_markets'] == 1]
elif private_market_option == "No":
    filtered_df = filtered_df[filtered_df['is_private_markets'] == 0]

if selected_asset_classes:
    filtered_df = filtered_df[filtered_df['asset_class_to_report'].isin(selected_asset_classes)]

if selected_ptf != 'All':
    ptf_codes = ptf_df[ptf_df['kpi_portfolio_id'] == selected_ptf]['codiceProdotto_frontoffice'].astype(str)
    filtered_df = filtered_df[filtered_df['codiceProdotto_frontoffice'].astype(str).isin(ptf_codes)]

product_options = filtered_df.apply(
    lambda x: f"{x['nomeProdotto_frontoffice']} ({x['ISIN']})", axis=1
).tolist()
selected_product = st.sidebar.selectbox("Select a product:", product_options)

# Add a button to find similar products
find_similar = st.sidebar.button("Find Similar Products")
if selected_product:
    # Extract ISIN from the selected product string
    selected_isin = selected_product.split("(")[-1].strip(")")
    selected_product_data = df[df['ISIN'] == selected_isin].iloc[0]
        
    # Display selected product details
    st.subheader("Selected Product Details")
        
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
            
        # Calculate distances and similarities
        distances = []
        for _, row in df.iterrows():
            dist = weighted_distance(selected_product_data, row)
            distances.append(dist)
            
        distances = np.array(distances)
        similarities = 1 - distances  # Convert distances to similarities
            
        # Get top 5 similar products (excluding the selected product)
        # First, get the index of the selected product
        selected_idx = df[df['ISIN'] == selected_isin].index[0]
            
        # Create a mask to exclude the selected product
        mask = np.ones(len(similarities), dtype=bool)
        mask[selected_idx] = False
            
        # Get top 5 similar products from the remaining products
        top_indices = np.argsort(similarities[mask])[-5:][::-1]
        # Convert masked indices back to original indices
        original_indices = np.where(mask)[0][top_indices]
            
        # Display similar products with similarity scores
        similar_products = df.iloc[original_indices]
            
        for idx, product in similar_products.iterrows():
            similarity_score = similarities[idx]
            with st.expander(f"{product['nomeProdotto_frontoffice']} ({product['ISIN']}) - Similarity: {similarity_score:.2%}"):
                # Display each category for similar products
                for category, columns in display_categories.items():
                    st.write(f"**{category}**")
                    # Create columns for each category
                    cols = st.columns(3)  # 3 columns for each category
                    for i, (col, display_name) in enumerate(columns.items()):
                        value = product[col]
                        # Round numerical values to integers
                        if isinstance(value, (int, float)):
                            value = int(round(value))
                        # Use modulo to distribute items across columns
                        cols[i % 3].write(f"**{display_name}:** {value}")
                    st.write("---")
