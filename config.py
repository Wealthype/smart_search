# Feature groups and their weights
FEATURE_WEIGHTS = {
    'satisfactions': {
        'weight': 0.5,  # 50% of total distance
        'features': ['income', 'multimanager', 'growth', 'lifestyle', 
                    'ig_planning', 'protection', 'retirement',
                    'active_management', 'tax_optimization', 'esg']
    },
    'risk_level': {
        'weight': 0.2,  # 20% of total distance
        'features': ['risk_level']
    },
    'private_markets': {
        'weight': 0.2,  # 20% of total distance
        'features': ['is_private_markets']
    },
    'asset_class': {
        'weight': 0.05,  # 5% of total distance
        'features': ['Asset Class']
    },
    'product_type': {
        'weight': 0.05,  # 5% of total distance
        'features': ['Tipologia Prodotto']
    }
}

# Value ranges for normalization
VALUE_RANGES = {
    'satisfactions': (0, 100),
    'risk_level': (1, 7)
} 