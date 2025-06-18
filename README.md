# Smart Product Search

A Streamlit-based application for finding similar financial products based on weighted feature comparison. The application uses a custom distance metric that takes into account both numerical and categorical features with different weights.

## Features

- Search products by name or ISIN
- View detailed product information
- Find similar products using a weighted distance metric
- Interactive web interface
- Support for both numerical and categorical features

## Feature Weights

The similarity calculation uses different weights for different types of features:

### High Weight Features (Weight: 1.0)
- Income
- Multimanager
- Growth
- Lifestyle
- IG Planning
- Protection
- Retirement
- Active Management
- Tax Optimization
- ESG
- Risk Level
- Private Markets

### Low Weight Features (Weight: 0.5)
- Asset Class
- Product Type (Tipologia Prodotto)

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd smart_search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For better performance on macOS, install Watchdog:
```bash
xcode-select --install
pip install watchdog
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. The application will open in your default web browser

3. To find similar products:
   - Use the search box to filter products by name or ISIN
   - Select a product from the dropdown menu
   - Click "Find Similar Products" to see the 5 most similar products

## Data

The application uses a CSV file (`gamma_funds_20250611.csv`) containing financial product data. The data is filtered to only include investable products (`is_universo_investibile == 1`).

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## License

