# Smart Product Search

A Streamlit-based application for finding similar financial products based on weighted feature comparison. The application uses a custom distance metric that takes into account both numerical and categorical features with different weights.

## Features

- Search products by name or ISIN
- Filter results by Ptf Modello (including an option to show only products present in any portfolio)
- Asset Class (`asset_class_to_report`) and Private Markets filters available in a collapsed "Advanced Filters" section
- View detailed product information organized in categories:
  - Basic Information (ISIN, Product Name, Asset Class (`asset_class_to_report`), Product Type)
  - Risk and Market Information (Risk Level, Minimum Lot, Private Markets)
  - Satisfactions (Income, Multi Manager, Growth, etc.)
- Find similar products using a weighted distance metric
- Display similarity scores as percentages
- Compact and organized display format
- Search results shown in a scrollable table with a drop-down selector for choosing a product
- Interactive web interface

## Similarity Calculation

The similarity calculation uses different weights and distance metrics for different types of features:

### Satisfactions (50% weight)
- Uses L2 distance on normalized values (0-100)
- Features: Income, Multi Manager, Growth, Lifestyle, IG Planning, Protection, Retirement, Active Management, Tax Optimization, ESG

### Risk Level (20% weight)
- Uses absolute distance on normalized values (1-7)

### Private Markets (20% weight)
- Uses Jaccard distance (binary comparison)

### Asset Class and Product Type (5% each)
- Uses Jaccard distance for categorical comparison

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

3. The app now includes two pages:
   - **Smart Product Search** (default) for finding similar products.
   - **Gamma Funds Search** for a free search on the `gamma_funds` dataset with additional filters by asset class, product type and a multi-select filter for satisfactions above 70.

4. To find similar products:
   - Use the search box to filter products by name or ISIN
   - Choose a Ptf Modello or the "Any Portfolio" option to restrict results
   - Expand **Advanced Filters** to filter by Asset Class or Private Markets if needed
   - Scroll through the results table on the left and choose a product by ISIN using the drop-down list below it
   - View the selected product's details on the right and click **Find Similar Products** to see the five most similar products

## Data

The application uses a CSV file (`gamma_funds.csv`) containing financial product data. The data is filtered to only include investable products (`is_universo_investibile == 1`).

### Updating the investible flag

If you need to update the `is_universo_investibile` column based on a portfolio
file, use the `update_investible.py` script:

```bash
python update_investible.py goalbased_ptfs.csv gamma_funds.csv -o gamma_funds_updated.csv
```

The script reads product codes from `goalbased_ptfs.csv` (column
`codiceProdotto_frontoffice`) and sets the flag to `1` for any matching rows in
the gamma funds dataset. The updated file is written to the path provided with
`-o` (if omitted, the original file is overwritten).

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt:
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - watchdog (optional, for better performance)

## Configuration

The application uses a `config.py` file to store:
- Feature weights for different categories
- Value ranges for normalization
- Feature groupings

You can adjust the weights and other parameters by modifying this file.

## License

[Add your license information here]

