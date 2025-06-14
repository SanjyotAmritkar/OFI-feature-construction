# Order Flow Imbalance (OFI) Feature Construction

This repository contains a modular and well-documented implementation of Order Flow Imbalance (OFI) features and a cross-asset impact analysis, based on the paper:  
*"Cross-Impact of Order Flow Imbalance in Equity Markets"*.

## Objective

To compute various OFI-based features from limit order book data, and analyze their predictive power and cross-asset impact on short-term returns.

## OFI Features Implemented

| Feature              | Description |
|----------------------|-------------|
| ✅ Best-Level OFI     | OFI using bid/ask level 0 |
| ✅ Multi-Level OFI    | OFI across levels 0–9 |
| ✅ Integrated OFI     | PCA-compressed OFI using all levels |
| ✅ Cross-Asset OFI    | LASSO-based estimation of cross-symbol OFI impact |
| ✅ Log Returns        | Computed from mid-price over time |

Each output is timestamp-aligned per symbol.

## How to Run

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/ofi-feature-construction.git
cd ofi-feature-construction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the computation

```bash
python main.py
```

Outputs will be saved as `.csv` files (e.g., `ofi_best_level_output.csv`, `cross_asset_ofi_coefficients.csv`).


## Notes

- For cross-asset OFI to work, the dataset must include multiple symbols.
- If only one symbol (e.g., AAPL) is present, the LASSO model will skip with a warning.

## Technologies Used

- Python 3.10+
- pandas, numpy
- scikit-learn (LassoCV, PCA)

## Author

**Sanjyot Satish Amritkar**  
MS in Data Science, Stony Brook University  
[LinkedIn](https://linkedin.com/in/sanjyotamritkar)

## License

This project is intended for educational and research purposes.
