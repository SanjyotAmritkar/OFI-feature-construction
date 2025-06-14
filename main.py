import pandas as pd
from ofi.ofi_calculator import OFICalculator

def main():
    # Load data
    df = pd.read_csv("data/first_25000_rows.csv")

    # Initialize calculator
    calc = OFICalculator(df)

    # Best-Level OFI
    best_ofi = calc.compute_best_level_ofi()
    best_ofi.to_csv("ofi_best_level_output.csv", index=False)

    # Multi-Level OFI
    multi_ofi = calc.compute_multi_level_ofi()
    multi_ofi.to_csv("ofi_multi_level_output.csv", index=False)

    # Integrated OFI
    integrated_ofi = calc.compute_integrated_ofi()
    integrated_ofi.to_csv("ofi_integrated_output.csv", index=False)

    # Log returns
    returns = calc.compute_log_returns()
    returns.to_csv("log_returns_output.csv", index=False)

    # Cross-Asset OFI via LASSO
    cross_asset_matrix = calc.compute_cross_asset_ofi()
    cross_asset_matrix.to_csv("cross_asset_ofi_coefficients.csv")

    print("All OFI computations (including cross-asset) completed.")

if __name__ == "__main__":
    main()
