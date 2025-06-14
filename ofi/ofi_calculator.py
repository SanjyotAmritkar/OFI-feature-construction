import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

class OFICalculator:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the calculator with the raw order book dataframe.
        """
        self.df = df.copy()
        self.df['ts_event'] = pd.to_datetime(self.df['ts_event'])
        self.df.sort_values(['symbol', 'ts_event'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def compute_best_level_ofi(self) -> pd.DataFrame:
        """
        Compute best-level OFI (level 0) for each symbol over time.
        """
        ofi_all = []

        for sym, grp in self.df.groupby('symbol'):
            grp = grp.sort_values('ts_event').reset_index(drop=True)
            ofi_values = [0]

            for i in range(1, len(grp)):
                prev = grp.iloc[i - 1]
                curr = grp.iloc[i]

                # Bid side logic
                if curr['bid_px_00'] > prev['bid_px_00']:
                    bid_ofi = curr['bid_sz_00']
                elif curr['bid_px_00'] == prev['bid_px_00']:
                    bid_ofi = curr['bid_sz_00'] - prev['bid_sz_00']
                else:
                    bid_ofi = -prev['bid_sz_00']

                # Ask side logic
                if curr['ask_px_00'] < prev['ask_px_00']:
                    ask_ofi = curr['ask_sz_00']
                elif curr['ask_px_00'] == prev['ask_px_00']:
                    ask_ofi = prev['ask_sz_00'] - curr['ask_sz_00']
                else:
                    ask_ofi = -prev['ask_sz_00']

                ofi = bid_ofi - ask_ofi
                ofi_values.append(ofi)

            grp['ofi_best_level'] = ofi_values
            ofi_all.append(grp)

        self.df = pd.concat(ofi_all).sort_values(['symbol', 'ts_event']).reset_index(drop=True)
        return self.df[['ts_event', 'symbol', 'ofi_best_level']]

    def compute_multi_level_ofi(self, levels: int = 10) -> pd.DataFrame:
        """
        Compute OFI for multiple depth levels (default: 0–9).
        """
        for m in range(levels):
            col = f'ofi_lvl_{m}'
            bid_px_col = f'bid_px_0{m}'
            ask_px_col = f'ask_px_0{m}'
            bid_sz_col = f'bid_sz_0{m}'
            ask_sz_col = f'ask_sz_0{m}'

            ofi_all = []

            for sym, grp in self.df.groupby('symbol'):
                grp = grp.sort_values('ts_event').reset_index(drop=True)
                ofi_values = [0]

                for i in range(1, len(grp)):
                    prev = grp.iloc[i - 1]
                    curr = grp.iloc[i]

                    # Bid OFI
                    if curr[bid_px_col] > prev[bid_px_col]:
                        bid_ofi = curr[bid_sz_col]
                    elif curr[bid_px_col] == prev[bid_px_col]:
                        bid_ofi = curr[bid_sz_col] - prev[bid_sz_col]
                    else:
                        bid_ofi = -prev[bid_sz_col]

                    # Ask OFI
                    if curr[ask_px_col] < prev[ask_px_col]:
                        ask_ofi = curr[ask_sz_col]
                    elif curr[ask_px_col] == prev[ask_px_col]:
                        ask_ofi = prev[ask_sz_col] - curr[ask_sz_col]
                    else:
                        ask_ofi = -prev[ask_sz_col]

                    ofi = bid_ofi - ask_ofi
                    ofi_values.append(ofi)

                grp[col] = ofi_values
                ofi_all.append(grp)

            self.df = pd.concat(ofi_all).sort_values(['symbol', 'ts_event']).reset_index(drop=True)

        return self.df[['ts_event', 'symbol'] + [f'ofi_lvl_{m}' for m in range(levels)]]

    def compute_integrated_ofi(self, levels: int = 10) -> pd.DataFrame:
        """
        Compute integrated OFI using PCA on normalized multi-level OFIs.
        """
        self.compute_multi_level_ofi(levels)

        for m in range(levels):
            bid_sz_col = f'bid_sz_0{m}'
            ask_sz_col = f'ask_sz_0{m}'
            self.df[f'depth_avg_{m}'] = 0.5 * (self.df[bid_sz_col] + self.df[ask_sz_col])
            self.df[f'norm_ofi_{m}'] = self.df[f'ofi_lvl_{m}'] / (self.df[f'depth_avg_{m}'] + 1e-6)

        norm_ofi_cols = [f'norm_ofi_{m}' for m in range(levels)]
        X = self.df[norm_ofi_cols].fillna(0).values
        pca = PCA(n_components=1)
        pca.fit(X)
        pc1 = pca.components_[0]
        pc1_norm = pc1 / np.sum(np.abs(pc1))

        self.df['ofi_integrated'] = X.dot(pc1_norm)
        return self.df[['ts_event', 'symbol', 'ofi_integrated']]

    def compute_log_returns(self) -> pd.DataFrame:
        """
        Compute log returns using mid-price changes over time.
        """
        return_all = []

        for sym, grp in self.df.groupby('symbol'):
            grp = grp.sort_values('ts_event').reset_index(drop=True)
            mid_price = (grp['bid_px_00'] + grp['ask_px_00']) / 2
            grp['log_return'] = np.log(mid_price).diff().fillna(0)
            return_all.append(grp)

        self.df = pd.concat(return_all).sort_values(['symbol', 'ts_event']).reset_index(drop=True)
        return self.df[['ts_event', 'symbol', 'log_return']]

    def compute_cross_asset_ofi(self) -> pd.DataFrame:
        """
        Compute cross-asset OFI impact using LASSO regression for each symbol's return.
        Returns a coefficient matrix with each row as target asset and columns as influencing assets.
        """
        self.compute_integrated_ofi()
        self.compute_log_returns()

        # Use pivot_table to handle potential duplicates
        pivot_ofi = self.df.pivot_table(index='ts_event', columns='symbol', values='ofi_integrated', aggfunc='mean')
        pivot_ret = self.df.pivot_table(index='ts_event', columns='symbol', values='log_return', aggfunc='mean')

        results = []

        for target in pivot_ret.columns:
            y = pivot_ret[target].fillna(0).values
            X = pivot_ofi.fillna(0).copy()

            # Skip if only one asset exists
            if target not in X.columns or len(X.columns) <= 1:
                print(f"Skipping {target}: not enough other assets for cross-impact.")
                continue

            X_target = X[target].values
            X = X.drop(columns=[target])

            scaler = StandardScaler()
            try:
                X_scaled = scaler.fit_transform(X)
                model = LassoCV(cv=5).fit(X_scaled, y)

                coefs = pd.Series(model.coef_, index=X.columns)
                coefs.name = target
                results.append(coefs)
            except ValueError as e:
                print(f"Skipping {target} due to error: {e}")

        if results:
            coef_df = pd.concat(results, axis=1).T
            return coef_df
        else:
            print("No cross-asset coefficients were computed.")
            return pd.DataFrame()
