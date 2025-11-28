# run.py
import multiprocessing
#import numpy as np
#import pandas as pd

from swaptions_aid import *

OUTPUT_PATH = "opt_data/sample_Simulated_Swaption_Price_imputed.xlsx"


def impute_column(col_index: int):
    """
    Worker: take column index, return (col_index, imputed_values_array).
    """
    series = select_timeseries(df, col_index)
    imputed = impute_nans_reservoir_stepwise(
        series,
        w=3,
        mem_size=3,
        washout_length=3,
        max_train_windows=10,
    )
    return col_index, imputed


if __name__ == "__main__":
    # choose up to 20 numeric columns (or first 20 columns blindly)
    n_cols = min(10, df.shape[1])
    col_indices = list(range(n_cols))

    with multiprocessing.Pool(processes=n_cols) as pool:
        results = pool.map(impute_column, col_indices)

    # rebuild DataFrame with imputed columns
    df_imputed = df.copy()
    for col_index, imputed_vals in results:
        col_name = df.columns[col_index]
        df_imputed[col_name] = imputed_vals

    # write to file
    df_imputed.to_excel(OUTPUT_PATH, index=False)
    print(f"Imputed data written to {OUTPUT_PATH}")