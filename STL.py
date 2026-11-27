import os
from typing import Optional

import pandas as pd
from statsmodels.tsa.seasonal import STL


def run_stl_decomposition(
    input_excel: str,
    output_excel: Optional[str] = None,
    period: int = 12,
    robust: bool = True,
    seasonal: int = 13,
    trend: Optional[int] = None,
    low_pass: Optional[int] = None,
    value_column: str = "Discharge (m3/s)",
    date_column: str = "date",
) -> pd.DataFrame:

    if output_excel is None:
        base, _ = os.path.splitext(input_excel)
        output_excel = f"{base}_STL.xlsx"

    df = pd.read_excel(input_excel)
    df.columns = [str(c).strip() for c in df.columns]

    if date_column not in df.columns:
        raise KeyError(f"Missing date column: '{date_column}'. Found {list(df.columns)}")
    if value_column not in df.columns:
        raise KeyError(
            f"Missing runoff column: '{value_column}'. Found {list(df.columns)}"
        )

    df[date_column] = pd.to_datetime(df[date_column])
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

    series = (
        df.dropna(subset=[date_column, value_column])
          .sort_values(date_column)
          .drop_duplicates(date_column, keep="first")
          .set_index(date_column)[value_column]
          .asfreq("MS")
    )

    if series.isna().any():
        series = series.interpolate(method="time").bfill().ffill()

    if series.isna().any():
        raise ValueError("Time series still contains missing values after interpolation.")
    if len(series) < 2 * period:
        raise ValueError(
            f"Series length {len(series)} is insufficient for period={period}."
        )

    stl = STL(
        series,
        period=period,
        seasonal=seasonal,
        trend=trend,
        low_pass=low_pass,
        robust=robust,
    )
    result = stl.fit()

    result_df = pd.DataFrame(
        {
            "date": series.index,
            "Trend": result.trend.values,
            "Seasonal": result.seasonal.values,
            "Residual": result.resid.values,
        }
    )

    result_df.to_excel(output_excel, index=False)
    print(f"STL decomposition saved to: {output_excel}")
    return result_df


if __name__ == "__main__":
    input_file = r"F:shuju.xlsx"
    output_dir = r"F:\jieguo"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_excel = os.path.join(output_dir, f"{base_name}_STL.xlsx")

    df_out = run_stl_decomposition(
        input_excel=input_file,
        output_excel=output_excel,
        period=12,
        robust=True,
        seasonal=13,
        value_column="Discharge (m3/s)",
        date_column="date",
    )

    print(df_out.head())