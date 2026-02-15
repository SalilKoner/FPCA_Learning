import numpy as np
import pandas as pd

from mfpca import fit_mfpca, predict_functions_from_scores


def _validate_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns in dataframe: {missing}")


def dataframe_to_sparse_multivariate(
    df: pd.DataFrame,
    subject_col: str,
    time_col: str,
    value_cols: list[str],
) -> dict:
    """
    Convert wide asynchronous data to sparse multivariate format used by mfpca.py.

    Expected long table layout:
    - one row per subject-time record
    - one shared time column
    - one value column per functional variable
    - NA means that variable is not observed at that row/time
    """
    _validate_columns(df, [subject_col, time_col], "id/time")
    _validate_columns(df, value_cols, "value")

    work = df[[subject_col, time_col] + value_cols].copy()
    work = work.sort_values([subject_col, time_col]).reset_index(drop=True)

    subjects = work[subject_col].dropna().unique().tolist()
    n_vars = len(value_cols)

    t_all: list[list[np.ndarray]] = []
    y_all: list[list[np.ndarray]] = []

    for sid in subjects:
        sub = work.loc[work[subject_col] == sid]
        t_i: list[np.ndarray] = []
        y_i: list[np.ndarray] = []
        for col in value_cols:
            keep = sub[col].notna() & sub[time_col].notna()
            t_ip = sub.loc[keep, time_col].to_numpy(dtype=float)
            y_ip = sub.loc[keep, col].to_numpy(dtype=float)
            order = np.argsort(t_ip)
            t_i.append(t_ip[order])
            y_i.append(y_ip[order])
        t_all.append(t_i)
        y_all.append(y_i)

    return {
        "t": t_all,
        "y": y_all,
        "n_vars": n_vars,
        "subject_ids": subjects,
        "value_cols": value_cols,
    }


def build_prediction_timepoints(
    df: pd.DataFrame,
    subject_ids: list,
    subject_col: str,
    time_col: str,
    response_cols: list[str],
    n_covariates: int,
    mode: str = "matched",
) -> list[list[np.ndarray]]:
    """
    Build subject-specific new timepoints for covariate prediction.

    mode="matched":
      covariate p is predicted at times where response p is observed.
      Requires len(response_cols) == n_covariates.

    mode="union":
      all covariates are predicted at union times where any response is observed.
    """
    _validate_columns(df, [subject_col, time_col], "id/time")
    _validate_columns(df, response_cols, "response")

    work = df[[subject_col, time_col] + response_cols].copy()
    work = work.sort_values([subject_col, time_col]).reset_index(drop=True)

    if mode not in {"matched", "union"}:
        raise ValueError("mode must be 'matched' or 'union'.")
    if mode == "matched" and len(response_cols) != n_covariates:
        raise ValueError("For mode='matched', len(response_cols) must equal n_covariates.")

    new_timepoints: list[list[np.ndarray]] = []
    for sid in subject_ids:
        sub = work.loc[work[subject_col] == sid]
        t_i: list[np.ndarray] = []

        if mode == "union":
            keep_union = sub[response_cols].notna().any(axis=1) & sub[time_col].notna()
            t_union = np.sort(sub.loc[keep_union, time_col].to_numpy(dtype=float))
            for _ in range(n_covariates):
                t_i.append(t_union.copy())
        else:
            for p in range(n_covariates):
                keep = sub[response_cols[p]].notna() & sub[time_col].notna()
                t_p = np.sort(sub.loc[keep, time_col].to_numpy(dtype=float))
                t_i.append(t_p)

        new_timepoints.append(t_i)

    return new_timepoints


def predict_covariates_at_response_times(
    df: pd.DataFrame,
    subject_col: str,
    time_col: str,
    covariate_cols: list[str],
    response_cols: list[str],
    prediction_mode: str = "matched",
    pve_threshold: float = 0.95,
) -> dict:
    """
    End-to-end pipeline:
    1) Fit multivariate fPCA on covariates
    2) Predict covariates at response-observation times
    3) Return tidy long dataframe of predictions
    """
    cov_data = dataframe_to_sparse_multivariate(
        df=df,
        subject_col=subject_col,
        time_col=time_col,
        value_cols=covariate_cols,
    )
    fit = fit_mfpca(data=cov_data, pve_threshold=pve_threshold)

    new_timepoints = build_prediction_timepoints(
        df=df,
        subject_ids=cov_data["subject_ids"],
        subject_col=subject_col,
        time_col=time_col,
        response_cols=response_cols,
        n_covariates=len(covariate_cols),
        mode=prediction_mode,
    )
    pred = predict_functions_from_scores(
        mean_model=fit["mean_model"],
        eig_model=fit["eig_model"],
        scores=fit["scores_hat"],
        new_timepoints=new_timepoints,
    )

    rows = []
    for i, sid in enumerate(cov_data["subject_ids"]):
        for p, cov_name in enumerate(covariate_cols):
            t_ip = new_timepoints[i][p]
            y_ip = pred[i][p]
            for tt, yy in zip(t_ip, y_ip):
                rows.append(
                    {
                        subject_col: sid,
                        time_col: float(tt),
                        "covariate": cov_name,
                        "predicted_value": float(yy),
                    }
                )
    pred_long = pd.DataFrame(rows)

    return {
        "fit": fit,
        "new_timepoints": new_timepoints,
        "predictions": pred,
        "predictions_long": pred_long,
    }


if __name__ == "__main__":
    # Template usage on your real data.
    # Replace path and column names below.
    csv_path = "your_dataset.csv"
    try:
        df_in = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(
            "Set csv_path in real_data_pipeline.py::__main__ to run on your data. "
            "No file loaded."
        )
    else:
        out = predict_covariates_at_response_times(
            df=df_in,
            subject_col="subject_id",
            time_col="time",
            covariate_cols=["x1", "x2", "x3"],
            response_cols=["y1", "y2", "y3"],
            prediction_mode="matched",
            pve_threshold=0.95,
        )
        print("Done. Components kept:", out["fit"]["eig_model"]["n_keep"])
        print(out["predictions_long"].head(10))

