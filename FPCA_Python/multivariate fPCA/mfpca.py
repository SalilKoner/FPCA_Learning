import numpy as np
from scipy.interpolate import BSpline


def build_open_uniform_knots(n_basis: int, degree: int) -> np.ndarray:
    if n_basis <= degree:
        raise ValueError("n_basis must be greater than degree.")
    n_interior = n_basis - degree - 1
    if n_interior > 0:
        interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    else:
        interior = np.array([])
    return np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])


def bspline_basis_matrix(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    n_basis = len(knots) - degree - 1
    basis = np.zeros((x.shape[0], n_basis))
    for j in range(n_basis):
        coef = np.zeros(n_basis)
        coef[j] = 1.0
        basis[:, j] = BSpline(knots, coef, degree, extrapolate=False)(x)
    return basis


def fit_penalized_spline_1d(
    x: np.ndarray,
    y: np.ndarray,
    n_basis: int,
    degree: int,
    penalty_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    knots = build_open_uniform_knots(n_basis=n_basis, degree=degree)
    bmat = bspline_basis_matrix(x, knots, degree)
    d2 = np.diff(np.eye(n_basis), n=2, axis=0)
    penalty = d2.T @ d2
    lhs = bmat.T @ bmat + penalty_lambda * penalty
    rhs = bmat.T @ y
    coef = np.linalg.solve(lhs, rhs)
    return knots, coef


def select_lambda_cv_1d(
    x: np.ndarray,
    y: np.ndarray,
    n_basis: int,
    degree: int,
    lambda_grid: list[float],
    n_folds: int = 5,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(123)
    idx = rng.permutation(x.shape[0])
    folds = np.array_split(idx, n_folds)

    best_lam = lambda_grid[0]
    best_mse = np.inf
    table = []
    for lam in lambda_grid:
        mse_folds = []
        for k in range(n_folds):
            val = folds[k]
            trn = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            knots, coef = fit_penalized_spline_1d(
                x=x[trn], y=y[trn], n_basis=n_basis, degree=degree, penalty_lambda=lam
            )
            pred = bspline_basis_matrix(x[val], knots, degree) @ coef
            mse_folds.append(np.mean((y[val] - pred) ** 2))
        mse = float(np.mean(mse_folds))
        table.append((lam, mse))
        if mse < best_mse:
            best_mse = mse
            best_lam = lam
    return {"lambda": best_lam, "mse": best_mse, "table": table}


def _stack_var_obs(data: dict, var_idx: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([data["t"][i][var_idx] for i in range(len(data["t"]))])
    y = np.concatenate([data["y"][i][var_idx] for i in range(len(data["y"]))])
    return x, y


def fit_multivariate_mean(
    data: dict,
    n_basis: int = 12,
    degree: int = 3,
    lambda_grid: list[float] | None = None,
    n_folds: int = 5,
) -> dict:
    if lambda_grid is None:
        lambda_grid = [0.01, 0.03, 0.1, 0.3, 1.0]

    n_vars = data["n_vars"]
    means = []
    cv_info = []
    for p in range(n_vars):
        x, y = _stack_var_obs(data, p)
        cv = select_lambda_cv_1d(
            x=x, y=y, n_basis=n_basis, degree=degree, lambda_grid=lambda_grid, n_folds=n_folds
        )
        knots, coef = fit_penalized_spline_1d(
            x=x, y=y, n_basis=n_basis, degree=degree, penalty_lambda=cv["lambda"]
        )
        means.append({"knots": knots, "coef": coef, "degree": degree})
        cv_info.append(cv)
    return {"means": means, "cv": cv_info}


def predict_mean(mean_model: dict, var_idx: int, x: np.ndarray) -> np.ndarray:
    m = mean_model["means"][var_idx]
    return bspline_basis_matrix(x, m["knots"], m["degree"]) @ m["coef"]


def compute_residuals(data: dict, mean_model: dict) -> list[list[np.ndarray]]:
    residuals = []
    for i in range(len(data["t"])):
        res_i = []
        for p in range(data["n_vars"]):
            mu_ip = predict_mean(mean_model, p, data["t"][i][p])
            res_i.append(data["y"][i][p] - mu_ip)
        residuals.append(res_i)
    return residuals


def _cross_products_pair(
    data: dict, residuals: list[list[np.ndarray]], p: int, q: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_all, t_all, z_all = [], [], []
    for i in range(len(data["t"])):
        t_p = data["t"][i][p]
        t_q = data["t"][i][q]
        r_p = residuals[i][p]
        r_q = residuals[i][q]

        if p == q:
            if len(t_p) < 2:
                continue
            a, b = np.triu_indices(len(t_p), k=1)
            s = t_p[a]
            t = t_p[b]
            z = r_p[a] * r_p[b]
            s_all.append(np.concatenate([s, t]))
            t_all.append(np.concatenate([t, s]))
            z_all.append(np.concatenate([z, z]))
        else:
            if len(t_p) == 0 or len(t_q) == 0:
                continue
            s = np.repeat(t_p, len(t_q))
            t = np.tile(t_q, len(t_p))
            z = np.repeat(r_p, len(r_q)) * np.tile(r_q, len(r_p))
            s_all.append(s)
            t_all.append(t)
            z_all.append(z)

    if not s_all:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(s_all), np.concatenate(t_all), np.concatenate(z_all)


def fit_bivariate_bspline(
    s: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    n_basis: int,
    degree: int,
    lambda_s: float,
    lambda_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    knots = build_open_uniform_knots(n_basis=n_basis, degree=degree)
    bs = bspline_basis_matrix(s, knots, degree)
    bt = bspline_basis_matrix(t, knots, degree)
    design = (bs[:, :, None] * bt[:, None, :]).reshape(s.shape[0], n_basis * n_basis)

    d2 = np.diff(np.eye(n_basis), n=2, axis=0)
    ps = d2.T @ d2
    pt = d2.T @ d2
    penalty = lambda_s * np.kron(ps, np.eye(n_basis)) + lambda_t * np.kron(
        np.eye(n_basis), pt
    )
    lhs = design.T @ design + penalty
    rhs = design.T @ z
    coef = np.linalg.solve(lhs, rhs).reshape(n_basis, n_basis)
    return knots, coef


def predict_bivariate_bspline(
    s: np.ndarray, t: np.ndarray, knots: np.ndarray, degree: int, coef: np.ndarray
) -> np.ndarray:
    n_basis = coef.shape[0]
    bs = bspline_basis_matrix(s, knots, degree)
    bt = bspline_basis_matrix(t, knots, degree)
    design = (bs[:, :, None] * bt[:, None, :]).reshape(s.shape[0], n_basis * n_basis)
    return design @ coef.ravel()


def select_lambda_cv_bivariate(
    s: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    n_basis: int,
    degree: int,
    lambda_grid: list[float],
    n_folds: int = 5,
    max_pairs: int = 50000,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(123)
    if len(z) > max_pairs:
        keep = rng.choice(len(z), size=max_pairs, replace=False)
        s, t, z = s[keep], t[keep], z[keep]

    idx = rng.permutation(len(z))
    folds = np.array_split(idx, n_folds)
    best = {"lambda_s": lambda_grid[0], "lambda_t": lambda_grid[0], "mse": np.inf}

    for lam_s in lambda_grid:
        for lam_t in lambda_grid:
            fold_mse = []
            for k in range(n_folds):
                val = folds[k]
                trn = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                knots, coef = fit_bivariate_bspline(
                    s=s[trn],
                    t=t[trn],
                    z=z[trn],
                    n_basis=n_basis,
                    degree=degree,
                    lambda_s=lam_s,
                    lambda_t=lam_t,
                )
                pred = predict_bivariate_bspline(
                    s=s[val], t=t[val], knots=knots, degree=degree, coef=coef
                )
                fold_mse.append(np.mean((z[val] - pred) ** 2))
            mse = float(np.mean(fold_mse))
            if mse < best["mse"]:
                best = {"lambda_s": lam_s, "lambda_t": lam_t, "mse": mse}
    return best


def fit_multivariate_covariance(
    data: dict,
    residuals: list[list[np.ndarray]],
    grid_size: int = 60,
    n_basis: int = 10,
    degree: int = 3,
    lambda_grid: list[float] | None = None,
) -> dict:
    if lambda_grid is None:
        lambda_grid = [0.01, 0.03, 0.1, 0.3, 1.0]

    n_vars = data["n_vars"]
    grid = np.linspace(0.0, 1.0, grid_size)

    # Global CV on pooled cross-products from all (p,q) pairs.
    s_pool, t_pool, z_pool = [], [], []
    for p in range(n_vars):
        for q in range(p, n_vars):
            s, t, z = _cross_products_pair(data, residuals, p, q)
            if len(z) == 0:
                continue
            s_pool.append(s)
            t_pool.append(t)
            z_pool.append(z)
    if not s_pool:
        raise ValueError("No valid cross-products for covariance estimation.")

    cv = select_lambda_cv_bivariate(
        s=np.concatenate(s_pool),
        t=np.concatenate(t_pool),
        z=np.concatenate(z_pool),
        n_basis=n_basis,
        degree=degree,
        lambda_grid=lambda_grid,
        n_folds=5,
        max_pairs=50000,
    )

    cov_blocks = [[None for _ in range(n_vars)] for _ in range(n_vars)]
    for p in range(n_vars):
        for q in range(p, n_vars):
            s, t, z = _cross_products_pair(data, residuals, p, q)
            if len(z) == 0:
                cov_pq = np.zeros((grid_size, grid_size))
            else:
                knots, coef = fit_bivariate_bspline(
                    s=s,
                    t=t,
                    z=z,
                    n_basis=n_basis,
                    degree=degree,
                    lambda_s=cv["lambda_s"],
                    lambda_t=cv["lambda_t"],
                )
                bg = bspline_basis_matrix(grid, knots, degree)
                cov_pq = bg @ coef @ bg.T
            cov_blocks[p][q] = cov_pq
            cov_blocks[q][p] = cov_pq.T

    return {"grid": grid, "cov_blocks": cov_blocks, "cv": cv}


def estimate_noise_variances(
    data: dict,
    residuals: list[list[np.ndarray]],
    cov_model: dict,
    n_basis: int = 10,
    degree: int = 3,
    lambda_diag: float = 0.3,
) -> np.ndarray:
    grid = cov_model["grid"]
    n_vars = data["n_vars"]
    noise_vars = np.zeros(n_vars)

    for p in range(n_vars):
        x = np.concatenate([data["t"][i][p] for i in range(len(data["t"]))])
        y = np.concatenate([residuals[i][p] ** 2 for i in range(len(residuals))])
        knots, coef = fit_penalized_spline_1d(
            x=x, y=y, n_basis=n_basis, degree=degree, penalty_lambda=lambda_diag
        )
        total_var = bspline_basis_matrix(grid, knots, degree) @ coef
        sigma2 = np.mean(total_var - np.diag(cov_model["cov_blocks"][p][p]))
        noise_vars[p] = max(float(sigma2), 1e-6)
    return noise_vars


def eigendecompose_multivariate_covariance(
    cov_model: dict, pve_threshold: float = 0.95
) -> dict:
    cov_blocks = cov_model["cov_blocks"]
    n_vars = len(cov_blocks)
    grid = cov_model["grid"]
    g = len(grid)
    delta = grid[1] - grid[0]

    c_big = np.zeros((n_vars * g, n_vars * g))
    for p in range(n_vars):
        for q in range(n_vars):
            r0, r1 = p * g, (p + 1) * g
            c0, c1 = q * g, (q + 1) * g
            c_big[r0:r1, c0:c1] = cov_blocks[p][q]
    c_big = 0.5 * (c_big + c_big.T)

    w_diag = np.full(n_vars * g, delta)
    w_sqrt = np.sqrt(w_diag)
    ws = np.diag(w_sqrt)
    a = ws @ c_big @ ws
    a = 0.5 * (a + a.T)

    eigvals, eigvecs = np.linalg.eigh(a)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    pos = eigvals > 1e-10
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]

    pve = eigvals / np.sum(eigvals)
    cum_pve = np.cumsum(pve)
    n_keep = int(np.searchsorted(cum_pve, pve_threshold) + 1)

    psi_big = eigvecs[:, :n_keep] / w_sqrt[:, None]
    # Re-orthonormalize under weighted inner product.
    qmat, _ = np.linalg.qr(ws @ psi_big)
    psi_big = qmat / w_sqrt[:, None]

    eigenfunctions = []
    for p in range(n_vars):
        r0, r1 = p * g, (p + 1) * g
        phi_p = psi_big[r0:r1, :]
        for k in range(phi_p.shape[1]):
            idx = np.argmax(np.abs(phi_p[:, k]))
            if phi_p[idx, k] < 0:
                phi_p[:, k] *= -1.0
        eigenfunctions.append(phi_p)

    return {
        "grid": grid,
        "eigenvalues": eigvals,
        "pve": pve,
        "cum_pve": cum_pve,
        "n_keep": n_keep,
        "eigenfunctions": eigenfunctions,
    }


def estimate_scores_blup(
    data: dict,
    mean_model: dict,
    eig_model: dict,
    noise_vars: np.ndarray,
    n_components: int | None = None,
) -> np.ndarray:
    if n_components is None:
        n_components = eig_model["n_keep"]

    grid = eig_model["grid"]
    lam = eig_model["eigenvalues"][:n_components]
    lam_mat = np.diag(lam)
    n_subj = len(data["t"])
    n_vars = data["n_vars"]
    scores = np.zeros((n_subj, n_components))

    for i in range(n_subj):
        y_center_list = []
        phi_rows = []
        var_ids = []
        for p in range(n_vars):
            t_ip = data["t"][i][p]
            y_ip = data["y"][i][p]
            mu_ip = predict_mean(mean_model, p, t_ip)
            y_center_list.append(y_ip - mu_ip)
            phi_p = eig_model["eigenfunctions"][p][:, :n_components]
            phi_interp = np.column_stack(
                [np.interp(t_ip, grid, phi_p[:, k]) for k in range(n_components)]
            )
            phi_rows.append(phi_interp)
            var_ids.extend([p] * len(t_ip))

        y_center = np.concatenate(y_center_list)
        phi_i = np.vstack(phi_rows)
        eps_diag = np.array([noise_vars[v] for v in var_ids])
        v_i = phi_i @ lam_mat @ phi_i.T + np.diag(eps_diag) + 1e-8 * np.eye(len(eps_diag))

        alpha = np.linalg.solve(v_i, y_center)
        scores[i, :] = lam_mat @ phi_i.T @ alpha

    return scores


def predict_functions_from_scores(
    mean_model: dict,
    eig_model: dict,
    scores: np.ndarray,
    new_timepoints: list[list[np.ndarray]],
    n_components: int | None = None,
) -> list[list[np.ndarray]]:
    """
    Predict multivariate functional trajectories at subject-specific new time points.

    Prediction formula:
      X_hat_i^(p)(t) = mu_hat_p(t) + sum_{k=1}^K xi_hat_ik * psi_hat_k^(p)(t)

    Parameters
    ----------
    mean_model : dict
        Output from fit_multivariate_mean.
    eig_model : dict
        Output from eigendecompose_multivariate_covariance.
    scores : np.ndarray
        Estimated subject scores, shape (n_subjects, K_max).
    new_timepoints : list[list[np.ndarray]]
        new_timepoints[i][p] are prediction times for subject i, variable p.
    n_components : int | None
        Number of components to use in reconstruction; default uses eig_model["n_keep"].
    """
    if n_components is None:
        n_components = eig_model["n_keep"]
    n_subjects = len(new_timepoints)
    n_vars = len(eig_model["eigenfunctions"])
    if scores.shape[0] != n_subjects:
        raise ValueError("scores row count must match number of subjects in new_timepoints.")
    if scores.shape[1] < n_components:
        raise ValueError("scores has fewer columns than requested n_components.")

    grid = eig_model["grid"]
    predictions: list[list[np.ndarray]] = []
    for i in range(n_subjects):
        pred_i: list[np.ndarray] = []
        xi_i = scores[i, :n_components]
        for p in range(n_vars):
            t_new = new_timepoints[i][p]
            mu_new = predict_mean(mean_model, p, t_new)
            phi_p = eig_model["eigenfunctions"][p][:, :n_components]
            phi_interp = np.column_stack(
                [np.interp(t_new, grid, phi_p[:, k]) for k in range(n_components)]
            )
            pred_i.append(mu_new + phi_interp @ xi_i)
        predictions.append(pred_i)
    return predictions


def fit_mfpca(
    data: dict,
    mean_n_basis: int = 12,
    cov_n_basis: int = 10,
    pve_threshold: float = 0.95,
) -> dict:
    mean_model = fit_multivariate_mean(
        data=data, n_basis=mean_n_basis, degree=3, lambda_grid=[0.01, 0.03, 0.1, 0.3, 1.0]
    )
    residuals = compute_residuals(data, mean_model)
    cov_model = fit_multivariate_covariance(
        data=data,
        residuals=residuals,
        grid_size=60,
        n_basis=cov_n_basis,
        degree=3,
        lambda_grid=[0.01, 0.03, 0.1, 0.3, 1.0],
    )
    noise_vars = estimate_noise_variances(
        data=data,
        residuals=residuals,
        cov_model=cov_model,
        n_basis=10,
        degree=3,
        lambda_diag=0.3,
    )
    eig_model = eigendecompose_multivariate_covariance(cov_model, pve_threshold=pve_threshold)
    scores_hat = estimate_scores_blup(
        data=data,
        mean_model=mean_model,
        eig_model=eig_model,
        noise_vars=noise_vars,
    )

    return {
        "mean_model": mean_model,
        "cov_model": cov_model,
        "eig_model": eig_model,
        "noise_vars_hat": noise_vars,
        "scores_hat": scores_hat,
    }
