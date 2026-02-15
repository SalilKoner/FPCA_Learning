import matplotlib.pyplot as plt
import numpy as np

from bspline_mean_estimation import (
    bspline_basis_matrix,
    build_open_uniform_knots,
    estimate_mean_bspline,
    fit_mean_bspline_coef,
    predict_mean,
    select_mean_penalty_cv,
)
from functional_data_generator import generate_functional_data, mean_function


def compute_residuals(data: dict, mean_model: dict) -> list[np.ndarray]:
    """Residuals r_ij = y_ij - mu_hat(t_ij) for each curve."""
    residuals = []
    for t_i, y_i in zip(data["t"], data["y"]):
        mu_i = predict_mean(mean_model, t_i)
        residuals.append(y_i - mu_i)
    return residuals


def off_diagonal_cross_products(
    times: list[np.ndarray], residuals: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build off-diagonal cross-products r_ij * r_ij' for j != j'.
    Adds both (t_ij, t_ij') and (t_ij', t_ij) to enforce symmetry.
    """
    s_all = []
    t_all = []
    z_all = []

    for t_i, r_i in zip(times, residuals):
        m_i = len(t_i)
        if m_i < 2:
            continue

        j_idx, k_idx = np.triu_indices(m_i, k=1)  # only j < j', excludes diagonal
        s = t_i[j_idx]
        t = t_i[k_idx]
        z = r_i[j_idx] * r_i[k_idx]

        s_all.append(np.concatenate([s, t]))
        t_all.append(np.concatenate([t, s]))
        z_all.append(np.concatenate([z, z]))

    if not s_all:
        raise ValueError("No off-diagonal pairs found. Increase points per curve.")

    return np.concatenate(s_all), np.concatenate(t_all), np.concatenate(z_all)


def fit_bivariate_bspline_coef(
    s: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    n_basis: int,
    degree: int,
    penalty_lambda_s: float,
    penalty_lambda_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit penalized tensor-product B-spline coefficients."""
    knots = build_open_uniform_knots(n_basis=n_basis, degree=degree)
    bs = bspline_basis_matrix(s, knots, degree)
    bt = bspline_basis_matrix(t, knots, degree)
    design = (bs[:, :, None] * bt[:, None, :]).reshape(s.shape[0], n_basis * n_basis)

    d2 = np.diff(np.eye(n_basis), n=2, axis=0)
    ps = d2.T @ d2
    pt = d2.T @ d2
    penalty = penalty_lambda_s * np.kron(ps, np.eye(n_basis))
    penalty += penalty_lambda_t * np.kron(np.eye(n_basis), pt)

    lhs = design.T @ design + penalty
    rhs = design.T @ z
    coef_vec = np.linalg.solve(lhs, rhs)
    return knots, coef_vec.reshape(n_basis, n_basis)


def predict_bivariate_bspline(
    s: np.ndarray, t: np.ndarray, knots: np.ndarray, degree: int, coef: np.ndarray
) -> np.ndarray:
    """Predict covariance values at arbitrary (s, t) pairs."""
    n_basis = coef.shape[0]
    bs = bspline_basis_matrix(s, knots, degree)
    bt = bspline_basis_matrix(t, knots, degree)
    design = (bs[:, :, None] * bt[:, None, :]).reshape(s.shape[0], n_basis * n_basis)
    return design @ coef.ravel()


def select_penalties_cv(
    s: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    n_basis: int,
    degree: int,
    lambda_grid: list[float],
    n_folds: int = 5,
    random_state: int = 123,
    max_pairs: int = 40000,
) -> dict:
    """K-fold CV for (lambda_s, lambda_t) based on prediction MSE."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must be non-empty.")

    rng = np.random.default_rng(random_state)
    n_total = z.shape[0]
    if n_total > max_pairs:
        keep = rng.choice(n_total, size=max_pairs, replace=False)
        s_cv, t_cv, z_cv = s[keep], t[keep], z[keep]
    else:
        s_cv, t_cv, z_cv = s, t, z

    n = z_cv.shape[0]
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    best_lambda_s = lambda_grid[0]
    best_lambda_t = lambda_grid[0]
    best_mse = np.inf
    cv_table = []

    for lam_s in lambda_grid:
        for lam_t in lambda_grid:
            fold_mse = []
            for k in range(n_folds):
                val_idx = folds[k]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])

                knots, coef = fit_bivariate_bspline_coef(
                    s=s_cv[train_idx],
                    t=t_cv[train_idx],
                    z=z_cv[train_idx],
                    n_basis=n_basis,
                    degree=degree,
                    penalty_lambda_s=lam_s,
                    penalty_lambda_t=lam_t,
                )
                pred = predict_bivariate_bspline(
                    s=s_cv[val_idx], t=t_cv[val_idx], knots=knots, degree=degree, coef=coef
                )
                fold_mse.append(float(np.mean((z_cv[val_idx] - pred) ** 2)))

            mse = float(np.mean(fold_mse))
            cv_table.append((lam_s, lam_t, mse))
            if mse < best_mse:
                best_mse = mse
                best_lambda_s = lam_s
                best_lambda_t = lam_t

    return {
        "lambda_s": best_lambda_s,
        "lambda_t": best_lambda_t,
        "cv_mse": best_mse,
        "cv_table": cv_table,
        "n_pairs_used": n,
    }


def estimate_covariance_bivariate_bspline(
    s: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    n_basis: int = 10,
    degree: int = 3,
    penalty_lambda_s: float = 0.2,
    penalty_lambda_t: float = 0.2,
    grid_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate covariance by penalized bivariate B-splines on off-diagonal cross-products.
    Solves:
      min_c ||z - B c||^2 + lambda_s ||(D2 \u2297 I)c||^2 + lambda_t ||(I \u2297 D2)c||^2
    where B is the tensor-product B-spline design matrix.
    """
    if penalty_lambda_s < 0 or penalty_lambda_t < 0:
        raise ValueError("penalty_lambda_s and penalty_lambda_t must be nonnegative.")

    knots, coef = fit_bivariate_bspline_coef(
        s=s,
        t=t,
        z=z,
        n_basis=n_basis,
        degree=degree,
        penalty_lambda_s=penalty_lambda_s,
        penalty_lambda_t=penalty_lambda_t,
    )

    grid = np.linspace(0.0, 1.0, grid_size)
    bg = bspline_basis_matrix(grid, knots, degree)
    cov_hat = bg @ coef @ bg.T
    cov_hat = 0.5 * (cov_hat + cov_hat.T)

    return grid, cov_hat


def fpca_from_covariance(
    grid: np.ndarray,
    cov_hat: np.ndarray,
    pve_threshold: float = 0.95,
) -> dict:
    """Eigen-decomposition of covariance operator on a uniform grid."""
    if not (0.0 < pve_threshold <= 1.0):
        raise ValueError("pve_threshold must be in (0, 1].")

    delta = grid[1] - grid[0]
    operator_mat = cov_hat * delta
    operator_mat = 0.5 * (operator_mat + operator_mat.T)

    eigvals, eigvecs = np.linalg.eigh(operator_mat)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = eigvals > 1e-10
    eigvals = eigvals[positive]
    eigvecs = eigvecs[:, positive]

    if eigvals.size == 0:
        raise ValueError("No positive eigenvalues found from covariance estimate.")

    pve = eigvals / np.sum(eigvals)
    cum_pve = np.cumsum(pve)
    n_keep = int(np.searchsorted(cum_pve, pve_threshold) + 1)

    # Convert discrete eigenvectors to approximately L2-normalized eigenfunctions.
    eigenfunctions = eigvecs[:, :n_keep] / np.sqrt(delta)

    # Re-orthonormalize in the discrete L2 inner product.
    q, _ = np.linalg.qr(np.sqrt(delta) * eigenfunctions)
    eigenfunctions = q / np.sqrt(delta)

    # Fix sign for stable plotting (deterministic orientation).
    for k in range(eigenfunctions.shape[1]):
        pivot = np.argmax(np.abs(eigenfunctions[:, k]))
        if eigenfunctions[pivot, k] < 0:
            eigenfunctions[:, k] *= -1.0

    return {
        "eigenvalues": eigvals,
        "pve": pve,
        "cum_pve": cum_pve,
        "n_keep": n_keep,gg
        "eigenfunctions": eigenfunctions,
    }


def estimate_noise_variance(
    times: list[np.ndarray],
    residuals: list[np.ndarray],
    grid: np.ndarray,
    cov_hat: np.ndarray,
    n_basis: int = 12,
    degree: int = 3,
    penalty_lambda: float = 0.3,
) -> dict:
    """
    Estimate measurement error variance sigma^2.
    Smooth E[r(t)^2] on the diagonal, then subtract covariance diagonal.
    """
    x_diag = np.concatenate(times)
    y_diag = np.concatenate([r_i**2 for r_i in residuals])

    knots, coef = fit_mean_bspline_coef(
        x=x_diag,
        y=y_diag,
        n_basis=n_basis,
        degree=degree,
        penalty_lambda=penalty_lambda,
    )
    bgrid = bspline_basis_matrix(grid, knots, degree)
    total_var_hat = bgrid @ coef
    total_var_hat = np.maximum(total_var_hat, 1e-8)

    cov_diag_hat = np.diag(cov_hat)
    sigma2 = float(np.mean(total_var_hat - cov_diag_hat))
    sigma2 = max(sigma2, 1e-6)

    return {
        "sigma2": sigma2,
        "total_var_hat": total_var_hat,
        "cov_diag_hat": cov_diag_hat,
    }


def estimate_pace_scores(
    data: dict,
    mean_model: dict,
    grid: np.ndarray,
    fpca: dict,
    sigma2: float,
    n_components: int | None = None,
) -> np.ndarray:
    """
    PACE/BLUP score estimator under working Gaussian model:
      xi_hat_i = Lambda Phi_i^T (Phi_i Lambda Phi_i^T + sigma^2 I)^(-1) (y_i - mu_i)
    """
    if n_components is None:
        n_components = fpca["n_keep"]

    lam = fpca["eigenvalues"][:n_components]
    phi_grid = fpca["eigenfunctions"][:, :n_components]
    lambda_mat = np.diag(lam)

    n_subjects = len(data["t"])
    scores_hat = np.zeros((n_subjects, n_components))

    for i, (t_i, y_i) in enumerate(zip(data["t"], data["y"])):
        mu_i = predict_mean(mean_model, t_i)
        y_center = y_i - mu_i

        phi_i = np.column_stack(
            [np.interp(t_i, grid, phi_grid[:, k]) for k in range(n_components)]
        )

        cov_i = phi_i @ lambda_mat @ phi_i.T + sigma2 * np.eye(len(t_i))
        cov_i += 1e-8 * np.eye(len(t_i))

        weights = np.linalg.solve(cov_i, y_center)
        scores_hat[i, :] = lambda_mat @ phi_i.T @ weights

    return scores_hat


if __name__ == "__main__":
    data = generate_functional_data(
        n_curves=1200,
        min_points=8,
        max_points=12,
        lambdas=(1.0, 0.6),
        noise_variance=0.05,
        random_state=12,
    )

    mean_lambda_grid = [0.01, 0.03, 0.1, 0.3, 1.0]
    mean_cv = select_mean_penalty_cv(
        data=data,
        n_basis=12,
        degree=3,
        lambda_grid=mean_lambda_grid,
        n_folds=5,
        random_state=123,
        max_points=20000,
    )
    print(
        "Selected mean penalty by CV: "
        f"lambda={mean_cv['penalty_lambda']} "
        f"(MSE={mean_cv['cv_mse']:.6f}, points={mean_cv['n_points_used']})"
    )

    mean_model = estimate_mean_bspline(
        data=data,
        n_basis=12,
        degree=3,
        penalty_lambda=mean_cv["penalty_lambda"],
    )

    residuals = compute_residuals(data, mean_model)
    s_all, t_all, z_all = off_diagonal_cross_products(data["t"], residuals)

    lambda_grid = [0.01, 0.03, 0.1, 0.3, 1.0]
    cv = select_penalties_cv(
        s=s_all,
        t=t_all,
        z=z_all,
        n_basis=10,
        degree=3,
        lambda_grid=lambda_grid,
        n_folds=5,
        random_state=123,
        max_pairs=40000,
    )
    print(
        "Selected roughness penalties by CV: "
        f"lambda_s={cv['lambda_s']}, lambda_t={cv['lambda_t']} "
        f"(MSE={cv['cv_mse']:.6f}, pairs={cv['n_pairs_used']})"
    )

    grid, cov_hat = estimate_covariance_bivariate_bspline(
        s_all,
        t_all,
        z_all,
        n_basis=10,
        degree=3,
        penalty_lambda_s=cv["lambda_s"],
        penalty_lambda_t=cv["lambda_t"],
        grid_size=50,
    )

    fpca = fpca_from_covariance(grid, cov_hat, pve_threshold=0.95)
    noise_fit = estimate_noise_variance(
        times=data["t"],
        residuals=residuals,
        grid=grid,
        cov_hat=cov_hat,
        n_basis=12,
        degree=3,
        penalty_lambda=0.3,
    )
    scores_hat = estimate_pace_scores(
        data=data,
        mean_model=mean_model,
        grid=grid,
        fpca=fpca,
        sigma2=noise_fit["sigma2"],
    )

    print(f"Number of components for 95% PVE: {fpca['n_keep']}")
    print("First 5 eigenvalues:", np.round(fpca["eigenvalues"][:5], 5))
    print("First 5 cumulative PVE:", np.round(fpca["cum_pve"][:5], 5))
    print(f"Estimated noise variance (sigma^2): {noise_fit['sigma2']:.6f}")
    print("First 5 estimated score vectors:")
    print(np.round(scores_hat[:5], 4))
    score_cov = np.cov(scores_hat, rowvar=False)
    print("Covariance matrix of estimated scores:")
    print(np.round(score_cov, 6))
    if "scores" in data and data["scores"].shape[1] >= 2 and scores_hat.shape[1] >= 2:
        corr1 = np.corrcoef(data["scores"][:, 0], scores_hat[:, 0])[0, 1]
        corr2 = np.corrcoef(data["scores"][:, 1], scores_hat[:, 1])[0, 1]
        print(f"Score correlation (true vs estimated): comp1={corr1:.3f}, comp2={corr2:.3f}")

    g1, g2 = np.meshgrid(grid, grid, indexing="xy")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    mean_hat = predict_mean(mean_model, grid)
    mean_true = mean_function(grid)
    for i in range(4):
        axes[0].scatter(data["t"][i], data["y"][i], s=10, alpha=0.25, color="gray")
    axes[0].plot(grid, mean_true, linewidth=2.2, label="True mean")
    axes[0].plot(grid, mean_hat, linewidth=2.2, linestyle="--", label="Estimated mean")
    axes[0].set_title("Mean Function")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("X(t)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    im = axes[1].contourf(g1, g2, cov_hat, levels=20, cmap="viridis")
    axes[1].set_title("Smoothed Covariance Estimate")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("t")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    for k in range(fpca["n_keep"]):
        axes[2].plot(grid, fpca["eigenfunctions"][:, k], linewidth=2, label=f"phi_{k+1}")
    axes[2].set_title(f"Estimated Eigenfunctions (95% PVE -> {fpca['n_keep']} comps)")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("phi(t)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("covariance_fpca_estimate.png", dpi=180)
    if "agg" not in plt.get_backend().lower():
        plt.show()
