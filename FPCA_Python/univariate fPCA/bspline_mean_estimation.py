import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

from functional_data_generator import generate_functional_data, mean_function


def build_open_uniform_knots(n_basis: int, degree: int) -> np.ndarray:
    """Build an open-uniform knot vector on [0, 1]."""
    if n_basis <= degree:
        raise ValueError("n_basis must be greater than degree.")

    n_interior = n_basis - degree - 1
    if n_interior > 0:
        interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    else:
        interior = np.array([])

    knots = np.concatenate(
        [
            np.zeros(degree + 1),
            interior,
            np.ones(degree + 1),
        ]
    )
    return knots


def bspline_basis_matrix(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate all B-spline basis functions at x."""
    n_basis = len(knots) - degree - 1
    basis = np.zeros((x.shape[0], n_basis))

    for j in range(n_basis):
        coeff = np.zeros(n_basis)
        coeff[j] = 1.0
        spline_j = BSpline(knots, coeff, degree, extrapolate=False)
        basis[:, j] = spline_j(x)

    return basis


def pooled_observations(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pool irregular observations from all curves."""
    x = np.concatenate(data["t"])
    y = np.concatenate(data["y"])
    return x, y


def fit_mean_bspline_coef(
    x: np.ndarray,
    y: np.ndarray,
    n_basis: int,
    degree: int,
    penalty_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit penalized B-spline coefficients for pooled observations."""
    knots = build_open_uniform_knots(n_basis=n_basis, degree=degree)
    bmat = bspline_basis_matrix(x, knots, degree)

    if n_basis >= 3:
        dmat = np.diff(np.eye(n_basis), n=2, axis=0)
        penalty = dmat.T @ dmat
    else:
        penalty = np.zeros((n_basis, n_basis))

    lhs = bmat.T @ bmat + penalty_lambda * penalty
    rhs = bmat.T @ y
    coef = np.linalg.solve(lhs, rhs)
    return knots, coef


def select_mean_penalty_cv(
    data: dict,
    n_basis: int,
    degree: int,
    lambda_grid: list[float],
    n_folds: int = 5,
    random_state: int = 123,
    max_points: int = 20000,
) -> dict:
    """K-fold CV to select the mean-smoother penalty lambda."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must be non-empty.")

    x, y = pooled_observations(data)
    rng = np.random.default_rng(random_state)

    if x.shape[0] > max_points:
        keep = rng.choice(x.shape[0], size=max_points, replace=False)
        x = x[keep]
        y = y[keep]

    perm = rng.permutation(x.shape[0])
    folds = np.array_split(perm, n_folds)

    best_lambda = lambda_grid[0]
    best_mse = np.inf
    cv_table = []

    for lam in lambda_grid:
        fold_mse = []
        for k in range(n_folds):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])

            knots, coef = fit_mean_bspline_coef(
                x=x[train_idx],
                y=y[train_idx],
                n_basis=n_basis,
                degree=degree,
                penalty_lambda=lam,
            )
            bval = bspline_basis_matrix(x[val_idx], knots, degree)
            pred = bval @ coef
            fold_mse.append(float(np.mean((y[val_idx] - pred) ** 2)))

        mse = float(np.mean(fold_mse))
        cv_table.append((lam, mse))
        if mse < best_mse:
            best_mse = mse
            best_lambda = lam

    return {
        "penalty_lambda": best_lambda,
        "cv_mse": best_mse,
        "cv_table": cv_table,
        "n_points_used": x.shape[0],
    }


def estimate_mean_bspline(
    data: dict,
    n_basis: int = 12,
    degree: int = 3,
    penalty_lambda: float = 0.1,
) -> dict:
    """
    Penalized B-spline mean estimator on pooled data.
    Solves: min_c ||y - Bc||^2 + lambda ||D c||^2
    where D is the second-difference matrix.
    """
    if penalty_lambda < 0:
        raise ValueError("penalty_lambda must be nonnegative.")

    x, y = pooled_observations(data)
    knots, coef = fit_mean_bspline_coef(
        x=x,
        y=y,
        n_basis=n_basis,
        degree=degree,
        penalty_lambda=penalty_lambda,
    )

    return {
        "knots": knots,
        "degree": degree,
        "coef": coef,
        "n_basis": n_basis,
        "penalty_lambda": penalty_lambda,
    }


def predict_mean(model: dict, x: np.ndarray) -> np.ndarray:
    """Evaluate fitted mean function on points x."""
    bmat = bspline_basis_matrix(x, model["knots"], model["degree"])
    return bmat @ model["coef"]


if __name__ == "__main__":
    data = generate_functional_data(
        n_curves=150,
        min_points=15,
        max_points=45,
        lambdas=(1.0, 0.6),
        noise_variance=0.06,
        random_state=5,
    )

    lambda_grid = [0.01, 0.03, 0.1, 0.3, 1.0]
    cv = select_mean_penalty_cv(
        data=data,
        n_basis=12,
        degree=3,
        lambda_grid=lambda_grid,
        n_folds=5,
        random_state=123,
        max_points=20000,
    )
    print(
        "Selected mean penalty by CV: "
        f"lambda={cv['penalty_lambda']} "
        f"(MSE={cv['cv_mse']:.6f}, points={cv['n_points_used']})"
    )

    model = estimate_mean_bspline(
        data=data,
        n_basis=12,
        degree=3,
        penalty_lambda=cv["penalty_lambda"],
    )

    grid = np.linspace(0.0, 1.0, 300)
    mean_hat = predict_mean(model, grid)
    true_mean = mean_function(grid)

    plt.figure(figsize=(9, 5))
    for i in range(5):
        plt.scatter(data["t"][i], data["y"][i], s=10, alpha=0.35, color="gray")
    plt.plot(grid, true_mean, linewidth=2.2, label="True mean")
    plt.plot(grid, mean_hat, linewidth=2.2, linestyle="--", label="B-spline mean estimate")
    plt.title("Mean Function Estimation via Penalized B-splines")
    plt.xlabel("t")
    plt.ylabel("X(t)")
    plt.xlim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mean_bspline_estimate.png", dpi=180)
    if "agg" not in plt.get_backend().lower():
        plt.show()
