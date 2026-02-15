import matplotlib.pyplot as plt
import numpy as np

from mfpca import fit_mfpca, predict_functions_from_scores, predict_mean
from simulation import simulate_multivariate_sparse_data


def main() -> None:
    data = simulate_multivariate_sparse_data(
        n_subjects=800,
        min_points=15,
        max_points=20,
        lambdas=(6.0, 3.0, 1.5),
        noise_vars=(0.01, 0.01, 0.01),
        random_state=11,
    )

    fit = fit_mfpca(data, mean_n_basis=12, cov_n_basis=10, pve_threshold=0.9)

    eig = fit["eig_model"]
    mean_model = fit["mean_model"]
    scores_hat = fit["scores_hat"]
    grid = eig["grid"]

    print("Mean CV lambdas by variable:", [round(m["lambda"], 4) for m in mean_model["cv"]])
    print("Covariance CV lambda_s:", fit["cov_model"]["cv"]["lambda_s"])
    print("Covariance CV lambda_t:", fit["cov_model"]["cv"]["lambda_t"])
    print("Estimated noise variances:", np.round(fit["noise_vars_hat"], 6))
    print(f"Components for 95% PVE: {eig['n_keep']}")
    print("First 5 eigenvalues:", np.round(eig["eigenvalues"][:5], 6))
    print("Score covariance matrix:")
    print(np.round(np.cov(scores_hat, rowvar=False), 6))
    delta = grid[1] - grid[0]
    mv_norms = []
    for k in range(eig["n_keep"]):
        sq_sum = 0.0
        for p in range(data["n_vars"]):
            sq_sum += float(np.sum(eig["eigenfunctions"][p][:, k] ** 2) * delta)
        mv_norms.append(np.sqrt(sq_sum))
    print("Multivariate L2 norms of estimated eigenfunctions:")
    print(np.round(np.array(mv_norms), 6))
    inner = np.zeros((eig["n_keep"], eig["n_keep"]))
    for k in range(eig["n_keep"]):
        for l in range(eig["n_keep"]):
            val = 0.0
            for p in range(data["n_vars"]):
                val += float(
                    np.sum(
                        eig["eigenfunctions"][p][:, k] * eig["eigenfunctions"][p][:, l]
                    )
                    * delta
                )
            inner[k, l] = val
    print("Multivariate inner-product matrix of estimated eigenfunctions:")
    print(np.round(inner, 6))

    if "scores_true" in data and scores_hat.shape[1] >= 2:
        corr1 = np.corrcoef(data["scores_true"][:, 0], scores_hat[:, 0])[0, 1]
        corr2 = np.corrcoef(data["scores_true"][:, 1], scores_hat[:, 1])[0, 1]
        print(f"Score correlation (comp1, comp2): ({corr1:.3f}, {corr2:.3f})")

    # Subject-specific prediction at irregular new time points (different per subject).
    rng = np.random.default_rng(2026)
    new_timepoints = []
    for i in range(len(data["t"])):
        times_i = []
        for p in range(data["n_vars"]):
            m_new = int(rng.integers(5, 12))
            t_new = np.sort(rng.uniform(0.0, 1.0, size=m_new))
            times_i.append(t_new)
        new_timepoints.append(times_i)
    pred_new = predict_functions_from_scores(
        mean_model=mean_model,
        eig_model=eig,
        scores=scores_hat,
        new_timepoints=new_timepoints,
    )
    print("Prediction example (subject 1, variable 1):")
    print("new t:", np.round(new_timepoints[0][0], 4))
    print("pred :", np.round(pred_new[0][0], 4))

    # Plot one subject: observed vs predicted points.
    subject_idx = 0
    var_idx = 0
    plt.figure(figsize=(8, 4.5))
    plt.scatter(
        data["t"][subject_idx][var_idx],
        data["y"][subject_idx][var_idx],
        s=40,
        alpha=0.8,
        label="Observed points",
    )
    plt.scatter(
        new_timepoints[subject_idx][var_idx],
        pred_new[subject_idx][var_idx],
        s=36,
        marker="x",
        linewidths=2.0,
        label="Predicted points",
    )
    plt.title("One Subject: Observed vs Predicted (Variable 1)")
    plt.xlabel("t")
    plt.ylabel("X(t)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("subject_prediction_vs_observed.png", dpi=180)

    fig, axes = plt.subplots(2, data["n_vars"], figsize=(5 * data["n_vars"], 8))
    for p in range(data["n_vars"]):
        ax = axes[0, p]
        for i in range(4):
            ax.scatter(data["t"][i][p], data["y"][i][p], s=12, alpha=0.25, color="gray")
        mu_hat = predict_mean(mean_model, p, grid)
        mu_true = data["mean_functions_true"][p](grid)
        ax.plot(grid, mu_true, linewidth=2.2, label="True mean")
        ax.plot(grid, mu_hat, linestyle="--", linewidth=2.2, label="Estimated mean")
        ax.set_title(f"Variable {p + 1} Mean")
        ax.grid(alpha=0.25)
        ax.legend()

    for p in range(data["n_vars"]):
        ax = axes[1, p]
        phi_p = eig["eigenfunctions"][p]
        for k in range(min(3, eig["n_keep"])):
            ax.plot(grid, phi_p[:, k], linewidth=2.0, label=f"phi_{k + 1}")
        ax.set_title(f"Variable {p + 1} Eigenfunctions")
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout()
    plt.savefig("multivariate_fpca_demo.png", dpi=180)
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
