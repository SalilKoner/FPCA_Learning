from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def add_page(pdf: PdfPages, title: str, lines: list[str], fontsize: int = 12) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.95
    ax.text(0.07, y, title, fontsize=16, fontweight="bold", va="top")
    y -= 0.05

    for line in lines:
        if line == "":
            y -= 0.02
            continue
        chunks = wrap(line, width=100) if len(line) > 100 else [line]
        for ch in chunks:
            ax.text(0.07, y, ch, fontsize=fontsize, va="top")
            y -= 0.03
            if y < 0.07:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    out_path = "score_flow_model_math.pdf"
    with PdfPages(out_path) as pdf:
        add_page(
            pdf,
            "Univariate Sparse fPCA: PACE vs Flow-MAP Scores",
            [
                "This document describes the mathematical model implemented in score_estimation_flow_framework.py.",
                "",
                "1) Data model (sparse irregular functional observations)",
                "For subject i and measurement j:",
                "Y_ij = X_i(T_ij) + eps_ij,   eps_ij ~ N(0, sigma^2),",
                "with irregular times T_ij in [0,1].",
                "",
                "Latent process decomposition:",
                "X_i(t) = mu(t) + sum_{k=1}^K xi_ik * phi_k(t),",
                "where phi_k are orthonormal eigenfunctions and xi_ik are FPC scores.",
                "",
                "2) Mean estimation",
                "The mean mu(t) is estimated by penalized B-splines:",
                "mu_hat(t) = sum_{b=1}^B c_b B_b(t),",
                "with coefficients c solving:",
                "min_c ||y - B c||^2 + lambda_mu ||D2 c||^2,",
                "where B is the spline design matrix and D2 is the second-difference penalty matrix.",
                "lambda_mu is selected by K-fold cross-validation.",
            ],
        )

        add_page(
            pdf,
            "Covariance, Eigenfunctions, and PACE",
            [
                "3) Covariance estimation from off-diagonal residual products",
                "Residuals: r_ij = Y_ij - mu_hat(T_ij).",
                "For j != j': E[r_ij r_ij'] = C(T_ij, T_ij'), so diagonal noise is avoided.",
                "",
                "Bivariate penalized spline model for covariance:",
                "C_hat(s,t) = sum_{a=1}^B sum_{b=1}^B theta_ab B_a(s) B_b(t).",
                "Coefficients theta solve:",
                "min_theta ||z - X theta||^2 + lambda_s ||(D2 ⊗ I) theta||^2 + lambda_t ||(I ⊗ D2) theta||^2.",
                "lambda_s, lambda_t are selected by K-fold cross-validation.",
                "",
                "4) Eigen-decomposition",
                "Discretize C_hat on grid {t_g}. Define operator matrix A = C_hat * Delta.",
                "Solve A v_k = lambda_k v_k, keep positive eigenvalues.",
                "Choose K as smallest number with cumulative PVE >= threshold (e.g., 95%).",
                "Eigenfunctions are normalized from discrete eigenvectors to satisfy L2 orthonormality.",
                "",
                "5) PACE score estimator (Gaussian working model)",
                "For subject i with centered vector y_i = (Y_ij - mu_hat(T_ij))_j and",
                "Phi_i(j,k) = phi_hat_k(T_ij), Lambda = diag(lambda_1,...,lambda_K):",
                "xi_hat_i^PACE = Lambda Phi_i^T (Phi_i Lambda Phi_i^T + sigma^2 I)^(-1) y_i.",
            ],
        )

        add_page(
            pdf,
            "Normalizing Flow Prior and Flow-MAP Scores",
            [
                "6) Non-Gaussian score modeling with a normalizing flow",
                "PACE scores are used as initial score samples to fit a flexible density p_flow(xi).",
                "A RealNVP flow is used:",
                "z ~ N(0, I),   xi = f_theta(z),",
                "log p_flow(xi) = log p_Z(f_theta^{-1}(xi)) + log |det J_{f_theta^{-1}}(xi)|.",
                "",
                "7) Subject-specific score estimation via MAP with flow prior",
                "For each subject i, estimate xi_i by maximizing posterior objective:",
                "xi_hat_i^Flow = argmax_xi { log p(y_i | xi) + log p_flow(xi) }",
                "with Gaussian observation likelihood",
                "log p(y_i | xi) = const - (1/(2 sigma^2)) || y_i - Phi_i xi ||^2.",
                "Equivalent minimization:",
                "min_xi (1/(2 sigma^2)) || y_i - Phi_i xi ||^2 - log p_flow(xi).",
                "The code solves this using gradient-based optimization (Adam) initialized at PACE scores.",
                "",
                "8) Prediction and comparison",
                "Predicted trajectory at time t:",
                "X_hat_i(t) = mu_hat(t) + sum_{k=1}^K xi_hat_ik phi_hat_k(t),",
                "computed using either xi_hat_i^PACE or xi_hat_i^Flow.",
                "Performance comparison uses RMSE on observed points and (in simulations) latent signal RMSE.",
            ],
        )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

