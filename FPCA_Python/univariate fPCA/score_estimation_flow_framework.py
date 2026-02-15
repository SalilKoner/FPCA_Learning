import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from covariance_fpca_estimation import (
    compute_residuals,
    estimate_covariance_bivariate_bspline,
    estimate_noise_variance,
    estimate_pace_scores,
    fpca_from_covariance,
    off_diagonal_cross_products,
    select_penalties_cv,
)
from bspline_mean_estimation import estimate_mean_bspline, predict_mean, select_mean_penalty_cv
from functional_data_generator import generate_functional_data


def build_univariate_fpca_model(data: dict, pve_threshold: float = 0.95) -> dict:
    mean_cv = select_mean_penalty_cv(
        data=data,
        n_basis=12,
        degree=3,
        lambda_grid=[0.01, 0.03, 0.1, 0.3, 1.0],
        n_folds=5,
        random_state=123,
    )
    mean_model = estimate_mean_bspline(
        data=data, n_basis=12, degree=3, penalty_lambda=mean_cv["penalty_lambda"]
    )

    residuals = compute_residuals(data, mean_model)
    s_all, t_all, z_all = off_diagonal_cross_products(data["t"], residuals)
    cov_cv = select_penalties_cv(
        s=s_all,
        t=t_all,
        z=z_all,
        n_basis=10,
        degree=3,
        lambda_grid=[0.01, 0.03, 0.1, 0.3, 1.0],
        n_folds=5,
        random_state=123,
        max_pairs=40000,
    )
    grid, cov_hat = estimate_covariance_bivariate_bspline(
        s_all,
        t_all,
        z_all,
        n_basis=10,
        degree=3,
        penalty_lambda_s=cov_cv["lambda_s"],
        penalty_lambda_t=cov_cv["lambda_t"],
        grid_size=60,
    )
    fpca = fpca_from_covariance(grid, cov_hat, pve_threshold=pve_threshold)
    noise_fit = estimate_noise_variance(
        times=data["t"],
        residuals=residuals,
        grid=grid,
        cov_hat=cov_hat,
        n_basis=12,
        degree=3,
        penalty_lambda=0.3,
    )
    scores_pace = estimate_pace_scores(
        data=data, mean_model=mean_model, grid=grid, fpca=fpca, sigma2=noise_fit["sigma2"]
    )
    return {
        "mean_model": mean_model,
        "grid": grid,
        "cov_hat": cov_hat,
        "fpca": fpca,
        "sigma2": noise_fit["sigma2"],
        "scores_pace": scores_pace,
    }


def design_matrix_for_subject(t_i: np.ndarray, grid: np.ndarray, fpca: dict, n_comp: int) -> np.ndarray:
    phi_grid = fpca["eigenfunctions"][:, :n_comp]
    return np.column_stack([np.interp(t_i, grid, phi_grid[:, k]) for k in range(n_comp)])


def reconstruct_at_times(
    t_i: np.ndarray, score_i: np.ndarray, mean_model: dict, grid: np.ndarray, fpca: dict
) -> np.ndarray:
    mu_i = predict_mean(mean_model, t_i)
    phi_i = design_matrix_for_subject(t_i, grid, fpca, n_comp=len(score_i))
    return mu_i + phi_i @ score_i


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        st = self.net(x_masked)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        logdet = torch.sum(s, dim=-1)
        return y, logdet

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_masked = y * self.mask
        st = self.net(y_masked)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        x = y_masked + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        logdet = -torch.sum(s, dim=-1)
        return x, logdet


class RealNVP(nn.Module):
    def __init__(self, dim: int, n_layers: int = 6, hidden: int = 64):
        super().__init__()
        self.dim = dim
        layers = []
        for l in range(n_layers):
            mask_np = np.array([(j + l) % 2 for j in range(dim)], dtype=np.float32)
            if np.all(mask_np == 0) or np.all(mask_np == 1):
                mask_np[::2] = 1.0
            mask = torch.from_numpy(mask_np)
            layers.append(AffineCoupling(dim=dim, mask=mask, hidden=hidden))
        self.layers = nn.ModuleList(layers)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        logdet = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            x, ld = layer.forward(x)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        logdet = torch.zeros(x.shape[0], device=x.device)
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            logdet = logdet + ld
        return z, logdet

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, logdet = self.inverse(x)
        log_base = -0.5 * torch.sum(z**2, dim=-1) - 0.5 * self.dim * math.log(2.0 * math.pi)
        return log_base + logdet


@dataclass
class FlowPriorModel:
    flow: RealNVP
    mean: np.ndarray
    std: np.ndarray
    device: str

    def log_prob_score(self, score: torch.Tensor) -> torch.Tensor:
        mean_t = torch.tensor(self.mean, dtype=torch.float32, device=self.device)
        std_t = torch.tensor(self.std, dtype=torch.float32, device=self.device)
        z = (score - mean_t) / std_t
        log_prob_z = self.flow.log_prob(z.unsqueeze(0))[0]
        log_jac = -torch.sum(torch.log(std_t))
        return log_prob_z + log_jac


def fit_flow_prior_on_scores(
    scores: np.ndarray,
    n_layers: int = 6,
    hidden: int = 64,
    epochs: int = 1200,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 123,
) -> FlowPriorModel:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = scores.mean(axis=0)
    std = scores.std(axis=0) + 1e-6
    z = (scores - mean) / std
    x_train = torch.tensor(z, dtype=torch.float32, device=device)

    flow = RealNVP(dim=scores.shape[1], n_layers=n_layers, hidden=hidden).to(device)
    opt = optim.Adam(flow.parameters(), lr=lr)

    n = x_train.shape[0]
    idx = np.arange(n)
    for _ in range(epochs):
        rng.shuffle(idx)
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            xb = x_train[batch_idx]
            loss = -flow.log_prob(xb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    return FlowPriorModel(flow=flow, mean=mean, std=std, device=device)


def estimate_scores_flow_map(
    data: dict,
    mean_model: dict,
    grid: np.ndarray,
    fpca: dict,
    sigma2: float,
    flow_prior: FlowPriorModel,
    init_scores: np.ndarray,
    n_components: int | None = None,
    steps: int = 600,
    lr: float = 0.03,
) -> np.ndarray:
    if n_components is None:
        n_components = fpca["n_keep"]

    mean_t = torch.tensor(flow_prior.mean[:n_components], dtype=torch.float32, device=flow_prior.device)
    std_t = torch.tensor(flow_prior.std[:n_components], dtype=torch.float32, device=flow_prior.device)

    scores_hat = np.zeros((len(data["t"]), n_components))
    for i, (t_i, y_i) in enumerate(zip(data["t"], data["y"])):
        mu_i = predict_mean(mean_model, t_i)
        phi_i = design_matrix_for_subject(t_i, grid, fpca, n_comp=n_components)

        y_t = torch.tensor(y_i - mu_i, dtype=torch.float32, device=flow_prior.device)
        phi_t = torch.tensor(phi_i, dtype=torch.float32, device=flow_prior.device)
        x0 = torch.tensor(
            init_scores[i, :n_components], dtype=torch.float32, device=flow_prior.device
        )
        x = nn.Parameter(x0.clone())
        opt = optim.Adam([x], lr=lr)

        for _ in range(steps):
            pred = phi_t @ x
            ll = -0.5 / sigma2 * torch.sum((y_t - pred) ** 2)
            z = (x - mean_t) / std_t
            lp = flow_prior.flow.log_prob(z.unsqueeze(0))[0] - torch.sum(torch.log(std_t))
            loss = -(ll + lp)
            opt.zero_grad()
            loss.backward()
            opt.step()

        scores_hat[i, :] = x.detach().cpu().numpy()

    return scores_hat


def compute_prediction_rmse(
    data: dict, mean_model: dict, grid: np.ndarray, fpca: dict, scores: np.ndarray
) -> dict:
    sq_obs = []
    sq_sig = []
    for i, t_i in enumerate(data["t"]):
        y_hat = reconstruct_at_times(t_i, scores[i], mean_model, grid, fpca)
        sq_obs.append((y_hat - data["y"][i]) ** 2)
        if "signal" in data:
            sq_sig.append((y_hat - data["signal"][i]) ** 2)
    out = {"rmse_obs": float(np.sqrt(np.mean(np.concatenate(sq_obs))))}
    if sq_sig:
        out["rmse_signal"] = float(np.sqrt(np.mean(np.concatenate(sq_sig))))
    return out


if __name__ == "__main__":
    # Increase these for more stable final experiments; keep moderate for iteration speed.
    n_curves = 300
    flow_epochs = 500
    map_steps = 250

    data = generate_functional_data(
        n_curves=n_curves,
        min_points=6,
        max_points=12,
        lambdas=(1.0, 0.6),
        noise_variance=0.05,
        random_state=41,
    )

    fpca_fit = build_univariate_fpca_model(data, pve_threshold=0.95)
    pace_scores = fpca_fit["scores_pace"]
    k = fpca_fit["fpca"]["n_keep"]

    flow_prior = fit_flow_prior_on_scores(
        scores=pace_scores[:, :k],
        n_layers=6,
        hidden=64,
        epochs=flow_epochs,
        batch_size=64,
        lr=1e-3,
        seed=123,
    )

    flow_scores = estimate_scores_flow_map(
        data=data,
        mean_model=fpca_fit["mean_model"],
        grid=fpca_fit["grid"],
        fpca=fpca_fit["fpca"],
        sigma2=fpca_fit["sigma2"],
        flow_prior=flow_prior,
        init_scores=pace_scores,
        n_components=k,
        steps=map_steps,
        lr=0.03,
    )

    metrics_pace = compute_prediction_rmse(
        data=data,
        mean_model=fpca_fit["mean_model"],
        grid=fpca_fit["grid"],
        fpca=fpca_fit["fpca"],
        scores=pace_scores[:, :k],
    )
    metrics_flow = compute_prediction_rmse(
        data=data,
        mean_model=fpca_fit["mean_model"],
        grid=fpca_fit["grid"],
        fpca=fpca_fit["fpca"],
        scores=flow_scores,
    )

    print(f"Components retained: {k}")
    print(f"Estimated sigma^2: {fpca_fit['sigma2']:.6f}")
    print("PACE RMSE (obs/signal):", metrics_pace)
    print("Flow-MAP RMSE (obs/signal):", metrics_flow)
    print("Covariance of PACE scores:")
    print(np.round(np.cov(pace_scores[:, :k], rowvar=False), 6))
    print("Covariance of Flow-MAP scores:")
    print(np.round(np.cov(flow_scores, rowvar=False), 6))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(pace_scores[:, 0], bins=35, alpha=0.7, label="PACE")
    axes[0].hist(flow_scores[:, 0], bins=35, alpha=0.7, label="Flow-MAP")
    axes[0].set_title("Score Distribution: Component 1")
    axes[0].legend()
    if k > 1:
        axes[1].scatter(pace_scores[:, 0], pace_scores[:, 1], s=8, alpha=0.35, label="PACE")
        axes[1].scatter(flow_scores[:, 0], flow_scores[:, 1], s=8, alpha=0.35, label="Flow-MAP")
        axes[1].set_title("Score Scatter (Comp1, Comp2)")
        axes[1].legend()
    else:
        axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("score_estimation_flow_vs_pace.png", dpi=180)
    if "agg" not in plt.get_backend().lower():
        plt.show()
