import matplotlib.pyplot as plt
import numpy as np

from functional_data_generator import generate_functional_data


def plot_functional_data(data: dict, n_show: int = 6) -> None:
    """Plot noisy irregular observations and latent smooth signals."""
    total = len(data["t"])
    n_show = min(n_show, total)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes = np.array(axes).ravel()

    for i in range(n_show):
        ax = axes[i]
        t_i = data["t"][i]
        y_i = data["y"][i]
        signal_i = data["signal"][i]

        ax.scatter(t_i, y_i, s=14, alpha=0.65, label="Noisy obs")
        ax.plot(t_i, signal_i, linewidth=2.0, label="Latent signal")
        ax.set_title(f"Curve {i + 1}")
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.25)

    for j in range(n_show, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Irregular Functional Data (Noisy vs Latent)", y=0.98)
    fig.text(0.5, 0.03, "t in [0, 1]", ha="center")
    fig.text(0.02, 0.5, "X(t)", va="center", rotation="vertical")
    fig.tight_layout(rect=(0.03, 0.05, 1, 0.93))


if __name__ == "__main__":
    generated = generate_functional_data(
        n_curves=12,
        min_points=20,
        max_points=45,
        lambdas=(1.2, 0.7),
        noise_variance=0.04,
        random_state=123,
    )

    plot_functional_data(generated, n_show=6)
    plt.savefig("functional_data_plot.png", dpi=180)
    if "agg" not in plt.get_backend().lower():
        plt.show()

