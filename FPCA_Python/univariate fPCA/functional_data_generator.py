
import numpy as np


def mean_function(x: np.ndarray) -> np.ndarray:
    """Smooth mean function on [0, 1]."""
    return 1.0 + 0.5 * np.sin(np.pi * x)


def eigenfunction_1(x: np.ndarray) -> np.ndarray:
    """Normalized eigenfunction: sqrt(2) * sin(2*pi*x)."""
    return np.sqrt(2.0) * np.sin(2.0 * np.pi * x)


def eigenfunction_2(x: np.ndarray) -> np.ndarray:
    """Normalized eigenfunction: sqrt(2) * cos(2*pi*x)."""
    return np.sqrt(2.0) * np.cos(2.0 * np.pi * x)


def covariance_function(s: np.ndarray, t: np.ndarray, lambdas=(1.0, 0.5)) -> np.ndarray:
    """Covariance from the 2-term Karhunen-Loeve expansion."""
    l1, l2 = lambdas
    return (
        l1 * eigenfunction_1(s) * eigenfunction_1(t)
        + l2 * eigenfunction_2(s) * eigenfunction_2(t)
    )


def generate_functional_data(
    n_curves: int = 100,
    min_points: int = 30,
    max_points: int = 60,
    lambdas=(1.0, 0.5),
    noise_variance: float = 0.05,
    random_state: int | None = 42,
):
    """
    Generate irregular functional observations:
      X_i(t) = mu(t) + xi_i1*phi_1(t) + xi_i2*phi_2(t) + eps_i(t)
    where xi_ik ~ N(0, lambda_k) and eps_i(t) ~ N(0, noise_variance).
    """
    if min_points <= 0 or max_points < min_points:
        raise ValueError("Require 0 < min_points <= max_points.")
    if noise_variance < 0:
        raise ValueError("noise_variance must be nonnegative.")

    rng = np.random.default_rng(random_state)
    l1, l2 = lambdas
    noise_sd = np.sqrt(noise_variance)

    observation_points = []
    observed_values = []
    latent_values = []
    scores = np.zeros((n_curves, 2))

    for i in range(n_curves):
        m_i = int(rng.integers(min_points, max_points + 1))
        t_i = np.sort(rng.uniform(0.0, 1.0, size=m_i))  # irregular random grid

        xi_i = np.array(
            [rng.normal(0.0, np.sqrt(l1)), rng.normal(0.0, np.sqrt(l2))]
        )
        scores[i, :] = xi_i

        signal_i = (
            mean_function(t_i)
            + xi_i[0] * eigenfunction_1(t_i)
            + xi_i[1] * eigenfunction_2(t_i)
        )
        y_i = signal_i + rng.normal(0.0, noise_sd, size=m_i)  # common noise variance

        observation_points.append(t_i)
        latent_values.append(signal_i)
        observed_values.append(y_i)

    return {
        "t": observation_points,
        "y": observed_values,
        "signal": latent_values,
        "scores": scores,
        "lambdas": np.array([l1, l2]),
        "noise_variance": noise_variance,
    }


if __name__ == "__main__":
    data = generate_functional_data(
        n_curves=5,
        min_points=15,
        max_points=25,
        lambdas=(1.2, 0.7),
        noise_variance=0.04,
        random_state=7,
    )

    print("Generated curves:", len(data["t"]))
    print("First curve points:", data["t"][0].shape[0])
    print("First 5 observation times:", np.round(data["t"][0][:5], 4))
    print("First 5 observed values:", np.round(data["y"][0][:5], 4))
    print("First curve scores:", np.round(data["scores"][0], 4))
