import numpy as np


def _mean1(x: np.ndarray) -> np.ndarray:
    return 5.0 * np.sin(2.0 * np.pi * x)


def _mean2(x: np.ndarray) -> np.ndarray:
    return 5.0 * np.cos(2.0 * np.pi * x)


def _mean3(x: np.ndarray) -> np.ndarray:
    return 5.0 * (x - 1.0) ** 2


def _psi11(x: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / 3.0) * np.sin(2.0 * np.pi * x)


def _psi12(x: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / 3.0) * np.cos(4.0 * np.pi * x)


def _psi13(x: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / 3.0) * np.sin(4.0 * np.pi * x)


def _psi21(x: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / 3.0) * np.sin(0.5 * np.pi * x)


def _psi22(x: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / 3.0) * np.sin(1.5 * np.pi * x)


def _psi23(x: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / 3.0) * np.sin(2.5 * np.pi * x)


def _psi31(x: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / 3.0) * np.sin(1.0 * np.pi * x)


def _psi32(x: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / 3.0) * np.sin(2.0 * np.pi * x)


def _psi33(x: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / 3.0) * np.sin(3.0 * np.pi * x)


def simulate_multivariate_sparse_data(
    n_subjects: int = 250,
    min_points: int = 3,
    max_points: int = 7,
    lambdas: tuple[float, ...] = (6.0, 3.0, 1.5),
    noise_vars: tuple[float, ...] = (0.01, 0.01, 0.01),
    random_state: int | None = 42,
) -> dict:
    """
    Simulate sparse multivariate data from a KL expansion in the mfaces example style:
      Y_i^(p)(t) = mu_p(t) + sum_k xi_ik * psi_k^(p)(t) + eps_i^(p)(t)
    with shared subject scores xi_ik across variables.
    """
    rng = np.random.default_rng(random_state)
    n_vars = 3
    n_comp = len(lambdas)

    def eval_mean(var_idx: int, x: np.ndarray) -> np.ndarray:
        if var_idx == 0:
            return _mean1(x)
        if var_idx == 1:
            return _mean2(x)
        return _mean3(x)

    def eval_eigen(var_idx: int, comp_idx: int, x: np.ndarray) -> np.ndarray:
        if var_idx == 0:
            basis = [_psi11(x), _psi12(x), _psi13(x)]
        elif var_idx == 1:
            basis = [_psi21(x), _psi22(x), _psi23(x)]
        else:
            basis = [_psi31(x), _psi32(x), _psi33(x)]
        return basis[comp_idx]

    scores = np.zeros((n_subjects, n_comp))
    t_obs: list[list[np.ndarray]] = []
    y_obs: list[list[np.ndarray]] = []
    signal_obs: list[list[np.ndarray]] = []

    for i in range(n_subjects):
        xi_i = np.array([rng.normal(0.0, np.sqrt(lam)) for lam in lambdas])
        scores[i, :] = xi_i

        t_i_vars = []
        y_i_vars = []
        s_i_vars = []
        for p in range(n_vars):
            m_ip = int(rng.integers(min_points, max_points + 1))
            t_ip = np.sort(rng.uniform(0.0, 1.0, size=m_ip))

            signal_ip = eval_mean(p, t_ip)
            for k in range(n_comp):
                signal_ip = signal_ip + xi_i[k] * eval_eigen(p, k, t_ip)

            y_ip = signal_ip + rng.normal(0.0, np.sqrt(noise_vars[p]), size=m_ip)

            t_i_vars.append(t_ip)
            y_i_vars.append(y_ip)
            s_i_vars.append(signal_ip)

        t_obs.append(t_i_vars)
        y_obs.append(y_i_vars)
        signal_obs.append(s_i_vars)

    return {
        "t": t_obs,
        "y": y_obs,
        "signal": signal_obs,
        "scores_true": scores,
        "lambdas_true": np.array(lambdas),
        "noise_vars_true": np.array(noise_vars),
        "n_vars": n_vars,
        "n_components_true": n_comp,
        "mean_functions_true": [_mean1, _mean2, _mean3],
        "eigenfunction_eval_true": eval_eigen,
    }
