# Multivariate fPCA (Sparse Data)

This folder provides a Python module for multivariate sparse functional PCA with:

- multivariate mean estimation via penalized B-splines (CV-selected smoothing),
- multivariate covariance estimation via penalized bivariate B-splines,
- multivariate eigen-decomposition from the estimated block covariance operator,
- subject-level latent score estimation using BLUP under a working Gaussian model (PACE-style).

## Files

- `simulation.py`: sparse multivariate KL simulation generator.
- `mfpca.py`: main estimation pipeline and utilities.
- `demo_mfpca.py`: end-to-end example run and plots.

## Run demo

```bash
cd "FPCA_Python/multivariate fPCA"
python demo_mfpca.py
```

Output figure: `multivariate_fpca_demo.png`.

## Notes

- The simulation defaults are configurable and designed for sparse settings.
- If you want exact replication of a specific simulation setting from a paper, update parameters in `simulate_multivariate_sparse_data`.

