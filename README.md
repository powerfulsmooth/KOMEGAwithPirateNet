# k-ω PINN on a Backward-Facing Step — MLP vs PirateNet

PINN solving the steady 2D incompressible **RANS + Wilcox(1988) k-ω** model for the
backward-facing step (Re_h = 5100), reproducing **Pioch et al. (2023)** as the MLP baseline
and comparing against **PirateNet** (Wang–Sankaran–Perdikaris 2024). Only the network is
swapped; geometry / physics / loss / data / evaluation / postprocess are shared.

## Workflow (laptop → GitHub → Colab GPU)
Edit + `git push` on the laptop; heavy training/evaluation run on **Colab GPU**.
See `colab/run.ipynb`.

## Quickstart (Colab)
```bash
git clone <REPO_URL> && cd bfs_komega_pinn
pip install -r requirements.txt
pip install -U "jax[cuda12]"
python data/download_case031.py            # DNS (ERCOFTAC Case031)
python -m src.train  --config=configs/mlp.py
python -m src.train  --config=configs/pirate.py
python -m experiments.run_comparison --config=configs/base.py
```

## Repository
```
configs/      base.py + mlp.py / pirate.py  (every option: omega_param, weighting, data mode, ...)
data/         download_case031.py, load_dns.py, raw/  (DNS; gitignored)
src/          geometry, physics (k-omega residuals), networks (MLP/PirateNet),
              losses (uniform/grad_norm/ntk), model, data_pipeline, train
evaluation/   metrics.py  (NMSE, FAC2, FB, V, reattachment, select_best)
postprocess/  fields, plots (contour/streamline/profiles/convergence), compare
experiments/  run_comparison.py  (load ckpts -> evaluate -> compare)
colab/        run.ipynb
```

## Key design choices (see *_spec.md design docs)
- **Pioch frame**: domain x/h∈[0,22], y/h∈[0,6], step notch at x/h=3 (= DNS x/h=0).
  DNS→frame: `x_pioch = x_ercoftac + 3`, inlet station `y += h`.
- **omega_param = "direct"** (Pioch 1:1; network outputs ω). Set `"g"` (ω=1/g²) for wall stability.
- BC: slip top, no-slip bottom+step, inlet/outlet DNS U,V only (k,ω not imposed — Pioch).
- Loss weighting: `uniform | grad_norm | ntk` (+ optional RBA), selectable in config.
- Init-sensitivity study: `run.n_seeds=10`, select lowest-NMSE physical model, report distributions.

## Status
Verified in CPU sandbox (no-holes check): geometry sampling, **k-ω residuals (2nd-order
nested autodiff) finite over batches**, and the full `config → network → physics → 7-term
loss → gradient` pipeline. GPU training runs on Colab.

## Notes
- Networks are implemented in **Flax** directly (runs as-is); `jaxpi` is listed and its
  `archs.PirateNet` can be swapped into `src/networks.build_network` if preferred.
- DNS data: ERCOFTAC Case031 (Le–Moin–Kim 1997), **CC BY-NC-SA 4.0** — cite + non-commercial.
