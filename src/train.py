"""Training: Adam(+decay) loop, N-seed sweep, periodic weight update, wandb, checkpoints.

Run (from repo root, after downloading DNS data):
    python -m src.train --config=configs/mlp.py
    python -m src.train --config=configs/pirate.py
"""
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from ml_collections import config_flags

from src import networks
from src import model as model_mod
from src import data_pipeline
from src.losses import LOSS_KEYS, update_weights

try:
    import wandb
except Exception:
    wandb = None

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/mlp.py", "Training config.")


def make_optimizer(cfg):
    sched = optax.exponential_decay(
        init_value=float(cfg.optim.learning_rate),
        transition_steps=int(cfg.optim.decay_steps),
        decay_rate=float(cfg.optim.decay_rate),
    )
    return optax.adam(sched)


def train_one_seed(cfg, seed, points, log_fn=None):
    key = jax.random.PRNGKey(int(seed))
    net = networks.build_network(cfg)
    params = networks.init_params(net, key)
    pinn = model_mod.KOmegaPINN(cfg, net, points)
    opt = make_optimizer(cfg)
    opt_state = opt.init(params)
    weights = {k: jnp.asarray(float(cfg.weighting.init_weights[k])) for k in LOSS_KEYS}

    @jax.jit
    def step(params, opt_state, weights):
        (loss, terms), grads = jax.value_and_grad(
            lambda p: pinn.total_loss(p, weights), has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, terms

    history = []
    scheme = cfg.weighting.scheme
    upd_every = int(cfg.weighting.update_every_steps)
    for it in range(int(cfg.training.max_steps)):
        if scheme != "uniform" and it > 0 and it % upd_every == 0:
            weights = update_weights(pinn.loss_terms, params, scheme, weights,
                                     float(cfg.weighting.momentum))
        params, opt_state, loss, terms = step(params, opt_state, weights)
        if it % 500 == 0:
            rec = {"step": it, "loss": float(loss)}
            rec.update({f"loss/{k}": float(terms[k]) for k in LOSS_KEYS})
            rec.update({f"w/{k}": float(weights[k]) for k in LOSS_KEYS})
            history.append(rec)
            if log_fn:
                log_fn(rec)
    return params, history


def save_checkpoint(cfg, params, tag):
    d = os.path.join(cfg.saving.checkpoint_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{tag}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(jax.device_get(params), fh)
    return path


def main(argv):
    cfg = FLAGS.config
    data = data_pipeline.build_data_points(cfg)        # requires DNS downloaded
    points = model_mod.build_points(cfg, data)
    arch = cfg.wandb.name or cfg.arch.arch_name

    for s in range(int(cfg.run.n_seeds)):
        seed = int(cfg.run.seed) + s
        use_wandb = wandb is not None and cfg.wandb.mode != "disabled"
        if use_wandb:
            wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                       name=f"{arch}-seed{seed}", group=arch, mode=cfg.wandb.mode,
                       tags=list(cfg.wandb.tags), reinit=True,
                       config=cfg.to_dict())

        def log_fn(rec):
            if use_wandb:
                wandb.log(rec)

        params, history = train_one_seed(cfg, seed, points, log_fn=log_fn)
        ckpt = save_checkpoint(cfg, params, tag=f"{arch}_seed{seed}")
        print(f"[seed {seed}] done -> {ckpt}  final_loss={history[-1]['loss']:.4e}")
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    app.run(main)
