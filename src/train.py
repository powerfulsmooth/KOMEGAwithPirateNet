# """Training: Adam(+decay) loop, N-seed sweep, periodic weight update, wandb, checkpoints.

# Run (from repo root, after downloading DNS data):
#     python -m src.train --config=configs/mlp.py
#     python -m src.train --config=configs/pirate.py
# """
# import os
# import pickle

# import jax
# import jax.numpy as jnp
# import numpy as np
# import optax
# from absl import app, flags
# from ml_collections import config_flags

# from src import networks
# from src import model as model_mod
# from src import data_pipeline
# from src.losses import LOSS_KEYS, update_weights

# try:
#     import wandb
# except Exception:
#     wandb = None

# FLAGS = flags.FLAGS
# config_flags.DEFINE_config_file("config", "configs/mlp.py", "Training config.")


# def make_optimizer(cfg):
#     sched = optax.exponential_decay(
#         init_value=float(cfg.optim.learning_rate),
#         transition_steps=int(cfg.optim.decay_steps),
#         decay_rate=float(cfg.optim.decay_rate),
#     )
#     return optax.adam(sched)


# def train_one_seed(cfg, seed, points, log_fn=None):
#     key = jax.random.PRNGKey(int(seed))
#     net = networks.build_network(cfg)
#     params = networks.init_params(net, key)
#     pinn = model_mod.KOmegaPINN(cfg, net, points)
#     opt = make_optimizer(cfg)
#     opt_state = opt.init(params)
#     weights = {k: jnp.asarray(float(cfg.weighting.init_weights[k])) for k in LOSS_KEYS}

#     @jax.jit
#     def step(params, opt_state, weights):
#         (loss, terms), grads = jax.value_and_grad(
#             lambda p: pinn.total_loss(p, weights), has_aux=True)(params)
#         updates, opt_state = opt.update(grads, opt_state, params)
#         params = optax.apply_updates(params, updates)
#         return params, opt_state, loss, terms

#     history = []
#     scheme = cfg.weighting.scheme
#     upd_every = int(cfg.weighting.update_every_steps)
#     for it in range(int(cfg.training.max_steps)):
#         if scheme != "uniform" and it > 0 and it % upd_every == 0:
#             weights = update_weights(pinn.loss_terms, params, scheme, weights,
#                                      float(cfg.weighting.momentum))
#         params, opt_state, loss, terms = step(params, opt_state, weights)
#         if it % 500 == 0:
#             rec = {"step": it, "loss": float(loss)}
#             rec.update({f"loss/{k}": float(terms[k]) for k in LOSS_KEYS})
#             rec.update({f"w/{k}": float(weights[k]) for k in LOSS_KEYS})
#             history.append(rec)
#             if log_fn:
#                 log_fn(rec)
#     return params, history


# def save_checkpoint(cfg, params, tag):
#     d = os.path.join(cfg.saving.checkpoint_dir)
#     os.makedirs(d, exist_ok=True)
#     path = os.path.join(d, f"{tag}.pkl")
#     with open(path, "wb") as fh:
#         pickle.dump(jax.device_get(params), fh)
#     return path


# def main(argv):
#     cfg = FLAGS.config
#     data = data_pipeline.build_data_points(cfg)        # requires DNS downloaded
#     points = model_mod.build_points(cfg, data)
#     arch = cfg.wandb.name or cfg.arch.arch_name

#     for s in range(int(cfg.run.n_seeds)):
#         seed = int(cfg.run.seed) + s
#         use_wandb = wandb is not None and cfg.wandb.mode != "disabled"
#         if use_wandb:
#             wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
#                        name=f"{arch}-seed{seed}", group=arch, mode=cfg.wandb.mode,
#                        tags=list(cfg.wandb.tags), reinit=True,
#                        config=cfg.to_dict())

#         def log_fn(rec):
#             if use_wandb:
#                 wandb.log(rec)

#         params, history = train_one_seed(cfg, seed, points, log_fn=log_fn)
#         ckpt = save_checkpoint(cfg, params, tag=f"{arch}_seed{seed}")
#         print(f"[seed {seed}] done -> {ckpt}  final_loss={history[-1]['loss']:.4e}")
#         if use_wandb:
#             wandb.finish()


# if __name__ == "__main__":
#     app.run(main)



"""Training: Adam(+decay) loop, N-seed sweep, periodic weight update, wandb, checkpoints.

Run (from repo root, after downloading DNS data):
    python -m src.train --config=configs/mlp.py
    python -m src.train --config=configs/pirate.py
"""
import os
import csv
import pickle
import time

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

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, *a, **k):
        return x

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/mlp.py", "Training config.")


def make_optimizer(cfg):
    sched = optax.exponential_decay(
        init_value=float(cfg.optim.learning_rate),
        transition_steps=int(cfg.optim.decay_steps),
        decay_rate=float(cfg.optim.decay_rate),
    )
    clip = float(cfg.optim.get("grad_clip_norm", 0.0) or 0.0)
    chain = [optax.zero_nans()]          # NaN 그래디언트 -> 0 (오염 방어)
    if clip > 0:
        chain.append(optax.clip_by_global_norm(clip))   # 거대 그래디언트 캡
    chain.append(optax.adam(sched))
    return optax.chain(*chain)


def save_contour_png(cfg, net, params, arch, seed, step):
    """현재 파라미터로 |U|(speed) 등고선을 그려 figs/progress 에 PNG로 저장.

    CFD에서 iter마다 contour 보듯, 학습 중 흐름장이 발전하는 걸 보기 위한 스냅샷.
    모니터링이 학습을 죽이면 안 되므로 어떤 에러든 삼키고 None을 반환한다.
    반환: 저장된 경로 (실패 시 None)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from postprocess import fields

        def predict_fn(xy):
            xy = jnp.asarray(xy)
            return np.asarray(jax.vmap(lambda pt: net.apply(params, pt))(xy))

        fl = fields.evaluate(predict_fn, cfg)
        outdir = os.path.join(cfg.saving.figs_dir, "progress")
        os.makedirs(outdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 2.8))
        pc = ax.contourf(fl["X"], fl["Y"], fl["speed"], levels=30, cmap="viridis")
        # 속도장 위에 streamline 오버레이 (고체 영역의 NaN은 0으로 대체)
        U = np.nan_to_num(fl["U"], nan=0.0)
        V = np.nan_to_num(fl["V"], nan=0.0)
        try:
            ax.streamplot(fl["X"], fl["Y"], U, V, density=1.3,
                          color="white", linewidth=0.6, arrowsize=0.7)
        except Exception:
            pass
        ax.set_aspect("equal"); ax.set_xlabel("x/h"); ax.set_ylabel("y/h")
        ax.set_xlim(fl["X"].min(), fl["X"].max())
        ax.set_ylim(fl["Y"].min(), fl["Y"].max())
        ax.set_title(f"{arch} |U| + streamlines  seed{seed}  step {step}")
        fig.colorbar(pc, ax=ax, shrink=0.9)
        path = os.path.join(outdir, f"{arch}_seed{seed}_step{step:06d}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        print(f"  [contour] step {step} 스냅샷 실패(학습은 계속): {e}")
        return None


def save_progress_ckpt(cfg, params, arch, seed, step):
    """중간 체크포인트 저장(붕괴 전 좋은 모델 보존용).

    run_comparison의 `{arch}_seed*.pkl` glob에 안 걸리도록 checkpoints/progress/ 하위에 저장."""
    try:
        d = os.path.join(cfg.saving.checkpoint_dir, "progress")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{arch}_seed{seed}_step{step:06d}.pkl")
        with open(path, "wb") as fh:
            pickle.dump(jax.device_get(params), fh)
        return path
    except Exception as e:
        print(f"  [ckpt] step {step} 중간 저장 실패(학습은 계속): {e}")
        return None


def append_history_csv(cfg, arch, seed, rec, write_header):
    """학습 로그(rec)를 구글 드라이브 CSV에 누적 저장 (wandb 수동 export 불필요).

    write_header=True(=run 시작, it==0)면 새 파일로 헤더부터 덮어쓰고, 이후엔 한 줄씩 append.
    실패해도 학습은 계속되도록 예외를 삼킨다."""
    try:
        d = cfg.saving.get("logs_dir", None) or os.path.join(cfg.saving.figs_dir, "logs")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{arch}_seed{seed}_history.csv")
        fields = (["step", "loss"]
                  + [f"loss/{k}" for k in LOSS_KEYS]
                  + [f"w/{k}" for k in LOSS_KEYS])
        with open(path, "w" if write_header else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(rec)
        return path
    except Exception as e:
        print(f"  [log] step {rec.get('step')} CSV 저장 실패(학습은 계속): {e}")
        return None


def train_one_seed(cfg, seed, points, log_fn=None):
    key = jax.random.PRNGKey(int(seed))
    net = networks.build_network(cfg)
    params = networks.init_params(net, key)
    # PirateNet: physics-informed init of the final layer (u,v least-squares on DNS data)
    if str(cfg.arch.arch_name) == "PirateNet" and bool(cfg.arch.get("pi_init", False)):
        params = networks.pi_init(net, params, points)
    pinn = model_mod.KOmegaPINN(cfg, net, points)
    opt = make_optimizer(cfg)
    opt_state = opt.init(params)
    weights = {k: jnp.asarray(float(cfg.weighting.init_weights[k])) for k in LOSS_KEYS}
    arch = cfg.wandb.name or cfg.arch.arch_name
    contour_every = int(cfg.saving.get("contour_every_steps", 0) or 0)
    save_every = int(cfg.saving.get("save_every_steps", 0) or 0)

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
    # Colab 진행바: % 완료 + 경과/예상남은시간(ETA) 표시 (loss 메트릭은 wandb에서)
    pbar = tqdm(range(int(cfg.training.max_steps)), desc=f"seed{seed}",
                ncols=90, mininterval=1.0)
    for it in pbar:
        if scheme != "uniform" and it > 0 and it % upd_every == 0:
            weights = update_weights(pinn.loss_terms, params, scheme, weights,
                                     float(cfg.weighting.momentum))
        params, opt_state, loss, terms = step(params, opt_state, weights)
        if it % 500 == 0:
            rec = {"step": it, "loss": float(loss)}
            rec.update({f"loss/{k}": float(terms[k]) for k in LOSS_KEYS})
            rec.update({f"w/{k}": float(weights[k]) for k in LOSS_KEYS})
            history.append(rec)
            append_history_csv(cfg, arch, seed, rec, write_header=(it == 0))
            img_path = None
            if contour_every and it % contour_every == 0:
                img_path = save_contour_png(cfg, net, params, arch, seed, it)
            if log_fn:
                log_fn(rec, img_path)
        if save_every and it > 0 and it % save_every == 0:
            save_progress_ckpt(cfg, params, arch, seed, it)
    return params, history


def save_checkpoint(cfg, params, tag):
    d = os.path.join(cfg.saving.checkpoint_dir)
    # 💡 구글 드라이브 동기화 지연으로 인한 경로 에러를 방지하기 위해 예외 처리 추가
    try:
        os.makedirs(d, exist_ok=True)
    except FileExistsError:
        pass
        
    path = os.path.join(d, f"{tag}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(jax.device_get(params), fh)
    
    # 💡 코랩에서 구글 드라이브 플러시(즉시 반영)를 유도하여 파일 유실 방지
    if hasattr(os, "sync"):
        os.sync()
    return path


def main(argv):
    cfg = FLAGS.config
    data = data_pipeline.build_data_points(cfg)        # requires DNS downloaded
    points = model_mod.build_points(cfg, data)
    arch = cfg.wandb.name or cfg.arch.arch_name

    for s in range(int(cfg.run.n_seeds)):
        seed = int(cfg.run.seed) + s
        
        # 💡 [핵심 추가] 이미 구글 드라이브에 완료된 가중치 파일이 있는지 검사
        ckpt_path = os.path.join(cfg.saving.checkpoint_dir, f"{arch}_seed{seed}.pkl")
        if os.path.exists(ckpt_path):
            print(f"--> [seed {seed}] 이미 완료된 파일이 존재하여 건너뜁니다: {ckpt_path}")
            continue

        use_wandb = wandb is not None and cfg.wandb.mode != "disabled"
        if use_wandb:
            wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                       name=f"{arch}-seed{seed}", group=arch, mode=cfg.wandb.mode,
                       tags=list(cfg.wandb.tags), reinit=True,
                       config=cfg.to_dict())

        def log_fn(rec, img_path=None):
            if use_wandb:
                data = dict(rec)
                if img_path:
                    data["contour"] = wandb.Image(img_path)
                wandb.log(data)

        print(f"--> [seed {seed}] 학습을 시작합니다...")
        params, history = train_one_seed(cfg, seed, points, log_fn=log_fn)
        ckpt = save_checkpoint(cfg, params, tag=f"{arch}_seed{seed}")
        print(f"[seed {seed}] done -> {ckpt}  final_loss={history[-1]['loss']:.4e}")
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    app.run(main)






