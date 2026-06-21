"""PirateNet config: Fourier embedding + adaptive residual (alpha init 0) + pi-init."""
import ml_collections
from configs import base


def get_config():
    config = base.get_config()
    config.arch.arch_name = "PirateNet"
    config.arch.num_layers = 3        # number of residual-adaptive blocks
    config.arch.hidden_dim = 256
    config.arch.activation = "tanh"
    config.arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 10.0, "embed_dim": 256}   # jaxpi BFS 권장값 (정규화 좌표 기준)
    )
    config.arch.nonlinearity = 0.0    # PirateNet adaptive-residual alpha init (= 0 -> identity)
    config.arch.pi_init = True        # physics-informed init (least-squares final layer)
    config.wandb.name = "pirate"
    config.wandb.tags = ("pirate",)
    return config
