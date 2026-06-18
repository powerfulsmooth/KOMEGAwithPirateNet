"""MLP config = Pioch baseline reproduction (no embedding / no residual / no pi-init)."""
from configs import base


def get_config():
    config = base.get_config()
    config.arch.arch_name = "Mlp"
    config.arch.num_layers = 5        # Pioch: 5 hidden layers
    config.arch.hidden_dim = 128      # Pioch: 128 neurons
    config.arch.activation = "tanh"
    config.arch.fourier_emb = None
    config.wandb.name = "mlp"
    config.wandb.tags = ("mlp", "pioch_baseline")
    return config
