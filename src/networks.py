"""Networks for the comparison: MLP (Pioch) vs PirateNet (Wang-Sankaran-Perdikaris 2024).

Implemented in Flax directly so the code runs as-is. Both share the SAME I/O:
  input  (x, y) -> output [u, v, p, k, w_raw]  (5-vector)
so swapping `config.arch.arch_name` is the only difference (controlled comparison).

NOTE: to use jaxpi's archs instead, replace build_network() body with
  from jaxpi import archs; return archs.Mlp(**...) / archs.PirateNet(**...)
matching the installed jaxpi version's signature.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = jnp.tanh(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.out_dim)(x)


class FourierEmbed(nn.Module):
    embed_dim: int          # total embedding dim (even); uses embed_dim//2 frequencies
    embed_scale: float

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        B = self.param("B",
                       lambda key, shape: jax.random.normal(key, shape) * self.embed_scale,
                       (self.embed_dim // 2, in_dim))
        proj = x @ B.T
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


class PirateNet(nn.Module):
    """Fourier embedding + gated adaptive-residual blocks (alpha init 0) + linear out."""
    num_blocks: int
    hidden_dim: int
    out_dim: int
    embed_dim: int
    embed_scale: float
    alpha_init: float = 0.0

    @nn.compact
    def __call__(self, x):
        act = jnp.tanh
        phi = FourierEmbed(self.embed_dim, self.embed_scale)(x)
        U = act(nn.Dense(self.hidden_dim)(phi))
        V = act(nn.Dense(self.hidden_dim)(phi))
        xl = act(nn.Dense(self.hidden_dim)(phi))
        for i in range(self.num_blocks):
            f = act(nn.Dense(self.hidden_dim)(xl))
            z1 = f * U + (1.0 - f) * V
            g = act(nn.Dense(self.hidden_dim)(z1))
            z2 = g * U + (1.0 - g) * V
            h = act(nn.Dense(self.hidden_dim)(z2))
            alpha = self.param(f"alpha_{i}",
                               nn.initializers.constant(self.alpha_init), ())
            xl = alpha * h + (1.0 - alpha) * xl
        return nn.Dense(self.out_dim)(xl)


def build_network(cfg):
    a = cfg.arch
    if a.arch_name == "PirateNet":
        emb = a.fourier_emb
        return PirateNet(num_blocks=int(a.num_layers), hidden_dim=int(a.hidden_dim),
                         out_dim=int(a.out_dim), embed_dim=int(emb.embed_dim),
                         embed_scale=float(emb.embed_scale),
                         alpha_init=float(a.get("nonlinearity", 0.0)))
    return MLP(num_layers=int(a.num_layers), hidden_dim=int(a.hidden_dim),
               out_dim=int(a.out_dim))


def init_params(net, key, in_dim=2):
    return net.init(key, jnp.zeros((in_dim,)))
