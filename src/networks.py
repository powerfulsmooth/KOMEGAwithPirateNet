"""Networks for the comparison: MLP (Pioch) vs PirateNet (Wang-Sankaran-Perdikaris 2024).

MLP = pure Pioch baseline (plain Dense, tanh, raw coords) — intentionally left untouched.

PirateNet is assembled from jaxpi-vendored primitives (src.archs_jaxpi: RWF Dense,
random Fourier features) plus the paper's adaptive-residual blocks and pi-init:
  - input coords affine-normalized to [-1, 1] (required for Fourier features; transparent
    to physics.py because autodiff carries the constant 1/scale Jacobian),
  - first block input = Fourier embedding phi  (paper eq. 4.8: u = W_L phi at init),
  - L adaptive-residual blocks, trainable alpha init 0 -> identity map at start (eq. 4.6),
  - final linear layer named "output_layer" (overwritten by pi_init on u,v).
I/O matches this project: input (x, y); output [u, v, p, k, omega] (out_dim = 5).
"""
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn

from src import archs_jaxpi as aj


class MLP(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = jnp.tanh(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.out_dim)(x)


class PirateNet(nn.Module):
    num_blocks: int
    hidden_dim: int
    out_dim: int
    embed_dim: int
    embed_scale: float
    activation: str = "tanh"
    alpha_init: float = 0.0
    use_rwf: bool = True
    reparam_mean: float = 1.0
    reparam_stddev: float = 0.1
    # affine input-normalization bounds (Pioch frame) -> [-1, 1]
    x0: float = 0.0
    x1: float = 22.0
    y0: float = 0.0
    y1: float = 6.0

    @nn.compact
    def __call__(self, x, return_features=False):
        act = aj._get_activation(self.activation)
        reparam = ({"type": "weight_fact", "mean": self.reparam_mean,
                    "stddev": self.reparam_stddev} if self.use_rwf else None)

        # affine input normalization to [-1, 1]; autodiff carries 1/scale so the
        # PDE residuals in physics.py stay in physical (Pioch) coordinates.
        xn = 2.0 * (x[0] - self.x0) / (self.x1 - self.x0) - 1.0
        yn = 2.0 * (x[1] - self.y0) / (self.y1 - self.y0) - 1.0
        z = jnp.stack([xn, yn])

        phi = aj.FourierEmbs(embed_scale=self.embed_scale, embed_dim=self.embed_dim)(z)
        U = act(aj.Dense(self.hidden_dim, reparam=reparam)(phi))
        V = act(aj.Dense(self.hidden_dim, reparam=reparam)(phi))

        xl = phi  # paper x^(1) = phi (requires embed_dim == hidden_dim)
        for i in range(self.num_blocks):
            f = act(aj.Dense(self.hidden_dim, reparam=reparam)(xl))
            z1 = f * U + (1.0 - f) * V
            g = act(aj.Dense(self.hidden_dim, reparam=reparam)(z1))
            z2 = g * U + (1.0 - g) * V
            h = act(aj.Dense(self.hidden_dim, reparam=reparam)(z2))
            alpha = self.param(f"alpha_{i}", nn.initializers.constant(self.alpha_init), ())
            xl = alpha * h + (1.0 - alpha) * xl

        if return_features:
            return xl
        return aj.Dense(self.out_dim, reparam=reparam, name="output_layer")(xl)


def build_network(cfg):
    a = cfg.arch
    if a.arch_name == "PirateNet":
        emb = a.fourier_emb
        if int(emb.embed_dim) != int(a.hidden_dim):
            raise ValueError(
                "PirateNet requires fourier_emb.embed_dim == arch.hidden_dim "
                "(the first block input is the embedding phi).")
        rp = a.get("reparam", None)
        use_rwf = rp is not None
        g = cfg.geom
        return PirateNet(
            num_blocks=int(a.num_layers), hidden_dim=int(a.hidden_dim), out_dim=int(a.out_dim),
            embed_dim=int(emb.embed_dim), embed_scale=float(emb.embed_scale),
            activation=str(a.get("activation", "tanh")),
            alpha_init=float(a.get("nonlinearity", 0.0)),
            use_rwf=use_rwf,
            reparam_mean=float(rp.mean) if use_rwf else 1.0,
            reparam_stddev=float(rp.stddev) if use_rwf else 0.1,
            x0=float(g.x_range[0]), x1=float(g.x_range[1]),
            y0=float(g.y_range[0]), y1=float(g.y_range[1]),
        )
    return MLP(num_layers=int(a.num_layers), hidden_dim=int(a.hidden_dim),
               out_dim=int(a.out_dim))


def init_params(net, key, in_dim=2):
    return net.init(key, jnp.zeros((in_dim,)))


def pi_init(net, params, points, out_cols=(0, 1)):
    """Physics-informed initialization (PirateNet eq. 4.9).

    Least-squares fit the final linear layer so that at init (alpha=0, u = W phi) the
    observed outputs best match the available DNS data. This project only has (u, v)
    data, so only those output columns are fitted; p, k, omega keep their random init.
    Returns updated params.
    """
    xs, ys = [], []
    for key in ("inlet", "outlet", "interior"):
        d = points.get(key)
        if d is not None:
            xs.append(np.asarray(d[0]))
            ys.append(np.asarray(d[1]))
    if not xs:
        return params
    X = jnp.asarray(np.concatenate(xs, 0))   # (N, 2) physical coords
    Y = jnp.asarray(np.concatenate(ys, 0))   # (N, 2) = [u, v]
    feats = jax.vmap(lambda xy: net.apply(params, xy, return_features=True))(X)  # (N, hidden)
    W = jnp.linalg.lstsq(feats, Y, rcond=None)[0]   # (hidden, len(out_cols))

    cols = jnp.asarray(list(out_cols))
    p = flax.core.unfreeze(params)
    kernel = p["params"]["output_layer"]["kernel"]
    if isinstance(kernel, (tuple, list)):            # RWF: kernel_eff = g * v
        g_fac, v_fac = kernel
        g_fac = g_fac.at[cols].set(1.0)
        v_fac = v_fac.at[:, cols].set(W)
        p["params"]["output_layer"]["kernel"] = (g_fac, v_fac)
    else:
        p["params"]["output_layer"]["kernel"] = kernel.at[:, cols].set(W)
    return flax.core.freeze(p)
