"""Loss weighting schemes: uniform | grad_norm | ntk  (selectable via config)."""
import jax
import jax.numpy as jnp

LOSS_KEYS = ["r_cont", "r_momx", "r_momy", "r_k", "r_omega", "bc", "data"]


def global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def _grad_norms(loss_terms_fn, params, keys):
    norms = {}
    for k in keys:
        g = jax.grad(lambda p: loss_terms_fn(p)[k])(params)
        norms[k] = global_norm(g)
    return norms


def update_weights(loss_terms_fn, params, scheme, weights, momentum, keys=LOSS_KEYS):
    """Return updated per-term weights (EMA).

    uniform   : weights unchanged.
    grad_norm : w_i = (sum_j ||grad L_j||) / ||grad L_i||      (Wang et al.)
    ntk       : NTK-inspired proxy, tr(K_i) ~ ||grad L_i||^2   (refine to full diag-NTK
                with per-point residual jacobians if needed).
    """
    if scheme == "uniform":
        return weights
    norms = _grad_norms(loss_terms_fn, params, keys)
    if scheme == "grad_norm":
        total = sum(norms.values())
        new = {k: total / (norms[k] + 1e-8) for k in keys}
    elif scheme == "ntk":
        sq = {k: norms[k] ** 2 for k in keys}
        total = sum(sq.values())
        new = {k: total / (sq[k] + 1e-8) for k in keys}
    else:
        return weights
    return {k: momentum * weights[k] + (1.0 - momentum) * new[k] for k in keys}
