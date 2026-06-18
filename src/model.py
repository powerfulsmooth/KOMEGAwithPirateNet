"""KOmegaPINN: assembles PDE/BC/data losses from physics + geometry + data points.

Network only is swapped (MLP vs PirateNet); everything here is shared.
"""
import jax
import jax.numpy as jnp
import numpy as np

from src import physics
from src import geometry
from src.losses import LOSS_KEYS


def build_points(cfg, data_points, rng=None):
    """Assemble all collocation + boundary + data points as jnp arrays (Pioch frame)."""
    coll = geometry.sample_collocation(cfg, rng)
    walls = geometry.wall_points(cfg)
    walls_noslip = np.concatenate([walls["bottom"], walls["step_top"], walls["step_face"]], 0)
    wall_top = walls["top"]

    inlet_xy, inlet_uv = data_points["inlet"]
    outlet_xy, outlet_uv = data_points["outlet"]
    interior = data_points["interior"]

    j = jnp.asarray
    pts = {
        "coll": j(coll),
        "walls_noslip": j(walls_noslip),
        "wall_top": j(wall_top),
        "inlet": (j(inlet_xy), j(inlet_uv)),
        "outlet": (j(outlet_xy), j(outlet_uv)),
        "interior": None if interior is None else (j(interior[0]), j(interior[1])),
    }
    return pts


class KOmegaPINN:
    def __init__(self, cfg, net, points):
        self.cfg = cfg
        self.net = net
        self.apply_fn = lambda p, xy: net.apply(p, xy)
        self.fns = physics.make_model_fns(self.apply_fn, cfg)
        self.pts = points

    def _vfields(self, params, xy):
        return jax.vmap(lambda pt: jnp.stack(self.fns["fields"](params, pt[0], pt[1])))(xy)

    def loss_terms(self, params):
        f = self.fns
        p = self.pts

        # --- PDE residuals at collocation points ---
        res = jax.vmap(lambda pt: f["residuals"](params, pt[0], pt[1]))(p["coll"])
        terms = {k: jnp.mean(res[k] ** 2) for k in
                 ["r_cont", "r_momx", "r_momy", "r_k", "r_omega"]}

        # --- BC: no-slip walls (u = v = 0) ---
        fns_ns = self._vfields(params, p["walls_noslip"])
        bc = jnp.mean(fns_ns[:, 0] ** 2 + fns_ns[:, 1] ** 2)

        # --- BC: slip top (v = 0, du/dy = 0) ---
        ft = self._vfields(params, p["wall_top"])
        uy = jax.vmap(lambda pt: f["u_y"](params, pt[0], pt[1]))(p["wall_top"])
        bc = bc + jnp.mean(ft[:, 1] ** 2 + uy ** 2)

        # --- BC: inlet Dirichlet (U, V) ---
        xi, ui = p["inlet"]
        fi = self._vfields(params, xi)
        bc = bc + jnp.mean((fi[:, 0] - ui[:, 0]) ** 2 + (fi[:, 1] - ui[:, 1]) ** 2)

        # --- BC: outlet (Dirichlet DNS  |  Neumann d/dx = 0) ---
        xo, uo = p["outlet"]
        if self.cfg.bc.outlet == "neumann":
            ux = jax.vmap(lambda pt: f["u_x"](params, pt[0], pt[1]))(xo)
            vx = jax.vmap(lambda pt: f["v_x"](params, pt[0], pt[1]))(xo)
            bc = bc + jnp.mean(ux ** 2 + vx ** 2)
        else:
            fo = self._vfields(params, xo)
            bc = bc + jnp.mean((fo[:, 0] - uo[:, 0]) ** 2 + (fo[:, 1] - uo[:, 1]) ** 2)
        terms["bc"] = bc

        # --- data: interior lines (U, V) ---
        if p["interior"] is not None:
            xd, ud = p["interior"]
            fd = self._vfields(params, xd)
            terms["data"] = jnp.mean((fd[:, 0] - ud[:, 0]) ** 2 + (fd[:, 1] - ud[:, 1]) ** 2)
        else:
            terms["data"] = jnp.asarray(0.0)
        return terms

    def total_loss(self, params, weights):
        terms = self.loss_terms(params)
        total = sum(weights[k] * terms[k] for k in LOSS_KEYS)
        return total, terms

    def predict_grid(self, params, xy):
        """(N,2) -> (N,5) fields [u,v,p,k,omega] for postprocessing."""
        return self._vfields(params, jnp.asarray(xy))
