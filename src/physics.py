"""k-omega RANS residuals (pure JAX), Wilcox(1988), Pioch-faithful.

Network output at (x, y):  [u, v, p, k, w_raw]
  - omega_param="direct" : omega = w_raw           (Pioch 1:1, default)
  - omega_param="g"      : omega = 1 / (w_raw^2)    (g-transform, wall-stable)

Closure:  nu_t = k/omega
  tau_xx = 2 nu_t u_x - (2/3) k
  tau_yy = 2 nu_t v_y - (2/3) k
  tau_xy = nu_t (u_y + v_x)
  P      = tau_ij dU_i/dX_j

Residuals (= 0):
  R_cont  = u_x + v_y
  R_momx  = u u_x + v u_y + (1/rho) p_x - nu (u_xx+u_yy) - (d tau_xx/dx + d tau_xy/dy)
  R_momy  = u v_x + v v_y + (1/rho) p_y - nu (v_xx+v_yy) - (d tau_xy/dx + d tau_yy/dy)
  R_k     = u k_x + v k_y - P + beta* k omega - d/dx[(nu+s* nu_t) k_x] - d/dy[...]
  R_omega = u w_x + v w_y - alpha (omega/k) P + beta omega^2 - d/dx[(nu+s nu_t) w_x] - d/dy[...]

`apply_fn(params, xy)` must map a length-2 vector -> length-5 vector.
All per-point functions are scalar-valued so jax.grad nests cleanly (up to 2nd order).
"""
from functools import partial
import jax
import jax.numpy as jnp


def make_model_fns(apply_fn, cfg):
    nu = float(cfg.problem.nu)
    rho = float(cfg.problem.rho)
    bstar = float(cfg.problem.beta_star)
    sstar = float(cfg.problem.sigma_star)
    sig = float(cfg.problem.sigma)
    alpha = float(cfg.problem.alpha)
    beta = float(cfg.problem.beta)
    eps = float(cfg.problem.omega_floor)
    omega_param = str(cfg.problem.omega_param)

    # ---- field accessors (scalar outputs for autodiff) ----
    def fields(params, x, y):
        out = apply_fn(params, jnp.stack([x, y]))
        u, v, p, k, wraw = out[0], out[1], out[2], out[3], out[4]
        omega = 1.0 / (wraw ** 2 + eps) if omega_param == "g" else wraw
        return u, v, p, k, omega

    u_ = lambda pr, x, y: fields(pr, x, y)[0]
    v_ = lambda pr, x, y: fields(pr, x, y)[1]
    p_ = lambda pr, x, y: fields(pr, x, y)[2]
    k_ = lambda pr, x, y: fields(pr, x, y)[3]
    w_ = lambda pr, x, y: fields(pr, x, y)[4]   # omega (already transformed)

    dx = lambda f: jax.grad(f, argnums=1)
    dy = lambda f: jax.grad(f, argnums=2)

    ux, uy = dx(u_), dy(u_)
    vx, vy = dx(v_), dy(v_)
    px, py = dx(p_), dy(p_)
    kx, ky = dx(k_), dy(k_)
    wx, wy = dx(w_), dy(w_)

    def nu_t_(pr, x, y):
        k = jnp.maximum(k_(pr, x, y), 0.0)
        om = jnp.maximum(w_(pr, x, y), eps)
        return k / om

    # ---- Reynolds stresses (Boussinesq) ----
    def tau_xx_(pr, x, y):
        return 2.0 * nu_t_(pr, x, y) * ux(pr, x, y) - (2.0 / 3.0) * k_(pr, x, y)

    def tau_yy_(pr, x, y):
        return 2.0 * nu_t_(pr, x, y) * vy(pr, x, y) - (2.0 / 3.0) * k_(pr, x, y)

    def tau_xy_(pr, x, y):
        return nu_t_(pr, x, y) * (uy(pr, x, y) + vx(pr, x, y))

    def production_(pr, x, y):
        return (tau_xx_(pr, x, y) * ux(pr, x, y)
                + tau_xy_(pr, x, y) * (uy(pr, x, y) + vx(pr, x, y))
                + tau_yy_(pr, x, y) * vy(pr, x, y))

    # ---- diffusion fluxes for k, omega ----
    def kflux_x(pr, x, y):
        return (nu + sstar * nu_t_(pr, x, y)) * kx(pr, x, y)

    def kflux_y(pr, x, y):
        return (nu + sstar * nu_t_(pr, x, y)) * ky(pr, x, y)

    def wflux_x(pr, x, y):
        return (nu + sig * nu_t_(pr, x, y)) * wx(pr, x, y)

    def wflux_y(pr, x, y):
        return (nu + sig * nu_t_(pr, x, y)) * wy(pr, x, y)

    # second derivatives of velocity (viscous term)
    uxx, uyy = dx(ux), dy(uy)
    vxx, vyy = dx(vx), dy(vy)

    def reynolds_force(pr, x, y):
        fx = dx(tau_xx_)(pr, x, y) + dy(tau_xy_)(pr, x, y)
        fy = dx(tau_xy_)(pr, x, y) + dy(tau_yy_)(pr, x, y)
        return fx, fy

    def residuals(pr, x, y):
        u, v, p, k, omega = fields(pr, x, y)
        k_safe = jnp.maximum(k, eps)
        P = production_(pr, x, y)
        fx, fy = reynolds_force(pr, x, y)

        r_cont = ux(pr, x, y) + vy(pr, x, y)
        r_momx = (u * ux(pr, x, y) + v * uy(pr, x, y) + (1.0 / rho) * px(pr, x, y)
                  - nu * (uxx(pr, x, y) + uyy(pr, x, y)) - fx)
        r_momy = (u * vx(pr, x, y) + v * vy(pr, x, y) + (1.0 / rho) * py(pr, x, y)
                  - nu * (vxx(pr, x, y) + vyy(pr, x, y)) - fy)
        diff_k = dx(kflux_x)(pr, x, y) + dy(kflux_y)(pr, x, y)
        r_k = (u * kx(pr, x, y) + v * ky(pr, x, y) - P + bstar * k * omega - diff_k)
        diff_w = dx(wflux_x)(pr, x, y) + dy(wflux_y)(pr, x, y)
        r_omega = (u * wx(pr, x, y) + v * wy(pr, x, y)
                   - alpha * (omega / k_safe) * P + beta * omega ** 2 - diff_w)
        return {"r_cont": r_cont, "r_momx": r_momx, "r_momy": r_momy,
                "r_k": r_k, "r_omega": r_omega}

    return {
        "fields": fields,           # (u,v,p,k,omega)
        "nu_t": nu_t_,
        "production": production_,
        "reynolds_force": reynolds_force,
        "residuals": residuals,     # per-point dict of 5 residuals
        # derivative helpers (for slip-wall BC du/dy etc.)
        "u_y": uy, "k_y": ky, "w_y": wy, "u_x": ux, "v_x": vx,
    }
