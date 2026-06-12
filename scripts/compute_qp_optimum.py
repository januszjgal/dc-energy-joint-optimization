"""Offline full-information optimum (peer-review M3): a convex QP lower bound.

The environment's episode cost is convex in the serving decisions: power is
linear in served CPU, energy cost linear in power, the peak penalty a convex
quadratic (alpha * d * g^2), and the service-backlog / batch-queue dynamics
are linear. A clairvoyant planner with perfect information over the full
31-day episode and unconstrained (fluid) allocation therefore solves a QP
whose optimum LOWER-BOUNDS the cost of any policy in the environment.
Reporting PPO's gap to this bound replaces "beats the best heuristic we
wrote" with a defensible optimality statement.

Formulation (batch mode). Variables per (t, i), all nonnegative:
  asg  service assigned to site i at t          (sum_i asg[t,i] = D_t)
  s    service served                            B = backlog state
  q    batch volume from ORIGIN stream i served anywhere at t
  C    running sum of q (auxiliary, keeps the deadline constraints sparse)
  y    batch executed at SITE j at t             (sum_j y = sum_i q per step)
  u = s + y <= kappa_i

Deadline feasibility per origin stream (single offset per stream => FIFO/
agreeable windows; EDF realizes any cumulative-feasible schedule):
  A_i(t - off_i) <= C[t,i] <= A_i(t),   A_i = cumsum of arrivals.

Objective (matches MultiDCEnv exactly, incl. always-paid idle power):
  sum [ pi * g * 1000 * dt_h + alpha * d * g^2 + lambda_b * B ],
  g = (idle_i + slope_i * u) * R.

Implementation note: cvxpy's conic reduction overflows 32-bit ints on
Windows at this scale, so the sparse QP is assembled directly for the
Clarabel solver; a small-T cvxpy cross-check validates the assembly
(--validate).

Usage:
  python scripts/compute_qp_optimum.py                 # all four configs
  python scripts/compute_qp_optimum.py --validate      # T=240 cross-check
Writes output/review_campaign/qp_optimum.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate import _make_env  # noqa: E402

ALPHA = 0.015
DT_H = 5.0 / 60.0
CONFIGS = {
    "us_spatial": ("env/scenarios/us_model.yaml", False),
    "us_batch": ("env/scenarios/us_model.yaml", True),
    "global_spatial": ("env/scenarios/global_model.yaml", False),
    "global_batch": ("env/scenarios/global_model.yaml", True),
}


def build_inputs(scenario: str, batch: bool, t_cap: int | None = None):
    env = _make_env(Path(scenario), batch_enabled=batch, peak_penalty_weight=ALPHA)
    T = env.max_steps if t_cap is None else min(t_cap, env.max_steps)
    N = env.n_dc
    sites = env.sites
    pi = np.array([[s.get_price(t) for s in sites] for t in range(T)])
    d = np.array([[s.get_net_demand(t) for s in sites] for t in range(T)])
    kappa = np.array([s.capacity for s in sites])
    idle = np.array([(s.power_model or env.power_model).idle_power for s in sites])
    slope = np.array([(s.power_model or env.power_model).slope for s in sites])
    R = np.array([s.rated_power_mw for s in sites])
    if batch:
        svc = np.array([[s.get_service_demand(t) for s in sites] for t in range(T)])
        arr = np.array([[s.get_batch_demand(t) for s in sites] for t in range(T)])
        off = np.array(env._deadline_offsets)
    else:
        svc = np.array([[s.get_local_demand(t) for s in sites] for t in range(T)])
        arr = np.zeros((T, N))
        off = np.zeros(N, dtype=int)
    return dict(T=T, N=N, pi=pi, d=d, kappa=kappa, idle=idle, slope=slope, R=R,
                svc=svc, arr=arr, off=off, lam_b=env.backlog_weight)


# ----------------------------------------------------------------------
# Direct sparse assembly for Clarabel:  min 1/2 x'Px + c'x
#   s.t. A_eq x = b_eq (ZeroCone),  A_in x <= b_in (NonnegativeCone)
# Variable layout (column blocks of size T*N each, flattened t-major):
#   batch :  [asg | s | B | q | C | y]      n = 6*T*N
#   spatial: [asg | s | B]                  n = 3*T*N  (u = s)
# ----------------------------------------------------------------------

def idx(block: int, t: np.ndarray, i: np.ndarray, T: int, N: int) -> np.ndarray:
    return block * T * N + t * N + i


def solve_clarabel(inp: dict, batch: bool) -> dict:
    import clarabel

    T, N = inp["T"], inp["N"]
    TN = T * N
    nblocks = 6 if batch else 3
    n = nblocks * TN
    D = inp["svc"].sum(axis=1)

    tt, ii = np.meshgrid(np.arange(T), np.arange(N), indexing="ij")
    tt, ii = tt.ravel(), ii.ravel()
    ASG, S, B = 0, 1, 2
    Q, C, Y = 3, 4, 5

    # ---- objective ----
    # g = idle*R + slope*R*u ;  u = s (+ y)
    sR = (inp["slope"] * inp["R"])[ii]          # per (t,i)
    iR = (inp["idle"] * inp["R"])[ii]
    pi_f = inp["pi"].ravel()
    d_f = inp["d"].ravel()
    # energy: pi*1000*dt*g  -> linear in u
    lin_u = pi_f * 1000.0 * DT_H * sR
    const_cost = float(np.sum(pi_f * 1000.0 * DT_H * iR))
    # peak: alpha*d*g^2 = alpha*d*(iR + sR*u)^2
    #   quad coef on u^2: alpha*d*sR^2 ; linear: 2*alpha*d*iR*sR ; const: alpha*d*iR^2
    qcoef = ALPHA * d_f * sR**2
    lin_u = lin_u + 2.0 * ALPHA * d_f * iR * sR
    const_cost += float(np.sum(ALPHA * d_f * iR**2))

    c = np.zeros(n)
    s_idx = idx(S, tt, ii, T, N)
    c[s_idx] += lin_u
    c[idx(B, tt, ii, T, N)] += inp["lam_b"]
    rows_P, cols_P, vals_P = [s_idx], [s_idx], [2.0 * qcoef]  # 1/2 x'Px convention
    if batch:
        y_idx = idx(Y, tt, ii, T, N)
        c[y_idx] += lin_u
        rows_P += [y_idx, s_idx, y_idx]
        cols_P += [y_idx, y_idx, s_idx]
        vals_P += [2.0 * qcoef, 2.0 * qcoef, 2.0 * qcoef]
    P = sp.csc_matrix(
        (np.concatenate(vals_P), (np.concatenate(rows_P), np.concatenate(cols_P))),
        shape=(n, n))
    P = sp.triu(P).tocsc()

    # ---- equality constraints ----
    eq_r, eq_c, eq_v, b_eq = [], [], [], []
    row = 0
    # E1: sum_i asg[t,i] = D_t
    for t in range(T):
        eq_r += [row] * N
        eq_c += list(idx(ASG, np.full(N, t), np.arange(N), T, N))
        eq_v += [1.0] * N
        b_eq.append(D[t])
        row += 1
    # E2: B[t] - B[t-1] - asg[t] + s[t] = 0
    r = np.arange(TN) + row
    eq_r += [r, r, r]
    eq_c += [idx(B, tt, ii, T, N), idx(ASG, tt, ii, T, N), idx(S, tt, ii, T, N)]
    eq_v += [np.ones(TN), -np.ones(TN), np.ones(TN)]
    prev = tt > 0
    eq_r += [r[prev]]
    eq_c += [idx(B, tt[prev] - 1, ii[prev], T, N)]
    eq_v += [-np.ones(prev.sum())]
    b_eq += [0.0] * TN
    row += TN
    if batch:
        # E3: C[t] - C[t-1] - q[t] = 0
        r = np.arange(TN) + row
        eq_r += [r, r]
        eq_c += [idx(C, tt, ii, T, N), idx(Q, tt, ii, T, N)]
        eq_v += [np.ones(TN), -np.ones(TN)]
        eq_r += [r[prev]]
        eq_c += [idx(C, tt[prev] - 1, ii[prev], T, N)]
        eq_v += [-np.ones(prev.sum())]
        b_eq += [0.0] * TN
        row += TN
        # E4: sum_j y[t,j] - sum_i q[t,i] = 0
        for t in range(T):
            eq_r += [row] * (2 * N)
            eq_c += (list(idx(Y, np.full(N, t), np.arange(N), T, N))
                     + list(idx(Q, np.full(N, t), np.arange(N), T, N)))
            eq_v += [1.0] * N + [-1.0] * N
            b_eq.append(0.0)
            row += 1

    def flat(parts):
        out = []
        for p in parts:
            out.append(np.asarray(p, dtype=float if parts is eq_v else int).ravel())
        return np.concatenate(out)

    A_eq = sp.csc_matrix(
        (np.concatenate([np.atleast_1d(np.asarray(v, dtype=float)) for v in eq_v]),
         (np.concatenate([np.atleast_1d(np.asarray(x, dtype=np.int64)) for x in eq_r]),
          np.concatenate([np.atleast_1d(np.asarray(x, dtype=np.int64)) for x in eq_c]))),
        shape=(row, n))

    # ---- inequality constraints  A_in x <= b_in ----
    in_r, in_c, in_v, b_in = [], [], [], []
    rowi = 0
    # I1: s + y <= kappa
    r = np.arange(TN) + rowi
    in_r += [r]; in_c += [idx(S, tt, ii, T, N)]; in_v += [np.ones(TN)]
    if batch:
        in_r += [r]; in_c += [idx(Y, tt, ii, T, N)]; in_v += [np.ones(TN)]
    b_in += list(inp["kappa"][ii])
    rowi += TN
    if batch:
        Acum = np.cumsum(inp["arr"], axis=0)
        # I2: C[t,i] <= A_i(t)
        r = np.arange(TN) + rowi
        in_r += [r]; in_c += [idx(C, tt, ii, T, N)]; in_v += [np.ones(TN)]
        b_in += list(Acum.ravel())
        rowi += TN
        # I3: -C[t,i] <= -A_i(t-off_i)   for t >= off_i
        mask = tt >= inp["off"][ii]
        r = np.arange(mask.sum()) + rowi
        in_r += [r]; in_c += [idx(C, tt[mask], ii[mask], T, N)]
        in_v += [-np.ones(mask.sum())]
        b_in += list(-Acum[tt[mask] - inp["off"][ii[mask]], ii[mask]])
        rowi += int(mask.sum())
    # I4: x >= 0  ->  -x <= 0
    r = np.arange(n) + rowi
    in_r += [r]; in_c += [np.arange(n)]; in_v += [-np.ones(n)]
    b_in += [0.0] * n
    rowi += n

    A_in = sp.csc_matrix(
        (np.concatenate([np.asarray(v, dtype=float).ravel() for v in in_v]),
         (np.concatenate([np.asarray(x, dtype=np.int64).ravel() for x in in_r]),
          np.concatenate([np.asarray(x, dtype=np.int64).ravel() for x in in_c]))),
        shape=(rowi, n))

    A = sp.vstack([A_eq, A_in]).tocsc()
    b = np.concatenate([np.asarray(b_eq, dtype=float), np.asarray(b_in, dtype=float)])
    cones = [clarabel.ZeroConeT(A_eq.shape[0]), clarabel.NonnegativeConeT(A_in.shape[0])]

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    t0 = time.time()
    solver = clarabel.DefaultSolver(P, c, A, b, cones, settings)
    sol = solver.solve()
    status = str(sol.status)
    obj = float(sol.obj_val) + const_cost
    return {
        "status": status,
        "optimum_total": obj,
        "objective_constant_idle_and_phi": const_cost,
        "n_vars": n,
        "n_constraints": int(A.shape[0]),
        "solve_seconds": round(time.time() - t0, 1),
    }


def solve_cvxpy_small(inp: dict, batch: bool) -> float:
    """Reference implementation for cross-validation at small T."""
    import cvxpy as cp
    T, N = inp["T"], inp["N"]
    D = inp["svc"].sum(axis=1)
    asg = cp.Variable((T, N), nonneg=True)
    s = cp.Variable((T, N), nonneg=True)
    B = cp.Variable((T, N), nonneg=True)
    cons = [cp.sum(asg, axis=1) == D,
            B[0, :] == asg[0, :] - s[0, :],
            B[1:, :] == B[:-1, :] + asg[1:, :] - s[1:, :]]
    if batch:
        q = cp.Variable((T, N), nonneg=True)
        y = cp.Variable((T, N), nonneg=True)
        cons += [cp.sum(y, axis=1) == cp.sum(q, axis=1)]
        A = np.cumsum(inp["arr"], axis=0)
        cum_q = cp.cumsum(q, axis=0)
        for i in range(N):
            off_i = int(inp["off"][i])
            cons += [cum_q[:, i] <= A[:, i]]
            if off_i < T:
                cons += [cum_q[off_i:, i] >= A[: T - off_i, i]]
        u = s + y
    else:
        u = s
    cons += [u <= np.tile(inp["kappa"], (T, 1))]
    g = cp.multiply(u, np.tile(inp["slope"] * inp["R"], (T, 1))) + \
        np.tile(inp["idle"] * inp["R"], (T, 1))
    obj = (cp.sum(cp.multiply(inp["pi"] * 1000.0 * DT_H, g))
           + ALPHA * cp.sum(cp.multiply(inp["d"], cp.square(g)))
           + inp["lam_b"] * cp.sum(B))
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    return float(prob.value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--validate", action="store_true",
                    help="cross-check manual assembly vs cvxpy at T=240")
    args = ap.parse_args()

    if args.validate:
        for key in ["us_spatial", "us_batch"]:
            scenario, batch = CONFIGS[key]
            inp = build_inputs(scenario, batch, t_cap=240)
            ours = solve_clarabel(inp, batch)["optimum_total"]
            ref = solve_cvxpy_small(inp, batch)
            rel = abs(ours - ref) / max(abs(ref), 1)
            print(f"[validate {key}] manual={ours:,.2f} cvxpy={ref:,.2f} "
                  f"rel_diff={rel:.2e} {'OK' if rel < 1e-5 else 'MISMATCH'}")
        return

    stats_path = ROOT / "output" / "review_campaign" / "stats.json"
    ppo_means = {}
    if stats_path.exists():
        st = json.load(open(stats_path, encoding="utf-8"))
        ppo_means = {k: v["per_algo"]["ppo"]["mean"] for k, v in st.items()
                     if "ppo" in v.get("per_algo", {})}

    results = {}
    for key in args.configs:
        scenario, batch = CONFIGS[key]
        print(f"[{key}] building inputs ...", flush=True)
        inp = build_inputs(scenario, batch)
        print(f"[{key}] solving (T={inp['T']}, batch={batch}) ...", flush=True)
        res = solve_clarabel(inp, batch)
        if key in ppo_means and res["optimum_total"] > 0:
            res["ppo_mean"] = ppo_means[key]
            res["ppo_optimality_gap_pct"] = float(
                100 * (ppo_means[key] - res["optimum_total"]) / res["optimum_total"])
        results[key] = res
        print(f"[{key}] {res['status']}  optimum={res['optimum_total']:,.0f}  "
              f"({res['solve_seconds']}s, {res['n_vars']:,} vars)"
              + (f"  PPO gap={res['ppo_optimality_gap_pct']:.2f}%"
                 if "ppo_optimality_gap_pct" in res else ""), flush=True)

    out = ROOT / "output" / "review_campaign" / "qp_optimum.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
