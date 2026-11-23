"""Single-ion barriers in the pinned cube-graph instance (Sec. IV B of the paper).

Rebuilds the b = 0.02 instance of simulations.py (same layout and field compensation) and resolves
the single-selector-flip route of Fig. 7 into single-ion moves:

* route states: the 681 domain-wall configurations visited by flipping each tree from a leaf;
* each reversal passes through one of four intermediate placements (either ion of the occupied
  diagonal moves first, to either vacant corner); the lowest of the four is that reversal's saddle;
* upper bound U: the highest saddle along the route, relative to the starting configuration;
* lower bound L: every path between the endpoints must visit a single-ion neighbour of each
  endpoint, so the barrier is at least the larger of the two lowest neighbour energies;
* every route state is checked against all 1360 x 1360 single-ion moves (strict local minimum).

Energies are direct Coulomb pair sums in units of e^2/(4 pi eps0 a), a = gadget spacing, and are
also reported in units of the fitted energy unit lambda. Floating-point (double precision)
computation, not an interval-certified bound.

Usage: python single_ion_barriers.py          -> single_ion_barriers.json
"""
import json
import pathlib
import time

import numpy as np

from simulations import (V, VERTS, EDGES, SPIN_GS, build_instance, selector_points, couplings)

HERE = pathlib.Path(__file__).resolve().parent
B, D, S = 0.02, 8, 12                       # demonstration parameters of the pinned instance


def instance():
    """the b = 0.02 instance with four Newton steps of field compensation, as in simulations.py"""
    c, tree, role, boxes = build_instance(S, D)
    M = len(c)
    eta = np.zeros(M)
    J, H, const = couplings(c, B, eta)
    lam_box = np.mean([J[np.ix_(p, q)].sum() for p, q in boxes])
    g = np.where(role == 1, lam_box, 0.0)
    beta = -1.0 / (4 * np.sqrt(2) * B ** 2)
    for _ in range(4):
        eta = eta + (g - H) / beta
        J, H, const = couplings(c, B, eta)
    # fitted energy unit: slope of the 256 tree-uniform energies against H_B (Fig. 6)
    n = len(V)
    HB, E = [], []
    for code in range(2 ** n):
        sig = np.array([1 if (code >> k) & 1 else -1 for k in range(n)])
        s = sig[tree].astype(float)
        E.append(const + H @ s + 0.5 * s @ J @ s)
        HB.append(sum(sig[u] * sig[v] for u, v in EDGES) + sig.sum())
    lam_fit = np.polyfit(HB, E, 1)[0]
    pts = selector_points(c, B, eta)                       # (M, state, point, xy)
    return c, tree, J, lam_fit, pts


def route(J, tree):
    """order of single-selector flips of Fig. 7: each tree from a leaf, breadth first"""
    b4 = B ** 4
    M = len(tree)
    adj = {i: [] for i in range(M)}
    for i in range(M):
        for j in np.nonzero(J[i] < -40 * b4)[0]:
            if i < j and tree[i] == tree[j]:
                adj[i].append(j)
                adj[j].append(i)
    order = []
    for v in range(len(V)):
        cells = [i for i in range(M) if tree[i] == v]
        leaf = next(i for i in cells if len(adj[i]) == 1)
        seen, queue = {leaf}, [leaf]
        while queue:
            i = queue.pop(0)
            order.append(i)
            for j in adj[i]:
                if j not in seen:
                    seen.add(j)
                    queue.append(j)
    return order


def main():
    t0 = time.time()
    c, tree, J, lam, pts = instance()
    M = len(c)
    sites = pts.reshape(-1, 2)                              # site 4i+2k+p: cell i, state k, point p
    d = sites[:, None, :] - sites[None, :, :]
    K = 1.0 / np.sqrt((d ** 2).sum(-1) + np.eye(len(sites)))
    np.fill_diagonal(K, 0.0)                                # pair kernel 1/r between sites

    def ion_sites(state):                                   # state[i] = 0 (+ diagonal) or 1 (-)
        return (4 * np.arange(M) + 2 * state)[:, None] + np.array([0, 1])[None, :]

    sig0 = np.array([SPIN_GS[v] for v in VERTS])[tree]
    state = np.where(sig0 > 0, 0, 1)
    order = route(J, tree)
    occ = np.zeros(len(sites), bool)
    occ[ion_sites(state).ravel()] = True
    Vpot = K[:, occ].sum(1)                                 # Coulomb potential at every site
    E_rel = 0.0                                             # energy relative to the starting state

    def neighbour_min():
        o, e = np.nonzero(occ)[0], np.nonzero(~occ)[0]
        dE = Vpot[e][None, :] - Vpot[o][:, None] - K[np.ix_(o, e)]
        return float(dE.min())

    def move(a, t):                                         # single-ion move a -> t, update state
        nonlocal E_rel, Vpot
        dE = Vpot[t] - Vpot[a] - K[a, t]
        occ[a], occ[t] = False, True
        Vpot = Vpot - K[:, a] + K[:, t]
        E_rel += dE
        return dE

    start_min = neighbour_min()
    min_margin = start_min
    saddles, route_E = [], [0.0]
    for i in order:
        held = ion_sites(state)[i]
        empty = 4 * i + 2 * (1 - state[i]) + np.array([0, 1])
        inter = [E_rel + Vpot[t] - Vpot[a] - K[a, t] for a in held for t in empty]
        saddles.append(min(inter))
        # carry out the reversal (the order of the two moves does not change the end state)
        move(held[0], empty[0])
        move(held[1], empty[1])
        state[i] = 1 - state[i]
        route_E.append(E_rel)
        min_margin = min(min_margin, neighbour_min())
    end_min = neighbour_min()
    saddles = np.array(saddles)
    U = float(saddles.max())
    L = float(max(start_min, E_rel + end_min))
    side_pair = (0.5 - 1 / (2 * np.sqrt(2))) / B
    out = dict(
        b=B, D=D, S=S, selectors=M, ions=2 * M, sites=len(sites), route_states=len(route_E),
        lambda_fit=float(lam), units="e^2/(4 pi eps0 a), a = gadget spacing",
        lower_bound=L, upper_bound=U, lower_over_lambda=L / lam, upper_over_lambda=U / lam,
        highest_saddle_reversal=int(saddles.argmax()) + 1,
        all_route_states_strict_local_minima=bool(min_margin > 0),
        smallest_single_ion_move_from_route_states=float(min_margin),
        end_minus_start_over_lambda=float(E_rel / lam),
        side_pair_cost=float(side_pair), upper_over_side_pair=U / side_pair,
        route_energies_over_lambda=[float(x / lam) for x in route_E],
        saddles_over_lambda=[float(x / lam) for x in saddles],
        precision="IEEE double; not interval-certified",
        runtime_s=round(time.time() - t0, 1),
    )
    (HERE / "single_ion_barriers.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if not isinstance(v, list):
            print(f"{k:42s} {v}")


if __name__ == "__main__":
    main()
