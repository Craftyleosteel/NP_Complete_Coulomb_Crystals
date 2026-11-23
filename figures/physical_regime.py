"""How far toward laboratory parameters does the pinned construction survive?

The proof needs the selector half-size b (in units of the gadget spacing a) to shrink
polynomially with the instance size. Here we fix the cube-graph instance of the paper and ask,
for finite b, whether the three properties the proof relies on still hold exactly:

  collapse   : tree-uniform energies lie on E0 + lambda * H_B   (max deviation / lambda)
  dominance  : every domain wall can be removed by a branch flip, using the proof's
               worst-case cut criterion evaluated on the actual couplings
  forcing    : no single-ion move from the ground state lowers the energy
               (pure Coulomb, and with a per-selector one-body offset that a smooth
               trap envelope or tunable well depths would supply)

Then, for finite well stiffness, we add the linear-response relaxation energy
-|F_i|^2 / (2 kappa) of ions in harmonic wells, kappa = (omega_w / omega_a)^2, and convert to
Sr+ at the Doppler limit.

Lengths are in units of a, energies in units of E_C = e^2 / (4 pi eps0 a).
"""
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simulations import V, EDGES, build_instance, couplings, selector_points  # noqa: E402

D, S = 8, 12
N_V = len(V)


def compensate(c, b, role, newton=8):
    M = len(c)
    eta = np.zeros(M)
    J, H, const = couplings(c, b, eta)
    lam = None
    for _ in range(newton):
        lam = box_lambda(J)
        g = np.where(role == 1, lam, 0.0)
        # exact internal field B(eta) = (1/(2 sqrt2))[(b+eta)^-1 - b^-1]/2 ; Newton on eta
        beta = -1.0 / (4 * np.sqrt(2) * (b + eta) ** 2)
        eta = eta + (g - H) / beta
        eta = np.clip(eta, -0.5 * b, 0.5 * b)
        J, H, const = couplings(c, b, eta)
    lam = box_lambda(J)
    g = np.where(role == 1, lam, 0.0)
    return eta, J, H, const, lam, g


BOXES = None


def box_lambda(J):
    return np.mean([J[np.ix_(p, q)].sum() for p, q in BOXES])


def tree_states():
    out = []
    for code in range(2 ** N_V):
        sig = np.array([1 if (code >> k) & 1 else -1 for k in range(N_V)])
        hb = sum(sig[u] * sig[v] for u, v in EDGES) + sig.sum()
        out.append((sig, hb))
    return out


def tree_graph(c, tree):
    """unit-distance adjacency inside each tree; returns parent pointers toward the root"""
    M = len(c)
    d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
    adj = (np.abs(d - 1) < 1e-9) & (tree[:, None] == tree[None])
    roots = list(range(N_V))
    parent = -np.ones(M, int)
    order = []
    for r in roots:
        seen = {r}
        stack = [r]
        while stack:
            u = stack.pop()
            order.append(u)
            for w in np.nonzero(adj[u])[0]:
                if w not in seen:
                    seen.add(w)
                    parent[w] = u
                    stack.append(w)
    assert (parent >= 0).sum() == M - N_V, "trees not connected"
    nedges = adj.sum() // 2
    assert nedges == M - N_V, f"tree graph has cycles: {nedges} vs {M - N_V}"
    return parent, order


def subtree_masks(parent, order):
    M = len(parent)
    children = [[] for _ in range(M)]
    for v in range(M):
        if parent[v] >= 0:
            children[parent[v]].append(v)
    masks = {}
    for v in reversed(order):
        m = np.zeros(M, bool)
        m[v] = True
        for w in children[v]:
            m |= masks[w]
        masks[v] = m
    return masks


def dominance_margin(J, H, g, parent, masks):
    """min over tree edges (a=parent, v) of |J_av| - sum of other |J| across the cut - residual fields in C"""
    A = np.abs(J)
    rf = np.abs(H - g)
    worst, worst_rel = np.inf, np.inf
    for v, m in masks.items():
        a = parent[v]
        if a < 0:
            continue
        cross = A[np.ix_(m, ~m)].sum() - A[v, a]
        margin = A[v, a] - cross - rf[m].sum()
        if margin < worst:
            worst = margin
            worst_rel = margin / A[v, a]
    return worst, worst_rel


def ion_positions(pts, s):
    """occupied ion coordinates for selector spins s (+1 -> state 0)"""
    st = np.where(s > 0, 0, 1)
    return pts[np.arange(len(s)), st].reshape(-1, 2)


def mono_potential(c, sites, per_site):
    """potential of the other selectors' charge 2 (at their centers), at each site.
    per_site=False evaluates it at the selector center and repeats it over the four sites, so it
    shifts both legal states equally; per_site=True evaluates it at each site."""
    M = len(c)
    own = np.repeat(np.arange(M), 4)
    at = sites if per_site else np.repeat(c, 4, axis=0)
    d = np.sqrt(((at[:, None] - c[None]) ** 2).sum(-1))
    d[np.arange(4 * M), own] = np.inf
    return (2 / d).sum(1)


def forcing_margin(pts, s, off):
    """cheapest single-ion move from configuration s, with per-site one-body offsets off (4M)"""
    M = len(s)
    ions = ion_positions(pts, s)
    sites = pts.reshape(-1, 2)                      # (4M, 2): selector i -> rows 4i..4i+3, + diagonal first
    occ = np.zeros(4 * M, bool)
    st = np.where(s > 0, 0, 1)
    for i in range(M):
        occ[4 * i + 2 * st[i]] = occ[4 * i + 2 * st[i] + 1] = True
    d = np.sqrt(((sites[:, None] - ions[None]) ** 2).sum(-1))
    with np.errstate(divide="ignore"):
        inv = np.where(d > 1e-12, 1 / d, 0.0)
    phi = inv.sum(1) + off                          # energy of an ion at each site (self excluded)
    occ_idx = np.nonzero(occ)[0]
    emp_idx = np.nonzero(~occ)[0]
    dst = np.sqrt(((sites[emp_idx][None] - sites[occ_idx][:, None]) ** 2).sum(-1))
    dE = phi[emp_idx][None] - phi[occ_idx][:, None] - 1 / dst
    same = (occ_idx[:, None] // 4) == (emp_idx[None] // 4)
    return dict(min=dE.min(), intra=dE[same].min(), inter=dE[~same].min())


def relax_energy(pts, s, c, subtract_mono):
    """sum |F_i|^2 over ions; in harmonic wells of stiffness kappa the ions relax and lower the
    energy by this / (2 kappa) (linear response). With subtract_mono, the smooth envelope that
    cancels the mean potential also cancels its force: F_i -> F_i - F_mono(x_i)."""
    x = ion_positions(pts, s)
    d = x[:, None] - x[None]
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    F = (d / r[..., None] ** 3).sum(1)
    if subtract_mono:
        own = np.repeat(np.arange(len(c)), 2)
        dc = x[:, None] - c[None]
        rc = np.sqrt((dc ** 2).sum(-1))
        rc[np.arange(len(x)), own] = np.inf
        F = F - (2 * dc / rc[..., None] ** 3).sum(1)
    return (F ** 2).sum()


def stiffness_study(b, mode, kappas):
    """max deviation from E0 + lambda' H_B over the 2^8 tree states, after re-fitting one field per
    tree (re-calibrating the field compensation), as a function of the well stiffness kappa"""
    c, tree, role, boxes = build_instance(S, D)
    r = analyze(b, mode=mode)
    global BOXES
    BOXES = boxes
    eta = np.zeros(len(c)) if mode != "coulomb" else None
    if eta is None:
        eta, *_ = compensate(c, b, role)
    pts = selector_points(c, b, eta)
    states = tree_states()
    HB = np.array([hb for _, hb in states])
    sig = np.array([s for s, _ in states], float)
    E = r["E_states"]
    F2 = np.array([relax_energy(pts, s[tree], c, mode != "coulomb") for s in sig])
    X = np.column_stack([np.ones_like(HB), HB, sig])            # c0, lambda', one field per tree
    out = []
    for k in kappas:
        Ek = E - F2 / (2 * k)
        coef, *_ = np.linalg.lstsq(X, Ek, rcond=None)
        lam_k = coef[1]
        dev = np.abs(Ek - X @ coef).max() / abs(lam_k)
        coef0, *_ = np.linalg.lstsq(X[:, :2], Ek, rcond=None)
        dev_nofit = np.abs(Ek - X[:, :2] @ coef0).max() / abs(coef0[1])
        out.append((k, dev, dev_nofit, lam_k / r["lam"]))
    return r, out


MODES = {
    "coulomb":  "pure Coulomb sites; fields cancelled by diagonal stretches (the proof's model)",
    "selector": "+ one depth offset per selector (cancels the mean potential) and field-cancelling "
                "depth offsets on each + diagonal",
    "site":     "+ every well depth set to cancel the mean Coulomb potential at that site",
}


def analyze(b, mode="coulomb"):
    """'coulomb': the instance of the theorem. 'selector'/'site': eta = 0, and one-body well-depth
    offsets do the field compensation (a depth w_i on the two + sites adds w_i (1 + sigma_i)) and
    flatten the mean potential, leaving the geometry and every coupling J untouched."""
    global BOXES
    c, tree, role, boxes = build_instance(S, D)
    BOXES = boxes
    M = len(c)
    if mode == "coulomb":
        eta, J, H, const, lam, g = compensate(c, b, role)
        off = np.zeros(4 * M)
    else:
        eta = np.zeros(M)
        J, H, const = couplings(c, b, eta)
        lam = box_lambda(J)
        g = np.where(role == 1, lam, 0.0)
        sites = selector_points(c, b, eta).reshape(-1, 2)
        off = -mono_potential(c, sites, per_site=(mode == "site"))
        o = off.reshape(M, 4)
        H = H + (o[:, 0] + o[:, 1] - o[:, 2] - o[:, 3]) / 2      # offsets seen by the legal states
        const = const + ((o[:, 0] + o[:, 1] + o[:, 2] + o[:, 3]) / 2).sum()
        w = g - H                                                 # field-cancelling depth on + sites
        off = off + np.stack([w, w, 0 * w, 0 * w], 1).reshape(-1)
        const, H = const + w.sum(), g.copy()
    resid_field = np.abs(H - g).sum()
    pts = selector_points(c, b, eta)
    HB, E = [], []
    states = tree_states()
    for sig, hb in states:
        s = sig[tree].astype(float)
        HB.append(hb)
        E.append(const + H @ s + 0.5 * s @ J @ s)
    HB, E = np.array(HB), np.array(E)
    slope, icpt = np.polyfit(HB, E, 1)
    dev = np.abs(E - icpt - slope * HB).max() / slope
    gs = np.argmin(E)
    hb_min_ok = HB[gs] == HB.min()
    parent, order = tree_graph(c, tree)
    masks = subtree_masks(parent, order)
    dom, dom_rel = dominance_margin(J, H, g, parent, masks)
    s_gs = states[gs][0][tree].astype(float)
    fm = forcing_margin(pts, s_gs, off)
    out = dict(mode=mode, b=b, M=len(c), lam=lam, lam_fit=slope, dev_over_lam=dev, gs_is_HBmin=bool(hb_min_ok),
               resid_field_over_lam=resid_field / lam, max_eta_over_b=float(np.abs(eta).max() / b),
               dominance=dom, dominance_rel=dom_rel,
               **{f"force_{k}": float(v) for k, v in fm.items()})
    out["E_states"] = E
    return out


# ------------------------------------------------------------------ Sr+ at the Doppler limit
KB, HBAR, AMU = 1.380649e-23, 1.054571817e-34, 1.66053907e-27
COUL = 2.307077e-28                              # e^2 / (4 pi eps0), J m
M_SR = 87.9056 * AMU
GAMMA_SR = 2 * np.pi * 21.5e6                    # 5s-5p1/2 linewidth, 1/s
T_DOPPLER = HBAR * GAMMA_SR / (2 * KB)           # ~0.52 mK


def E_C(a):
    """Coulomb energy scale at gadget spacing a (m), in kelvin"""
    return COUL / a / KB


def omega_a(a):
    return np.sqrt(COUL / (M_SR * a ** 3))


def full_study():
    bs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.33, 0.35, 0.4]
    sweep = {}
    for mode in MODES:
        sweep[mode] = []
        for b in bs:
            r = analyze(b, mode=mode)
            r.pop("E_states")
            r["ok"] = bool(r["dev_over_lam"] < 0.05 and r["dominance_rel"] > 0 and r["force_min"] > 0)
            sweep[mode].append(r)
            print(mode, b, "ok" if r["ok"] else "fail", round(r["force_min"], 4), round(r["dominance_rel"], 4))
    kappas = np.logspace(0, 5, 26)
    stiff = {}
    for b in [0.1, 0.15, 0.2, 0.25, 0.3]:
        r, out = stiffness_study(b, "site", kappas)
        rows = [dict(kappa=float(k), dev_refit=float(d1), dev_norefit=float(d0), lam_ratio=float(lr))
                for k, d1, d0, lr in out]
        kstar = next(row["kappa"] for row in rows if abs(row["lam_ratio"] - 1) < 0.1 and row["dev_refit"] < 0.05)
        stiff[str(b)] = dict(lam=r["lam"], rows=rows, kappa_star=kstar)
        print("stiffness", b, "kappa*", kstar)
    phys = dict(T_doppler_K=T_DOPPLER, m_Sr=M_SR)
    return dict(b_grid=bs, sweep=sweep, stiffness=stiff, physical=phys, modes=MODES)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from simulations import UP, DOWN, AF, INK, MUTED, GRID, clean, save
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw=dict(width_ratios=[1, 1.05]))

    # (a) window of b in which the cube-graph instance still encodes H_B
    labels = {"coulomb": "pure Coulomb\n(Theorem 2)", "selector": "+ one depth\nper selector",
              "site": "+ depth\nper site"}
    colors = {"coulomb": MUTED, "selector": DOWN, "site": UP}
    bs = np.array(res["b_grid"])
    for y, mode in enumerate(["coulomb", "selector", "site"]):
        ok = np.array([r["ok"] for r in res["sweep"][mode]])
        bmax = bs[ok].max()
        nxt = bs[bs > bmax].min() if (bs > bmax).any() else bmax
        ax1.barh(y, bmax - bs[0], left=bs[0], height=0.5, color=colors[mode], lw=0)
        ax1.barh(y, nxt - bmax, left=bmax, height=0.5, color=colors[mode], alpha=0.3, lw=0)
        ax1.text(nxt + 0.006, y, f"$b\\leq{bmax:g}$", va="center", fontsize=7, color=INK)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels([labels[m] for m in ["coulomb", "selector", "site"]], fontsize=7)
    ax1.set_xlim(0, 0.45)
    ax1.set_ylim(-0.6, 3.0)
    ax1.axvline(0.25, color=GRID, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax1.text(0.25, 2.62, "sites form a half-filled\nsquare lattice", fontsize=6.3, color=MUTED,
             va="center", ha="center")
    ax1.set_xlabel(r"selector half-size $b$ (units of gadget spacing $a$)")
    ax1.set_title("(a)", loc="left")
    clean(ax1)

    # (b) laboratory window at b = 0.25 for Doppler-cooled Sr+
    b = 0.25
    st = res["stiffness"][str(b)]
    lam_EC = st["lam"]
    kstar = st["kappa_star"]
    d = np.logspace(0, np.log10(300), 200) * 1e-6            # site spacing = a / 2
    a = 2 * d
    w_min = np.sqrt(kstar) * omega_a(a) / (2 * np.pi)      # well frequency needed (Hz)
    gap = lam_EC * E_C(a) / T_DOPPLER                       # lambda / k_B T_D
    ax2.fill_between(d * 1e6, w_min / 1e6, 1e3, where=gap >= 10, color=UP, alpha=0.18, lw=0)
    ax2.plot(d * 1e6, w_min / 1e6, color=UP, lw=1.4)
    dmax = d[gap >= 10].max() * 1e6
    ax2.axvline(dmax, color=DOWN, lw=1.2, ls=(0, (4, 2)))
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1, 300)
    ax2.set_ylim(0.01, 100)
    ax2.set_xlabel(r"site spacing $a/2$ ($\mu$m)")
    ax2.set_ylabel(r"well frequency $\omega_w/2\pi$ (MHz)")
    ax2.text(1.3, 0.025, f"wells too soft:\nrelaxation shifts $\\lambda$ by $>10\\%$", fontsize=6.5, color=INK)
    ax2.text(dmax * 0.92, 30, f"$\\lambda=10\\,k_BT_D$", fontsize=6.5, color=DOWN, ha="right")
    ax2.text(12, 12, "encoding holds", fontsize=7, color=UP, ha="center")
    ax2.set_title(r"(b)  Sr$^+$ at the Doppler limit, $b=0.25$", loc="left")
    clean(ax2)
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig_physical")


if __name__ == "__main__":
    if "full" in sys.argv:
        path = HERE / "physical_regime.json"
        if "replot" in sys.argv and path.exists():
            res = json.loads(path.read_text())
        else:
            res = full_study()
            path.write_text(json.dumps(res, indent=1, default=float))
        plot(res)
        sys.exit()
    mode = next((m for m in MODES if m in sys.argv), "coulomb")
    bs = [float(x) for x in sys.argv[1:] if x not in MODES and x != "stiff"] or [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    if "stiff" in sys.argv:
        for b in [0.1, 0.25]:
            r, out = stiffness_study(b, "site", [1, 3, 10, 30, 100, 300, 1000, 1e4])
            for k, dev, dev0, lr in out:
                print(f"b={b} kappa={k:g} dev_refit={dev:.4f} dev_norefit={dev0:.4f} lam_ratio={lr:.4f}")
        sys.exit()
    for b in bs:
        r = analyze(b, mode=mode)
        r.pop("E_states")
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}))
