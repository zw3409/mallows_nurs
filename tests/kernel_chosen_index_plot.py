import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from nurs import logsumexp2, sample_unif, sub_block_stops_log, apply_perm
from distance import dist, invert, DIST_NAME_TO_ID

def _apply_pub_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (7.5, 4.75),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

def nurs_kernel_chosen_index(n, start, beta, eps, max_doublings, rho, dist_id):
    leps = math.log(eps)
    left = start.copy(); right = start.copy(); best = start.copy()
    best_chosen_idx = 0
    tot = -beta * dist(dist_id, start)
    fwd = rho; bwd = invert(fwd)
    lpos = 0; rpos = 0
    perm_buffer = np.empty(n, dtype=np.int64)
    doubling_bits = np.random.randint(0, 2, size=max_doublings)
    for j in range(max_doublings):
        ext = 1 << j
        grow = doubling_bits[j] == 1
        anchor = right if grow else left
        anchor_off = rpos if grow else lpos
        step = fwd if grow else bwd
        perm = anchor.copy()
        ext_logw = np.empty(ext, dtype=np.float64)
        for t in range(ext):
            apply_perm(perm_buffer, perm, step)
            perm, perm_buffer = perm_buffer, perm
            ext_logw[t] = -beta * dist(dist_id, perm)
        if sub_block_stops_log(ext_logw, leps):
            break
        last = perm.copy()
        perm = anchor.copy()
        for t in range(ext):
            apply_perm(perm_buffer, perm, step)
            perm, perm_buffer = perm_buffer, perm
            pos = anchor_off + (t + 1) if grow else anchor_off - (t + 1)
            lw = ext_logw[t]
            nt = logsumexp2(tot, lw)
            if np.random.random() < math.exp(min(0.0, lw - nt)):
                best = perm.copy()
                best_chosen_idx = pos
            tot = nt
        if grow:
            right = last; rpos += ext
        else:
            left = last; lpos -= ext
        if max(-beta * dist(dist_id, left), -beta * dist(dist_id, right)) <= leps + tot:
            break
    return best, best_chosen_idx

def sample_chosen_index(n, sigma0, beta, eps, max_doublings, dist_id):
    s0 = np.fromiter(sigma0, dtype=np.int64, count=n)
    rho = sample_unif(n)
    nxt, chosen_idx = nurs_kernel_chosen_index(n, s0, beta, eps, max_doublings, rho, dist_id)
    return tuple(int(x) for x in nxt), int(chosen_idx)

def run_chain(n, steps, beta, eps, max_doublings, dist_id):
    st = tuple(range(n))
    idxs = np.empty(steps, dtype=np.int64)
    for k in range(steps):
        st, idxs[k] = sample_chosen_index(n, st, beta, eps, max_doublings, dist_id)
    return idxs

if __name__ == "__main__":
    n = 1000
    beta = 1.0 / (n**3)
    beta_expr = "1/n³"
    steps = 20_000
    eps = 1e-2
    max_doublings = 7
    dist_id = DIST_NAME_TO_ID["L2"]
    rng_seed = 0

    np.random.seed(rng_seed)
    idxs = run_chain(n, steps, beta, eps, max_doublings, dist_id)

    _apply_pub_style()
    fig, ax = plt.subplots(constrained_layout=True)
    bins = np.arange(int(idxs.min()) - 0.5, int(idxs.max()) + 1.5, 1)
    ax.hist(idxs, bins=bins, edgecolor="black", linewidth=0.8, alpha=0.35)
    ax.set_title(f"Chosen index distribution — n={n}, β={beta_expr}={beta:.3g}, ε={eps}, steps={steps:,}, max doublings={max_doublings}")
    ax.set_xlabel("Chosen index k (signed)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()