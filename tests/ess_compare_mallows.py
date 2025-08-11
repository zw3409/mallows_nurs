import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from distance import DIST_NAME_TO_ID
import hit_run
import nurs

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

def ess_1d(x):
    N = len(x)
    x = np.asarray(x, dtype=float)
    x -= x.mean()
    var = (x**2).mean()
    if var == 0.0:
        return 0.0
    fft = np.fft.fft(np.concatenate([x, np.zeros_like(x)]))
    acov = np.fft.ifft(fft * np.conjugate(fft)).real[:N] / N
    rho = acov / var
    tau = 0.0
    k = 1
    while k + 1 < N and (rho[k] + rho[k + 1]) > 0.0:
        tau += rho[k] + rho[k + 1]
        k += 2
    return N / (1 + 2 * tau)

def _zero_based(s):
    s = np.asarray(s)
    if s.min() == 1 and s.max() == len(s):
        return s - 1
    return s

def l2_distance(s):
    s0 = _zero_based(s)
    return np.linalg.norm(s0 - np.arange(len(s0), dtype=s0.dtype))

def pos_vector(s):
    s0 = _zero_based(s)
    p = np.empty_like(s0)
    p[s0] = np.arange(len(s0))
    return p

def five_before_eight(s):
    p = pos_vector(s)
    return int(p[5] < p[8])

def zero_top10(s):
    return int(pos_vector(s)[0] < 10)

def one_adjacent_two(s):
    p = pos_vector(s)
    return int(abs(p[1] - p[2]) == 1)

if __name__ == "__main__":
    n = 1000
    beta = 1.0 / (n**3)
    beta_expr = "1/n³"
    steps = 20_000
    chains = 1
    eps = 1e-2
    max_doublings = 7
    dist_id = DIST_NAME_TO_ID["L2"]
    rng_seed = 0

    def step_hr(s):
        return np.asarray(hit_run.hit_and_run_L2_step(np.asarray(s), beta))

    def step_nurs(s):
        return np.asarray(nurs.sample(n, tuple(np.asarray(s)), beta, eps, max_doublings, dist_id))

    kernels = {"Hit-and-Run": step_hr, "NURS": step_nurs}

    for f in kernels.values():
        _ = f(np.arange(n, dtype=int))

    stats = {
        "L2 distance": l2_distance,
        "5 before 8": five_before_eight,
        "0 in top-10": zero_top10,
        "1 adj 2": one_adjacent_two,
    }
    stat_names = list(stats.keys())

    np.random.seed(rng_seed)
    results = {}
    for kname, step in kernels.items():
        ess_accum = {sname: [] for sname in stat_names}
        for _ in range(chains):
            state = np.arange(n, dtype=int)
            traces = {sname: np.empty(steps) for sname in stat_names}
            for t in range(steps):
                state = step(state)
                for sname, fn in stats.items():
                    traces[sname][t] = fn(state)
            for sname in stat_names:
                ess_accum[sname].append(ess_1d(traces[sname]))
        results[kname] = {sname: float(np.mean(vals)) for sname, vals in ess_accum.items()}

    _apply_pub_style()

    x = np.arange(len(stat_names))
    klist = list(results.keys())
    W = 0.36
    offsets = np.linspace(-(len(klist)-1)/2, (len(klist)-1)/2, len(klist)) * W

    fig, ax = plt.subplots(constrained_layout=True)
    bars = []
    for o, kname in zip(offsets, klist):
        vals = [results[kname][s] for s in stat_names]
        b = ax.bar(x + o, vals, width=W, label=kname)
        bars.append((b, vals))

    ax.set_xticks(x)
    ax.set_xticklabels(stat_names, rotation=0, ha="center")
    ax.set_ylabel(f"Effective sample size (per {steps:,} draws)")
    ax.set_title(f"ESS by statistic — n={n}, β={beta_expr}={beta:.3g}, steps={steps:,}, max doublings={max_doublings}")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.margins(x=0.04)
    ax.legend(loc="upper left", ncols=len(klist))

    ymax = max(max(vals) for _, vals in bars)
    ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)

    for b, vals in bars:
        for rect, v in zip(b, vals):
            ax.annotate(f"{v:,.0f}", xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    plt.show()