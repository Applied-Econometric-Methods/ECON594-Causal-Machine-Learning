# -*- coding: utf-8 -*-
"""Shared helpers for the Causal ML course lecture decks.

DAGs are drawn by graphviz when it is available, which lays them out far better
than anything hand rolled. If graphviz is missing or its `dot` binary cannot
run, draw_dag silently falls back to a matplotlib version, so a lecture never
dies on a broken install. Run check_setup() to see which one you are getting.

  * draw_dag(...)                 -- DAGs, graphviz with a matplotlib fallback
  * check_setup()                 -- one-line student environment check
  * cumulative_gain_curve(...)    -- effect-ranking evaluation (replaces fklearn)
    relative_cumulative_gain_curve(...)
    area_under_the_relative_cumulative_gain_curve(...)
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch
import networkx as nx

# Robert's horizontal y-axis label: flat, above the axis, never rotated.
# Falls back to an inline copy of pywrangling.graphing_functions.set_top_ylabel.
try:
    from pywrangling.graphing_functions import set_top_ylabel
except Exception:
    def set_top_ylabel(ax, label, pad=0.02, **text_kwargs):
        ax.text(0.0, 1.0 + pad, label, transform=ax.transAxes,
                ha="center", va="bottom", rotation=0, **text_kwargs)

MADRID = "#2A3B8F"   # node fill
INK    = "#1A1A1A"
ACCENT  = "#D98A00"   # second group in the animations, and edge labels
GRIDCOL = "#C4C4D0"   # gridlines read as grey on a projector, not as nothing
GROUP_COLORS  = ["#E69F00", "#0072B2", "#00A087"]   # orange, blue, teal
GROUP_MARKERS = ["o", "^", "s"]
FITLINE = "#B3121F"

# video settings: constant-quality h264 via the ffmpeg bundled in imageio-ffmpeg
matplotlib.rcParams["animation.embed_limit"] = 100
matplotlib.rcParams["animation.bitrate"] = -1
matplotlib.rcParams["animation.ffmpeg_args"] = ["-crf", "18", "-preset", "slow"]
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass


# ============================================================== DAGs
def _wrap_label(label, width=13):
    """Break a long node label across lines.

    A label like "Default at yr=1" on one line makes an ellipse wide enough to
    swallow its neighbours, and edges then vanish behind it. Wrapping keeps
    nodes compact, which is what the layout code assumes.
    """
    s = str(label)
    if "\n" in s or len(s) <= width:
        return s
    lines, cur = [], ""
    for word in s.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _node_rx(label, ry):
    """Ellipse half-width sized to the label (widest line)."""
    widest = max((len(line) for line in str(label).split("\n")), default=1)
    return max(ry * 1.5, 0.16 * widest + 0.32)


def _node_ry(label, ry):
    """Ellipse half-height, grown for each extra line of a wrapped label."""
    return ry * (1 + 0.62 * str(label).count("\n"))


def _layered_pos(G, rxs, ry, xpad=1.6, ygap=1.7):
    """Left-to-right layout: one column per topological generation, with each
    column x-placed so wide ellipses in adjacent columns never collide."""
    gens = [list(g) for g in nx.topological_generations(G)]
    col_hw = [max((rxs[n] for n in gen), default=1.0) for gen in gens]
    pos, x = {}, 0.0
    for xi, gen in enumerate(gens):
        if xi > 0:
            x += col_hw[xi - 1] + col_hw[xi] + xpad
        for yi, node in enumerate(gen):
            pos[node] = (x, (yi - (len(gen) - 1) / 2) * ygap)
    return pos


def _fit_positions(pos, rxs, ry, xpad=0.55, ypad=0.5):
    """Stretch hand-placed positions until no two nodes overlap.

    Hand coordinates are written in arbitrary units, but ellipse widths come
    from the label text, so a long label like "Default at yr=1" can easily be
    wider than the gap it was given. Rather than force whoever writes the cell
    to know that, scale the whole layout up until everything clears. Relative
    arrangement, which is the reason for placing by hand, is preserved.
    """
    nodes = list(pos)
    sx = sy = 1.0
    for _ in range(4):                       # a few passes to settle
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                dx = abs(pos[a][0] - pos[b][0])
                dy = abs(pos[a][1] - pos[b][1])
                need_x = rxs[a] + rxs[b] + xpad
                need_y = 2 * ry + ypad
                if dx * sx >= need_x or dy * sy >= need_y:
                    continue                 # already clear on one axis
                if dx > 1e-9:
                    sx = max(sx, need_x / dx)
                elif dy > 1e-9:
                    sy = max(sy, need_y / dy)
    return {n: (p[0] * sx, p[1] * sy) for n, p in pos.items()}


def _ellipse_hit(center, toward, rx, ry):
    """Point on the ellipse around `center` in the direction of `toward`."""
    dx, dy = toward[0] - center[0], toward[1] - center[1]
    if dx == 0 and dy == 0:
        return center
    t = 1.0 / np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    return (center[0] + dx * t, center[1] + dy * t)


_GV = None          # cached graphviz module, or False once known unusable
CONDITIONED = "#D9D9D9"


def _graphviz():
    """Return the graphviz module, or False. Caches the answer.

    Importing is not enough: the Python package is a wrapper around the `dot`
    executable, and the classic failure is having one without the other. So we
    actually render a throwaway graph and only trust graphviz if that works.
    """
    global _GV
    if _GV is None:
        try:
            import graphviz as _gv
            _gv.Digraph().node("x")
            _gv.Digraph(body=["x;"]).pipe(format="svg")
            _GV = _gv
        except Exception:
            _GV = False
    return _GV


def _draw_dag_graphviz(gv, edges, boxed=(), edge_labels=None):
    """Facure's graphviz look: left-to-right, grey fill for conditioned nodes,
    penwidth for causal strength."""
    edge_labels = edge_labels or {}
    g = gv.Digraph(
        graph_attr={"rankdir": "LR", "bgcolor": "transparent",
                    "ranksep": "0.55", "nodesep": "0.30", "margin": "0.05"},
        node_attr={"shape": "ellipse", "fontname": "Helvetica", "fontsize": "15",
                   "color": INK, "fontcolor": INK, "penwidth": "1.5",
                   "margin": "0.09,0.05"},
        edge_attr={"color": INK, "penwidth": "1.5", "arrowsize": "0.85"})

    # A node NAME is used as a label and may hold a newline (for a two-line
    # label) or an apostrophe. Those break DOT if used as the node identifier,
    # and the Digraph then fails to render, so Jupyter silently shows its repr
    # instead of the diagram. Give every node a safe id and carry the real text
    # as the label, converting a literal newline into DOT's line-break escape.
    names = []
    for e in edges:
        for n in (e[0], e[1]):
            if n not in names:
                names.append(n)
    nid = {n: "n%d" % i for i, n in enumerate(names)}
    label = lambda n: n.replace("\n", "\\n")

    fills = boxed if isinstance(boxed, dict) else {n: CONDITIONED for n in boxed}
    for n in names:
        attrs = {"label": label(n)}
        if n in fills:
            attrs["style"] = "filled"
            attrs["fillcolor"] = fills[n]
        g.node(nid[n], **attrs)

    for e in edges:
        attrs = {}
        if len(e) > 2:
            if e[2] == "dashed":
                attrs["style"] = "dashed"
            else:
                attrs["penwidth"] = str(e[2])
        if (e[0], e[1]) in edge_labels:
            attrs["label"] = f" {edge_labels[(e[0], e[1])]} "
            attrs["fontcolor"] = ACCENT
            attrs["fontname"] = "Helvetica bold"
            attrs["fontsize"] = "13"
        g.edge(nid[e[0]], nid[e[1]], **attrs)
    return g


def draw_dag(edges, boxed=(), pos=None, ry=0.52, fontsize=13, figscale=0.66,
             engine=None, edge_labels=None):
    """Draw a DAG in the course style.

    edges  : (source, target) pairs, optionally (source, target, penwidth) to
             encode causal strength, or (source, target, "dashed")
    boxed  : nodes to draw as conditioned-on (grey fill). May be a dict of
             {node: fill colour} when two levels of conditioning need to differ
    pos    : hand positions {node: (x, y)}; used only by the matplotlib fallback,
             since graphviz lays the graph out itself
    engine : force "graphviz" or "matplotlib"; default picks whatever works
    """
    gv = _graphviz()
    if engine != "matplotlib" and gv is not False:
        g = _draw_dag_graphviz(gv, edges, boxed, edge_labels)
        # Render to SVG right now and hand back a display object. Returning the
        # raw Digraph makes Jupyter render it lazily at display time, and that
        # lazy path intermittently fails and shows the plain "<Digraph ...>"
        # text instead of the diagram. Baking the SVG here makes it always draw.
        from IPython.display import SVG
        return SVG(data=g.pipe(format="svg").decode("utf-8"))
    if engine == "graphviz":
        raise RuntimeError("graphviz is not usable here; run chh.check_setup()")
    if isinstance(boxed, dict):
        boxed = tuple(boxed)
    return _draw_dag_mpl(edges, boxed, pos, ry, fontsize, figscale, edge_labels)


def _draw_dag_mpl(edges, boxed=(), pos=None, ry=0.52, fontsize=13, figscale=0.66,
                  edge_labels=None):
    """Fallback DAG renderer: matplotlib + networkx, no system binary needed.

    Ellipses size to their labels, and arrows stop on each node's boundary, so
    nothing collides regardless of figure size.
    """
    widths = {(e[0], e[1]): (e[2] if len(e) > 2 and not isinstance(e[2], str) else 1.9)
              for e in edges}
    dashed = {(e[0], e[1]) for e in edges if len(e) > 2 and e[2] == "dashed"}
    edges = [(e[0], e[1]) for e in edges]
    G = nx.DiGraph()
    G.add_nodes_from({n for e in edges for n in e})
    G.add_edges_from(edges)
    labels = {n: _wrap_label(n) for n in G.nodes()}
    rxs = {n: _node_rx(labels[n], ry) for n in G.nodes()}
    rys = {n: _node_ry(labels[n], ry) for n in G.nodes()}
    maxry = max(rys.values())
    if pos is None:
        pos = _layered_pos(G, rxs, maxry, ygap=2 * maxry + 0.85)
    else:
        pos = _fit_positions(pos, rxs, maxry)

    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    maxrx = max(rxs.values())
    W = (max(xs) - min(xs)) + 2 * maxrx + 2.0
    H = (max(ys) - min(ys)) + 2 * maxry + 2.2
    fig, ax = plt.subplots(figsize=(W * figscale, H * figscale))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(min(xs) - maxrx - 0.6, max(xs) + maxrx + 0.6)
    # headroom for arcs that bend around blocking nodes
    ax.set_ylim(min(ys) - maxry - 1.0, max(ys) + maxry + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    def _blocked(a, b):
        """Does the straight a->b segment pass too close to another node?
        Returns an arc curvature (0 = straight)."""
        (x1, y1), (x2, y2) = pos[a], pos[b]
        seg = np.array([x2 - x1, y2 - y1]); L = np.hypot(*seg)
        worst = 0.0
        for n in G.nodes():
            if n in (a, b):
                continue
            p = np.array(pos[n]) - np.array([x1, y1])
            t = np.clip(p @ seg / (L * L), 0, 1)
            d = np.hypot(*(p - t * seg))
            half = max(rxs[n], rys[n])
            if 0.05 < t < 0.95 and d < half + 0.25:
                # Bend away from the blocking node, harder when that node is
                # big relative to the edge: a wide box needs a wide detour, or
                # the edge vanishes behind it and looks like it starts there.
                side = np.sign(np.cross(seg, p)) or 1.0
                bend = min(0.8, 0.34 + 1.15 * (half + 0.25 - d) / max(L, 1e-6))
                if abs(bend) > abs(worst):
                    worst = -bend * side
        return worst

    for a, b in G.edges():
        rad = _blocked(a, b)
        pa = _ellipse_hit(pos[a], pos[b], rxs[a], rys[a])
        pb = _ellipse_hit(pos[b], pos[a], rxs[b], rys[b])
        style = f"arc3,rad={rad}" if rad else "arc3"
        ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=16,
                                     lw=widths.get((a, b), 1.9), color=INK,
                                     connectionstyle=style,
                                     linestyle=(0, (5, 4)) if (a, b) in dashed else "-",
                                     shrinkA=0, shrinkB=0, zorder=2))
        if edge_labels and (a, b) in edge_labels:
            ax.text((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2,
                    edge_labels[(a, b)], ha="center", va="center", zorder=5,
                    fontsize=fontsize - 2, color=ACCENT, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none"))

    for n, (x, y) in pos.items():
        rx, nry = rxs[n], rys[n]
        if n in boxed:
            m = 0.12
            ax.add_patch(FancyBboxPatch((x - rx, y - nry), 2 * rx, 2 * nry,
                                        boxstyle="square,pad=0", lw=2.2,
                                        edgecolor=INK, facecolor=MADRID, zorder=3))
            ax.add_patch(FancyBboxPatch((x - rx - m, y - nry - m), 2 * rx + 2 * m,
                                        2 * nry + 2 * m, boxstyle="square,pad=0",
                                        lw=1.1, edgecolor=INK, facecolor="none", zorder=3))
        else:
            ax.add_patch(Ellipse((x, y), 2 * rx, 2 * nry, facecolor=MADRID,
                                  edgecolor=MADRID, zorder=3))
        ax.text(x, y, labels[n], ha="center", va="center", color="white",
                fontsize=fontsize, zorder=4, linespacing=0.95)

    fig.tight_layout()
    plt.close(fig)
    return fig


# ============================================================== setup check
def check_setup():
    """Print a one-line-per-package report of the course environment.

    Students run this once after setup and paste the output if anything is
    wrong. Nothing here raises, so it always finishes and always tells you
    something useful.
    """
    import importlib
    import shutil
    import sys

    print(f"python       {sys.version.split()[0]}")
    for name in ("numpy", "pandas", "scipy", "matplotlib", "seaborn", "sklearn",
                 "statsmodels", "linearmodels", "networkx", "pywrangling",
                 "fklearn", "torch"):
        try:
            mod = importlib.import_module(name)
            print(f"  ok   {name:<13}{getattr(mod, '__version__', '')}")
        except Exception:
            print(f"  MISSING {name}")

    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
        print("  ok   imageio-ffmpeg   (animations will render)")
    except Exception:
        print("  MISSING imageio-ffmpeg   (animations will not render)")

    gv = _graphviz()
    if gv is not False:
        print(f"  ok   graphviz     {gv.__version__}, dot at {shutil.which('dot')}")
        print("\nAll set. DAGs will be drawn by graphviz.")
    else:
        try:
            importlib.import_module("graphviz")
            why = ("the Python package is installed but the `dot` program is "
                   "not on your PATH")
        except Exception:
            why = "the graphviz package is not installed"
        print(f"  note graphviz     not usable: {why}")
        print("\nEverything will still run: DAGs fall back to matplotlib.")
        print("For the nicer diagrams, in your activated cml environment run:")
        print("    conda install -c conda-forge python-graphviz -y")


# ============================================================== effect-ranking eval
# Portable replacements for the two fklearn functions used in chapter 7. Built
# so students can see how the cumulative-gain metric is computed rather than
# importing it as a black box.
def _effect_by_quantile(df, prediction, y, t):
    d = df.sort_values(prediction, ascending=False).reset_index(drop=True)
    n = len(d)
    out = []
    for k in range(1, n + 1):
        head = d.iloc[:k]
        tt, yy = head[t].values, head[y].values
        # simple slope of y on t within the top-k (the estimated effect there)
        if tt.std() == 0:
            eff = 0.0
        else:
            eff = np.cov(tt, yy)[0, 1] / np.var(tt)
        out.append(eff)
    return np.array(out)


def cumulative_effect_curve(df, prediction, y, t):
    """Estimated treatment effect within the top-ranked fraction, as the
    fraction grows from a small head to the whole sample."""
    return _effect_by_quantile(df, prediction, y, t)


def cumulative_gain_curve(df, prediction, y, t, normalize=True):
    """Cumulative effect weighted by the fraction of sample covered.
    A model that ranks effects well rises fast and stays above the diagonal."""
    eff = _effect_by_quantile(df, prediction, y, t)
    n = len(eff)
    frac = np.arange(1, n + 1) / n
    gain = eff * frac
    return gain


def relative_cumulative_gain_curve(df, prediction, y, t):
    """Cumulative gain minus the random-baseline (average-effect) line."""
    eff = _effect_by_quantile(df, prediction, y, t)
    n = len(eff)
    frac = np.arange(1, n + 1) / n
    ate = eff[-1]                      # effect over the whole sample
    return (eff - ate) * frac


def area_under_the_relative_cumulative_gain_curve(df, prediction, y, t):
    """Single-number summary: area under the relative cumulative gain curve."""
    rcg = relative_cumulative_gain_curve(df, prediction, y, t)
    frac = np.arange(1, len(rcg) + 1) / len(rcg)
    # numpy 2 renamed trapz; students on either version get the same answer
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(rcg, frac))


# ============================================================== ch1 schematics
# Reproductions of the book-only illustrations in chapter 1. Semantics follow
# the book: FILLED CIRCLES are realized (observed) outcomes, OPEN TRIANGLES are
# unrealized potential outcomes. Colour marks treatment: blue treated, orange
# untreated.
_TREAT, _CTRL = "#0072B2", "#E69F00"


def _schem_ax(ax, xlabel="Business size", ylabel="Amount sold"):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=12)
    set_top_ylabel(ax, ylabel, fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _schem_units(seed=7, n=14, effect=1.6):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    y0 = 1.5 + 0.55 * x + rng.normal(0, 0.45, n)
    y1 = y0 + effect
    return x, y0, y1


def fig_bias_schematic():
    """The book's pair: (left) both potential outcomes per unit, the individual
    effect is the small vertical gap; (right) Y0 only, where the treated sit
    higher even untreated. That gap IS the bias."""
    x, y0, y1 = _schem_units()
    t = x > 5.5                       # bigger businesses get treated
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    yobs = np.where(t, y1, y0)
    ycf = np.where(t, y0, y1)
    for xi, yo, yc, ti in zip(x, yobs, ycf, t):
        col = _TREAT if ti else _CTRL
        ax.plot([xi, xi], [yo, yc], color="#B9B9C2", lw=1.0, zorder=1)
        ax.scatter([xi], [yo], marker="o", color=col, s=52, zorder=3)
        ax.scatter([xi], [yc], marker="^", facecolors="none", edgecolors=col,
                   s=58, lw=1.6, zorder=3)
    ax.set_title("Both potential outcomes\n(gap = $Y_1 - Y_0$, the effect)", fontsize=12)
    _schem_ax(ax)
    ax = axes[1]
    for xi, yy, ti in zip(x, y0, t):
        if ti:
            ax.scatter([xi], [yy], marker="^", facecolors="none",
                       edgecolors=_TREAT, s=58, lw=1.6, zorder=3)
        else:
            ax.scatter([xi], [yy], marker="o", color=_CTRL, s=52, zorder=3)
    m1, m0 = y0[t].mean(), y0[~t].mean()
    ax.axhline(m1, color=_TREAT, ls=(0, (6, 4)), lw=1.6)
    ax.axhline(m0, color=_CTRL, ls=(0, (6, 4)), lw=1.6)
    ax.annotate("", xy=(9.7, m1), xytext=(9.7, m0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(9.9, (m1 + m0) / 2, "Bias", fontsize=12, va="center", color=INK)
    ax.text(0.2, m1 + 0.15, "$E[Y_0|T=1]$", fontsize=11, color=_TREAT)
    ax.text(0.2, m0 - 0.42, "$E[Y_0|T=0]$", fontsize=11, color=_CTRL)
    ax.set_title("$Y_0$ only: treated differ\neven WITHOUT treatment", fontsize=12)
    _schem_ax(ax)
    ax.set_xlim(-0.3, 11.6)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_randomization_schematic():
    """The book's pair: before treatment both potential outcomes wait to be
    realized (triangles); randomization then picks one AT RANDOM, so treatment
    ends up unrelated to size."""
    x, y0, y1 = _schem_units(seed=12)
    rng = np.random.default_rng(3)
    t = rng.integers(0, 2, len(x)).astype(bool)      # coin flip
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    for xi, a, b in zip(x, y0, y1):
        ax.scatter([xi], [b], marker="^", facecolors="none", edgecolors="#55555F",
                   s=58, lw=1.5, zorder=3)
        ax.scatter([xi], [a], marker="v", facecolors="none", edgecolors="#9E9EA8",
                   s=58, lw=1.5, zorder=3)
    ax.set_title("Before: a world of\npotential outcomes", fontsize=12)
    _schem_ax(ax)
    ax = axes[1]
    for xi, a, b, ti in zip(x, y0, y1, t):
        if ti:
            ax.scatter([xi], [b], marker="o", color=_TREAT, s=52, zorder=3)
            ax.scatter([xi], [a], marker="v", facecolors="none",
                       edgecolors="#9E9EA8", s=58, lw=1.3, zorder=2)
        else:
            ax.scatter([xi], [a], marker="o", color=_CTRL, s=52, zorder=3)
            ax.scatter([xi], [b], marker="^", facecolors="none",
                       edgecolors="#9E9EA8", s=58, lw=1.3, zorder=2)
    ax.set_title("After randomization:\na coin flip realizes one", fontsize=12)
    _schem_ax(ax)
    fig.text(0.5, 0.97, "Randomization", ha="center", fontsize=13,
             weight="bold", color=MADRID)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); plt.close(fig)
    return fig


def fig_randomized_outcomes():
    """The book's pair: after randomization the observed difference in means IS
    the ATE, and the Y0-only view shows no gap at all."""
    x, y0, y1 = _schem_units(seed=12)
    rng = np.random.default_rng(3)
    t = rng.integers(0, 2, len(x)).astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    yobs = np.where(t, y1, y0)
    for xi, yy, ti in zip(x, yobs, t):
        ax.scatter([xi], [yy], marker="o", color=_TREAT if ti else _CTRL,
                   s=52, zorder=3)
    ax.axhline(yobs[t].mean(), color=_TREAT, ls=(0, (6, 4)), lw=1.6)
    ax.axhline(yobs[~t].mean(), color=_CTRL, ls=(0, (6, 4)), lw=1.6)
    ax.text(0.2, yobs[t].mean() + 0.15, "$E[Y|T=1]$", fontsize=11, color=_TREAT)
    ax.text(0.2, yobs[~t].mean() - 0.42, "$E[Y|T=0]$", fontsize=11, color=_CTRL)
    ax.set_title("Observed outcomes:\ndifference in means $=$ ATE", fontsize=12)
    _schem_ax(ax)
    ax = axes[1]
    for xi, yy, ti in zip(x, y0, t):
        if ti:
            ax.scatter([xi], [yy], marker="^", facecolors="none",
                       edgecolors=_TREAT, s=58, lw=1.6, zorder=3)
        else:
            ax.scatter([xi], [yy], marker="o", color=_CTRL, s=52, zorder=3)
    ax.axhline(y0[t].mean(), color=_TREAT, ls=(0, (6, 4)), lw=1.6)
    ax.axhline(y0[~t].mean(), color=_CTRL, ls=(0, (6, 4)), lw=1.6)
    ax.set_title("$Y_0$ only: no gap.\n$E[Y_0|T=0] = E[Y_0|T=1]$", fontsize=12)
    _schem_ax(ax)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_selection_vs_intervention():
    """The book's Figure 1-2: conditioning selects a subsample; do() forces the
    treatment on everyone."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    for ax, mode in zip(axes, ("select", "do")):
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
        ax.add_patch(plt.Rectangle((2.6, 0.6), 7.0, 4.8, fill=False,
                                   edgecolor=INK, lw=1.4))
        ax.plot([2.6, 9.6], [3.0, 3.0], color=INK, lw=1.0)
        ax.text(1.3, 4.0, "On sale", ha="center", fontsize=12)
        ax.text(1.3, 1.6, "Not on\nsale", ha="center", fontsize=12)
        if mode == "select":
            ax.add_patch(plt.Rectangle((2.6, 3.0), 7.0, 2.4,
                                       facecolor="#F5C97B", edgecolor=INK, lw=1.6))
            ax.text(6.1, 4.2, "$E[Y\\,|\\,\\mathrm{on\\ sale}]$",
                    ha="center", fontsize=13)
            ax.set_title("Conditioning: keep only the\nunits that CHOSE treatment",
                         fontsize=12)
        else:
            ax.add_patch(plt.Rectangle((2.6, 0.6), 7.0, 4.8,
                                       facecolor="#F5C97B", edgecolor=INK, lw=1.6))
            ax.text(6.1, 2.9, "$E[Y\\,|\\,do(\\mathrm{on\\ sale})]$",
                    ha="center", fontsize=13)
            ax.set_title("Intervening: force treatment\non the ENTIRE sample",
                         fontsize=12)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_simpson_panels():
    """The book's Figure 1-3 flavour: the pooled slope overstates the effect;
    within business size the slope is smaller but still positive."""
    rng = np.random.default_rng(5)
    n = 11
    x_s = rng.uniform(0.5, 4.5, n)                    # small firms: small discounts
    y_s = 1.0 + 0.35 * x_s + rng.normal(0, 0.28, n)
    x_l = rng.uniform(5.0, 9.5, n)                    # large firms: big discounts
    y_l = 3.6 + 0.35 * x_l + rng.normal(0, 0.28, n)
    X = np.concatenate([x_s, x_l]); Y = np.concatenate([y_s, y_l])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for ax in axes:
        ax.scatter(x_s, y_s, marker="o", color=_CTRL, s=48, zorder=3,
                   label="Small business")
        ax.scatter(x_l, y_l, marker="^", facecolors="none", edgecolors=_TREAT,
                   s=56, lw=1.6, zorder=3, label="Large business")
        _schem_ax(ax, xlabel="Price discount", ylabel="Amount sold")
    b, a = np.polyfit(X, Y, 1)
    xs = np.array([X.min() - 0.3, X.max() + 0.3])
    axes[0].plot(xs, a + b * xs, color=FITLINE, lw=2.4, zorder=4)
    axes[0].set_title(f"Pooled slope: {b:+.2f}\n(way too steep)", fontsize=12)
    for xg, yg in ((x_s, y_s), (x_l, y_l)):
        bg, ag = np.polyfit(xg, yg, 1)
        xs = np.array([xg.min() - 0.3, xg.max() + 0.3])
        axes[1].plot(xs, ag + bg * xs, color=FITLINE, lw=2.4, zorder=4)
    axes[1].set_title("Within size: smaller slope,\nstill positive", fontsize=12)
    axes[1].legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============================================================== ch3 cheat sheet
def fig_association_cheatsheet():
    """The book's Figure 3-1: how association flows. Top row unconditioned,
    bottom row conditioned on the highlighted node. Chain and fork transmit
    association until you condition; the collider is the reverse."""
    cols = [
        dict(name="Chain", struct="T $\\to$ X $\\to$ Y",
             nodes=[(0, 0), (1, 0), (2, 0)], arrows=[(0, 1), (1, 2)],
             labels=["T", "X", "Y"], cond=1,
             top="associated", bot="independent"),
        dict(name="Fork", struct="T $\\leftarrow$ X $\\to$ Y",
             nodes=[(0, 0), (1, 0.9), (2, 0)], arrows=[(1, 0), (1, 2)],
             labels=["T", "X", "Y"], cond=1,
             top="associated", bot="independent"),
        dict(name="Collider", struct="T $\\to$ X $\\leftarrow$ Y",
             nodes=[(0, 0.9), (1, 0), (2, 0.9)], arrows=[(0, 1), (2, 1)],
             labels=["T", "X", "Y"], cond=1,
             top="independent", bot="associated"),
        dict(name="Collider + descendant", struct="T $\\to$ X $\\leftarrow$ Y,  X $\\to$ D",
             nodes=[(0, 1.5), (1, 0.75), (2, 1.5), (1, -0.1)],
             arrows=[(0, 1), (2, 1), (1, 3)],
             labels=["T", "X", "Y", "D"], cond=3,
             top="independent", bot="partly associated"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.0))
    for row in (0, 1):
        for j, col in enumerate(cols):
            ax = axes[row][j]
            ax.set_xlim(-0.6, 2.6); ax.set_ylim(-1.15, 2.15)
            ax.set_aspect("equal"); ax.axis("off")
            cond = col["cond"] if row else None
            for a, b in col["arrows"]:
                (x1, y1), (x2, y2) = col["nodes"][a], col["nodes"][b]
                ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                             mutation_scale=13, lw=1.8, color=INK,
                                             shrinkA=13, shrinkB=13, zorder=2))
            for k, (x, y) in enumerate(col["nodes"]):
                fill = CONDITIONED if k == cond else "white"
                ax.add_patch(plt.Circle((x, y), 0.28, facecolor=fill,
                                        edgecolor=INK, lw=1.4, zorder=3))
                ax.text(x, y, col["labels"][k], ha="center", va="center",
                        fontsize=11, color=INK, zorder=4)
            if row == 0:
                ax.set_title(f"{col['name']}\n{col['struct']}", fontsize=11,
                             color=INK, linespacing=1.5)
            verdict = col["top"] if row == 0 else col["bot"]
            ax.text(1, -0.95, f"T and Y\n{verdict}", ha="center", va="center",
                    fontsize=10, weight="bold", linespacing=1.35,
                    color="#B3121F" if verdict == "independent" else "#1A7A3C")

    fig.tight_layout(rect=(0.10, 0, 1, 0.99))
    for row, label in ((0, "not\nconditioned"), (1, "conditioned\non the grey node")):
        box = axes[row][0].get_position()
        fig.text(0.055, (box.y0 + box.y1) / 2, label, ha="center", va="center",
                 fontsize=11, weight="bold", color=MADRID, linespacing=1.4)
    plt.close(fig)
    return fig


# ============================================================== ch2 schematic
def fig_power_diagram(delta=2.8, alpha=1.96):
    """The book's power picture: null and alternative sampling distributions,
    the 5% critical value, and the power as the shaded mass of the alternative
    beyond it (about 80% when the true effect sits 2.8 SEs out)."""
    from scipy import stats
    x = np.linspace(-4, 7, 400)
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.plot(x, stats.norm.pdf(x, 0, 1), color=_CTRL, lw=2.4)
    ax.plot(x, stats.norm.pdf(x, delta, 1), color=_TREAT, lw=2.4)
    xs = x[x >= alpha]
    ax.fill_between(xs, stats.norm.pdf(xs, delta, 1), color=_TREAT, alpha=0.25)
    ax.fill_between(xs, stats.norm.pdf(xs, 0, 1), color=_CTRL, alpha=0.45)
    ax.axvline(alpha, color=INK, lw=1.4, ls=(0, (6, 4)))
    power = 1 - stats.norm.cdf(alpha - delta)
    ax.text(0, 0.42, "$H_0$: no effect", ha="center", fontsize=12, color=_CTRL)
    ax.text(delta, 0.42, "true effect $\\delta$", ha="center", fontsize=12, color=_TREAT)
    ax.text(alpha + 0.1, 0.30, "critical value\n($1.96\\,SE$)", fontsize=10, color=INK)
    ax.annotate(f"Power $\\approx$ {power:.0%}", xy=(delta + 0.7, 0.05),
                xytext=(4.9, 0.22), fontsize=12, color=_TREAT,
                arrowprops=dict(arrowstyle="->", color=_TREAT))
    ax.text(alpha + 0.35, 0.012, "$\\alpha/2$", fontsize=10, color="#8a6d1d")
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlabel("Estimated difference (in SEs)", fontsize=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============================================================== FWL animation
# Animated version of the book's Figures 4-1 and 4-2: orthogonalization first
# removes bias (residualize T on X), then removes noise (residualize Y on X).
# Data are simulated so the truth is known: the raw slope is NEGATIVE, the true
# effect is POSITIVE, and three X groups confound the picture.
_FPS, _HOLD, _TWEEN = 15, 24, 14


def _ease(t):
    return t * t * (3 - 2 * t)


def _fwl_demo_data(seed=42, n_per=60, beta=1.2):
    rng = np.random.default_rng(seed)
    g = np.repeat([0, 1, 2], n_per)
    t = rng.normal(2.0 + 2.0 * g, 0.55)
    y = 22.0 - 7.0 * g + beta * t + rng.normal(0, 1.1, len(g))
    return g, t, y


def fwl_animation(seed=42):
    """The FWL / orthogonalization story in motion, in seven stages.

    1 raw cloud, 2 colour by group, 3 naive fit (wrong sign), 4 estimate
    E[T|X], 5 debias (T minus E[T|X]), 6 estimate E[Y|X], 7 denoise and the
    true positive slope. Residuals are shown AS residuals, centred on zero,
    with the axes travelling between stages.
    """
    g, t, y = _fwl_demo_data(seed)
    tm = np.array([t[g == k].mean() for k in range(3)])
    ym = np.array([y[g == k].mean() for k in range(3)])
    tr = t - tm[g]              # debiased treatment, mean zero
    yr = y - ym[g]              # denoised outcome, mean zero
    b_naive = np.polyfit(t, y, 1)
    b_deb = np.polyfit(tr, y, 1)
    b_final = np.polyfit(tr, yr, 1)

    def _lims(v, pad=0.10):
        lo, hi = float(v.min()), float(v.max())
        p = pad * (hi - lo)
        return (lo - p, hi + p)

    #        x    y   caption                                              naive  gT     gY     deb    fin    vl     hl
    stages = [
        (t,  y,  "1. Treatment vs outcome, raw cloud.",                    0,     0,     0,     0,     0,     0,     0),
        (t,  y,  "2. Three groups of X drive both $T$ and $Y$.",           0,     0,     0,     0,     0,     0,     0),
        (t,  y,  f"3. Naive fit, slope {b_naive[0]:+.2f}. Wrong sign!",    1,     0,     0,     0,     0,     0,     0),
        (t,  y,  "4. Estimate $E[T|X]$: each group's average $T$.",        0,     1,     0,     0,     0,     0,     0),
        (tr, y,  f"5. Debias: $T - E[T|X]$. Slope now {b_deb[0]:+.2f}.",   0,     0,     0,     1,     0,     1,     0),
        (tr, y,  "6. Estimate $E[Y|X]$: each group's average $Y$.",        0,     0,     1,     1,     0,     1,     0),
        (tr, yr, f"7. Denoise: $Y - E[Y|X]$. Slope {b_final[0]:+.2f}, tight fit.", 0, 0, 0,     0,     1,     1,     1),
    ]
    st_xlim = [_lims(s[0]) for s in stages]
    st_ylim = [_lims(s[1]) for s in stages]
    st_xlab = ["$T$"] * 4 + ["$T - E[T|X]$"] * 3
    st_ylab = ["$Y$"] * 6 + ["$Y - E[Y|X]$"]
    color_on = [0, 1, 1, 1, 1, 1, 1]        # stage 1 is uncoloured

    def _blend(p, c, a):
        return (p[0] + (c[0] - p[0]) * a, p[1] + (c[1] - p[1]) * a)

    frames = []
    for i, st in enumerate(stages):
        if i > 0:
            px, py = stages[i - 1][0], stages[i - 1][1]
            for f in range(_TWEEN):
                a = _ease((f + 1) / _TWEEN)
                frames.append(dict(x=px + (st[0] - px) * a, y=py + (st[1] - py) * a,
                                   cap=st[2], naive=st[3], gT=st[4], gY=st[5],
                                   deb=st[6], fin=st[7], vl=st[8], hl=st[9],
                                   xlim=_blend(st_xlim[i - 1], st_xlim[i], a),
                                   ylim=_blend(st_ylim[i - 1], st_ylim[i], a),
                                   xlab=st_xlab[i], ylab=st_ylab[i],
                                   colored=color_on[i], alpha=a, line_alpha=0.0))
        # the fit line appears only once the cloud has settled, fading in at the
        # start of the hold and out at the end, so it never slides in with the
        # moving axis
        for h in range(_HOLD):
            fade = 6
            la = (_ease((h + 1) / fade) if h < fade else
                  _ease((_HOLD - h) / fade) if h >= _HOLD - fade else 1.0)
            frames.append(dict(x=st[0], y=st[1], cap=st[2], naive=st[3], gT=st[4],
                               gY=st[5], deb=st[6], fin=st[7], vl=st[8], hl=st[9],
                               xlim=st_xlim[i], ylim=st_ylim[i],
                               xlab=st_xlab[i], ylab=st_ylab[i],
                               colored=color_on[i], alpha=1.0, line_alpha=la))

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=150)
    fig.patch.set_facecolor("white")
    # reserve room for the title and caption up front; tight_layout cannot run
    # per frame without the axes jittering
    fig.subplots_adjust(left=0.13, right=0.97, top=0.81, bottom=0.12)

    def draw(fr):
        ax.clear()
        ax.set_facecolor("white")
        ax.grid(True, color="#C4C4D0", linewidth=0.9, linestyle=(0, (1, 4)))
        ax.set_axisbelow(True)
        if fr["vl"]:
            ax.axvline(0, color="#7A7A85", linewidth=1.0, zorder=1)
        if fr["hl"]:
            ax.axhline(0, color="#7A7A85", linewidth=1.0, zorder=1)
        if fr["colored"]:
            for k in range(3):
                m = g == k
                ax.scatter(fr["x"][m], fr["y"][m], s=26, alpha=0.75,
                           color=GROUP_COLORS[k], marker=GROUP_MARKERS[k],
                           edgecolors="white", linewidths=0.4, zorder=3,
                           label=f"$X = {k + 1}$")
            ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
        else:
            ax.scatter(fr["x"], fr["y"], s=26, alpha=0.75, color="#55555F",
                       edgecolors="white", linewidths=0.4, zorder=3)
        if fr["gT"]:
            for k in range(3):
                ax.axvline(tm[k], color=GROUP_COLORS[k], linewidth=2.0,
                           linestyle=(0, (6, 4)), alpha=fr["alpha"], zorder=2)
        if fr["gY"]:
            for k in range(3):
                ax.axhline(ym[k], color=GROUP_COLORS[k], linewidth=2.0,
                           linestyle=(0, (6, 4)), alpha=fr["alpha"], zorder=2)
        xx = np.array(fr["xlim"])
        if fr["naive"] and fr["line_alpha"] > 0:
            ax.plot(xx, b_naive[1] + b_naive[0] * xx, color=FITLINE,
                    linewidth=2.4, alpha=fr["line_alpha"], zorder=4)
        if fr["deb"] and fr["line_alpha"] > 0:
            ax.plot(xx, b_deb[1] + b_deb[0] * xx, color=FITLINE,
                    linewidth=2.4, alpha=fr["line_alpha"], zorder=4)
        if fr["fin"] and fr["line_alpha"] > 0:
            ax.plot(xx, b_final[1] + b_final[0] * xx, color=FITLINE,
                    linewidth=2.4, alpha=fr["line_alpha"], zorder=4)
        ax.set_xlim(*fr["xlim"]); ax.set_ylim(*fr["ylim"])
        ax.set_xlabel(fr["xlab"], fontsize=13)
        ax.text(0.0, 1.02, fr["ylab"], transform=ax.transAxes, ha="left",
                va="bottom", rotation=0, fontsize=12, color=INK)
        ax.set_title("Orthogonalization: debias, then denoise", fontsize=15,
                     color=MADRID, pad=24, weight="bold")
        ax.text(0.5, 1.02, fr["cap"], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=11, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / _FPS)
    plt.close(fig)
    return anim


# ================================================ ch3 structure animations
# Ported from the archived Lecture 1 helper so the Lectures folder stands on its
# own. Nick Huntington-Klein's "controlling for a variable" sequence: the same
# residualize-X, residualize-Y machinery on screen, with opposite lessons for a
# confounder (removes a spurious link) and a collider (manufactures one).
from pathlib import Path

_POINT_SIZE = 34
_ANIM_DPI = 150       # figsize * dpi must stay even in both dimensions for h264
_DATA = Path(__file__).resolve().parent / "data"
_real_cache = {}


def _corr(x, y):
    return np.corrcoef(x, y)[0, 1]


def _residualize(v, g):
    """Within-group demeaning: the residual for a discrete control."""
    out = v.astype(float).copy()
    for lvl in np.unique(g):
        m = g == lvl
        out[m] = out[m] - out[m].mean()
    return out


def _partial(v, ctrl):
    """FWL residual of v on a constant and ctrl.

    ctrl may be one control (1-D) or several (2-D, one column each).
    """
    C = np.asarray(ctrl, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    A = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def _make_collider(seed=3444, n=200):
    """Cunningham's Mixtape example (his Stata used seed 3444). Beauty and
    talent are INDEPENDENT, but "movie star" is the top 15% of beauty+talent.
    Conditioning on stardom, a collider, manufactures a negative correlation."""
    rng = np.random.default_rng(seed)
    beauty = rng.normal(0, 1, n)
    talent = rng.normal(0, 1, n)
    star = (beauty + talent >= np.quantile(beauty + talent, 0.85)).astype(int)
    return star, beauty, talent


def collider_animation(seed=3444):
    """Conditioning on a collider CREATES a correlation that was never there.

    Two fit lines in the group colours make the point the pooled cloud hides:
    the downward tilt shows up inside BOTH groups, not only among the stars.
    """
    G, x, y = _make_collider(seed)
    xr, yr = _residualize(x, G), _residualize(y, G)
    naive, controlled = _corr(x, y), _corr(xr, yr)

    #      X    Y   caption                                              naive  ctrl   vl     hl     groups
    seq = [
        (x,  y,  f"Raw data. Correlation: {naive:+.2f}",                 False, False, False, False, False),
        (x,  y,  "But stars are the top 15% on beauty + talent.",        False, False, False, False, False),
        (x,  y,  "A naive line through all the points.",                 True,  False, False, False, False),
        (x,  y,  "Both groups tilt down, not just the stars.",           True,  False, False, False, True),
        (xr, y,  "Remove the part of beauty explained by stardom.",      False, False, True,  False, False),
        (xr, yr, "Remove the part of talent explained by stardom.",      False, False, True,  True,  False),
        (xr, yr, "Both residuals now average exactly zero.",             False, True,  True,  True,  True),
        (xr, yr, f"Correlation controlling for stardom: {controlled:+.2f}", False, True, True, True, True),
    ]
    stages = [(a, b, f"{i}. {t}", nf, cf, vl, hl, gg)
              for i, (a, b, t, nf, cf, vl, hl, gg) in enumerate(seq, start=1)]

    allx = np.concatenate([x, xr]); ally = np.concatenate([y, yr])
    xlim = (allx.min() - 0.6, allx.max() + 0.6)
    ylim = (ally.min() - 0.6, ally.max() + 0.6)
    colors = np.where(G == 1, ACCENT, MADRID)

    frames = []
    for i, st in enumerate(stages):
        if i > 0:
            px, py = stages[i - 1][0], stages[i - 1][1]
            for f in range(_TWEEN):
                a = _ease((f + 1) / _TWEEN)
                frames.append((px + (st[0] - px) * a, py + (st[1] - py) * a,
                               st[2], st[3], st[4], st[5], st[6], st[7], a))
        for _ in range(_HOLD):
            frames.append((st[0], st[1], st[2], st[3], st[4], st[5], st[6],
                           st[7], 1.0))

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=_ANIM_DPI)
    fig.patch.set_facecolor("white")

    def draw(fr):
        ax.clear()
        Xa, Ya, cap, nf, cf, vl, hl, gg, af = fr
        ax.set_facecolor("white")
        ax.grid(True, color=GRIDCOL, linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        if vl:
            ax.axvline(0, color="#7A7A85", linewidth=1.0, zorder=1)
        if hl:
            ax.axhline(0, color="#7A7A85", linewidth=1.0, zorder=1)
        ax.scatter(Xa, Ya, s=_POINT_SIZE, c=colors, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=3)
        xx = np.array(xlim)
        if nf:
            b, a0 = np.polyfit(x, y, 1)
            ax.plot(xx, a0 + b * xx, color=FITLINE, lw=2.4, alpha=af, zorder=4)
        if cf:
            b, a0 = np.polyfit(xr, yr, 1)
            ax.plot(xx, a0 + b * xx, color=FITLINE, lw=2.4, alpha=af, zorder=4)
        if gg:
            # fit on what is currently drawn so the lines track through tweens
            for lvl, col in ((0, MADRID), (1, ACCENT)):
                m = G == lvl
                b, a0 = np.polyfit(Xa[m], Ya[m], 1)
                ax.plot(xx, a0 + b * xx, color=col, lw=2.6, alpha=af, zorder=5)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("Beauty", fontsize=13)
        ax.text(0.0, 1.02, "Talent", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=12, color=INK)
        ax.set_title("Controlling for movie-star status", fontsize=15,
                     color=MADRID, pad=44, weight="bold")
        ax.text(0.5, 1.075, cap, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=11.5, color=INK)
        ax.scatter([], [], c=MADRID, s=_POINT_SIZE, label="Not a star")
        ax.scatter([], [], c=ACCENT, s=_POINT_SIZE, label="Movie star")
        ax.legend(loc="upper right", frameon=True, fontsize=10, framealpha=0.9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, draw, frames=frames,
                                   interval=1000 / _FPS)
    plt.close(fig)
    return anim


def load_icecream():
    """Monthly US panel, 1999-01 to 2020-12 (N=264).

    icecream  FRED IPN31152N, ice cream and frozen dessert production, NSA index
    drownings CDC WONDER D76, accidental drowning and submersion, W65-W74
    tempF     NOAA Climate at a Glance, contiguous-US monthly mean temperature
    """
    import pandas as pd
    if "df" not in _real_cache:
        _real_cache["df"] = pd.read_csv(_DATA / "icecream_drownings_temp.csv",
                                        parse_dates=["date"])
    return _real_cache["df"]


def fig_icecream_scatter():
    """The hook: real ice cream output against real drowning deaths, one colour.
    The confounder is deliberately NOT revealed yet."""
    df = load_icecream()
    x, y = df["icecream"].to_numpy(float), df["drownings"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.0, 4.9), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRIDCOL, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.scatter(x, y, s=_POINT_SIZE, c=MADRID, alpha=0.8, edgecolors="white",
               linewidths=0.5, zorder=3)
    b, a0 = np.polyfit(x, y, 1)
    xx = np.array([x.min() - 4, x.max() + 4])
    ax.plot(xx, a0 + b * xx, color=FITLINE, linewidth=2.4, zorder=4)
    ax.set_xlim(*xx)
    ax.set_xlabel("Ice cream production (index, 2017=100)", fontsize=12)
    ax.text(0.0, 1.02, "Drowning deaths", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("Ice cream and drowning, United States, 1999 to 2020",
                 fontsize=15, color=MADRID, pad=40, weight="bold")
    ax.text(0.5, 1.065,
            f"Monthly, N = {len(df)}.  Correlation: {_corr(x, y):+.2f}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=11.5,
            color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def _residual_animation(x, y, w, *, title, cbar_label, xlab_raw, xlab_res,
                        ylab_raw, ylab_res, captions, fixed_scale=False):
    """The residualize-X, residualize-Y sequence on real data, coloured by the
    control. Shared by the confounder and mediator animations, which differ only
    in their data, labels, and moral.

    captions     : the 7 stage captions, each already carrying its own number
    fixed_scale  : hold the axis SPAN constant and only translate the view.

    fixed_scale matters whenever the moral of the animation is the slope. The
    on-screen tilt of a line is the coefficient times the x span over the y
    span, so if each stage rescales to fit its own cloud, a coefficient can fall
    by half and the line will look untouched. Holding the span fixed makes the
    picture move with the number instead of hiding it.
    """
    xr, yr = _partial(x, w), _partial(y, w)

    def _lims(v, pad=0.07):
        lo, hi = float(np.min(v)), float(np.max(v))
        p = pad * (hi - lo)
        return (lo - p, hi + p)

    #        X    Y   naive  ctrl   vline  hline
    flags = [(x,  y,  False, False, False, False),
             (x,  y,  False, False, False, False),
             (x,  y,  True,  False, False, False),
             (xr, y,  False, False, True,  False),
             (xr, yr, False, False, True,  True),
             (xr, yr, False, True,  True,  True),
             (xr, yr, False, True,  True,  True)]
    stages = [(a, b, cap, nf, cf, vl, hl)
              for (a, b, nf, cf, vl, hl), cap in zip(flags, captions)]

    # The axes travel with the data: one fixed range spanning both the raw and
    # residual clouds would strand the residuals in a corner.
    st_xlim = [_lims(s[0]) for s in stages]
    st_ylim = [_lims(s[1]) for s in stages]
    if fixed_scale:
        def _recentre(lims):
            span = max(hi - lo for lo, hi in lims)
            return [((lo + hi) / 2 - span / 2, (lo + hi) / 2 + span / 2)
                    for lo, hi in lims]
        st_xlim, st_ylim = _recentre(st_xlim), _recentre(st_ylim)
    st_xlab = [xlab_raw if s[0] is x else xlab_res for s in stages]
    st_ylab = [ylab_raw if s[1] is y else ylab_res for s in stages]

    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(w.min(), w.max())

    def _blend(p, c, a):
        return (p[0] + (c[0] - p[0]) * a, p[1] + (c[1] - p[1]) * a)

    frames = []
    for i, st in enumerate(stages):
        if i > 0:
            px, py = stages[i - 1][0], stages[i - 1][1]
            for f in range(_TWEEN):
                a = _ease((f + 1) / _TWEEN)
                frames.append(dict(
                    X=px + (st[0] - px) * a, Y=py + (st[1] - py) * a,
                    cap=st[2], naive=st[3], ctrl=st[4], vl=st[5], hl=st[6],
                    xlim=_blend(st_xlim[i - 1], st_xlim[i], a),
                    ylim=_blend(st_ylim[i - 1], st_ylim[i], a),
                    xlab=st_xlab[i], ylab=st_ylab[i], alpha=a, line_alpha=0.0))
        # fit line fades in and out during the hold, never while the cloud moves
        for h in range(_HOLD):
            fade = 6
            la = (_ease((h + 1) / fade) if h < fade else
                  _ease((_HOLD - h) / fade) if h >= _HOLD - fade else 1.0)
            frames.append(dict(
                X=st[0], Y=st[1], cap=st[2], naive=st[3], ctrl=st[4],
                vl=st[5], hl=st[6], xlim=st_xlim[i], ylim=st_ylim[i],
                xlab=st_xlab[i], ylab=st_ylab[i], alpha=1.0, line_alpha=la))

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=_ANIM_DPI)
    fig.patch.set_facecolor("white")
    # the title sits 44pt above the axes, so reserve the room before the
    # colorbar steals its slice; tight_layout cannot run per frame without the
    # axes jittering
    fig.subplots_adjust(left=0.14, right=0.95, top=0.80, bottom=0.12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.ax.set_title(cbar_label, fontsize=9.5, color=INK, pad=6)
    cbar.ax.tick_params(labelsize=9)

    def draw(fr):
        ax.clear()
        ax.set_facecolor("white")
        ax.grid(True, color=GRIDCOL, linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        # zero lines appear as each variable becomes a residual, so the cloud is
        # visibly settling onto mean zero rather than just being told to
        if fr["vl"]:
            ax.axvline(0, color="#7A7A85", linewidth=1.0, zorder=1)
        if fr["hl"]:
            ax.axhline(0, color="#7A7A85", linewidth=1.0, zorder=1)
        ax.scatter(fr["X"], fr["Y"], s=_POINT_SIZE, c=w, cmap=cmap, norm=norm,
                   alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
        xx = np.array(fr["xlim"])
        if fr["naive"] and fr["line_alpha"] > 0:
            b, a0 = np.polyfit(x, y, 1)
            ax.plot(xx, a0 + b * xx, color=FITLINE, lw=2.4,
                    alpha=fr["line_alpha"], zorder=4)
        if fr["ctrl"] and fr["line_alpha"] > 0:
            b, a0 = np.polyfit(xr, yr, 1)
            ax.plot(xx, a0 + b * xx, color=FITLINE, lw=2.4,
                    alpha=fr["line_alpha"], zorder=4)
        ax.set_xlim(*fr["xlim"]); ax.set_ylim(*fr["ylim"])
        ax.set_xlabel(fr["xlab"], fontsize=12)
        ax.text(0.0, 1.02, fr["ylab"], transform=ax.transAxes, ha="left",
                va="bottom", fontsize=12, color=INK)
        ax.set_title(title, fontsize=15, color=MADRID, pad=28, weight="bold")
        ax.text(0.5, 1.075, fr["cap"], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=10.5, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    anim = animation.FuncAnimation(fig, draw, frames=frames,
                                   interval=1000 / _FPS)
    plt.close(fig)
    return anim


def confounder_animation():
    """FORK. Controlling for the confounder REMOVES a spurious association.

    Colour is the payoff the simulated version cannot show: the raw cloud is a
    blue to red diagonal, and by the last frame the colours are scrambled, which
    is what "temperature no longer explains where a point sits" looks like.

    Honest caveat for the slide: this lands at +0.15, not 0. Real monthly
    aggregates carry residual seasonality and trend that mean temperature alone
    does not absorb.
    """
    df = load_icecream()
    x = df["icecream"].to_numpy(float)
    y = df["drownings"].to_numpy(float)
    w = df["tempF"].to_numpy(float)
    naive, controlled = _corr(x, y), _corr(_partial(x, w), _partial(y, w))
    return _residual_animation(
        x, y, w,
        title="Controlling for temperature",
        cbar_label="Mean temp (F)",
        xlab_raw="Ice cream production (index, 2017=100)",
        xlab_res="Ice cream production, residual",
        ylab_raw="Drowning deaths", ylab_res="Drowning deaths, residual",
        captions=[
            f"1. Raw monthly data. Correlation: {naive:+.2f}",
            "2. But hot months push up both ice cream and drownings.",
            "3. A naive line through all the points.",
            "4. Remove the part of ice cream explained by temperature.",
            "5. Remove the part of drownings explained by temperature.",
            "6. Both residuals now average exactly zero.",
            f"7. Correlation controlling for temperature: {controlled:+.2f}",
        ])


def mediator_animation():
    """CHAIN. Controlling for the mediator DESTROYS part of a real effect.

    Same machinery as the confounder animation, opposite moral. The fitted line
    flattens from the total effect of age on log wage (+0.022) to the direct
    path only (+0.013), because conditioning on schooling and experience closes
    the two indirect routes. Those are exactly the numbers on the path diagram,
    which is the point: this is that arithmetic happening on screen.
    """
    import pandas as pd
    z = pd.read_csv(_DATA / "wage2_blackburn_neumark.csv")[
        ["lwage", "educ", "exper", "age"]].dropna()
    x = z["age"].to_numpy(float)
    y = z["lwage"].to_numpy(float)
    M = z[["educ", "exper"]].to_numpy(float)     # the two mediators
    w = z["educ"].to_numpy(float)                # colour by schooling

    total = np.polyfit(x, y, 1)[0]
    direct = np.polyfit(_partial(x, M), _partial(y, M), 1)[0]
    return _residual_animation(
        x, y, w, cbar_label="Years of school",
        title="Controlling for education and experience",
        xlab_raw="Age", xlab_res="Age, residual",
        ylab_raw="Log wage", ylab_res="Log wage, residual",
        fixed_scale=True,
        captions=[
            f"1. Raw data. Effect of age on log wage: {total:+.3f}",
            "2. But age also buys schooling and experience.",
            "3. The total effect: every route from age to wages.",
            "4. Remove the part of age explained by the mediators.",
            "5. Remove the part of log wage explained by the mediators.",
            "6. Both residuals now average exactly zero.",
            f"7. Direct path only: {direct:+.3f}. The rest was thrown away.",
        ])


def wage_paths():
    """Path coefficients from the Blackburn and Neumark data, in raw units.

    Estimated live rather than hard coded so the slide can never drift from the
    file in data/. Returns (paths dict, direct effect, total effect).
    """
    import pandas as pd
    z = pd.read_csv(_DATA / "wage2_blackburn_neumark.csv")[
        ["lwage", "educ", "exper", "age"]].dropna()

    def _ols(yname, xnames):
        A = np.column_stack([np.ones(len(z))] + [z[c].to_numpy() for c in xnames])
        beta, *_ = np.linalg.lstsq(A, z[yname].to_numpy(), rcond=None)
        return dict(zip(xnames, beta[1:]))

    full = _ols("lwage", ["age", "educ", "exper"])
    paths = {"age_edu": _ols("educ", ["age"])["age"],
             "age_exp": _ols("exper", ["age"])["age"],
             "edu_earn": full["educ"], "exp_earn": full["exper"]}
    return paths, full["age"], _ols("lwage", ["age"])["age"]


def dag_mediator_paths():
    """The way around over-control: label every path, then add them up.

    Coefficients are estimated from the Blackburn and Neumark (1992) sample,
    N = 935 young men from the NLS, outcome is log wage.
    """
    p, direct, _total = wage_paths()
    return draw_dag(
        [("Age", "Education"), ("Age", "Experience"),
         ("Education", "Earnings"), ("Experience", "Earnings"),
         ("Age", "Earnings")],
        edge_labels={
            ("Age", "Education"): f"{p['age_edu']:+.3f}",
            ("Age", "Experience"): f"{p['age_exp']:+.3f}",
            ("Education", "Earnings"): f"{p['edu_earn']:+.3f}",
            ("Experience", "Earnings"): f"{p['exp_earn']:+.3f}",
            ("Age", "Earnings"): f"{direct:+.3f} direct"})


# ================================================ degrees-of-freedom geometry
def fig_dof_geometry():
    """Why n-1: the deviation vector is perpendicular to the ones vector.

    Drawn for n=2 (the pizza {8,5} case), the only case that fits on a slide.
    Deviation space has axes (x1 - xbar) and (x2 - xbar). The constraint
    sum(x_i - xbar) = 0 is one line through the origin, perpendicular to the
    ones vector 1 = (1, 1). Every possible deviation vector must lie ON that
    line, so it has exactly one free direction: n - 1 = 1.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    L = 3.0
    ax.axhline(0, color="#B8B8C2", lw=1.0, zorder=0)
    ax.axvline(0, color="#B8B8C2", lw=1.0, zorder=0)

    # constraint line d1 + d2 = 0  (i.e. d2 = -d1): the allowed subspace
    t = np.array([-L, L])
    ax.plot(t, -t, color=MADRID, lw=2.6, zorder=2,
            label="constraint  $d_1 + d_2 = 0$")

    # the ones vector, the locked direction
    ax.annotate("", xy=(1.5, 1.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#7A7A85"))
    ax.text(1.62, 1.62, r"$\mathbf{1}=(1,1)$", color="#5A5A64",
            fontsize=13, ha="left", va="center")

    # the pizza {8,5} deviation vector: xbar=6.5, deviations (1.5, -1.5)
    ax.annotate("", xy=(1.5, -1.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=2.8, color=FITLINE))
    ax.plot([1.5], [-1.5], "o", color=FITLINE, ms=7, zorder=4)
    ax.text(1.62, -1.62, r"$\mathbf{d}=(1.5,\,-1.5)$", color=FITLINE,
            fontsize=13, ha="left", va="center")

    # right-angle tick where the two directions meet
    ax.plot([0.24, 0.42, 0.18], [-0.24, -0.06, 0.0], color="#5A5A64", lw=1.2,
            zorder=3)

    ax.set_xlim(-L, L); ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1 - \bar{x}$", fontsize=13)
    ax.text(-0.12, 0.98, r"$x_2 - \bar{x}$", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=13, color=INK)
    ax.set_title("One constraint locks one direction",
                 fontsize=15, color=MADRID, pad=30, weight="bold")
    ax.text(0.5, -0.16, "The deviation vector may point anywhere ON the blue "
            "line: one free direction, so $n-1 = 1$.",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            color=INK)
    ax.legend(loc="upper left", frameon=True, fontsize=10, framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============================================================ bias-variance
# The prediction-side bias-variance trade-off, simulated rather than sketched.
# Polynomials of increasing degree are fit to noisy draws from a fixed truth,
# and the expected test error is split into bias squared, variance, and the
# irreducible noise.
_BV_SIGMA = 0.35
_BV_DEGREES = list(range(1, 10))


def _bv_truth(x):
    return np.sin(2.0 * np.pi * x) + 0.4 * x


def _bv_simulate(n_train=40, n_reps=250, seed=0):
    """Return (degrees, bias2, variance, train_mse, test_mse)."""
    rng = np.random.default_rng(seed)
    xg = np.linspace(0.02, 0.98, 60)          # where we evaluate
    truth = _bv_truth(xg)
    bias2, var, trmse = [], [], []
    for d in _BV_DEGREES:
        preds = np.empty((n_reps, len(xg)))
        tr = np.empty(n_reps)
        for r in range(n_reps):
            xt = rng.uniform(0, 1, n_train)
            yt = _bv_truth(xt) + rng.normal(0, _BV_SIGMA, n_train)
            p = np.polynomial.Polynomial.fit(xt, yt, d)
            preds[r] = p(xg)
            tr[r] = np.mean((yt - p(xt)) ** 2)
        mean_pred = preds.mean(axis=0)
        bias2.append(float(np.mean((mean_pred - truth) ** 2)))
        var.append(float(np.mean(preds.var(axis=0))))
        trmse.append(float(tr.mean()))
    bias2 = np.array(bias2); var = np.array(var)
    test = bias2 + var + _BV_SIGMA ** 2
    return np.array(_BV_DEGREES), bias2, var, np.array(trmse), test


_bv_cache = {}


def _bv():
    if "r" not in _bv_cache:
        _bv_cache["r"] = _bv_simulate()
    return _bv_cache["r"]


def fig_bias_variance_curves():
    """Training error falls forever; test error is U shaped."""
    d, b2, v, tr, te = _bv()
    best = int(d[np.argmin(te)])
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(d, tr, "o-", color=ACCENT, lw=2.4, ms=6, zorder=3,
            label="training error")
    ax.plot(d, te, "o-", color=FITLINE, lw=2.6, ms=6, zorder=3,
            label="test error")
    ax.axhline(_BV_SIGMA ** 2, color="#7A7A85", ls=(0, (5, 4)), lw=1.2,
               zorder=2)
    ax.text(d[-1], _BV_SIGMA ** 2 * 1.35, "irreducible noise", ha="right",
            fontsize=10, color="#5A5A64")
    ax.axvline(best, color=MADRID, ls=(0, (4, 3)), lw=1.4)
    ax.text(best + 0.15, ax.get_ylim()[1] * 0.72, "best flexibility",
            fontsize=10.5, color=MADRID)
    ax.set_yscale("log")
    ax.set_xlabel("Model flexibility (polynomial degree)", fontsize=12)
    ax.text(0, 1.02, "Mean squared error (log scale)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, color=INK)
    ax.set_title("Underfitting on the left, overfitting on the right",
                 fontsize=14, color=MADRID, weight="bold", pad=26)
    ax.legend(loc="upper center", fontsize=10, framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_bias_variance_decomposition():
    """The same test error, split into its three parts."""
    d, b2, v, tr, te = _bv()
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.stackplot(d, b2, v, np.full_like(b2, _BV_SIGMA ** 2),
                 colors=[MADRID, ACCENT, "#C9C9D2"],
                 labels=["bias$^2$", "variance", "irreducible"], zorder=2)
    ax.plot(d, te, color=FITLINE, lw=2.6, zorder=4, label="test error")
    ax.set_ylim(0, min(te.max() * 1.6, np.percentile(te, 90) * 2.2))
    ax.set_xlabel("Model flexibility (polynomial degree)", fontsize=12)
    ax.text(0, 1.02, "Expected test error", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, color=INK)
    ax.set_title("Test error $=$ bias$^2$ $+$ variance $+$ irreducible noise",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    ax.legend(loc="upper center", fontsize=10, framealpha=0.92, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_bias_variance_fits(degree, n_show=12, seed=1):
    """Many fitted curves at one flexibility, against the truth. Low degree:
    all the curves agree with each other but miss the truth (bias). High
    degree: they scatter wildly (variance)."""
    rng = np.random.default_rng(seed)
    xg = np.linspace(0.02, 0.98, 200)
    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
    fits = []
    for _ in range(n_show):
        xt = rng.uniform(0, 1, 40)
        yt = _bv_truth(xt) + rng.normal(0, _BV_SIGMA, 40)
        f = np.polynomial.Polynomial.fit(xt, yt, degree)(xg)
        fits.append(f)
        ax.plot(xg, f, color=ACCENT, lw=1.3, alpha=0.5, zorder=3)
    ax.plot(xg, np.mean(fits, axis=0), color=FITLINE, lw=2.6, zorder=5,
            label="average fit")
    ax.plot(xg, _bv_truth(xg), color=MADRID, lw=2.8, zorder=4, label="truth")
    ax.set_ylim(-2.2, 2.4)
    ax.set_xlabel("x", fontsize=12)
    ax.text(0, 1.02, "y", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("Degree %d: %d fits on %d different samples"
                 % (degree, n_show, n_show), fontsize=13.5, color=MADRID,
                 weight="bold", pad=26)
    ax.legend(loc="lower center", fontsize=10, framealpha=0.9, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============ double machine learning and causal forests ======================
import pandas as pd

GRID = GRIDCOL
_GREEN = "#2E8B57"

# One partially linear design carries both decks:
#     T = m(X) + V,      Y = theta(X) * T + g(X) + U
# with m and g nonlinear in X, so a linear control strategy cannot work and a
# flexible first stage is not optional.
_DML_TRUE = 1.0


def _dml_m(X):
    return 1.0 * np.sin(X[:, 0]) + 0.5 * X[:, 1]


def _dml_g(X):
    return 2.0 * np.sin(X[:, 0]) + (X[:, 1] ** 2 - 1.0)


def dml_data(n=1000, p=6, seed=0, theta=None):
    """Partially linear data. Pass theta=callable for a heterogeneous effect."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    T = _dml_m(X) + rng.normal(0, 1.0, n)
    th = _DML_TRUE if theta is None else theta(X)
    Y = th * T + _dml_g(X) + rng.normal(0, 1.0, n)
    return X, T, Y


def _forest(trees=40, leaf=5):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=trees, min_samples_leaf=leaf,
                                 random_state=0, n_jobs=-1)


def dml_residualize(X, T, Y, k=5, model=None):
    """Cross-fit residuals: every prediction comes from a model that never saw
    the row it is predicting. This is the whole trick."""
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.base import clone
    m = _forest() if model is None else model
    kf = KFold(k, shuffle=True, random_state=1)
    t_hat = cross_val_predict(clone(m), X, T, cv=kf)
    y_hat = cross_val_predict(clone(m), X, Y, cv=kf)
    return T - t_hat, Y - y_hat


def dml_ate(X, T, Y, k=5):
    """Point estimate and standard error from the orthogonal moment."""
    rt, ry = dml_residualize(X, T, Y, k=k)
    theta = float(np.sum(rt * ry) / np.sum(rt ** 2))
    psi = rt * (ry - theta * rt)
    se = float(np.sqrt(np.mean(psi ** 2) / np.mean(rt ** 2) ** 2 / len(rt)))
    return theta, se


# ---------------------------------------------- the sampling-distribution study
_DML_SIM_CSV = _DATA / "dml_sim.csv"
_DML_SIM = None


def dml_simulate(reps=60, n=1000, p=6, force=False):
    """Three estimators of the same theta = 1.0, over repeated samples.

      ols   OLS of Y on T and X entered linearly
      naive removes g-hat(X) from Y but regresses on RAW T (not orthogonal)
      dml   residualizes both, cross-fit (orthogonal)
    """
    global _DML_SIM
    if _DML_SIM is not None and not force:
        return _DML_SIM
    if _DML_SIM_CSV.exists() and not force:
        _DML_SIM = pd.read_csv(_DML_SIM_CSV)
        return _DML_SIM
    from sklearn.linear_model import LinearRegression
    rows = []
    for r in range(reps):
        X, T, Y = dml_data(n=n, p=p, seed=1000 + r)
        rt, ry = dml_residualize(X, T, Y)
        rows.append(dict(
            ols=float(LinearRegression().fit(np.column_stack([T, X]), Y).coef_[0]),
            naive=float(np.sum(T * ry) / np.sum(T * T)),
            dml=float(np.sum(rt * ry) / np.sum(rt ** 2)),
        ))
    _DML_SIM = pd.DataFrame(rows)
    _DML_SIM.to_csv(_DML_SIM_CSV, index=False)
    return _DML_SIM


def fig_dml_sampling():
    """Where each estimator lands, over 60 fresh samples. Only the orthogonal,
    cross-fit one is centred on the truth."""
    d = dml_simulate()
    specs = [("ols", "OLS, X entered linearly", ACCENT),
             ("naive", "residualize Y only", FITLINE),
             ("dml", "residualize both (DML)", _GREEN)]
    fig, ax = plt.subplots(figsize=(8.0, 4.3), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    bins = np.linspace(min(d.min()) - 0.06, max(d.max()) + 0.06, 34)
    for col, lab, c in specs:
        ax.hist(d[col], bins=bins, color=c, alpha=0.62, zorder=3,
                label="%s\nmean %.2f" % (lab, d[col].mean()))
    ax.axvline(_DML_TRUE, color=INK, lw=2.2, ls=(0, (5, 4)), zorder=5)
    ax.text(_DML_TRUE + 0.015, ax.get_ylim()[1] * 0.92, "truth", fontsize=11.5,
            color=INK)
    ax.set_xlabel(r"estimate of $\theta$", fontsize=12)
    ax.text(0, 1.02, "count over 60 samples", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("Orthogonalization is what removes the bias", fontsize=14,
                 color=MADRID, weight="bold", pad=26)
    ax.legend(fontsize=9, framealpha=0.92, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_dml_first_stage():
    """The two nuisance functions the first stage has to learn, each shown
    against the covariate that bends it. A line can nearly manage the first and
    cannot manage the second at all."""
    from sklearn.linear_model import LinearRegression
    X, T, Y = dml_data(n=1500, seed=4)
    panels = [
        (0, T, "treatment T", r"$X_1$", r"$E[T \mid X]$",
         lambda gr: _dml_m(gr)),
        (1, Y, "outcome Y", r"$X_2$", r"$E[Y \mid X]$",
         lambda gr: _DML_TRUE * _dml_m(gr) + _dml_g(gr)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=110)
    fig.patch.set_facecolor("white")
    for ax, (j, target, ylab, xlab, ttl, truth) in zip(axes, panels):
        lin = LinearRegression().fit(X, target)
        rf = _forest(trees=200, leaf=10).fit(X, target)
        # hold the other covariates at zero so each panel is about one axis
        grid = np.zeros((120, X.shape[1]))
        grid[:, j] = np.linspace(np.quantile(X[:, j], 0.02),
                                 np.quantile(X[:, j], 0.98), 120)
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.scatter(X[:, j], target, s=10, color="#D6D6DE", zorder=2)
        ax.plot(grid[:, j], truth(grid), color=INK, lw=3.0, ls=(0, (5, 4)),
                zorder=6, label="truth")
        ax.plot(grid[:, j], lin.predict(grid), color=ACCENT, lw=2.6, zorder=4,
                label="linear control")
        ax.plot(grid[:, j], rf.predict(grid), color=_GREEN, lw=2.4, alpha=0.95,
                zorder=5, label="random forest")
        ax.set_xlabel(xlab, fontsize=12)
        ax.text(0, 1.03, ylab, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11.5, color=INK)
        ax.set_title(ttl, fontsize=13.5, color=MADRID, weight="bold", pad=26)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=9.5, framealpha=0.92, loc="upper left")
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_dml_residual_scatter():
    """Raw data gives the wrong slope; the residualized data gives the right
    one. Same 600 rows in both panels."""
    X, T, Y = dml_data(n=600, seed=4)
    rt, ry = dml_residualize(X, T, Y)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), dpi=110)
    fig.patch.set_facecolor("white")
    for ax, (a, b, lab_x, lab_y, ttl) in zip(axes, [
            (T, Y, "treatment T", "outcome Y", "Raw"),
            (rt, ry, "residualized T", "residualized Y", "Orthogonalized")]):
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.scatter(a, b, s=17, color="#9E9E9E", alpha=0.75, zorder=2)
        slope = float(np.sum((a - a.mean()) * (b - b.mean())
                             ) / np.sum((a - a.mean()) ** 2))
        gx = np.linspace(a.min(), a.max(), 50)
        ax.plot(gx, b.mean() + slope * (gx - a.mean()), color=FITLINE, lw=2.8,
                zorder=4)
        ax.plot(gx, b.mean() + _DML_TRUE * (gx - a.mean()), color=INK, lw=2.0,
                ls=(0, (5, 4)), zorder=5)
        ax.set_xlabel(lab_x, fontsize=11.5)
        ax.text(0, 1.03, lab_y, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11.5, color=INK)
        ax.set_title("%s: slope %.2f" % (ttl, slope), fontsize=13,
                     color=FITLINE if abs(slope - 1) > 0.1 else _GREEN,
                     weight="bold", pad=26)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Dashed line is the truth", fontsize=12, color=INK, y=1.02)
    fig.tight_layout(); plt.close(fig)
    return fig


# ---------------------------------------------------------- causal forests ----
def _cf_theta(X):
    """A smoothly heterogeneous effect: the treatment works better at high X1."""
    return 0.5 + 1.0 * X[:, 0]


def cf_data(n=2000, p=6, seed=11):
    return dml_data(n=n, p=p, seed=seed, theta=_cf_theta)


def fig_causal_tree_split(seed=3):
    """A regression tree splits where the OUTCOME differs. A causal tree splits
    where the EFFECT differs. Those are not the same place."""
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.uniform(-2, 2, n)
    t = rng.integers(0, 2, n)
    eff = 0.5 + 1.0 * x                     # effect rises with x
    level = 3.0 * np.exp(-((x + 1.0) ** 2))  # outcome level peaks at x = -1
    y = level + eff * t + rng.normal(0, 0.35, n)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), dpi=110)
    fig.patch.set_facecolor("white")
    for ax, mode in zip(axes, ("level", "effect")):
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.scatter(x[t == 0], y[t == 0], s=18, color=_CTRL, alpha=0.8, zorder=2,
                   label="control")
        ax.scatter(x[t == 1], y[t == 1], s=18, color=_TREAT, alpha=0.8, zorder=2,
                   label="treated")
        cut = -1.0 if mode == "level" else 0.0
        ax.axvline(cut, color=MADRID if mode == "effect" else ACCENT, lw=2.6,
                   zorder=5)
        ax.set_xlabel(r"$X_1$", fontsize=11.5)
        ax.set_title("Split on outcome level" if mode == "level"
                     else "Split on treatment effect", fontsize=13,
                     color=ACCENT if mode == "level" else MADRID,
                     weight="bold", pad=26)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(0, 1.04, "outcome Y", transform=axes[0].transAxes, ha="left",
                 va="bottom", fontsize=11.5, color=INK)
    axes[0].legend(fontsize=9.5, framealpha=0.92, loc="upper right")
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_honesty():
    """Honest splitting: one half of the data chooses the splits, the other half
    fills in the leaf effects."""
    fig, ax = plt.subplots(figsize=(8.8, 3.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white"); ax.axis("off")
    ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.2, 3.2)
    boxes = [(0.0, "Sample", "#DCE3F2", INK),
             (3.4, "Half A\nchooses the splits", ACCENT, "white"),
             (6.8, "Half B\nestimates leaf effects", _GREEN, "white")]
    for x0, lab, fc, tc in boxes:
        ax.add_patch(plt.Rectangle((x0, 1.0), 2.6, 1.2, facecolor=fc,
                                   edgecolor=INK, lw=1.2))
        ax.text(x0 + 1.3, 1.6, lab, ha="center", va="center", fontsize=11.5,
                color=tc, weight="bold")
    for x0 in (2.6, 6.0):
        ax.annotate("", xy=(x0 + 0.75, 1.6), xytext=(x0, 1.6),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=INK))
    ax.text(5.0, 0.45, "no observation does both jobs, so a split chosen by "
                       "chance cannot inflate its own estimate",
            ha="center", fontsize=11, color=INK, style="italic")
    ax.set_title("Honesty", fontsize=14, color=MADRID, weight="bold", pad=4)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_honesty_payoff(reps=200, seed=0):
    """With NO real heterogeneity, adaptive leaves that estimate their own
    effect spread out. Honest leaves do not."""
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(seed)
    adaptive, honest = [], []
    for _ in range(reps):
        n, p = 400, 5
        X = rng.normal(size=(n, p))
        T = rng.integers(0, 2, n).astype(float)
        Y = 1.0 * T + rng.normal(0, 1.0, n)          # constant effect, no het
        half = n // 2
        a, b = slice(0, half), slice(half, n)
        # pseudo outcome for a constant propensity of 0.5
        ps = (T - 0.5) / 0.25
        star = ps * Y
        tree = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=20,
                                     random_state=0).fit(X[a], star[a])
        la, lb = tree.apply(X[a]), tree.apply(X[b])
        for leaf in np.unique(la):
            adaptive.append(float(np.mean(star[a][la == leaf])))
            sel = lb == leaf
            if sel.sum() >= 5:
                honest.append(float(np.mean(star[b][sel])))
    fig, ax = plt.subplots(figsize=(7.6, 4.1), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    lo = min(min(adaptive), min(honest)); hi = max(max(adaptive), max(honest))
    bins = np.linspace(lo, hi, 40)
    ax.hist(adaptive, bins=bins, color=FITLINE, alpha=0.6, zorder=3,
            label="same data splits and estimates\nsd %.2f" % np.std(adaptive))
    ax.hist(honest, bins=bins, color=_GREEN, alpha=0.6, zorder=3,
            label="honest: held-out half estimates\nsd %.2f" % np.std(honest))
    ax.axvline(1.0, color=INK, lw=2.2, ls=(0, (5, 4)), zorder=5)
    ax.text(hi, ax.get_ylim()[1] * 0.62, "true effect, the same\nin every leaf",
            ha="right", fontsize=10.5, color=INK)
    ax.set_xlabel("estimated effect in a leaf", fontsize=12)
    ax.text(0, 1.02, "count over leaves", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("Heterogeneity that is not there", fontsize=14, color=MADRID,
                 weight="bold", pad=26)
    ax.legend(fontsize=9, framealpha=0.92, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_forest_weights(seed=2):
    """A forest is an adaptive neighbourhood: the rows that keep landing in the
    same leaf as the target get the weight."""
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(seed)
    n = 300
    X = rng.uniform(-2, 2, (n, 2))
    y = np.sin(1.5 * X[:, 0]) + 0.3 * X[:, 1] + rng.normal(0, 0.3, n)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=8,
                               random_state=0).fit(X, y)
    target = np.array([[0.8, -0.4]])
    leaves = rf.apply(X)
    tleaf = rf.apply(target)[0]
    w = (leaves == tleaf).mean(axis=1)
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    sc = ax.scatter(X[:, 0], X[:, 1], c=w, s=26 + 320 * w, cmap="viridis",
                    zorder=3, edgecolors="none")
    ax.scatter(target[:, 0], target[:, 1], marker="*", s=420, color=FITLINE,
               edgecolors=INK, lw=1.2, zorder=6)
    ax.annotate("target x", xy=(target[0, 0], target[0, 1]),
                xytext=(target[0, 0] + 0.25, target[0, 1] - 0.55),
                fontsize=11.5, color=FITLINE, weight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=FITLINE))
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("share of trees sharing a leaf", fontsize=10.5)
    ax.set_xlabel(r"$X_1$", fontsize=11.5)
    ax.text(0, 1.03, r"$X_2$", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11.5, color=INK)
    ax.set_title("The forest picks its own neighbourhood", fontsize=13.5,
                 color=MADRID, weight="bold", pad=24)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def causal_forest(X, T, Y, trees=300, leaf=20, k=5):
    """A causal forest, spelled out. Local centering by cross-fit residuals,
    then a weighted forest on the pseudo outcome. Returns a predict function."""
    from sklearn.ensemble import RandomForestRegressor
    rt, ry = dml_residualize(X, T, Y, k=k)
    w = rt ** 2
    star = ry / np.where(np.abs(rt) < 1e-8, 1e-8, rt)
    f = RandomForestRegressor(n_estimators=trees, min_samples_leaf=leaf,
                              random_state=0, n_jobs=-1)
    f.fit(X, star, sample_weight=w)
    return f


def fig_causal_forest_cate(n=2000, seed=11):
    """Recovered effect against the true effect, row by row."""
    X, T, Y = cf_data(n=n, seed=seed)
    f = causal_forest(X, T, Y)
    pred = f.predict(X)
    truth = _cf_theta(X)
    const, _ = dml_ate(X, T, Y)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.scatter(truth, pred, s=15, color=_GREEN, alpha=0.5, zorder=3,
               label="causal forest")
    lo, hi = truth.min(), truth.max()
    ax.plot([lo, hi], [lo, hi], color=INK, lw=2.2, ls=(0, (5, 4)), zorder=5,
            label="perfect recovery")
    ax.axhline(const, color=ACCENT, lw=2.4, zorder=4,
               label="one constant effect (%.2f)" % const)
    ax.set_xlabel(r"true $\theta(X)$", fontsize=12)
    ax.text(0, 1.02, r"estimated $\theta(X)$", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, color=INK)
    ax.set_title("The forest tracks who benefits", fontsize=14, color=MADRID,
                 weight="bold", pad=26)
    ax.legend(fontsize=9.5, framealpha=0.92, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ==================== deck 04: regression-chapter figures =====================
# Faithful course-styled versions of Facure's chapter graphs that were missing
# or unlabelled: positivity/extrapolation, the concave spend nonlinearity, the
# sqrt linearization, non-linear FWL, and de-meaning / fixed effects. All carry
# a title, axis labels, and (where a third variable is shown) a legend.
_SEQ = None


def _wage_binned(df, n=5):
    import pandas as pd
    wg = pd.IntervalIndex(pd.qcut(df["wage"], n)).mid
    return (df.assign(wage_group=wg)
              .groupby(["wage_group", "credit_limit"])["spend"].mean()
              .reset_index())


def _spend_scatter(ax, g):
    """Scatter of mean spend vs credit limit, coloured by wage group."""
    groups = sorted(g["wage_group"].unique())
    cols = plt.cm.viridis(np.linspace(0.12, 0.88, len(groups)))
    for col, grp in zip(cols, groups):
        s = g[g["wage_group"] == grp]
        ax.scatter(s["credit_limit"], s["spend"], color=col, s=20, alpha=0.85,
                   edgecolors="none", zorder=3)
    leg = ax.legend([plt.Line2D([], [], marker="o", ls="", color=c)
                     for c in cols], ["%.0f" % g for g in groups],
                    title="wage", fontsize=8.5, title_fontsize=9,
                    framealpha=0.9, loc="lower right")
    return leg


def fig_nonlinearity_spend():
    """Spend rises with the credit limit but with diminishing returns. A straight
    line cannot fit this concave cloud; that is the nonlinearity to spot."""
    import pandas as pd
    g = _wage_binned(pd.read_csv(_DATA / "spend_data.csv"))
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
    _spend_scatter(ax, g)
    ax.set_xlabel("credit limit", fontsize=12)
    ax.text(0, 1.02, "spend", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("Spend vs credit limit is concave, not linear", fontsize=14,
                 color=MADRID, weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_linearize_treatment():
    """The same cloud with a square-root fit overlaid: linear in sqrt(credit
    limit) captures the diminishing returns a raw-linear fit would miss."""
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    df = pd.read_csv(_DATA / "spend_data.csv")
    m = smf.ols("spend ~ np.sqrt(credit_limit)", data=df).fit()
    g = _wage_binned(df)
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
    leg1 = _spend_scatter(ax, g)
    x = np.linspace(g["credit_limit"].min(), g["credit_limit"].max(), 200)
    fitline, = ax.plot(x, m.params.iloc[0] + m.params.iloc[1] * np.sqrt(x),
                       color=FITLINE, lw=3.2, zorder=5,
                       label=r"fit on $\sqrt{\mathrm{credit\ limit}}$")
    ax.set_xlabel("credit limit", fontsize=12)
    ax.text(0, 1.02, "spend", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("Linearize the treatment: fit on the square root", fontsize=14,
                 color=MADRID, weight="bold", pad=26)
    ax.legend(handles=[fitline], fontsize=10, framealpha=0.92, loc="upper left")
    ax.add_artist(leg1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_nonlinear_fwl():
    """Non-linear FWL: debias sqrt(limit) on wage, denoise spend on wage, regress
    the residuals. The recovered curve matches the direct sqrt fit."""
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    df = pd.read_csv(_DATA / "spend_data.csv")
    deb = smf.ols("np.sqrt(credit_limit) ~ wage", data=df).fit()
    den = smf.ols("spend ~ wage", data=df).fit()
    d = df.assign(cl_deb=deb.resid + np.sqrt(df["credit_limit"]).mean(),
                  sp_den=den.resid + df["spend"].mean())
    final = smf.ols("sp_den ~ cl_deb", data=d).fit()
    g = _wage_binned(df)
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
    leg1 = _spend_scatter(ax, g)
    x = np.linspace(g["credit_limit"].min(), g["credit_limit"].max(), 200)
    fitline, = ax.plot(x, final.params.iloc[0] + final.params.iloc[1] * np.sqrt(x),
                       color=FITLINE, lw=3.2, zorder=5, label="FWL residual fit")
    ax.set_xlabel("credit limit", fontsize=12)
    ax.text(0, 1.02, "spend", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("Non-linear FWL recovers the same curve", fontsize=14,
                 color=MADRID, weight="bold", pad=26)
    ax.legend(handles=[fitline], fontsize=10, framealpha=0.92, loc="upper left")
    ax.add_artist(leg1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_positivity(seed=1):
    """Two designs. Left: treatment overlaps across x, so the group lines
    interpolate within data. Right: treatment is decided by x, so the groups do
    not overlap and each line must extrapolate (dashed) into a region with no
    data of that colour. Regression fills that gap with a functional-form guess."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = 400
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), dpi=110, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, overlap, ttl in ((axes[0], True, "Positivity: groups overlap"),
                             (axes[1], False, "No positivity: the fit extrapolates")):
        x = rng.normal(0, 1, n)
        if overlap:
            t = rng.integers(0, 2, n)
        else:
            t = (x + rng.normal(0, 0.25, n) > 0).astype(int)
        y = 0.35 * x + 1.0 * t + rng.normal(0, 0.28, n)
        ax.set_facecolor("white")
        ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
        gx = np.linspace(x.min(), x.max(), 100)
        for grp, col, mk in ((0, _CTRL, "o"), (1, _TREAT, "^")):
            m = t == grp
            ax.scatter(x[m], y[m], s=20, color=col, marker=mk, alpha=0.75,
                       edgecolors="white", lw=0.3, zorder=3,
                       label="control" if grp == 0 else "treated")
            b = np.polyfit(x[m], y[m], 1)
            lo, hi = x[m].min(), x[m].max()
            supp = (gx >= lo) & (gx <= hi)
            ax.plot(gx[supp], b[1] + b[0] * gx[supp], color=col, lw=2.6, zorder=4)
            ax.plot(gx[~supp], b[1] + b[0] * gx[~supp], color=col, lw=2.0,
                    ls=(0, (4, 3)), zorder=4)
        ax.set_xlabel("covariate x", fontsize=12)
        ax.set_title(ttl, fontsize=13, color=MADRID, weight="bold", pad=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(0, 1.03, "outcome y", transform=axes[0].transAxes, ha="left",
                 va="bottom", fontsize=12, color=INK)
    axes[0].legend(fontsize=9.5, framealpha=0.92, loc="upper left")
    fig.suptitle("Dashed = extrapolation beyond a group's data", fontsize=11,
                 color=INK, y=1.01)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_demeaning():
    """Fixed effects as de-meaning. Left: raw credit limit vs default, coloured by
    score bucket, the pooled slope confounded by the bucket. Right: after
    subtracting each bucket's mean limit, only within-bucket variation remains and
    the slope is the fixed-effects estimate."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(_DATA / "risk_data_rnd.csv")
    df = df.assign(cl_avg=df.groupby("credit_score1_buckets")["credit_limit"]
                   .transform("mean"))
    df = df.assign(cl_demean=df["credit_limit"] - df["cl_avg"])
    buckets = sorted(df["credit_score1_buckets"].unique())
    cols = plt.cm.viridis(np.linspace(0.12, 0.88, len(buckets)))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=110)
    fig.patch.set_facecolor("white")
    # bin the x for a readable mean-outcome scatter
    for ax, xcol, ttl, xlab in (
            (axes[0], "credit_limit", "Raw: pooled", "credit limit"),
            (axes[1], "cl_demean", "De-meaned: within bucket", "credit limit minus bucket mean")):
        ax.set_facecolor("white")
        ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
        for col, b in zip(cols, buckets):
            s = df[df["credit_score1_buckets"] == b]
            q = pd.qcut(s[xcol], 6, duplicates="drop")
            m = s.groupby(q, observed=True).agg(x=(xcol, "mean"),
                                                y=("default", "mean"))
            ax.scatter(m["x"], m["y"], color=col, s=26, alpha=0.9, zorder=3,
                       label="%.0f" % b)
        ax.set_xlabel(xlab, fontsize=11.5)
        ax.set_title(ttl, fontsize=12.5, color=MADRID, weight="bold", pad=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(0, 1.10, "default rate", transform=axes[0].transAxes, ha="left",
                 va="bottom", fontsize=11.5, color=INK)
    axes[1].legend(title="score bucket", fontsize=8.5, title_fontsize=9,
                   framealpha=0.9, loc="upper right")
    fig.suptitle("Fixed effects subtract each bucket's mean limit", fontsize=12,
                 color=INK, y=1.02)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_conditional_random():
    """A conditionally random experiment: within each credit-score bucket the
    credit limit was assigned at random, so it still spreads across a range. That
    within-bucket variation is the as-good-as-random treatment regression uses."""
    import numpy as np
    import pandas as pd
    d = pd.read_csv(_DATA / "risk_data_rnd.csv")
    buckets = sorted(d["credit_score1_buckets"].unique())
    cols = plt.cm.viridis(np.linspace(0.12, 0.88, len(buckets)))
    bins = np.linspace(d["credit_limit"].min(), d["credit_limit"].max(), 38)
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for col, b in zip(cols, buckets):
        s = d[d["credit_score1_buckets"] == b]
        ax.hist(s["credit_limit"], bins=bins, color=col, alpha=0.55, zorder=3,
                label="%.0f" % b)
    ax.set_xlabel("credit limit", fontsize=12)
    ax.text(0, 1.02, "count", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("Within each score bucket, the credit limit still varies",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    ax.legend(title="score bucket", fontsize=8.5, title_fontsize=9,
              framealpha=0.9, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_extrapolation():
    """The danger of extrapolation: a straight fit matches the data, then keeps
    climbing past it, while the true regression function bends away. Predicting
    at a new input beyond the data trusts the line's guess, not the truth."""
    import numpy as np
    rng = np.random.default_rng(1)
    true = lambda x: 2.4 * np.sin(0.62 * x) + 0.12 * x
    xd = rng.uniform(0.2, 3.6, 130)
    yd = true(xd) + rng.normal(0, 0.16, len(xd))
    b = np.polyfit(xd, yd, 1)
    xg = np.linspace(0, 9.2, 500)
    xstar = 7.6
    ystar = b[1] + b[0] * xstar

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.scatter(xd, yd, facecolors="none", edgecolors=MADRID, s=24, lw=1.0,
               zorder=3)
    ax.plot(xg, b[1] + b[0] * xg, color=MADRID, lw=2.4, zorder=4)
    ax.plot(xg, true(xg), color=FITLINE, lw=2.4, zorder=4)
    ax.plot([xstar], [ystar], marker="*", ms=20, color=MADRID,
            markeredgecolor="white", zorder=6)
    # freeze the vertical range so the new-input marker sits on the axis, not
    # floating above it on the auto y-margin
    y0, y1 = ax.get_ylim()
    y0 = min(y0, -2.9)
    ax.set_ylim(y0, y1)
    ax.plot([xstar, xstar], [y0, ystar], color=MADRID,
            lw=1.4, ls=(0, (5, 4)), zorder=2)
    ax.plot([xstar], [y0 + 0.08], marker="s", ms=8, color=MADRID, zorder=6,
            clip_on=False)

    ax.annotate("Regression line", xy=(2.3, b[1] + b[0] * 2.3),
                xytext=(0.4, 4.1), fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax.annotate("Prediction", xy=(xstar, ystar), xytext=(5.4, 5.3),
                fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax.annotate("True regression function\n" r"$r(x)=E[Y\,|\,X=x]$",
                xy=(6.9, true(6.9)), xytext=(3.4, -1.7), fontsize=11, color=INK,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax.text(xstar + 0.22, true(xstar) - 0.15, r"New input $X^{*}$", ha="left",
            va="center", fontsize=12, color=INK)

    ax.set_xlabel("Predictor X", fontsize=13)
    ax.text(0, 1.01, "Response Y", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=13, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.2, 9.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============ mediator diagrams (ported from Robert's intro deck) =============
_MED_DPI = 150
_MED_AGE, _MED_EDU = (0.0, 1.15), (2.9, 2.35)
_MED_EXP, _MED_EARN = (2.9, -0.05), (5.8, 1.15)


def _med_canvas():
    fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=_MED_DPI)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-0.85, 6.65); ax.set_ylim(-0.75, 3.05)
    ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def _med_ellipse(ax, x, y, label, w=1.15, h=0.72):
    ax.add_patch(Ellipse((x, y), w, h, facecolor=MADRID, edgecolor=MADRID, zorder=3))
    ax.text(x, y, label, ha="center", va="center", color="white",
            fontsize=12.5, zorder=4, linespacing=0.95)


def _med_box(ax, x, y, label, w=1.35, h=0.78):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="square,pad=0",
                                linewidth=2.4, edgecolor=INK, facecolor=MADRID, zorder=3))
    m = 0.11
    ax.add_patch(FancyBboxPatch((x - w / 2 - m, y - h / 2 - m), w + 2 * m, h + 2 * m,
                                boxstyle="square,pad=0", linewidth=1.2,
                                edgecolor=INK, facecolor="none", zorder=3))
    ax.text(x, y, label, ha="center", va="center", color="white",
            fontsize=12, zorder=4, linespacing=0.95)


def _med_edge(ax, p0, p1, shrinkA=20, shrinkB=20):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18,
                                 shrinkA=shrinkA, shrinkB=shrinkB, linewidth=1.9,
                                 color=INK, zorder=2))


def dag_mediator(conditioned=True):
    """Age reaches earnings directly and through education and experience.
    Boxing the mediators (controlling for them) blocks the indirect paths and
    leaves only the direct arrow: over-control."""
    fig, ax = _med_canvas()
    node = _med_box if conditioned else _med_ellipse
    _med_edge(ax, _MED_AGE, _MED_EDU, 44, 58)
    _med_edge(ax, _MED_AGE, _MED_EXP, 44, 58)
    _med_edge(ax, _MED_EDU, _MED_EARN, 58, 44)
    _med_edge(ax, _MED_EXP, _MED_EARN, 58, 44)
    _med_edge(ax, _MED_AGE, _MED_EARN, 44, 44)
    _med_ellipse(ax, *_MED_AGE, "Age")
    node(ax, *_MED_EDU, "Education")
    node(ax, *_MED_EXP, "Experience")
    _med_ellipse(ax, *_MED_EARN, "Earnings")
    fig.tight_layout(); plt.close(fig)
    return fig


def mediator_paths():
    """The way around over-control: label every path, then add them up.
    Coefficients estimated from Blackburn and Neumark (1992)."""
    p, direct, total = wage_paths()
    fig, ax = _med_canvas()
    _med_edge(ax, _MED_AGE, _MED_EDU, 44, 58)
    _med_edge(ax, _MED_AGE, _MED_EXP, 44, 58)
    _med_edge(ax, _MED_EDU, _MED_EARN, 58, 44)
    _med_edge(ax, _MED_EXP, _MED_EARN, 58, 44)
    _med_edge(ax, _MED_AGE, _MED_EARN, 44, 44)

    def _lab(p0, p1, text, dx=0.0, dy=0.0):
        ax.text((p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy, text,
                ha="center", va="center", fontsize=12, color=ACCENT,
                weight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none"))

    _lab(_MED_AGE, _MED_EDU,  f"{p['age_edu']:+.3f}",  dx=-0.30, dy=0.26)
    _lab(_MED_AGE, _MED_EXP,  f"{p['age_exp']:+.3f}",  dx=-0.30, dy=-0.26)
    _lab(_MED_EDU, _MED_EARN, f"{p['edu_earn']:+.3f}", dx=0.30,  dy=0.26)
    _lab(_MED_EXP, _MED_EARN, f"{p['exp_earn']:+.3f}", dx=0.30,  dy=-0.26)
    _lab(_MED_AGE, _MED_EARN, f"{direct:+.3f}", dy=0.24)

    _med_ellipse(ax, *_MED_AGE, "Age")
    _med_ellipse(ax, *_MED_EDU, "Education")
    _med_ellipse(ax, *_MED_EXP, "Experience")
    _med_ellipse(ax, *_MED_EARN, "Earnings")
    ax.text(0.5, -0.02, "Blackburn and Neumark (1992), N = 935. Outcome is log wage.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=INK)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============================================================== propensity score lecture
_PS_CACHE = {}


def _ps_frame():
    """Management training data with a fitted propensity score column."""
    import pandas as pd
    import statsmodels.formula.api as smf
    if "df" not in _PS_CACHE:
        df = pd.read_csv(_DATA / "management_training.csv")
        m = smf.logit(
            "intervention ~ tenure + last_engagement_score + department_score"
            " + C(n_of_reports) + C(gender) + C(role)", data=df).fit(disp=0)
        _PS_CACHE["df"] = df.assign(propensity_score=m.predict(df))
    return _PS_CACHE["df"]


def _size_legend(ax, values, scale, label, loc="upper left"):
    """A real marker-size legend: grey reference dots at the given weights."""
    handles = [ax.scatter([], [], s=v * scale, color="#8A8A94",
                          edgecolors="white", lw=0.4) for v in values]
    leg = ax.legend(handles, ["%g" % v for v in values], title=label,
                    fontsize=9, title_fontsize=9.5, framealpha=0.92,
                    loc=loc, labelspacing=1.0, borderpad=0.8)
    ax.add_artist(leg)
    return leg


def fig_ipw_weights():
    """Engagement against the propensity score, marker area equal to the IPW
    weight. The heavy dots sit where a unit's treatment status is surprising:
    treated managers with a low score, untreated managers with a high score.
    Each of those stands in for the many similar units we never observe."""
    df = _ps_frame()
    t = df["intervention"].to_numpy()
    e = df["propensity_score"].to_numpy()
    y = df["engagement_score"].to_numpy()
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    scale = 9.0
    fig, ax = plt.subplots(figsize=(9.6, 4.9), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for grp, col, mk, lab in ((0, _CTRL, "o", "untreated"),
                              (1, _TREAT, "^", "treated")):
        m = t == grp
        ax.scatter(e[m], y[m], s=w[m] * scale, color=col, marker=mk,
                   alpha=0.55, edgecolors="white", lw=0.3, zorder=3, label=lab)
    _size_legend(ax, [1.5, 3, 6], scale, "IPW weight", loc="upper left")
    ax.legend(fontsize=10, framealpha=0.92, loc="lower right")
    ax.set_xlabel("propensity score  $\\hat{e}(X)$", fontsize=12)
    ax.text(0, 1.03, "engagement score", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("Who gets a loud voice under inverse propensity weighting",
                 fontsize=13, color=MADRID, weight="bold", pad=22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_ps_balance():
    """Propensity score distributions by treatment group, raw and reweighted.
    Raw, the treated pile up at higher scores, which is confounding made
    visible. Weighted by IPW, the two histograms sit on top of each other:
    treatment looks as good as random with respect to the score."""
    df = _ps_frame()
    t = df["intervention"].to_numpy()
    e = df["propensity_score"].to_numpy()
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    bins = np.linspace(e.min(), e.max(), 32)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), dpi=110, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, use_w, ttl in ((axes[0], False, "Raw data"),
                           (axes[1], True, "After IPW reweighting")):
        ax.set_facecolor("white")
        ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
        for grp, col, lab in ((0, _CTRL, "untreated"), (1, _TREAT, "treated")):
            m = t == grp
            wt = w[m] if use_w else np.ones(m.sum())
            ax.hist(e[m], bins=bins, weights=wt / wt.sum(), color=col,
                    alpha=0.55, label=lab, zorder=3)
        ax.set_xlabel("propensity score  $\\hat{e}(X)$", fontsize=11.5)
        ax.set_title(ttl, fontsize=12.5, color=MADRID, weight="bold", pad=22)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(0, 1.03, "share of group", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=11.5, color=INK)
    axes[0].legend(fontsize=10, framealpha=0.92)
    fig.tight_layout(); plt.close(fig)
    return fig


def ps_positivity_data(seed=7, n=1000):
    """Simulated data where the confounder decides treatment almost outright,
    so the groups barely overlap. True ATE is 1 by construction."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    t = (x + rng.normal(0, 0.5, n) > 0).astype(int)
    y0 = -x
    y1 = y0 + 1
    y = np.where(t == 1, y1, y0) + rng.normal(0, 0.2, n)
    return pd.DataFrame(dict(x=x, t=t, y=y))


def fig_ps_positivity(seed=7):
    """Three views of a positivity failure. Left: the raw data, where x drives
    both treatment and outcome. Middle: the propensity distributions, with the
    groups separated and almost no overlap to reweight from. Right: the IPW
    view, where the few units in the overlap region carry enormous weights."""
    import statsmodels.formula.api as smf
    df = ps_positivity_data(seed)
    e = smf.logit("t ~ x", data=df).fit(disp=0).predict(df).to_numpy()
    t, x, y = df["t"].to_numpy(), df["x"].to_numpy(), df["y"].to_numpy()
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=110)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for grp, col, mk, lab in ((0, _CTRL, "o", "untreated"),
                              (1, _TREAT, "^", "treated")):
        m = t == grp
        axes[0].scatter(x[m], y[m], s=14, color=col, marker=mk, alpha=0.65,
                        edgecolors="white", lw=0.2, zorder=3, label=lab)
        axes[1].hist(e[m], bins=np.linspace(0, 1, 30),
                     weights=np.ones(m.sum()) / m.sum(),
                     color=col, alpha=0.55, zorder=3, label=lab)
        axes[2].scatter(x[m], y[m], s=np.clip(w[m], 0, 40) * 2.2, color=col,
                        marker=mk, alpha=0.5, edgecolors="white", lw=0.2,
                        zorder=3, label=lab)
    _size_legend(axes[2], [2, 10, 30], 2.2, "IPW weight", loc="upper right")
    axes[0].set_xlabel("confounder x", fontsize=11.5)
    axes[0].text(0, 1.03, "outcome y", transform=axes[0].transAxes, ha="left",
                 va="bottom", fontsize=11.5, color=INK)
    axes[0].set_title("The raw data", fontsize=12.5, color=MADRID,
                      weight="bold", pad=22)
    axes[0].legend(fontsize=9.5, framealpha=0.92, loc="upper right")
    axes[1].set_xlabel("propensity score  $\\hat{e}(x)$", fontsize=11.5)
    axes[1].text(0, 1.03, "share of group", transform=axes[1].transAxes,
                 ha="left", va="bottom", fontsize=11.5, color=INK)
    axes[1].set_title("The positivity check", fontsize=12.5, color=MADRID,
                      weight="bold", pad=22)
    axes[2].set_xlabel("confounder x", fontsize=11.5)
    axes[2].text(0, 1.03, "outcome y", transform=axes[2].transAxes, ha="left",
                 va="bottom", fontsize=11.5, color=INK)
    axes[2].set_title("What IPW has to work with", fontsize=12.5, color=MADRID,
                      weight="bold", pad=22)
    fig.tight_layout(); plt.close(fig)
    return fig


def dr_scenario_data(kind, seed=11, n=10000):
    """Two toy worlds for doubly robust estimation. kind="treatment": the
    treatment model is a plain logit but the outcome is a cubic, true ATE 2.
    kind="outcome": the outcome is linear but the treatment model has a cubic
    inside, true ATE is -1."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    x = np.round(rng.uniform(0, 1, n), 2) * 2
    if kind == "treatment":
        e = 1 / (1 + np.exp(-(1 + 1.5 * x)))
        y1 = np.ones(n)
        y0 = 1 - x ** 3
    else:
        e = 1 / (1 + np.exp(-(2 * x - x ** 3)))
        y1 = x
        y0 = x + 1
    t = rng.binomial(1, e)
    y = np.where(t == 1, y1, y0) + rng.normal(0, 1, n)
    return pd.DataFrame(dict(x=x, t=t, y=y)), float(np.mean(y1 - y0))


def fig_dr_scenario(kind="treatment"):
    """Left: the share treated at each x, which is the shape a propensity model
    must capture. Right: outcome against x by group, the shape an outcome model
    must capture. One of the panels is easy and the other is hard, and which is
    which flips between the two scenarios."""
    import pandas as pd
    df, ate = dr_scenario_data(kind)
    lab = {"treatment": "Scenario 1: the treatment model is the easy one",
           "outcome": "Scenario 2: the outcome model is the easy one"}[kind]
    bins = pd.cut(df["x"], 40)
    g = df.groupby(bins, observed=True).agg(x=("x", "mean"), t=("t", "mean"),
                                            n=("t", "size"))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), dpi=110)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].scatter(g["x"], g["t"], s=g["n"] * 0.55, color=MADRID, alpha=0.8,
                    edgecolors="white", lw=0.4, zorder=3)
    _size_legend(axes[0], [100, 200, 300], 0.55, "units in bin",
                 loc="lower right" if kind == "treatment" else "upper right")
    axes[0].set_xlabel("confounder x", fontsize=11.5)
    axes[0].text(0, 1.03, "share treated", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=11.5, color=INK)
    axes[0].set_title("$P(T=1 \\mid x)$", fontsize=12.5, color=MADRID,
                      weight="bold", pad=22)
    samp = df.sample(1200, random_state=3)
    for grp, col, mk, glab in ((0, _CTRL, "o", "untreated"),
                               (1, _TREAT, "^", "treated")):
        m = samp["t"] == grp
        axes[1].scatter(samp.loc[m, "x"], samp.loc[m, "y"], s=12, color=col,
                        marker=mk, alpha=0.5, edgecolors="none", zorder=3,
                        label=glab)
    axes[1].legend(fontsize=9.5, framealpha=0.92, loc="lower left")
    axes[1].set_xlabel("confounder x", fontsize=11.5)
    axes[1].text(0, 1.03, "outcome y", transform=axes[1].transAxes, ha="left",
                 va="bottom", fontsize=11.5, color=INK)
    axes[1].set_title("$E[Y \\mid x, T]$", fontsize=12.5, color=MADRID,
                      weight="bold", pad=22)
    fig.suptitle(lab, fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout(); plt.close(fig)
    return fig


def _gps_pieces():
    """Interest rate data with a one-confounder GPS, for the weight figures."""
    import pandas as pd
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    if "gps" not in _PS_CACHE:
        df = pd.read_csv(_DATA / "interest_rate.csv")
        mt = smf.ols("interest ~ ml_1", data=df).fit()
        gps = norm(loc=mt.fittedvalues,
                   scale=np.std(mt.resid)).pdf(df["interest"])
        stab = norm(loc=df["interest"].mean(),
                    scale=df["interest"].std()).pdf(df["interest"])
        _PS_CACHE["gps"] = (df, mt, gps, stab)
    return _PS_CACHE["gps"]


def fig_gps_weights(stabilized=False):
    """Continuous treatment weighting on the interest rate data, one confounder
    so it fits on a page. Left: the biased raw relationship, coloured by the
    confounder. Middle: the treatment model, marker area equal to the weight,
    largest far from the fitted line (raw) or far from both the line and the
    mean (stabilized). Right: the reweighted relationship and its WLS slope."""
    import statsmodels.formula.api as smf
    df, mt, gps, stab = _gps_pieces()
    w = (stab / gps) if stabilized else (1 / gps)
    wname = "$f(t)/f(t \\mid x)$" if stabilized else "$1/f(t \\mid x)$"
    naive = smf.ols("duration ~ interest", data=df).fit()
    wls = smf.wls("duration ~ interest", data=df, weights=w).fit()
    samp = df.sample(1500, random_state=5).index
    scale = 30.0 / w.max()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), dpi=110)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    gx = np.linspace(df["interest"].min(), df["interest"].max(), 50)
    sc = axes[0].scatter(df.loc[samp, "interest"], df.loc[samp, "duration"],
                         c=df.loc[samp, "ml_1"], cmap="viridis", s=10,
                         alpha=0.65, zorder=3)
    axes[0].plot(gx, naive.params["Intercept"] + naive.params["interest"] * gx,
                 color=FITLINE, lw=2.6, zorder=4)
    cb = fig.colorbar(sc, ax=axes[0], pad=0.02)
    cb.set_label("confounder ml_1", fontsize=10)
    axes[0].set_xlabel("interest rate", fontsize=11.5)
    axes[0].text(0, 1.03, "months to repay", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=11.5, color=INK)
    axes[0].set_title("Raw: slope %+.2f" % naive.params["interest"],
                      fontsize=12.5, color=MADRID, weight="bold", pad=22)
    axes[1].scatter(df.loc[samp, "ml_1"], df.loc[samp, "interest"],
                    s=np.asarray(w)[samp] * scale, color=MADRID, alpha=0.4,
                    edgecolors="white", lw=0.2, zorder=3)
    order = df.loc[samp].sort_values("ml_1")
    axes[1].plot(order["ml_1"], mt.predict(order), color=FITLINE, lw=2.6,
                 zorder=4, label="$E[t \\mid x]$")
    if stabilized:
        axes[1].axhline(df["interest"].mean(), color=ACCENT, lw=2.2,
                        ls=(0, (5, 3)), zorder=4, label="$E[t]$")
    axes[1].legend(fontsize=9.5, framealpha=0.92, loc="upper right")
    axes[1].set_xlabel("confounder ml_1", fontsize=11.5)
    axes[1].text(0, 1.03, "interest rate", transform=axes[1].transAxes,
                 ha="left", va="bottom", fontsize=11.5, color=INK)
    axes[1].set_title("The weights " + wname, fontsize=12.5, color=MADRID,
                      weight="bold", pad=22)
    axes[2].scatter(df.loc[samp, "interest"], df.loc[samp, "duration"],
                    s=np.asarray(w)[samp] * scale, color=MADRID, alpha=0.35,
                    edgecolors="white", lw=0.2, zorder=3)
    axes[2].plot(gx, wls.params["Intercept"] + wls.params["interest"] * gx,
                 color=FITLINE, lw=2.6, zorder=4)
    axes[2].set_xlabel("interest rate", fontsize=11.5)
    axes[2].text(0, 1.03, "months to repay", transform=axes[2].transAxes,
                 ha="left", va="bottom", fontsize=11.5, color=INK)
    axes[2].set_title("Weighted: slope %+.2f" % wls.params["interest"],
                      fontsize=12.5, color=MADRID, weight="bold", pad=22)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_ps_methods_compare(seed=12, n=420):
    """Orthogonalization and IPW debiasing the same simulated data, side by
    side with the biased original. The true effect of t on y is +1 but the
    group variable confounds it into a negative raw slope. Both fixes recover
    the positive slope by very different moves."""
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 4, n)
    t = 0.12 * x + rng.normal(0.25, 0.16, n)
    y = 2.0 + 1.0 * t - 0.6 * x + rng.normal(0, 0.10, n) + 1.2
    from scipy.stats import norm
    ey_x = np.array([y[x == k].mean() for k in (1, 2, 3)])[x - 1]
    et_x = np.array([t[x == k].mean() for k in (1, 2, 3)])[x - 1]
    sd_x = np.array([t[x == k].std() for k in (1, 2, 3)])[x - 1]
    y_o = y - ey_x + y.mean()
    t_o = t - et_x + t.mean()
    w = norm(t.mean(), t.std()).pdf(t) / norm(et_x, sd_x).pdf(t)
    w_show = np.clip(w, 0, np.percentile(w, 99))
    scale = 24.0 / np.percentile(w, 99)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=110, sharey=False)
    fig.patch.set_facecolor("white")
    panels = (
        (axes[0], t, y, None, "Biased data", "treatment t"),
        (axes[1], t_o, y_o, None, "Orthogonalization",
         "t residual, recentred"),
        (axes[2], t, y, w, "IPW", "treatment t"),
    )
    for ax, tt, yy, ww, ttl, xlab in panels:
        ax.set_facecolor("white")
        ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for k, col, mk in zip((1, 2, 3), GROUP_COLORS, GROUP_MARKERS):
            m = x == k
            size = 16 if ww is None else np.asarray(w_show)[m] * scale
            ax.scatter(tt[m], yy[m], s=size, color=col, marker=mk, alpha=0.6,
                       edgecolors="white", lw=0.2, zorder=3,
                       label="x = %d" % k)
        if ww is None:
            b = np.polyfit(tt, yy, 1)
        else:
            b = np.polyfit(tt, yy, 1, w=np.sqrt(ww))
        gx = np.linspace(tt.min(), tt.max(), 40)
        ax.plot(gx, b[1] + b[0] * gx, color=FITLINE, lw=2.6, zorder=4)
        ax.set_title("%s: slope %+.2f" % (ttl, b[0]), fontsize=12.5,
                     color=MADRID, weight="bold", pad=22)
        ax.set_xlabel(xlab, fontsize=11.5)
    axes[0].text(0, 1.03, "outcome y", transform=axes[0].transAxes, ha="left",
                 va="bottom", fontsize=11.5, color=INK)
    axes[0].legend(fontsize=9.5, framealpha=0.92, loc="upper left")
    _size_legend(axes[2], [1, 2, 4], scale, "weight", loc="lower right")
    fig.tight_layout(); plt.close(fig)
    return fig



# ============================================================== LLN + CLT in one animation
def clt_pi_animation(n_total=3000, n_frames=60, seed=4, fps=6):
    """Both limit theorems in one running experiment. Darts land uniformly on
    the square; the share inside the circle estimates pi/4. Left: the darts.
    Right: the running estimate of pi with its CLT 95% band. The line settling
    on pi is the law of large numbers; the band shrinking like one over root n
    is the central limit theorem doing the shrinking."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, size=(n_total, 2))
    inside = (pts ** 2).sum(axis=1) <= 1.0
    cum_in = np.cumsum(inside)
    ns = np.arange(1, n_total + 1)
    p_hat = cum_in / ns
    pi_hat = 4 * p_hat
    se = 4 * np.sqrt(np.clip(p_hat * (1 - p_hat), 1e-12, None) / ns)

    sched = np.unique(np.geomspace(1, n_total, n_frames).astype(int))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.5), dpi=100)
    fig.patch.set_facecolor("white")
    theta = np.linspace(0, 2 * np.pi, 200)
    axL.plot(np.cos(theta), np.sin(theta), color=INK, lw=2)
    axL.set_xlim(-1.05, 1.05)
    axL.set_ylim(-1.05, 1.05)
    axL.set_aspect("equal")
    axL.set_xticks([]); axL.set_yticks([])
    scat_in = axL.scatter([], [], s=7, color=_TREAT, alpha=0.7)
    scat_out = axL.scatter([], [], s=7, color=_CTRL, alpha=0.7)
    titleL = axL.set_title("", fontsize=12.5, color=MADRID, weight="bold",
                           pad=10)

    axR.set_facecolor("white")
    axR.grid(True, color=GRIDCOL, lw=0.8, zorder=0); axR.set_axisbelow(True)
    axR.set_xscale("log")
    axR.set_xlim(1, n_total)
    axR.set_ylim(2.0, 4.3)
    axR.axhline(np.pi, color=FITLINE, lw=1.8, ls=(0, (6, 4)), zorder=3)
    axR.text(n_total * 0.9, np.pi + 0.06, "true $\\pi$", ha="right",
             fontsize=10.5, color=FITLINE)
    line, = axR.plot([], [], color=MADRID, lw=2.2, zorder=4)
    band = [axR.fill_between([], [], [], color=MADRID, alpha=0.15)]
    axR.set_xlabel("darts thrown (log scale)", fontsize=11.5)
    axR.text(0, 1.03, "estimate of $\\pi$, with 95% band",
             transform=axR.transAxes, ha="left", va="bottom", fontsize=11.5,
             color=INK)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    fig.tight_layout()

    def update(k):
        n = sched[k]
        m = inside[:n]
        scat_in.set_offsets(pts[:n][m])
        scat_out.set_offsets(pts[:n][~m])
        titleL.set_text("n = %d darts, %.0f%% inside" % (n, 100 * p_hat[n - 1]))
        line.set_data(ns[:n], pi_hat[:n])
        band[0].remove()
        band[0] = axR.fill_between(ns[:n], pi_hat[:n] - 1.96 * se[:n],
                                   pi_hat[:n] + 1.96 * se[:n],
                                   color=MADRID, alpha=0.15, zorder=2)
        return [scat_in, scat_out, line]

    anim = animation.FuncAnimation(fig, update, frames=len(sched),
                                   blit=False, interval=1000 / fps)
    from IPython.display import HTML
    out = HTML(anim.to_html5_video())
    plt.close(fig)
    return out

