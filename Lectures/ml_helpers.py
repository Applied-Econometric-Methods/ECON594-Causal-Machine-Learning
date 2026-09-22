# -*- coding: utf-8 -*-
"""Figures and animations for the machine-learning survey decks (StatQuest
source). Concepts are re-created as clean course-styled graphics; none of Josh
Starmer's hand-drawn art is reproduced. Palette and animation settings are
shared with the causal decks via course_helpers.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

import course_helpers as chh
MADRID, ACCENT, FITLINE, INK, GRID = (chh.MADRID, chh.ACCENT, chh.FITLINE,
                                      chh.INK, chh.GRIDCOL)
_FPS, _HOLD, _TWEEN, _ease = chh._FPS, chh._HOLD, chh._TWEEN, chh._ease
_STEM = "#2E8B57"
_ANIM_DPI = 150   # figsize * dpi must stay even in both dims for h264

# StatQuest's running example: fit Height ~ Weight, 3 points.
_W = np.array([0.5, 2.3, 2.9])
_H = np.array([1.4, 1.9, 3.2])


def _ssr(b, m):
    return np.sum((_H - (b + m * _W)) ** 2)


def _grad(b, m):
    r = _H - (b + m * _W)
    return -2 * np.sum(r), -2 * np.sum(_W * r)   # d/db, d/dm


# ============================================================ one parameter
def gd_animation(seed=0, slope=0.64, lr=0.08, n_steps=10):
    """Flagship gradient-descent view: fit the intercept (slope fixed).

    Left  = the data with the current line and residual stems.
    Right = the loss SSR as a function of the intercept, a parabola, with a
            ball rolling downhill. The step shrinks on its own as the slope of
            the loss flattens near the bottom, which is the whole point.
    """
    b = 0.0
    path = [b]
    for _ in range(n_steps):
        g = -2 * np.sum(_H - (b + slope * _W))
        b -= lr * g
        path.append(b)
    path = np.array(path)
    b_opt = np.mean(_H - slope * _W)

    grid = np.linspace(-0.5, 2.3, 200)
    loss = np.array([_ssr(bb, slope) for bb in grid])
    xs = np.array([0.0, 3.4])

    # build frames: tween the intercept between successive GD steps
    frames = []
    for i in range(len(path)):
        if i > 0:
            for f in range(_TWEEN):
                a = _ease((f + 1) / _TWEEN)
                frames.append((path[i - 1] + (path[i] - path[i - 1]) * a, i))
        for _ in range(_HOLD):
            frames.append((path[i], i))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=_ANIM_DPI)
    fig.patch.set_facecolor("white")

    def draw(fr):
        b_cur, step = fr
        for ax in (axL, axR):
            ax.clear(); ax.set_facecolor("white")
            ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        # left: data + current line + residual stems
        axL.plot(xs, b_cur + slope * xs, color=MADRID, lw=2.6, zorder=2)
        for wi, hi in zip(_W, _H):
            axL.plot([wi, wi], [b_cur + slope * wi, hi], color=_STEM, lw=2, zorder=3)
        axL.scatter(_W, _H, s=90, color=ACCENT, edgecolors=INK, lw=1.3, zorder=4)
        axL.set_xlim(0, 3.4); axL.set_ylim(0, 3.7)
        axL.set_xlabel("Weight", fontsize=12)
        axL.text(0, 1.02, "Height", transform=axL.transAxes, ha="left",
                 va="bottom", fontsize=12, color=INK)
        axL.set_title(f"Height = {b_cur:.2f} + {slope} x Weight", fontsize=13,
                      color=MADRID, weight="bold", pad=10)
        # right: loss parabola + ball
        axR.plot(grid, loss, color=INK, lw=2, zorder=2)
        axR.plot(path[:step + 1], [_ssr(bb, slope) for bb in path[:step + 1]],
                 "o-", color=FITLINE, ms=5, lw=1.3, alpha=0.7, zorder=3)
        axR.scatter([b_cur], [_ssr(b_cur, slope)], s=110, color=ACCENT,
                    edgecolors=INK, lw=1.4, zorder=5)
        axR.set_xlim(-0.5, 2.3); axR.set_ylim(-0.3, loss.max() + 0.3)
        axR.set_xlabel("Intercept", fontsize=12)
        axR.text(0, 1.02, "SSR (loss)", transform=axR.transAxes, ha="left",
                 va="bottom", fontsize=12, color=INK)
        axR.set_title(f"Step {step} of {n_steps}   SSR = {_ssr(b_cur, slope):.2f}",
                      fontsize=13, color=MADRID, weight="bold", pad=10)
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / _FPS)
    plt.close(fig)
    return anim


# ============================================================ two parameters
def fig_two_parameter():
    """Our version of Josh's figure: the loss over BOTH parameters is a
    3-dimensional bowl, one axis for the intercept, one for the slope, and the
    vertical axis for SSR. No descent path (he does not draw one on the bowl)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d)
    b = np.linspace(-0.9, 2.8, 80)
    m = np.linspace(-0.1, 1.4, 80)
    B, M = np.meshgrid(b, m)
    Z = np.zeros_like(B)
    for wi, hi in zip(_W, _H):
        Z += (hi - (B + M * wi)) ** 2

    fig = plt.figure(figsize=(7.2, 5.4), dpi=110)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    from matplotlib.colors import PowerNorm
    ax.plot_surface(B, M, Z, cmap="YlGn", edgecolor="none", alpha=0.95,
                    rstride=1, cstride=1, antialiased=True,
                    norm=PowerNorm(0.45, vmin=Z.min(), vmax=Z.max()))
    ax.set_xlabel("Intercept", fontsize=11, labelpad=6)
    ax.set_ylabel("Slope", fontsize=11, labelpad=6)
    ax.set_zlabel("SSR", fontsize=11, labelpad=4)
    ax.set_title("The loss over two parameters is a bowl",
                 fontsize=14, color=MADRID, weight="bold", pad=6)
    ax.view_init(elev=34, azim=-54)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ==================================== neural network fundamentals (NN book ch1)
# Josh's running example: Dose -> 2 ReLU hidden nodes -> Sum -> Effectiveness.
_NW = dict(w1=1.43, b1=-0.61, o1=-3.89,      # top hidden node
           w2=2.63, b2=-0.27, o2=1.35)       # bottom hidden node
_DOSE = np.array([0.0, 0.5, 1.0])            # low / medium / high
_EFF = np.array([0.0, 1.0, 0.0])


def _relu(x):
    return np.maximum(x, 0.0)


def nn_top(d):
    return _NW["o1"] * _relu(_NW["w1"] * d + _NW["b1"])


def nn_bottom(d):
    return _NW["o2"] * _relu(_NW["w2"] * d + _NW["b2"])


def nn_output(d):
    return nn_top(d) + nn_bottom(d)


def fig_nn_architecture():
    """Our version of Josh's anatomy figure: one input, a hidden layer of two
    ReLU nodes with his weights and biases, summed into one output."""
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, label, color, w=1.5, h=0.95):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h,
                                   facecolor="white", edgecolor=color,
                                   lw=2.2, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=11, zorder=4)

    def relu_box(x, y, color, w=1.35, h=0.95):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h,
                                   facecolor="white", edgecolor=color,
                                   lw=2.4, zorder=3))
        xs = np.linspace(-1, 1, 40)
        ax.plot(x + xs * (w * 0.32), y - h * 0.26 + _relu(xs) * (h * 0.5),
                color=color, lw=2.0, zorder=4)

    def arrow(p0, p1, color=INK):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=color))

    box(1.05, 2.5, "Dose\n(input)", INK)
    relu_box(4.6, 3.75, MADRID)
    relu_box(4.6, 1.25, ACCENT)
    box(8.85, 2.5, "Effectiveness\n(output)", "#2E8B57")
    ax.text(7.45, 2.5, "Sum", ha="center", va="center", fontsize=12,
            color=INK, style="italic")

    arrow((1.85, 2.78), (3.88, 3.6), MADRID)
    arrow((1.85, 2.22), (3.88, 1.4), ACCENT)
    arrow((5.32, 3.75), (7.05, 2.78), MADRID)
    arrow((5.32, 1.25), (7.05, 2.22), ACCENT)
    arrow((7.85, 2.5), (8.08, 2.5))

    lab = dict(fontsize=10.5, ha="center", va="center", zorder=5,
               bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                         edgecolor="none"))
    ax.text(2.45, 3.52, r"$\times 1.43$", color=MADRID, **lab)
    ax.text(3.45, 3.02, r"$+\,(-0.61)$", color=MADRID, **lab)
    ax.text(2.45, 1.48, r"$\times 2.63$", color=ACCENT, **lab)
    ax.text(3.45, 1.98, r"$+\,(-0.27)$", color=ACCENT, **lab)
    ax.text(6.35, 3.45, r"$\times (-3.89)$", color=MADRID, **lab)
    ax.text(6.35, 1.55, r"$\times 1.35$", color=ACCENT, **lab)

    ax.set_title("One input, a hidden layer of two ReLU nodes, one output",
                 fontsize=14, color=MADRID, weight="bold", pad=6)
    fig.tight_layout(); plt.close(fig)
    return fig


def nn_squiggle_animation():
    """Josh's step-by-step: sweep the Dose from 0 to 1, watch each hidden node
    trace its own bent line, and watch the two add into the green squiggle."""
    d = np.linspace(0, 1, 110)
    top, bot, out = nn_top(d), nn_bottom(d), nn_output(d)
    frames = list(range(1, len(d) + 1)) + [len(d)] * (_HOLD * 2)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=_ANIM_DPI)
    fig.patch.set_facecolor("white")

    def draw(k):
        for ax in (axL, axR):
            ax.clear(); ax.set_facecolor("white")
            ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
            ax.set_xlim(-0.03, 1.03)
            ax.set_xlabel("Drug dose", fontsize=12)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axL.plot(d[:k], top[:k], color=MADRID, lw=2.6, zorder=3)
        axL.plot(d[:k], bot[:k], color=ACCENT, lw=2.6, zorder=3)
        axL.set_ylim(-3.4, 3.4)
        axL.text(0, 1.02, "Each hidden node, after its weight",
                 transform=axL.transAxes, ha="left", va="bottom",
                 fontsize=11.5, color=INK)
        axR.plot(d[:k], out[:k], color="#2E8B57", lw=3.0, zorder=3)
        axR.scatter(_DOSE, _EFF, s=110, color=ACCENT, edgecolors=INK,
                    lw=1.3, zorder=5)
        axR.set_ylim(-0.35, 1.35)
        axR.text(0, 1.02, "Their sum: the fitted squiggle",
                 transform=axR.transAxes, ha="left", va="bottom",
                 fontsize=11.5, color=INK)
        axR.set_title("Dose = %.2f" % d[min(k, len(d)) - 1], fontsize=13,
                      color=MADRID, weight="bold", pad=10)
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, draw, frames=frames,
                                   interval=1000 / _FPS)
    plt.close(fig)
    return anim


# ==================================== backpropagation (NN book ch2)
# Josh optimizes ONE parameter, w3 (the top node's output weight), holding the
# other five at their fitted values. He starts at w3 = 0 and descends to -3.89.
def _bp_pieces(d):
    """The fixed bottom path, and the top path BEFORE its output weight."""
    a = _relu(_NW["w1"] * d + _NW["b1"])          # top, pre-output-weight
    c = _NW["o2"] * _relu(_NW["w2"] * d + _NW["b2"])   # bottom, complete
    return a, c


def bp_output(d, w3):
    a, c = _bp_pieces(d)
    return w3 * a + c


def bp_ssr(w3):
    a, c = _bp_pieces(_DOSE)
    return float(np.sum((_EFF - (w3 * a + c)) ** 2))


def bp_optimal_w3():
    a, c = _bp_pieces(_DOSE)
    return float(np.sum(a * (_EFF - c)) / np.sum(a ** 2))


def bp_animation(lr=0.1, n_steps=12):
    """Gradient descent on a single weight, w3, exactly as Josh does it.

    Left  = the data with the network's current green curve.
    Right = SSR as a function of w3, with the ball rolling to the minimum.
    """
    a3, c3 = _bp_pieces(_DOSE)
    w3 = 0.0
    path = [w3]
    for _ in range(n_steps):
        grad = -2.0 * np.sum(a3 * (_EFF - (w3 * a3 + c3)))
        w3 -= lr * grad
        path.append(w3)
    path = np.array(path)
    w_opt = bp_optimal_w3()

    d = np.linspace(0, 1, 120)
    grid = np.linspace(-6.0, 1.0, 200)
    loss = np.array([bp_ssr(w) for w in grid])

    frames = []
    for i in range(len(path)):
        if i > 0:
            for f in range(_TWEEN):
                t = _ease((f + 1) / _TWEEN)
                frames.append((path[i - 1] + (path[i] - path[i - 1]) * t, i))
        for _ in range(_HOLD):
            frames.append((path[i], i))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=_ANIM_DPI)
    fig.patch.set_facecolor("white")

    def draw(fr):
        w_cur, step = fr
        for ax in (axL, axR):
            ax.clear(); ax.set_facecolor("white")
            ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axL.plot(d, bp_output(d, w_cur), color="#2E8B57", lw=3.0, zorder=3)
        axL.scatter(_DOSE, _EFF, s=110, color=ACCENT, edgecolors=INK,
                    lw=1.3, zorder=5)
        axL.set_xlim(-0.03, 1.03); axL.set_ylim(-0.6, 3.5)
        axL.set_xlabel("Drug dose", fontsize=12)
        axL.text(0, 1.02, "Effectiveness", transform=axL.transAxes, ha="left",
                 va="bottom", fontsize=12, color=INK)
        axL.set_title("$w_3$ = %.2f" % w_cur, fontsize=13, color=MADRID,
                      weight="bold", pad=10)
        axR.plot(grid, loss, color=INK, lw=2, zorder=2)
        axR.plot(path[:step + 1], [bp_ssr(w) for w in path[:step + 1]], "o-",
                 color=FITLINE, ms=5, lw=1.3, alpha=0.75, zorder=3)
        axR.scatter([w_cur], [bp_ssr(w_cur)], s=110, color=ACCENT,
                    edgecolors=INK, lw=1.4, zorder=5)
        axR.axvline(w_opt, color="#7A7A85", ls=(0, (5, 4)), lw=1)
        axR.set_xlim(-6.0, 1.0); axR.set_ylim(-0.6, loss.max() + 0.6)
        axR.set_xlabel("$w_3$", fontsize=12)
        axR.text(0, 1.02, "SSR (loss)", transform=axR.transAxes, ha="left",
                 va="bottom", fontsize=12, color=INK)
        axR.set_title("Step %d of %d    SSR = %.2f" % (step, n_steps,
                      bp_ssr(w_cur)), fontsize=13, color=MADRID,
                      weight="bold", pad=10)
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, draw, frames=frames,
                                   interval=1000 / _FPS)
    plt.close(fig)
    return anim


# ============================ multiple inputs and outputs (NN book ch3, iris)
# Josh's iris network: Petal Width and Sepal Width in, three species out.
_IRIS = dict(
    w_in=np.array([[-2.5, -1.5],      # petal -> h1, h2
                   [0.6, 0.4]]),      # sepal -> h1, h2
    b_hid=np.array([1.6, 0.7]),
    w_out=np.array([[-0.1, 2.4, -2.2],   # h1 -> setosa, versicolor, virginica
                    [1.5, -5.2, 3.7]]),  # h2 -> ...
    b_out=np.array([0.0, 0.0, 1.0]),
)
_SPECIES = ["Setosa", "Versicolor", "Virginica"]


def iris_forward(petal, sepal):
    """Raw (pre-ArgMax, pre-SoftMax) output for each species."""
    x = np.array([petal, sepal], dtype=float)
    hid = _relu(x @ _IRIS["w_in"] + _IRIS["b_hid"])
    return hid @ _IRIS["w_out"] + _IRIS["b_out"]


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def fig_iris_architecture():
    """Our version of Josh's ch3 diagram: two inputs, two ReLU hidden nodes,
    three outputs, one per iris species."""
    fig, ax = plt.subplots(figsize=(10.4, 5.0), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, label, color, w=1.55, h=0.8, fs=10.5):
        ax.add_patch(plt.Rectangle((x - w/2, y - h/2), w, h, facecolor="white",
                                   edgecolor=color, lw=2.0, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)

    def relu_box(x, y, color, w=1.15, h=0.85):
        ax.add_patch(plt.Rectangle((x - w/2, y - h/2), w, h, facecolor="white",
                                   edgecolor=color, lw=2.2, zorder=3))
        xs = np.linspace(-1, 1, 30)
        ax.plot(x + xs*(w*0.3), y - h*0.25 + _relu(xs)*(h*0.48), color=color,
                lw=1.8, zorder=4)

    def arrow(p0, p1, color, lw=1.5):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                    alpha=0.85))

    ins = [(1.1, 3.9, "Petal\nwidth", MADRID), (1.1, 1.8, "Sepal\nwidth", ACCENT)]
    for x, y, lab, col in ins:
        box(x, y, lab, col)
    hid = [(5.0, 4.1, MADRID), (5.0, 1.6, ACCENT)]
    for x, y, col in hid:
        relu_box(x, y, col)
    outs = [(9.4, 4.9, "Setosa", "#2E8B57"), (9.4, 2.9, "Versicolor", "#7B4EA8"),
            (9.4, 0.9, "Virginica", "#B3121F")]
    for x, y, lab, col in outs:
        box(x, y, lab, col, w=1.9)

    for (xi, yi, _, ci) in ins:
        for (xh, yh, _) in hid:
            arrow((xi + 0.8, yi), (xh - 0.62, yh), ci)
    for (xh, yh, ch) in hid:
        for (xo, yo, _, _) in outs:
            arrow((xh + 0.6, yh), (xo - 0.98, yo), ch, lw=1.3)

    ax.text(3.0, 4.55, r"$\times -2.5$", color=MADRID, fontsize=9.5, ha="center")
    ax.text(3.0, 2.55, r"$\times -1.5$", color=MADRID, fontsize=9.5, ha="center")
    ax.text(3.0, 3.15, r"$\times 0.6$", color=ACCENT, fontsize=9.5, ha="center")
    ax.text(3.0, 1.25, r"$\times 0.4$", color=ACCENT, fontsize=9.5, ha="center")
    ax.text(5.0, 4.85, r"bias $+1.6$", color=MADRID, fontsize=9.5, ha="center")
    ax.text(5.0, 0.85, r"bias $+0.7$", color=ACCENT, fontsize=9.5, ha="center")
    ax.text(7.1, 5.15, "six weights", color=INK, fontsize=9.5, ha="center",
            style="italic")

    ax.set_title("Two inputs, two hidden nodes, three outputs",
                 fontsize=14, color=MADRID, weight="bold", pad=6)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============================ cross entropy (NN book ch5)
# Josh's 3-row iris training set, and the Setosa output bias as the one
# parameter being optimized.
_IRIS_X = np.array([[0.04, 0.42],    # Setosa
                    [1.00, 0.54],    # Virginica
                    [0.50, 0.37]])   # Versicolor
_IRIS_Y = np.array([0, 2, 1])        # index of the true species


def iris_raw_with_bias(x, setosa_bias):
    """Raw outputs when the Setosa output bias is set to `setosa_bias`."""
    hid = _relu(x @ _IRIS["w_in"] + _IRIS["b_hid"])
    b = _IRIS["b_out"].copy()
    b[0] = setosa_bias
    return hid @ _IRIS["w_out"] + b


def total_cross_entropy(setosa_bias):
    """Sum of -log(p) for the true species over Josh's three rows."""
    tot = 0.0
    for x, y in zip(_IRIS_X, _IRIS_Y):
        p = softmax(iris_raw_with_bias(x, setosa_bias))
        tot += -np.log(p[y])
    return float(tot)


def ce_animation(lr=0.6, n_steps=14, start=-2.0):
    """Josh's ch5 figure: Total Cross Entropy as a function of the Setosa
    output bias, with gradient descent walking to the lowest point."""
    grid = np.linspace(-2.6, 2.6, 220)
    loss = np.array([total_cross_entropy(b) for b in grid])

    def dloss(b, h=1e-5):
        return (total_cross_entropy(b + h) - total_cross_entropy(b - h)) / (2 * h)

    b = float(start)
    path = [b]
    for _ in range(n_steps):
        b -= lr * dloss(b)
        path.append(b)
    path = np.array(path)
    b_opt = grid[int(np.argmin(loss))]

    frames = []
    for i in range(len(path)):
        if i > 0:
            for f in range(_TWEEN):
                t = _ease((f + 1) / _TWEEN)
                frames.append((path[i - 1] + (path[i] - path[i - 1]) * t, i))
        for _ in range(_HOLD):
            frames.append((path[i], i))

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=_ANIM_DPI)
    fig.patch.set_facecolor("white")

    def draw(fr):
        b_cur, step = fr
        ax.clear(); ax.set_facecolor("white")
        ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.plot(grid, loss, color=FITLINE, lw=2.4, zorder=2)
        ax.plot(path[:step + 1], [total_cross_entropy(v) for v in path[:step + 1]],
                "o-", color=INK, ms=5, lw=1.2, alpha=0.75, zorder=3)
        ax.scatter([b_cur], [total_cross_entropy(b_cur)], s=120, color=ACCENT,
                   edgecolors=INK, lw=1.4, zorder=5)
        ax.axvline(b_opt, color="#7A7A85", ls=(0, (5, 4)), lw=1)
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(0, loss.max() + 0.4)
        ax.set_xlabel("Setosa output bias", fontsize=12)
        ax.text(0, 1.02, "Total cross entropy", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=12, color=INK)
        ax.set_title("Step %d of %d    loss = %.2f" % (step, n_steps,
                     total_cross_entropy(b_cur)), fontsize=13, color=MADRID,
                     weight="bold", pad=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, draw, frames=frames,
                                   interval=1000 / _FPS)
    plt.close(fig)
    return anim


# ============================ convolutional neural networks (NN book ch6)
# Josh's example: a hand-drawn O on a 6x6 grid, white pixels 0 and black 1,
# and a 3x3 filter whose values are learned by backpropagation.
_IMG_O = np.array([
    [0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0]], dtype=float)

_FILTER = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]], dtype=float)


def convolve(img, filt):
    """Valid convolution: slide the filter, multiply overlapping cells, sum."""
    kh, kw = filt.shape
    out = np.zeros((img.shape[0] - kh + 1, img.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * filt)
    return out


def max_pool(fmap, size=2):
    h, w = fmap.shape[0] // size, fmap.shape[1] // size
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            out[i, j] = fmap[i * size:(i + 1) * size,
                             j * size:(j + 1) * size].max()
    return out


def _grid(ax, M, title, cmap="Blues", fs=11, fmt="%.0f"):
    ax.imshow(M, cmap=cmap, vmin=0, vmax=max(1.0, M.max()))
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, fmt % v, ha="center", va="center", fontsize=fs,
                color="white" if v > 0.6 * max(1.0, M.max()) else INK)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11.5, color=MADRID, weight="bold", pad=6)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_cnn_pipeline():
    """Our version of Josh's pipeline: image, filter, convolution, ReLU,
    max pooling. Every number is computed, not typed in."""
    fmap = convolve(_IMG_O, _FILTER)
    relud = _relu(fmap)
    pooled = max_pool(relud, 2)

    fig, axes = plt.subplots(1, 5, figsize=(13.6, 3.4), dpi=110)
    fig.patch.set_facecolor("white")
    _grid(axes[0], _IMG_O, "Input image (an O)")
    _grid(axes[1], _FILTER, "Filter (3 x 3)", cmap="Oranges")
    _grid(axes[2], fmap, "After convolution")
    _grid(axes[3], relud, "After ReLU")
    _grid(axes[4], pooled, "After max pooling", cmap="Greens")
    fig.tight_layout(); plt.close(fig)
    return fig


# ============================ RNN and LSTM (NN book ch7, ch8)
def fig_rnn_unrolled(n_steps=3):
    """Our version of Josh's figure: one cell with a feedback loop, and the
    same cell unrolled once per observation, reusing the SAME weights."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 3.9), dpi=110,
                                   gridspec_kw={"width_ratios": [1, 2.5]})
    fig.patch.set_facecolor("white")
    for ax in (axL, axR):
        ax.set_facecolor("white"); ax.axis("off")

    def cell(ax, x, y, w=1.25, h=1.0, color=MADRID, label=""):
        ax.add_patch(plt.Rectangle((x - w/2, y - h/2), w, h, facecolor="white",
                                   edgecolor=color, lw=2.2, zorder=3))
        xs = np.linspace(-1, 1, 30)
        ax.plot(x + xs*(w*0.28), y - h*0.22 + _relu(xs)*(h*0.45), color=color,
                lw=1.8, zorder=4)
        if label:
            ax.text(x, y - h*0.75, label, ha="center", va="top", fontsize=10,
                    color=INK)

    def arrow(ax, p0, p1, color=INK, lw=1.7, style="-|>", rad=0.0):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                    connectionstyle="arc3,rad=%s" % rad))

    # left: the compact form with a feedback loop
    axL.set_xlim(0, 4); axL.set_ylim(0, 3)
    cell(axL, 2.0, 1.5)
    arrow(axL, (0.5, 1.5), (1.3, 1.5))
    arrow(axL, (2.7, 1.5), (3.5, 1.5))
    axL.annotate("", xy=(1.75, 2.05), xytext=(2.25, 2.05),
                 arrowprops=dict(arrowstyle="-|>", lw=1.7, color=FITLINE,
                                 connectionstyle="arc3,rad=1.6"))
    axL.text(2.0, 2.55, "feedback loop", ha="center", fontsize=10.5,
             color=FITLINE)
    axL.text(0.45, 1.75, "input", ha="left", fontsize=10, color=INK)
    axL.text(3.55, 1.75, "output", ha="right", fontsize=10, color=INK)
    axL.set_title("One cell", fontsize=13, color=MADRID, weight="bold")

    # right: unrolled, same weights reused
    axR.set_xlim(0, 3 * n_steps + 1.6); axR.set_ylim(0, 3)
    for k in range(n_steps):
        x = 1.6 + 3 * k
        cell(axR, x, 1.5, label="day %d" % (k + 1))
        arrow(axR, (x, 2.35), (x, 2.05))
        axR.text(x, 2.45, "input", ha="center", fontsize=9.5, color=INK)
        if k < n_steps - 1:
            arrow(axR, (x + 0.7, 1.5), (x + 2.3, 1.5), FITLINE)
    arrow(axR, (1.6 + 3 * (n_steps - 1) + 0.7, 1.5),
          (1.6 + 3 * (n_steps - 1) + 1.5, 1.5))
    axR.text(1.6 + 3 * (n_steps - 1) + 1.6, 1.5, "prediction", ha="left",
             va="center", fontsize=10, color=INK)
    axR.set_title("The same cell, unrolled once per observation",
                  fontsize=13, color=MADRID, weight="bold")
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_vanishing_exploding(n=20):
    """Josh's ch8 problem figure: the weight between unrolled copies is applied
    once per step, so it is raised to a power. Below 1 it vanishes, above 1 it
    explodes."""
    steps = np.arange(0, n + 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.semilogy(steps, 0.5 ** steps, "o-", color=MADRID, lw=2.2, ms=4,
                label="$w = 0.5$  (vanishes)")
    ax.semilogy(steps, 2.0 ** steps, "o-", color=FITLINE, lw=2.2, ms=4,
                label="$w = 2.0$  (explodes)")
    ax.axhline(1.0, color="#7A7A85", ls=(0, (5, 4)), lw=1)
    ax.set_xlabel("Unrolled steps back in time", fontsize=12)
    ax.text(0, 1.02, "$w^{\\,steps}$  (log scale)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, color=INK)
    ax.set_title("Why basic RNNs struggle with long sequences",
                 fontsize=14, color=MADRID, weight="bold", pad=8)
    ax.legend(loc="center left", fontsize=10, framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_sigmoid_tanh():
    """The two activation functions LSTMs add, drawn as Josh draws them."""
    x = np.linspace(-6, 6, 400)
    sig = np.exp(x) / (np.exp(x) + 1)
    th = np.tanh(x)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.8), dpi=110)
    fig.patch.set_facecolor("white")
    for ax, y, name, col, lo, hi in ((a1, sig, "Sigmoid", MADRID, 0, 1),
                                     (a2, th, "Tanh", ACCENT, -1, 1)):
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.plot(x, y, color=col, lw=2.8, zorder=3)
        ax.axhline(lo, color="#7A7A85", ls=(0, (5, 4)), lw=1)
        ax.axhline(hi, color="#7A7A85", ls=(0, (5, 4)), lw=1)
        ax.set_xlim(-6, 6); ax.set_ylim(lo - 0.25, hi + 0.25)
        ax.set_xlabel("input", fontsize=11)
        ax.set_title("%s: output in [%g, %g]" % (name, lo, hi), fontsize=13,
                     color=col, weight="bold", pad=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a1.scatter([5], [np.exp(5) / (np.exp(5) + 1)], s=80, color=FITLINE,
               zorder=5)
    a1.text(4.6, 0.80, "$f(5)=0.99$", ha="right", fontsize=10, color=FITLINE)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============================ decision trees (ML book ch10)
# Josh's classification example, with the movie swapped to Evil Dead 2
def gini(counts):
    """Gini impurity of a leaf, given class counts."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return float(1.0 - np.sum(p ** 2))


def weighted_gini(left, right):
    nl, nr = sum(left), sum(right)
    return (nl / (nl + nr)) * gini(left) + (nr / (nl + nr)) * gini(right)


def fig_gini_split(question="Loves Popcorn", left=(1, 3), right=(2, 1),
                   stage="total", outcome="Loves Evil Dead 2", title=None):
    """One split, two leaves, and the arithmetic that scores it.

    left and right are (loves it, does not) counts. stage walks the slide:
    "counts" stops at the tally, "left" works out the left leaf, "leaves" works
    out both, "total" carries the two numbers into the weighted average.
    """
    gl, gr = gini(left), gini(right)
    tot = weighted_gini(left, right)
    nl, nr = sum(left), sum(right)

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    ax.add_patch(plt.Rectangle((3.4, 4.6), 3.2, 0.85, facecolor=ACCENT,
                               edgecolor=INK, lw=1.6, zorder=3))
    ax.text(5.0, 5.02, question, ha="center", va="center",
            fontsize=12, color="white", weight="bold", zorder=4)

    for x, cnt, g, lab, side in ((2.2, left, gl, "True", "left"),
                                 (7.8, right, gr, "False", "right")):
        n = sum(cnt)
        ax.annotate("", xy=(x + (0.7 if x < 5 else -0.7), 3.5),
                    xytext=(5.0 + (-0.6 if x < 5 else 0.6), 4.55),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=INK))
        ax.text((5.0 + x) / 2, 4.15, lab, ha="center", va="center",
                fontsize=11, color=INK, zorder=8,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none"))
        ax.add_patch(plt.Rectangle((x - 1.35, 2.35), 2.7, 1.15,
                                   facecolor="#2E8B57", edgecolor=INK,
                                   lw=1.6, zorder=3))
        ax.text(x, 3.16, outcome, ha="center", va="center",
                fontsize=10.5, color="white", weight="bold", zorder=4)
        ax.text(x - 0.62, 2.68, "True\n%d" % cnt[0], ha="center", va="center",
                fontsize=10.5, color="white", zorder=4)
        ax.text(x + 0.62, 2.68, "False\n%d" % cnt[1], ha="center", va="center",
                fontsize=10.5, color="white", zorder=4)
        ax.text(x, 1.98, "%d %s in this leaf" % (n, "person" if n == 1
                                                 else "people"),
                ha="center", fontsize=10, color=INK)
        worked = stage == "leaves" or (stage == "left" and side == "left")
        if worked:
            ax.text(x, 1.30,
                    "Gini $= 1 - \\left(\\frac{%d}{%d}\\right)^{2}"
                    " - \\left(\\frac{%d}{%d}\\right)^{2} = %.3f$"
                    % (cnt[0], n, cnt[1], n, g),
                    ha="center", fontsize=11.5, color=FITLINE, weight="bold")
        elif stage == "total":
            ax.text(x, 1.42, "Gini = %.3f" % g, ha="center", fontsize=11.5,
                    color=INK)

    if stage == "counts":
        ax.text(5.0, 1.30, "nothing scored yet, this is just where"
                           " the seven people land",
                ha="center", fontsize=11, color=INK)
        default = "Send everyone down one side or the other"
    elif stage in ("left", "leaves"):
        ax.text(5.0, 0.42, "one minus the squared share of each answer,"
                           " leaf by leaf",
                ha="center", fontsize=11, color=INK)
        default = "First, score each leaf on its own"
    else:
        ax.text(5.0, 0.52,
                "Total Gini  =  $\\frac{%d}{%d}$(%.3f) + $\\frac{%d}{%d}$(%.3f)"
                "  =  %.3f" % (nl, nl + nr, gl, nr, nl + nr, gr, tot),
                ha="center", fontsize=13, color=FITLINE, weight="bold")
        default = "Then average them, weighted by how many people are in each"

    ax.set_title(title or default, fontsize=13.5, color=MADRID,
                 weight="bold", pad=4)
    fig.tight_layout(); plt.close(fig)
    return fig
# Josh's regression example: drug dose against effectiveness, four clusters
_DT_DOSE = np.array([2., 4., 6., 8., 10., 13.,          # < 14.5   -> 4.2
                     16., 18., 20., 22.,                 # 14.5-23.5 -> 100
                     24., 25., 26., 27., 28.,            # 23.5-29  -> 52.8
                     30., 33., 36., 39.])                # >= 29    -> 2.5
_DT_EFF = np.array([0., 2., 3., 5., 6., 9.2,
                    100., 100., 100., 100.,
                    58., 55., 53., 50., 48.,
                    3., 2., 2., 3.])



def dt_predict(dose):
    dose = np.asarray(dose, dtype=float)
    out = np.empty_like(dose)
    out[dose < 14.5] = _DT_EFF[_DT_DOSE < 14.5].mean()
    m2 = (dose >= 14.5) & (dose < 23.5)
    out[m2] = _DT_EFF[(_DT_DOSE >= 14.5) & (_DT_DOSE < 23.5)].mean()
    m3 = (dose >= 23.5) & (dose < 29)
    out[m3] = _DT_EFF[(_DT_DOSE >= 23.5) & (_DT_DOSE < 29)].mean()
    out[dose >= 29] = _DT_EFF[_DT_DOSE >= 29].mean()
    return out


def fig_regression_tree():
    """Our version of Josh's figure: the tree's prediction is a step function,
    one flat level per leaf, each the average of the points in that leaf."""
    grid = np.linspace(0, 41, 600)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(grid, dt_predict(grid), color=MADRID, lw=2.8, zorder=3,
            label="regression tree")
    ax.scatter(_DT_DOSE, _DT_EFF, s=80, color=ACCENT, edgecolors=INK, lw=1.2,
               zorder=4)
    for b in (14.5, 23.5, 29):
        ax.axvline(b, color="#7A7A85", ls=(0, (5, 4)), lw=1)
        ax.text(b, 108, str(b), ha="center", fontsize=9.5, color="#5A5A64")
    ax.set_xlim(0, 41); ax.set_ylim(-6, 116)
    ax.set_xlabel("Drug dose", fontsize=12)
    ax.text(0, 1.02, "Drug effectiveness (%)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, color=INK)
    ax.set_title("A regression tree predicts a step function",
                 fontsize=14, color=MADRID, weight="bold", pad=16)
    ax.legend(loc="center right", fontsize=10, framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ================= step-by-step figure sequences (Josh gives each step a picture)
_GREEN = "#2E8B57"


def _curve_ax(ax, xlab="Drug dose", ylab=None, ylim=None):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.set_xlim(-0.03, 1.03)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlab, fontsize=12)
    if ylab:
        ax.text(0, 1.02, ylab, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=12, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _fig1(w=6.6, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    fig.patch.set_facecolor("white")
    return fig, ax


# ---------------------------------------------- ML-03: building the squiggle
def fig_nn_data():
    """Step 1: just the three observations."""
    fig, ax = _fig1()
    _curve_ax(ax, ylab="Effectiveness", ylim=(-0.35, 1.35))
    ax.scatter(_DOSE, _EFF, s=140, color=ACCENT, edgecolors=INK, lw=1.4, zorder=5)
    for d, e, lab in zip(_DOSE, _EFF, ["low", "medium", "high"]):
        ax.text(d, e + 0.12, lab, ha="center", fontsize=11, color=INK)
    ax.set_title("The data: only the medium dose works", fontsize=13.5,
                 color=MADRID, weight="bold", pad=26)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_nn_straight_line():
    """Step 2: the best straight line, which is useless here."""
    d = np.linspace(0, 1, 50)
    b, a = np.polyfit(_DOSE, _EFF, 1)
    fig, ax = _fig1()
    _curve_ax(ax, ylab="Effectiveness", ylim=(-0.35, 1.35))
    ax.plot(d, a + b * d, color=FITLINE, lw=2.8, zorder=3)
    ax.scatter(_DOSE, _EFF, s=140, color=ACCENT, edgecolors=INK, lw=1.4, zorder=5)
    ax.set_title("No straight line fits", fontsize=13.5, color=FITLINE,
                 weight="bold", pad=26)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_nn_top_preactivation():
    """Step 3: top node, dose x 1.43 + (-0.61), before the activation."""
    d = np.linspace(0, 1, 200)
    fig, ax = _fig1()
    _curve_ax(ax, ylab="$1.43 \\times$ dose $- 0.61$", ylim=(-0.9, 1.0))
    ax.axhline(0, color="#7A7A85", lw=1)
    ax.plot(d, _NW["w1"] * d + _NW["b1"], color=MADRID, lw=2.8, zorder=3)
    ax.set_title("Top node: multiply by the weight, add the bias",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_nn_top_relu():
    """Step 4: ReLU flattens the negative part into a bend."""
    d = np.linspace(0, 1, 200)
    z = _NW["w1"] * d + _NW["b1"]
    fig, ax = _fig1()
    _curve_ax(ax, ylab="ReLU output", ylim=(-0.9, 1.0))
    ax.axhline(0, color="#7A7A85", lw=1)
    ax.plot(d, z, color=MADRID, lw=1.4, ls=(0, (5, 4)), alpha=0.6, zorder=2)
    ax.plot(d, _relu(z), color=MADRID, lw=3.0, zorder=3)
    ax.axvline(-_NW["b1"] / _NW["w1"], color=ACCENT, lw=1.4, ls=(0, (4, 3)))
    ax.text(-_NW["b1"] / _NW["w1"] + 0.02, -0.7, "the bend", fontsize=11,
            color=ACCENT)
    ax.set_title("ReLU: negatives become zero, and a bend appears",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_nn_top_scaled():
    """Step 5: multiply by the output weight -3.89, which flips and stretches."""
    d = np.linspace(0, 1, 200)
    fig, ax = _fig1()
    _curve_ax(ax, ylab="Top node contribution", ylim=(-3.5, 1.0))
    ax.axhline(0, color="#7A7A85", lw=1)
    ax.plot(d, _relu(_NW["w1"] * d + _NW["b1"]), color=MADRID, lw=1.4,
            ls=(0, (5, 4)), alpha=0.6, zorder=2)
    ax.plot(d, nn_top(d), color=MADRID, lw=3.0, zorder=3)
    ax.set_title("Multiply by $-3.89$: the bend flips and stretches",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_nn_bottom_scaled():
    """Step 6: the bottom node, all the way through its own weight."""
    d = np.linspace(0, 1, 200)
    fig, ax = _fig1()
    _curve_ax(ax, ylab="Bottom node contribution", ylim=(-0.5, 3.5))
    ax.axhline(0, color="#7A7A85", lw=1)
    ax.plot(d, _relu(_NW["w2"] * d + _NW["b2"]), color=ACCENT, lw=1.4,
            ls=(0, (5, 4)), alpha=0.6, zorder=2)
    ax.plot(d, nn_bottom(d), color=ACCENT, lw=3.0, zorder=3)
    ax.set_title("Bottom node: same three moves, its own numbers",
                 fontsize=13.5, color=ACCENT, weight="bold", pad=26)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_nn_both():
    """Step 7: both contributions on one pair of axes, before adding."""
    d = np.linspace(0, 1, 200)
    fig, ax = _fig1()
    _curve_ax(ax, ylab="Each node's contribution", ylim=(-3.5, 3.5))
    ax.axhline(0, color="#7A7A85", lw=1)
    ax.plot(d, nn_top(d), color=MADRID, lw=3.0, zorder=3, label="top node")
    ax.plot(d, nn_bottom(d), color=ACCENT, lw=3.0, zorder=3, label="bottom node")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.set_title("Two bent lines, neither useful on its own",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_nn_sum():
    """Step 8: add them, and the fitted squiggle appears."""
    d = np.linspace(0, 1, 200)
    fig, ax = _fig1()
    _curve_ax(ax, ylab="Effectiveness", ylim=(-0.5, 1.5))
    ax.axhline(0, color="#7A7A85", lw=1)
    ax.plot(d, nn_top(d), color=MADRID, lw=1.4, ls=(0, (5, 4)), alpha=0.55)
    ax.plot(d, nn_bottom(d), color=ACCENT, lw=1.4, ls=(0, (5, 4)), alpha=0.55)
    ax.plot(d, nn_output(d), color=_GREEN, lw=3.4, zorder=4)
    ax.scatter(_DOSE, _EFF, s=140, color=ACCENT, edgecolors=INK, lw=1.4, zorder=5)
    ax.set_title("Add them: the squiggle passes through every point",
                 fontsize=13.5, color=_GREEN, weight="bold", pad=26)
    fig.tight_layout(); plt.close(fig)
    return fig


# ---------------------------------------------- ML-07: sliding the filter
def fig_cnn_filter_at(i, j):
    """Josh slides the filter one position at a time. This draws one position:
    the image with the 3x3 window outlined, and the arithmetic for that cell."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 4.0), dpi=110,
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("white")
    _grid(axL, _IMG_O, "Input image", fs=10)
    axL.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 3, 3, fill=False,
                                edgecolor=FITLINE, lw=3.0, zorder=6))
    patch = _IMG_O[i:i + 3, j:j + 3]
    val = float(np.sum(patch * _FILTER))
    _grid(axR, patch * _FILTER, "patch $\\times$ filter  (sum = %d)" % val,
          cmap="Oranges", fs=11)
    fig.suptitle("Filter at row %d, column %d" % (i + 1, j + 1), fontsize=13.5,
                 color=MADRID, weight="bold", y=1.02)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_cnn_feature_map_partial(n_filled):
    """The feature map filling in, one cell at a time."""
    full = convolve(_IMG_O, _FILTER)
    shown = np.full(full.shape, np.nan)
    flat = 0
    for i in range(full.shape[0]):
        for j in range(full.shape[1]):
            if flat < n_filled:
                shown[i, j] = full[i, j]
            flat += 1
    fig, ax = _fig1(5.2, 4.4)
    ax.imshow(np.nan_to_num(shown), cmap="Blues", vmin=0, vmax=full.max())
    for (i, j), v in np.ndenumerate(shown):
        ax.text(j, i, "" if np.isnan(v) else "%d" % v, ha="center", va="center",
                fontsize=13, color="white" if v > 0.6 * full.max() else INK)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Feature map: %d of %d positions" % (n_filled, full.size),
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    fig.tight_layout(); plt.close(fig)
    return fig


# ---------------------------------------------- ML-02: growing the tree
def fig_gini_candidates():
    """Josh scores every candidate question. This shows all three side by side
    so the winner is visible."""
    cands = [("Loves popcorn", (1, 3), (2, 1)),
             ("Loves soda", (3, 1), (0, 3)),
             ("Age > 12.5", (2, 2), (1, 2))]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), dpi=110)
    fig.patch.set_facecolor("white")
    scores = [weighted_gini(l, r) for _, l, r in cands]
    best = int(np.argmin(scores))
    for k, (ax, (name, left, right)) in enumerate(zip(axes, cands)):
        ax.set_facecolor("white"); ax.axis("off")
        ax.set_xlim(0, 10); ax.set_ylim(0, 6)
        col = _GREEN if k == best else INK
        ax.add_patch(plt.Rectangle((3.0, 4.4), 4.0, 0.9, facecolor=col,
                                   edgecolor=INK, lw=1.6))
        ax.text(5.0, 4.85, name, ha="center", va="center", fontsize=11,
                color="white", weight="bold")
        for x, cnt in ((2.4, left), (7.6, right)):
            ax.add_patch(plt.Rectangle((x - 1.3, 2.5), 2.6, 1.0,
                                       facecolor="white", edgecolor=col, lw=1.8))
            ax.text(x, 3.0, "%d true / %d false" % cnt, ha="center", va="center",
                    fontsize=10.5, color=INK)
            ax.annotate("", xy=(x, 3.6), xytext=(5.0, 4.35),
                        arrowprops=dict(arrowstyle="-|>", lw=1.5, color=INK))
        ax.text(5.0, 1.6, "weighted Gini = %.3f" % scores[k], ha="center",
                fontsize=12, color=col, weight="bold")
        if k == best:
            ax.text(5.0, 0.8, "best split", ha="center", fontsize=11,
                    color=_GREEN, style="italic")
    fig.suptitle("Score every candidate question, keep the smallest",
                 fontsize=13.5, color=MADRID, weight="bold", y=1.04)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_regression_tree_steps(n_splits):
    """The regression tree deepening one split at a time."""
    bounds = [[0, 41], [0, 14.5, 41], [0, 14.5, 29, 41], [0, 14.5, 23.5, 29, 41]]
    edges = bounds[min(n_splits, 3)]
    grid = np.linspace(0, 41, 600)
    pred = np.empty_like(grid)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (_DT_DOSE >= a) & (_DT_DOSE < b)
        pred[(grid >= a) & (grid < b)] = _DT_EFF[m].mean() if m.any() else 0.0
    fig, ax = _fig1(7.4, 4.2)
    _curve_ax(ax, xlab="Drug dose", ylab="Effectiveness (%)")
    ax.set_xlim(0, 41); ax.set_ylim(-6, 116)
    ax.plot(grid, pred, color=MADRID, lw=2.8, zorder=3)
    ax.scatter(_DT_DOSE, _DT_EFF, s=80, color=ACCENT, edgecolors=INK, lw=1.2,
               zorder=4)
    for e in edges[1:-1]:
        ax.axvline(e, color="#7A7A85", ls=(0, (5, 4)), lw=1)
    ax.set_title("%d split%s" % (n_splits, "" if n_splits == 1 else "s"),
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    fig.tight_layout(); plt.close(fig)
    return fig


# ---------------------------------------------- ML-04: backprop steps
def fig_bp_fit_at(w3):
    """The network's curve for a given value of w3."""
    d = np.linspace(0, 1, 200)
    fig, ax = _fig1(6.4, 4.2)
    _curve_ax(ax, ylab="Effectiveness", ylim=(-0.6, 3.5))
    ax.plot(d, bp_output(d, w3), color=_GREEN, lw=3.0, zorder=3)
    ax.scatter(_DOSE, _EFF, s=120, color=ACCENT, edgecolors=INK, lw=1.3, zorder=5)
    ax.set_title("$w_3$ = %.2f    SSR = %.2f" % (w3, bp_ssr(w3)),
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_bp_loss_at(w3):
    """SSR as a function of w3, with the current value marked."""
    grid = np.linspace(-6, 1, 200)
    loss = np.array([bp_ssr(w) for w in grid])
    fig, ax = _fig1(6.4, 4.2)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(grid, loss, color=INK, lw=2.4, zorder=2)
    ax.scatter([w3], [bp_ssr(w3)], s=130, color=ACCENT, edgecolors=INK, lw=1.4,
               zorder=5)
    ax.axvline(bp_optimal_w3(), color="#7A7A85", ls=(0, (5, 4)), lw=1)
    ax.set_xlim(-6, 1); ax.set_ylim(-0.6, loss.max() + 0.6)
    ax.set_xlabel("$w_3$", fontsize=12)
    ax.text(0, 1.02, "SSR", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("$w_3$ = %.2f" % w3, fontsize=13.5, color=MADRID,
                 weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ---------------------------------------------- ML-05: one surface per species
def fig_iris_surface(k):
    """Josh draws one crinkled surface per output. This is species k."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    p = np.linspace(0, 1.2, 45)
    s = np.linspace(0, 1.2, 45)
    P, S = np.meshgrid(p, s)
    Z = np.zeros_like(P)
    for a in range(P.shape[0]):
        for b in range(P.shape[1]):
            Z[a, b] = iris_forward(P[a, b], S[a, b])[k]
    fig = plt.figure(figsize=(6.2, 4.6), dpi=110)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(P, S, Z, cmap=["Greens", "Purples", "Reds"][k],
                    edgecolor="none", alpha=0.95)
    ax.set_xlabel("Petal width", fontsize=10, labelpad=4)
    ax.set_ylabel("Sepal width", fontsize=10, labelpad=4)
    ax.set_zlabel("output", fontsize=10, labelpad=2)
    ax.set_title("%s output surface" % _SPECIES[k], fontsize=13.5,
                 color=MADRID, weight="bold", pad=4)
    ax.view_init(elev=24, azim=-60)
    ax.tick_params(labelsize=7)
    fig.tight_layout(); plt.close(fig)
    return fig


# ---------------------------------------------- ML-06: the -log curve
def fig_neglog():
    """Why -log: a confident wrong answer costs a great deal."""
    p = np.linspace(0.01, 1.0, 300)
    fig, ax = _fig1(6.6, 4.2)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(p, -np.log(p), color=FITLINE, lw=3.0, zorder=3)
    for pv in (0.15, 0.65, 0.95):
        ax.scatter([pv], [-np.log(pv)], s=90, color=ACCENT, edgecolors=INK,
                   lw=1.2, zorder=5)
        ax.text(pv + 0.02, -np.log(pv) + 0.12, "p=%.2f\nloss=%.2f"
                % (pv, -np.log(pv)), fontsize=9.5, color=INK)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 4.2)
    ax.set_xlabel("probability given to the true class", fontsize=12)
    ax.text(0, 1.02, "$-\\log(p)$", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("Confident and wrong is punished hardest", fontsize=13.5,
                 color=FITLINE, weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ---------------------------------------------- ML-09: one activation each
def fig_activation(which="sigmoid"):
    x = np.linspace(-6, 6, 400)
    if which == "sigmoid":
        y, name, col, lo, hi = np.exp(x) / (np.exp(x) + 1), "Sigmoid", MADRID, 0, 1
    else:
        y, name, col, lo, hi = np.tanh(x), "Tanh", ACCENT, -1, 1
    fig, ax = _fig1(6.4, 4.0)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(x, y, color=col, lw=3.0, zorder=3)
    ax.axhline(lo, color="#7A7A85", ls=(0, (5, 4)), lw=1)
    ax.axhline(hi, color="#7A7A85", ls=(0, (5, 4)), lw=1)
    if which == "sigmoid":
        ax.scatter([5], [np.exp(5) / (np.exp(5) + 1)], s=90, color=FITLINE,
                   zorder=5)
        ax.text(4.7, 0.80, "$f(5)=0.99$", ha="right", fontsize=11, color=FITLINE)
    ax.set_xlim(-6, 6); ax.set_ylim(lo - 0.25, hi + 0.25)
    ax.set_xlabel("input", fontsize=12)
    ax.text(0, 1.02, "output", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("%s: output squashed into [%g, %g]" % (name, lo, hi),
                 fontsize=13.5, color=col, weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_gradient_problem(kind="vanish"):
    n = np.arange(0, 21)
    w = 0.5 if kind == "vanish" else 2.0
    col = MADRID if kind == "vanish" else FITLINE
    fig, ax = _fig1(6.6, 4.0)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.semilogy(n, w ** n, "o-", color=col, lw=2.4, ms=5, zorder=3)
    ax.axhline(1.0, color="#7A7A85", ls=(0, (5, 4)), lw=1)
    ax.set_xlabel("Unrolled steps back in time", fontsize=12)
    ax.text(0, 1.02, "$w^{\\,steps}$ (log scale)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, color=INK)
    ttl = ("$w = 0.5$: after 20 steps, $0.0000010$"
           if kind == "vanish" else "$w = 2.0$: after 20 steps, $1{,}048{,}576$")
    ax.set_title(ttl, fontsize=13.5, color=col, weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============== NN ch1: the remaining figures, to complete the chapter =========
def fig_squiggle_examples():
    """Josh's 'a problem' panel: three datasets needing three different shapes."""
    x = np.linspace(0, 1, 200)
    sets = [("a smooth squiggle", 100 * np.sin(np.pi * x), MADRID),
            ("a bent shape", 100 * (1 - np.abs(2 * x - 1)), _GREEN),
            ("a complicated squiggle",
             50 + 50 * np.cos(3 * np.pi * x - np.pi), INK)]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), dpi=110)
    fig.patch.set_facecolor("white")
    for ax, (name, y, col) in zip(axes, sets):
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.plot(x, y, color=col, lw=3.0, zorder=3)
        pts = np.array([0.0, 0.5, 1.0])
        ax.scatter(pts, np.interp(pts, x, y), s=90, color=ACCENT,
                   edgecolors=INK, lw=1.2, zorder=5)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["Low", "Medium", "High"])
        ax.set_ylim(-12, 112); ax.set_xlabel("Drug dose", fontsize=11)
        ax.set_title(name, fontsize=12, color=col, weight="bold", pad=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(0, 1.06, "Drug effectiveness %", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=11, color=INK)
    fig.suptitle("Different datasets need different shapes", fontsize=13.5,
                 color=MADRID, weight="bold", y=1.06)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_activation_zoo():
    """The three activation functions Josh names: ReLU, SoftPlus, Sigmoid."""
    x = np.linspace(-4, 4, 400)
    fns = [("ReLU", np.maximum(x, 0), MADRID, "bent at $x=0$"),
           ("SoftPlus", np.log(1 + np.exp(x)), FITLINE, "a smooth curve"),
           ("Sigmoid", np.exp(x) / (np.exp(x) + 1), _GREEN, "an s-shaped squiggle")]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5), dpi=110)
    fig.patch.set_facecolor("white")
    for ax, (name, y, col, sub) in zip(axes, fns):
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.axhline(0, color="#7A7A85", lw=1); ax.axvline(0, color="#7A7A85", lw=1)
        ax.plot(x, y, color=col, lw=3.0, zorder=3)
        ax.set_xlim(-4, 4); ax.set_ylim(-0.6, 4.2)
        ax.set_xlabel("input", fontsize=11)
        ax.set_title("%s\n%s" % (name, sub), fontsize=12, color=col,
                     weight="bold", pad=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Three common activation functions", fontsize=13.5,
                 color=MADRID, weight="bold", y=1.04)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_relu_example(xv):
    """Josh plugs a number into ReLU and reads the output off the graph."""
    x = np.linspace(-3, 3, 300)
    out = max(0.0, xv)
    fig, ax = _fig1(5.8, 4.0)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.axhline(0, color="#7A7A85", lw=1); ax.axvline(0, color="#7A7A85", lw=1)
    ax.plot(x, np.maximum(x, 0), color=MADRID, lw=3.0, zorder=3)
    ax.plot([xv, xv], [0, out], color=FITLINE, lw=1.8, ls=(0, (4, 3)), zorder=4)
    ax.plot([0, xv], [out, out], color=FITLINE, lw=1.8, ls=(0, (4, 3)), zorder=4)
    ax.scatter([xv], [out], s=130, color=ACCENT, edgecolors=INK, lw=1.4, zorder=6)
    ax.set_xlim(-3, 3); ax.set_ylim(-0.8, 3.2)
    ax.set_xlabel("input $x$", fontsize=12)
    ax.text(0, 1.02, "ReLU$(x)$", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("ReLU(%.1f) = Max(0, %.1f) = %.1f" % (xv, xv, out),
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_nn_dots(node, upto):
    """Josh feeds doses in one at a time and the coloured dots accumulate.
    node is 'top' or 'bottom'; upto is the highest dose fed in so far."""
    doses = np.round(np.arange(0.0, 1.01, 0.1), 2)
    shown = doses[doses <= upto + 1e-9]
    col = MADRID if node == "top" else ACCENT
    f = nn_top if node == "top" else nn_bottom
    d = np.linspace(0, 1, 200)
    fig, ax = _fig1(6.4, 4.2)
    _curve_ax(ax, ylab="%s node output" % node.capitalize())
    ax.set_ylim(-3.6, 3.6)
    ax.axhline(0, color="#7A7A85", lw=1)
    if upto >= 1.0:
        ax.plot(d, f(d), color=col, lw=2.6, alpha=0.55, zorder=2)
    ax.scatter(shown, f(shown), s=110, color=col, edgecolors=INK, lw=1.2,
               zorder=5)
    ax.set_title("Doses fed in: 0 to %.1f" % upto, fontsize=13.5, color=col,
                 weight="bold", pad=26)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_combine_bends():
    """Josh's 'stretches, flips, crops and combines' panel: orange + blue = green."""
    d = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5), dpi=110)
    fig.patch.set_facecolor("white")
    series = [(nn_bottom(d), ACCENT, "bottom node", (-0.5, 3.5)),
              (nn_top(d), MADRID, "top node", (-3.5, 0.5)),
              (nn_output(d), _GREEN, "their sum", (-0.5, 1.5))]
    for ax, (y, col, name, ylim) in zip(axes, series):
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.axhline(0, color="#7A7A85", lw=1)
        ax.plot(d, y, color=col, lw=3.0, zorder=3)
        if name == "their sum":
            ax.scatter(_DOSE, _EFF, s=100, color=ACCENT, edgecolors=INK,
                       lw=1.2, zorder=5)
        ax.set_ylim(*ylim); ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("Drug dose", fontsize=11)
        ax.set_title(name, fontsize=12.5, color=col, weight="bold", pad=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Stretch, flip, and combine", fontsize=13.5, color=MADRID,
                 weight="bold", y=1.04)
    fig.tight_layout(); plt.close(fig)
    return fig


# ===== ML book ch10: Josh's full training data, with Evil Dead 2 as the movie =====
_ED_POP = np.array([1, 1, 0, 0, 1, 1, 0])          # loves popcorn
_ED_SODA = np.array([1, 0, 1, 1, 1, 0, 0])         # loves soda
_ED_AGE = np.array([7, 12, 18, 35, 38, 50, 83])
_ED_Y = np.array([0, 0, 1, 1, 1, 0, 0])            # loves Evil Dead 2


def _counts(mask):
    return (int(np.sum(_ED_Y[mask] == 1)), int(np.sum(_ED_Y[mask] == 0)))


def ed_split(feature, threshold=None):
    """Return ((yes_counts),(no_counts), weighted gini) for one candidate."""
    if feature == "popcorn":
        m = _ED_POP == 1
    elif feature == "soda":
        m = _ED_SODA == 1
    else:
        m = _ED_AGE < threshold
    left, right = _counts(m), _counts(~m)
    return left, right, weighted_gini(left, right)


def _age_thresholds():
    a = np.sort(_ED_AGE)
    return [(a[i] + a[i + 1]) / 2 for i in range(len(a) - 1)]


def fig_dt_data_table():
    """Josh's training data, drawn as he draws it."""
    fig, ax = _fig1(7.4, 4.2)
    ax.set_facecolor("white"); ax.axis("off")
    cols = ["Loves\nPopcorn", "Loves\nSoda", "Age", "Loves\nEvil Dead 2"]
    rows = list(zip(_ED_POP, _ED_SODA, _ED_AGE, _ED_Y))
    nR, nC = len(rows), len(cols)
    for j, c in enumerate(cols):
        ax.add_patch(plt.Rectangle((j, nR), 1, 1,
                                   facecolor=ACCENT if j < 3 else _GREEN,
                                   edgecolor="white", lw=2))
        ax.text(j + .5, nR + .5, c, ha="center", va="center", fontsize=10,
                color="white", weight="bold")
    for i, r in enumerate(rows):
        y = nR - 1 - i
        for j, v in enumerate(r):
            ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor="#F5F5F7",
                                       edgecolor="white", lw=2))
            if j == 2:
                txt, col = str(v), INK
            else:
                txt = "True" if v else "False"
                col = (MADRID if v else FITLINE) if j == 3 else INK
            ax.text(j + .5, y + .5, txt, ha="center", va="center", fontsize=11,
                    color=col, weight="bold" if j == 3 else "normal")
    ax.set_xlim(-.2, nC + .2); ax.set_ylim(-.2, nR + 1.2)
    ax.set_title("The training data: 7 people", fontsize=13.5, color=MADRID,
                 weight="bold", pad=10)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_dt_candidate(feature, threshold=None, show_gini=True):
    """One candidate question. With show_gini False it stops at the counts,
    which is where the video is before impurity has been named."""
    left, right, tot = ed_split(feature, threshold)
    name = {"popcorn": "Loves Popcorn", "soda": "Loves Soda"}.get(
        feature, "Age $<$ %.1f" % (threshold or 0))
    fig, ax = _fig1(7.4, 4.4)
    ax.set_facecolor("white"); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.add_patch(plt.Rectangle((3.2, 4.5), 3.6, 0.9, facecolor=ACCENT,
                               edgecolor=INK, lw=1.8))
    ax.text(5.0, 4.95, name, ha="center", va="center", fontsize=12.5,
            color="white", weight="bold")
    for x, cnt, lab in ((2.3, left, "True"), (7.7, right, "False")):
        ax.annotate("", xy=(x, 3.6), xytext=(5.0 + (-0.9 if x < 5 else 0.9), 4.45),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=INK))
        ax.text((5.0 + x) / 2, 4.05, lab, ha="center", va="center",
                fontsize=11, color=INK, zorder=8,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none"))
        pure = (cnt[0] == 0 or cnt[1] == 0)
        ax.add_patch(plt.Rectangle((x - 1.5, 2.3), 3.0, 1.25,
                                   facecolor=_GREEN if pure else "white",
                                   edgecolor=_GREEN, lw=2.0))
        c = "white" if pure else INK
        ax.text(x, 3.15, "Loves Evil Dead 2", ha="center", fontsize=10.5, color=c,
                weight="bold")
        ax.text(x, 2.65, "%d true    %d false" % cnt, ha="center", fontsize=11, color=c)
        if show_gini:
            ax.text(x, 1.85, "Gini = %.3f" % gini(cnt), ha="center", fontsize=11,
                    color=INK)
        if pure and show_gini:
            ax.text(x, 1.35, "pure", ha="center", fontsize=10, color=_GREEN,
                    style="italic")
    if show_gini:
        ax.text(5.0, 0.6, "weighted Gini = %.3f" % tot, ha="center",
                fontsize=13.5, color=FITLINE, weight="bold")
    ax.set_title("Candidate question", fontsize=13, color=MADRID,
                 weight="bold", pad=6)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_dt_age_search():
    """Josh tries every midpoint between adjacent ages and scores each."""
    ths = _age_thresholds()
    scores = [ed_split("age", t)[2] for t in ths]
    best = int(np.argmin(scores))
    fig, ax = _fig1(7.4, 4.2)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(ths, scores, "o-", color=MADRID, lw=2.2, ms=8, zorder=3)
    ax.scatter([ths[best]], [scores[best]], s=170, color=_GREEN,
               edgecolors=INK, lw=1.5, zorder=5)
    ax.text(ths[best], scores[best] - 0.035, "best age split\n%.1f"
            % ths[best], ha="center", va="top", fontsize=10.5, color=_GREEN)
    for t, s in zip(ths, scores):
        ax.text(t, s + 0.012, "%.3f" % s, ha="center", fontsize=9, color=INK)
    ax.set_xlabel("Age threshold (midpoint between adjacent ages)", fontsize=11.5)
    ax.text(0, 1.02, "weighted Gini", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_ylim(min(scores) - 0.08, max(scores) + 0.05)
    ax.set_title("Age is numeric, so every midpoint is a candidate",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_dt_scoreboard():
    """All three features scored side by side, winner highlighted."""
    ths = _age_thresholds()
    age_scores = [ed_split("age", t)[2] for t in ths]
    items = [("Loves Popcorn", ed_split("popcorn")[2]),
             ("Loves Soda", ed_split("soda")[2]),
             ("Age < %.1f" % ths[int(np.argmin(age_scores))], min(age_scores))]
    names = [n for n, _ in items]; vals = [v for _, v in items]
    best = int(np.argmin(vals))
    fig, ax = _fig1(7.4, 4.0)
    ax.set_facecolor("white")
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    cols = [_GREEN if k == best else MADRID for k in range(3)]
    ax.barh(names, vals, color=cols, edgecolor=INK, lw=1.2, height=0.55, zorder=3)
    for k, v in enumerate(vals):
        ax.text(v + 0.008, k, "%.3f" % v, va="center", fontsize=11.5,
                color=cols[k], weight="bold")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("weighted Gini impurity (lower is better)", fontsize=11.5)
    ax.set_title("Loves Soda wins the root", fontsize=13.5, color=_GREEN,
                 weight="bold", pad=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_rt_threshold_search():
    """Regression trees score candidate splits by SSR, not Gini."""
    d, y = _DT_DOSE, _DT_EFF
    cands = [(d[i] + d[i + 1]) / 2 for i in range(len(d) - 1)]
    ssrs = []
    for t in cands:
        lo, hi = y[d < t], y[d >= t]
        ssrs.append(float(np.sum((lo - lo.mean()) ** 2)
                          + np.sum((hi - hi.mean()) ** 2)))
    best = int(np.argmin(ssrs))
    fig, ax = _fig1(7.6, 4.2)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(cands, ssrs, "o-", color=MADRID, lw=2.2, ms=6, zorder=3)
    ax.scatter([cands[best]], [ssrs[best]], s=170, color=_GREEN,
               edgecolors=INK, lw=1.5, zorder=5)
    ax.text(cands[best], ssrs[best] * 1.06, "best first split: %.1f"
            % cands[best], ha="center", fontsize=11, color=_GREEN)
    ax.set_xlabel("candidate split point (drug dose)", fontsize=11.5)
    ax.text(0, 1.02, "total SSR after the split", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, color=INK)
    ax.set_title("Try every split, keep the one with the smallest SSR",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============== ML-05 / ML-06: outputs, ArgMax, SoftMax, cross entropy ========
def fig_output_conversion(petal=0.0, sepal=1.0):
    """Raw outputs, then ArgMax, then SoftMax, as three panels."""
    raw = iris_forward(petal, sepal)
    am = (raw == raw.max()).astype(float)
    sm = softmax(raw)
    panels = [(raw, "Raw output", MADRID, None),
              (am, "ArgMax", FITLINE, (0, 1.15)),
              (sm, "SoftMax", _GREEN, (0, 1.15))]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), dpi=110)
    fig.patch.set_facecolor("white")
    for ax, (v, name, col, ylim) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.bar(_SPECIES, v, color=col, edgecolor=INK, lw=1.2, width=0.6, zorder=3)
        ax.axhline(0, color=INK, lw=1)
        for k, val in enumerate(v):
            ax.text(k, val + (0.05 if val >= 0 else -0.14), "%.2f" % val,
                    ha="center", fontsize=11, color=col, weight="bold")
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.set_title(name, fontsize=13, color=col, weight="bold", pad=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Petal width %.1f, sepal width %.1f" % (petal, sepal),
                 fontsize=13, color=INK, y=1.04)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_ce_row(k, setosa_bias=-2.0):
    """One training flower: its SoftMax probabilities, with the true class
    marked and its -log(p) shown. Josh does this row by row."""
    names = ["Setosa", "Virginica", "Versicolor"]
    x, y = _IRIS_X[k], _IRIS_Y[k]
    p = softmax(iris_raw_with_bias(x, setosa_bias))
    ce = -np.log(p[y])
    cols = [_GREEN if i == y else "#B8B8C2" for i in range(3)]
    fig, ax = _fig1(7.0, 4.2)
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.bar(_SPECIES, p, color=cols, edgecolor=INK, lw=1.2, width=0.6, zorder=3)
    for i, v in enumerate(p):
        ax.text(i, v + 0.03, "%.2f" % v, ha="center", fontsize=11,
                color=cols[i] if i == y else INK, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.text(0, 1.02, "SoftMax probability", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("True species: %s     $-\\log(%.2f) = %.2f$"
                 % (names[k], p[y], ce), fontsize=13.5, color=_GREEN,
                 weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============== ML-08: Josh's four StatLand stock trends ======================
_STATLAND = [((0.0, 0.0), 0.0, "low for two days stays low"),
             ((0.0, 0.5), 1.0, "low then medium goes higher"),
             ((1.0, 0.5), 0.0, "high then medium goes lower"),
             ((1.0, 1.0), 1.0, "high for two days stays high")]


def fig_statland(k):
    """One of Josh's four stock scenarios: yesterday, today, tomorrow."""
    (yest, today), tom, caption = _STATLAND[k]
    fig, ax = _fig1(6.2, 4.0)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    xs = [0, 1, 2]
    ax.plot(xs[:2], [yest, today], "o-", color=MADRID, lw=2.6, ms=13, zorder=3)
    ax.plot([1, 2], [today, tom], "--", color=_GREEN, lw=2.4, zorder=3)
    ax.scatter([2], [tom], s=170, color=_GREEN, edgecolors=INK, lw=1.4, zorder=5)
    ax.set_xticks(xs); ax.set_xticklabels(["Yesterday", "Today", "Tomorrow"])
    ax.set_ylim(-0.18, 1.18); ax.set_yticks([0, 0.5, 1])
    ax.text(0, 1.02, "Price (scaled 0 to 1)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, color=INK)
    ax.set_title(caption.capitalize(), fontsize=13.5, color=MADRID,
                 weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============== ML-09: the two memory paths ==================================
def fig_lstm_paths():
    """Josh's 'two separate paths' picture: a long-term memory that crosses the
    cell with little interference, and a short-term memory updated each step."""
    fig, ax = _fig1(10.4, 4.4)
    ax.set_facecolor("white"); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    for k in range(3):
        x = 2.0 + 3.4 * k
        ax.add_patch(plt.Rectangle((x - 1.2, 1.5), 2.4, 3.0, facecolor="#F5F5F7",
                                   edgecolor=INK, lw=1.8, zorder=2))
        ax.text(x, 4.75, "step %d" % (k + 1), ha="center", fontsize=10.5,
                color=INK)
        for gy, gc in ((3.6, MADRID), (2.9, MADRID), (2.2, ACCENT)):
            ax.add_patch(plt.Circle((x, gy), 0.22, facecolor="white",
                                    edgecolor=gc, lw=1.8, zorder=4))
        ax.text(x + 0.45, 3.6, "forget", fontsize=8.5, color=MADRID, va="center")
        ax.text(x + 0.45, 2.9, "input", fontsize=8.5, color=MADRID, va="center")
        ax.text(x + 0.45, 2.2, "output", fontsize=8.5, color=ACCENT, va="center")
    ax.annotate("", xy=(11.4, 4.15), xytext=(0.4, 4.15),
                arrowprops=dict(arrowstyle="-|>", lw=3.0, color=_GREEN))
    ax.text(0.4, 4.45, "long-term memory: crosses with little interference",
            fontsize=11, color=_GREEN, weight="bold")
    ax.annotate("", xy=(11.4, 1.15), xytext=(0.4, 1.15),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color=ACCENT))
    ax.text(0.4, 0.72, "short-term memory: rewritten at every step",
            fontsize=11, color=ACCENT, weight="bold")
    ax.set_title("An LSTM carries two memories, not one", fontsize=14,
                 color=MADRID, weight="bold", pad=4)
    fig.tight_layout(); plt.close(fig)
    return fig


# ==================================== ensembles: bagging, forests, boosting ===
# A single noisy regression problem, used throughout so the decks can compare
# a lone tree against a bagged ensemble, a forest, and a boosted model.
_ENS_SIGMA = 0.30


def _ens_truth(x):
    return np.sin(2.0 * np.pi * x) + 0.4 * x


def ens_data(n=60, seed=3):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = _ens_truth(x) + rng.normal(0, _ENS_SIGMA, n)
    return x, y


def fig_bootstrap_samples(n_show=3, seed=5):
    """A bootstrap sample draws n rows WITH replacement, so some rows repeat
    and some are left out. Josh's course calls the left-out rows out-of-bag."""
    x, y = ens_data()
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, n_show + 1, figsize=(3.3 * (n_show + 1), 3.4),
                             dpi=110, sharey=True)
    fig.patch.set_facecolor("white")
    axes[0].scatter(x, y, s=34, color=INK, alpha=0.8, zorder=3)
    axes[0].set_title("Original sample", fontsize=12, color=INK,
                      weight="bold", pad=8)
    for k in range(n_show):
        ax = axes[k + 1]
        idx = rng.integers(0, len(x), len(x))
        used = np.bincount(idx, minlength=len(x))
        ax.scatter(x, y, s=20, color="#C9C9D2", zorder=2)
        ax.scatter(x[used > 0], y[used > 0], s=30 + 22 * used[used > 0],
                   color=ACCENT, alpha=0.75, edgecolors=INK, lw=0.6, zorder=4)
        oob = int(np.sum(used == 0))
        ax.set_title("Bootstrap %d\n%d rows left out" % (k + 1, oob),
                     fontsize=11.5, color=ACCENT, weight="bold", pad=8)
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.set_xlabel("x", fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(0, 1.06, "y", transform=axes[0].transAxes, ha="left",
                 va="bottom", fontsize=11, color=INK)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_bagging_variance(max_B=40, n_rep=120, seed=7):
    """Averaging B independent-ish predictors divides the variance by B.
    Measured, and compared with the sigma^2 / B curve."""
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(seed)
    xg = np.linspace(0.05, 0.95, 25).reshape(-1, 1)
    Bs = [1, 2, 5, 10, 20, 40]
    measured = []
    for B in Bs:
        preds = np.empty((n_rep, len(xg)))
        for r in range(n_rep):
            x, y = ens_data(seed=int(rng.integers(1e6)))
            acc = np.zeros(len(xg))
            for _ in range(B):
                idx = rng.integers(0, len(x), len(x))
                t = DecisionTreeRegressor(max_depth=None, random_state=0)
                t.fit(x[idx].reshape(-1, 1), y[idx])
                acc += t.predict(xg)
            preds[r] = acc / B
        measured.append(float(np.mean(preds.var(axis=0))))
    measured = np.array(measured)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(Bs, measured, "o-", color=MADRID, lw=2.4, ms=7, zorder=4,
            label="measured variance")
    ax.plot(Bs, measured[0] / np.array(Bs), "--", color=FITLINE, lw=2.0,
            zorder=3, label=r"$\sigma^2 / B$")
    ax.set_xlabel("B, number of trees averaged", fontsize=12)
    ax.text(0, 1.02, "variance of the prediction", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, color=INK)
    ax.annotate("the gap: bagged trees are correlated,\n"
                "so averaging helps less than\n"
                "independence would promise",
                xy=(20, measured[-2]), xytext=(13, measured[0] * 0.62),
                fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color=INK))
    ax.set_title(r"Averaging helps, but less than $\sigma^2/B$ promises",
                 fontsize=14, color=MADRID, weight="bold", pad=26)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_tree_vs_ensemble(kind="bagging", n_estimators=100, seed=3):
    """One deep tree against an averaged ensemble, on the same data."""
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (RandomForestRegressor, BaggingRegressor,
                                  GradientBoostingRegressor)
    x, y = ens_data(seed=seed)
    X = x.reshape(-1, 1)
    xg = np.linspace(0, 1, 400).reshape(-1, 1)
    single = DecisionTreeRegressor(random_state=0).fit(X, y)
    if kind == "bagging":
        ens = BaggingRegressor(DecisionTreeRegressor(), n_estimators=n_estimators,
                               random_state=0).fit(X, y)
        name, col = "Bagged trees", ACCENT
    elif kind == "forest":
        ens = RandomForestRegressor(n_estimators=n_estimators, random_state=0).fit(X, y)
        name, col = "Random forest", _GREEN
    else:
        ens = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=2,
                                        learning_rate=0.1,
                                        random_state=0).fit(X, y)
        name, col = "Gradient boosting", FITLINE
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(xg, _ens_truth(xg.ravel()), color=INK, lw=2.0, ls=(0, (5, 4)),
            zorder=3, label="truth")
    ax.plot(xg, single.predict(xg), color=MADRID, lw=1.7, alpha=0.85, zorder=4,
            label="one deep tree")
    ax.plot(xg, ens.predict(xg), color=col, lw=3.0, zorder=5, label=name)
    ax.scatter(x, y, s=30, color="#9E9E9E", alpha=0.7, zorder=2)
    ax.set_xlabel("x", fontsize=12)
    ax.text(0, 1.02, "y", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("%s vs a single tree" % name, fontsize=14, color=col,
                 weight="bold", pad=26)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_boosting_rounds(rounds=(1, 5, 25, 150), seed=3):
    """Boosting grows trees sequentially, each fitting what is left over."""
    from sklearn.ensemble import GradientBoostingRegressor
    x, y = ens_data(seed=seed)
    X = x.reshape(-1, 1)
    xg = np.linspace(0, 1, 400).reshape(-1, 1)
    fig, axes = plt.subplots(1, len(rounds), figsize=(3.4 * len(rounds), 3.5),
                             dpi=110, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, m in zip(axes, rounds):
        g = GradientBoostingRegressor(n_estimators=m, max_depth=2,
                                      learning_rate=0.1, random_state=0).fit(X, y)
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.scatter(x, y, s=24, color="#9E9E9E", alpha=0.7, zorder=2)
        ax.plot(xg, _ens_truth(xg.ravel()), color=INK, lw=1.6, ls=(0, (5, 4)),
                zorder=3)
        ax.plot(xg, g.predict(xg), color=FITLINE, lw=2.6, zorder=4)
        ax.set_xlabel("x", fontsize=11)
        ax.set_title("%d tree%s" % (m, "" if m == 1 else "s"), fontsize=12,
                     color=FITLINE, weight="bold", pad=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(0, 1.06, "y", transform=axes[0].transAxes, ha="left",
                 va="bottom", fontsize=11, color=INK)
    fig.suptitle("Each new tree fits what the previous ones missed",
                 fontsize=13.5, color=MADRID, weight="bold", y=1.04)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_variable_importance(seed=0):
    """Forests report which covariates actually drove the splits."""
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(seed)
    n = 400
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    x3 = rng.normal(size=n); x4 = rng.normal(size=n)
    y = 3.0 * x1 + 1.2 * np.sign(x2) * x2 ** 2 + 0.0 * x3 + 0.0 * x4 \
        + rng.normal(0, 1.0, n)
    X = np.column_stack([x1, x2, x3, x4])
    rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)
    imp = rf.feature_importances_
    names = ["x1 (strong)", "x2 (nonlinear)", "x3 (noise)", "x4 (noise)"]
    cols = [MADRID, ACCENT, "#C9C9D2", "#C9C9D2"]
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.barh([names[i] for i in order], imp[order],
            color=[cols[i] for i in order], edgecolor=INK, lw=1.1, height=0.6,
            zorder=3)
    for k, i in enumerate(order):
        ax.text(imp[i] + 0.008, k, "%.2f" % imp[i], va="center", fontsize=11,
                color=cols[i] if imp[i] > 0.05 else INK, weight="bold")
    ax.set_xlim(0, imp.max() * 1.25)
    ax.set_xlabel("variable importance", fontsize=11.5)
    ax.set_title("The forest finds the variables that matter", fontsize=13.5,
                 color=MADRID, weight="bold", pad=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ==================================== cross-validation and regularization =====
def fig_kfold_diagram(k=5):
    """The k-fold picture: each fold is held out once."""
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white"); ax.axis("off")
    ax.set_xlim(-1.6, k + 0.2); ax.set_ylim(-0.4, k + 0.5)
    for r in range(k):
        y = k - 1 - r
        for c in range(k):
            held = (c == r)
            ax.add_patch(plt.Rectangle((c, y), 0.94, 0.8,
                                       facecolor=FITLINE if held else "#DCE3F2",
                                       edgecolor=INK, lw=1.1))
            if held:
                ax.text(c + 0.47, y + 0.4, "test", ha="center", va="center",
                        fontsize=9.5, color="white", weight="bold")
        ax.text(-0.2, y + 0.4, "round %d" % (r + 1), ha="right", va="center",
                fontsize=10.5, color=INK)
    ax.text(k / 2.0, k + 0.15, "the data, split into %d folds" % k,
            ha="center", fontsize=11, color=INK)
    ax.set_title("Every observation is tested on exactly once",
                 fontsize=14, color=MADRID, weight="bold", pad=6)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_cv_selects_flexibility(seed=2):
    """Cross-validation on the training data reproduces the U shape, so it can
    pick the flexibility without ever touching a test set."""
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, 60)
    y = chh._bv_truth(x) + rng.normal(0, 0.35, 60)
    X = x.reshape(-1, 1)
    degs = list(range(1, 10))
    cv, tr = [], []
    for d in degs:
        m = make_pipeline(PolynomialFeatures(d), LinearRegression())
        cv.append(float(-cross_val_score(m, X, y, cv=5,
                                         scoring="neg_mean_squared_error").mean()))
        m.fit(X, y)
        tr.append(float(np.mean((y - m.predict(X)) ** 2)))
    best = degs[int(np.argmin(cv))]
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(degs, tr, "o-", color=ACCENT, lw=2.3, ms=6, label="training error")
    ax.plot(degs, cv, "o-", color=FITLINE, lw=2.6, ms=6, label="5-fold CV error")
    ax.axvline(best, color=MADRID, ls=(0, (4, 3)), lw=1.5)
    ax.text(best + 0.12, max(cv) * 0.85, "CV picks degree %d" % best,
            fontsize=10.5, color=MADRID)
    ax.set_yscale("log")
    ax.set_xlabel("Model flexibility (polynomial degree)", fontsize=12)
    ax.text(0, 1.02, "Mean squared error (log scale)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, color=INK)
    ax.set_title("Cross-validation finds the U without a test set",
                 fontsize=14, color=MADRID, weight="bold", pad=26)
    ax.legend(fontsize=10, framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_crossfitting():
    """DML's cross-fitting: the nuisance model that residualizes a fold is
    never fit on that fold."""
    fig, ax = plt.subplots(figsize=(9.0, 3.6), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white"); ax.axis("off")
    ax.set_xlim(-2.0, 5.4); ax.set_ylim(-0.4, 3.6)
    for r in range(3):
        y = 2 - r
        for c in range(3):
            held = (c == r)
            ax.add_patch(plt.Rectangle((c * 1.6, y), 1.5, 0.8,
                                       facecolor=_GREEN if held else "#DCE3F2",
                                       edgecolor=INK, lw=1.1))
            ax.text(c * 1.6 + 0.75, y + 0.4,
                    "residualize" if held else "fit nuisance",
                    ha="center", va="center", fontsize=9,
                    color="white" if held else INK,
                    weight="bold" if held else "normal")
        ax.text(-0.15, y + 0.4, "fold %d" % (r + 1), ha="right", va="center",
                fontsize=10.5, color=INK)
    ax.set_title("Cross-fitting: predict each fold with a model that never saw it",
                 fontsize=13.5, color=MADRID, weight="bold", pad=6)
    fig.tight_layout(); plt.close(fig)
    return fig


def _reg_data(n=80, p=12, seed=1):
    """A few real signals hidden among correlated noise columns."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 1] = X[:, 0] * 0.85 + rng.normal(0, 0.4, n)     # correlated pair
    beta = np.zeros(p); beta[[0, 2, 5]] = [2.5, -1.8, 1.2]
    y = X @ beta + rng.normal(0, 1.0, n)
    return X, y, beta


def fig_coefficient_paths(kind="ridge"):
    """How coefficients shrink as the penalty grows. Lasso sends them to
    exactly zero; ridge only shrinks them toward it."""
    from sklearn.linear_model import Ridge, Lasso
    X, y, beta = _reg_data()
    lams = np.logspace(-2, 2.2, 60)
    paths = []
    for lam in lams:
        m = (Ridge(alpha=lam) if kind == "ridge" else Lasso(alpha=lam, max_iter=20000))
        m.fit(X, y)
        paths.append(m.coef_)
    paths = np.array(paths)
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for j in range(paths.shape[1]):
        real = beta[j] != 0
        ax.plot(lams, paths[:, j], lw=2.4 if real else 1.2,
                color=MADRID if real else "#C9C9D2",
                zorder=4 if real else 2,
                label=("true signal" if (real and j == 0) else
                       ("noise" if (not real and j == 1) else None)))
    ax.axhline(0, color=INK, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"penalty strength $\lambda$ (log scale)", fontsize=12)
    ax.text(0, 1.02, "coefficient", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("%s: coefficient paths" % kind.capitalize(), fontsize=14,
                 color=MADRID, weight="bold", pad=26)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_lasso_selects():
    """How many coefficients survive as lambda grows, and what CV picks."""
    from sklearn.linear_model import LassoCV, Lasso
    X, y, beta = _reg_data()
    lams = np.logspace(-2, 1.2, 50)
    nz = [int(np.sum(Lasso(alpha=l, max_iter=20000).fit(X, y).coef_ != 0))
          for l in lams]
    cv = LassoCV(alphas=lams, cv=5, max_iter=20000).fit(X, y)
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.step(lams, nz, where="post", color=MADRID, lw=2.6, zorder=3)
    ax.axhline(int(np.sum(beta != 0)), color=_GREEN, ls=(0, (5, 4)), lw=2,
               zorder=2)
    ax.text(lams[0], np.sum(beta != 0) + 0.45, "true number of signals",
            fontsize=10.5, color=_GREEN)
    ax.axvline(cv.alpha_, color=FITLINE, ls=(0, (4, 3)), lw=1.8)
    ax.text(cv.alpha_ * 1.1, max(nz) * 0.8, "CV picks\n$\\lambda$ = %.3f"
            % cv.alpha_, fontsize=10.5, color=FITLINE)
    ax.set_xscale("log")
    ax.set_xlabel(r"penalty strength $\lambda$ (log scale)", fontsize=12)
    ax.text(0, 1.02, "coefficients kept (not zero)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, color=INK)
    ax.set_title("Lasso selects variables", fontsize=14, color=MADRID,
                 weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ==================================== k-nearest neighbors ====================
# KNN earns its place in a causal course through two bridges: nearest-neighbour
# MATCHING is KNN applied to treatment effects, and a causal forest is a KNN
# that learns its own neighbourhood (see course_helpers.fig_forest_weights).
_KNN_ORANGE = "#F2C79B"      # light fills for the two decision regions
_KNN_BLUE = "#AFC1E8"
_GREEN = "#2E8B57"


def _knn_data(n=200, noise=0.30, seed=1):
    from sklearn.datasets import make_moons
    return make_moons(n_samples=n, noise=noise, random_state=seed)


def fig_knn_boundary(k=1, seed=1):
    """The decision boundary at a given k. Small k is a jagged, high-variance
    boundary; large k is smooth and can wash out real structure."""
    from sklearn.neighbors import KNeighborsClassifier
    from matplotlib.colors import ListedColormap
    X, y = _knn_data(seed=seed)
    clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    x0, x1 = X[:, 0], X[:, 1]
    xx, yy = np.meshgrid(
        np.linspace(x0.min() - 0.4, x0.max() + 0.4, 300),
        np.linspace(x1.min() - 0.4, x1.max() + 0.4, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                colors=[_KNN_ORANGE, _KNN_BLUE], alpha=0.55, zorder=1)
    ax.contour(xx, yy, Z, levels=[0.5], colors=[INK], linewidths=1.6, zorder=2)
    for cls, col, mk in ((0, ACCENT, "o"), (1, MADRID, "^")):
        m = y == cls
        ax.scatter(x0[m], x1[m], s=26, color=col, marker=mk, edgecolors="white",
                   linewidths=0.5, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("k = %d" % k, fontsize=15, color=MADRID, weight="bold", pad=8)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_knn_bias_variance(seed=1):
    """Error against k. Reading left to right the model gets LESS flexible, so
    this is the bias-variance U with the flexibility knob spelled k."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    X, y = _knn_data(n=300, seed=seed)
    ks = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100]
    tr, cv = [], []
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)
        tr.append(1 - clf.score(X, y))
        cv.append(1 - cross_val_score(clf, X, y, cv=5).mean())
    best = ks[int(np.argmin(cv))]
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(ks, tr, "o-", color=ACCENT, lw=2.3, ms=6, label="training error")
    ax.plot(ks, cv, "o-", color=FITLINE, lw=2.6, ms=6, label="5-fold CV error")
    ax.axvline(best, color=MADRID, ls=(0, (4, 3)), lw=1.5)
    ax.text(best * 1.1, max(cv) * 0.55, "CV picks k = %d" % best, fontsize=10.5,
            color=MADRID)
    ax.set_xscale("log")
    ax.set_xlabel("k  (fewer neighbours = more flexible, to the left)", fontsize=12)
    ax.text(0, 1.02, "misclassification rate", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("k is the bias-variance knob", fontsize=14, color=MADRID,
                 weight="bold", pad=26)
    ax.legend(fontsize=10, framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_curse_of_dimensionality():
    """To trap a fixed fraction of the data you must cover almost the whole
    range of every axis once there are many axes, so nothing is ever local."""
    d = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for frac, col, lab in ((0.10, MADRID, "10% of the data"),
                           (0.01, ACCENT, "1% of the data")):
        ax.plot(d, frac ** (1.0 / d), "o-", color=col, lw=2.5, ms=5, label=lab)
    ax.axhline(1.0, color=INK, lw=1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("number of covariates (dimensions)", fontsize=12)
    ax.text(0, 1.02, "edge length of the neighbourhood", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, color=INK)
    ax.set_title("The curse of dimensionality", fontsize=14, color=MADRID,
                 weight="bold", pad=26)
    ax.legend(fontsize=10, framealpha=0.9, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_nn_matching(seed=4):
    """Nearest-neighbour matching: each treated unit is paired with the closest
    control. This is 1-NN doing causal inference."""
    rng = np.random.default_rng(seed)
    treat = rng.normal([0.9, 0.9], 0.55, size=(9, 2))
    ctrl = rng.normal([0.0, 0.0], 0.75, size=(28, 2))
    _TREAT, _CTRL = chh._TREAT, chh._CTRL
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for t in treat:
        j = int(np.argmin(((ctrl - t) ** 2).sum(1)))
        ax.plot([t[0], ctrl[j, 0]], [t[1], ctrl[j, 1]], color="#9E9E9E",
                lw=1.2, zorder=2)
    ax.scatter(ctrl[:, 0], ctrl[:, 1], s=42, color=_CTRL, marker="o",
               edgecolors="white", lw=0.6, zorder=3, label="control")
    ax.scatter(treat[:, 0], treat[:, 1], s=52, color=_TREAT, marker="^",
               edgecolors="white", lw=0.6, zorder=4, label="treated")
    ax.set_xlabel("covariate 1", fontsize=12)
    ax.text(0, 1.02, "covariate 2", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("Nearest-neighbour matching is 1-NN", fontsize=14, color=MADRID,
                 weight="bold", pad=26)
    ax.legend(fontsize=10, framealpha=0.92, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ==================================== logit / probit =========================
# Follows Robert's "OLS The Book" chapter on qualitative response: the LPM and
# its causal defense, the latent-variable link, logit drawn as a one-neuron
# computational graph, and multinomial logit as the softmax that becomes a
# neural network. Colours match the book: inputs yellow-orange, index/output
# green.
_IN_FILL = "#F4C542"      # inputs (his yellow!80!orange)
_ND_FILL = "#3F7D4E"      # linear index and output (his green)
_GREEN = "#2E8B57"


def _node(ax, xy, r, text, fill, tcol="white", fs=13):
    ax.add_patch(plt.Circle(xy, r, facecolor=fill, edgecolor=INK, lw=1.5,
                            zorder=4))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs,
            color=tcol, zorder=5)


def _arrow(ax, a, b, label=None, lcol=INK, shrink=20):
    ax.annotate("", xy=b, xytext=a, zorder=2,
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#6B6B6B",
                                shrinkA=shrink, shrinkB=shrink))
    if label:
        ax.text(0.55 * a[0] + 0.45 * b[0], 0.55 * a[1] + 0.45 * b[1] + 0.12,
                label, ha="center", va="bottom", fontsize=10.5, color=lcol,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"),
                zorder=6)


def fig_link_squash():
    """A linear index can be any real number. The LPM leaves it alone and the
    prediction escapes [0,1]; a link function squashes it back in."""
    from scipy.stats import norm
    z = np.linspace(-6, 6, 400)
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axhspan(1.0, 1.35, color=FITLINE, alpha=0.07, zorder=0)
    ax.axhspan(-0.35, 0.0, color=FITLINE, alpha=0.07, zorder=0)
    ax.grid(True, color=GRID, lw=0.8, zorder=1); ax.set_axisbelow(True)
    ax.plot(z, 0.5 + 0.11 * z, color=ACCENT, lw=2.6, zorder=4,
            label="LPM (no link): escapes $[0,1]$")
    ax.plot(z, 1 / (1 + np.exp(-z)), color=MADRID, lw=2.8, zorder=5,
            label=r"logit link $\Lambda(z)$")
    ax.plot(z, norm.cdf(z), color=_GREEN, lw=2.4, ls=(0, (5, 3)), zorder=5,
            label=r"probit link $\Phi(z)$")
    ax.axhline(0, color=INK, lw=1); ax.axhline(1, color=INK, lw=1)
    ax.set_ylim(-0.35, 1.35)
    ax.set_xlabel(r"linear index  $z = \mathbf{x}'\boldsymbol{\beta}$", fontsize=12)
    ax.text(0, 1.02, "predicted probability", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("A link function squashes the index into $[0,1]$", fontsize=14,
                 color=MADRID, weight="bold", pad=26)
    ax.legend(fontsize=9.5, framealpha=0.92, loc="center right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_logit_neuron():
    """Binary logit as a one-neuron computational graph: covariates feed a
    linear index, the sigmoid squashes it to a probability."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white"); ax.axis("off")
    ax.set_xlim(-0.8, 9.0); ax.set_ylim(-3.1, 3.1)
    ys = [2.4, 1.2, 0.0, -1.2, -2.4]
    labs = ["$1$", "$X_1$", "$X_2$", "$X_3$", "$X_k$"]
    bs = [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$", r"$\beta_3$", r"$\beta_k$"]
    for y, lab in zip(ys, labs):
        _node(ax, (0.0, y), 0.40, lab, _IN_FILL, tcol=INK, fs=12)
    ax.text(0.0, -1.75, r"$\vdots$", ha="center", va="center", fontsize=16,
            color=INK)
    _node(ax, (4.4, 0.0), 0.55, "$z$", _ND_FILL, fs=15)
    _node(ax, (7.9, 0.0), 0.58, r"$\hat{P}$", _ND_FILL, fs=15)
    for y, b in zip(ys, bs):
        _arrow(ax, (0.0, y), (4.4, 0.0), label=b, lcol=INK)
    _arrow(ax, (4.4, 0.0), (7.9, 0.0), label=r"$\Lambda(\cdot)$", lcol=MADRID)
    ax.text(4.4, -0.95, r"$z=\mathbf{x}'\boldsymbol{\beta}$", ha="center",
            fontsize=12, color=INK)
    ax.text(7.9, -0.95, r"$\hat{P}=\Lambda(z)$", ha="center", fontsize=12,
            color=INK)
    ax.set_title("Logit is a one-neuron network", fontsize=14, color=MADRID,
                 weight="bold")
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_softmax_graph():
    """Multinomial logit fans the covariates into K linear indices and runs the
    vector through a softmax. This is the picture that becomes a network."""
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white"); ax.axis("off")
    ax.set_xlim(-0.8, 9.6); ax.set_ylim(-3.1, 3.3)
    ys = [2.4, 1.2, 0.0, -1.2, -2.4]
    labs = ["$1$", "$X_1$", "$X_2$", "$X_3$", "$X_k$"]
    for y, lab in zip(ys, labs):
        _node(ax, (0.0, y), 0.38, lab, _IN_FILL, tcol=INK, fs=11)
    ax.text(0.0, -1.8, r"$\vdots$", ha="center", va="center", fontsize=15,
            color=INK)
    zy = [1.7, 0.0, -1.7]
    for j, y in enumerate(zy, 1):
        _node(ax, (3.9, y), 0.48, "$z_%d$" % j, _ND_FILL, fs=13)
        for iy in ys:
            ax.annotate("", xy=(3.9 - 0.48, y), xytext=(0.38, iy),
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="#B9B9C2"),
                        zorder=1)
    ax.add_patch(plt.Rectangle((5.7, -2.15), 0.9, 4.3, facecolor="#DCE3F2",
                               edgecolor=INK, lw=1.3, zorder=3))
    ax.text(6.15, 0.0, "softmax", ha="center", va="center", rotation=90,
            fontsize=12.5, color=INK, weight="bold", zorder=4)
    py = [1.7, 0.0, -1.7]
    for j, y in enumerate(py, 1):
        _node(ax, (8.4, y), 0.5, r"$\hat{P}_%d$" % j, _ND_FILL, fs=13)
        ax.annotate("", xy=(8.4 - 0.5, y), xytext=(6.6, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#6B6B6B"),
                    zorder=2)
        ax.annotate("", xy=(5.7, y), xytext=(3.9 + 0.48, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#6B6B6B"),
                    zorder=2)
    ax.text(3.9, -2.55, r"$z_j=\mathbf{x}'\boldsymbol{\beta}_j$", ha="center",
            fontsize=11.5, color=INK)
    ax.text(8.4, -2.55, r"$\hat{P}_j=\dfrac{e^{z_j}}{\sum_m e^{z_m}}$",
            ha="center", fontsize=11.5, color=INK)
    ax.set_title("Multinomial logit is softmax", fontsize=14, color=MADRID,
                 weight="bold")
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_lpm_logit_probit():
    """LPM, logit, and probit fit to the same binary outcome. They agree through
    the middle; only the LPM leaves the unit interval at the edges."""
    import pandas as pd
    import statsmodels.api as sm
    from scipy.stats import norm
    d = pd.read_csv(chh._DATA / "bliss_beetle_mortality.csv")
    xs, ys = [], []                         # expand grouped data to one row per beetle
    for _, r in d.iterrows():
        n, k = int(r["n_tested"]), int(r["n_killed"])
        xs += [r["log_dose"]] * n
        ys += [1] * k + [0] * (n - k)
    x = np.array(xs, float); y = np.array(ys, float)
    X = sm.add_constant(x)
    lpm = sm.OLS(y, X).fit()
    logit = sm.Logit(y, X).fit(disp=False)
    probit = sm.Probit(y, X).fit(disp=False)
    g = np.linspace(x.min() - 0.035, x.max() + 0.035, 200)
    G = sm.add_constant(g)
    props = (d["n_killed"] / d["n_tested"]).to_numpy()
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.scatter(d["log_dose"], props, s=52, color="#6B6B6B", zorder=5,
               label="observed mortality")
    ax.plot(g, lpm.predict(G), color=ACCENT, lw=2.6, zorder=4, label="LPM (OLS)")
    ax.plot(g, logit.predict(G), color=MADRID, lw=2.8, zorder=4, label="logit")
    ax.plot(g, norm.cdf(G @ probit.params), color=_GREEN, lw=2.2,
            ls=(0, (5, 3)), zorder=4, label="probit")
    ax.axhline(0, color=INK, lw=1); ax.axhline(1, color=INK, lw=1)
    ax.set_xlabel(r"log dose of CS$_2$", fontsize=12)
    ax.text(0, 1.02, "P(beetle dies)", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("Three models on Bliss's beetles", fontsize=14, color=MADRID,
                 weight="bold", pad=26)
    ax.legend(fontsize=9.5, framealpha=0.92, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============================================================== model assessment
_ASSESS_CACHE = {}


def _bliss_fit():
    """One row per beetle, plus fitted logit death probabilities."""
    import pandas as pd
    import statsmodels.api as sm
    if "bliss" not in _ASSESS_CACHE:
        d = pd.read_csv(chh._DATA / "bliss_beetle_mortality.csv")
        xs, ys = [], []
        for _, r in d.iterrows():
            n, k = int(r["n_tested"]), int(r["n_killed"])
            xs += [r["log_dose"]] * n
            ys += [1] * k + [0] * (n - k)
        x = np.array(xs, float)
        y = np.array(ys, int)
        p = sm.Logit(y, sm.add_constant(x)).fit(disp=False).predict()
        _ASSESS_CACHE["bliss"] = (x, y, p)
    return _ASSESS_CACHE["bliss"]


def confusion_counts(threshold=0.5):
    """(tp, fn, fp, tn) for the beetle logit at the given threshold."""
    _, y, p = _bliss_fit()
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fn, fp, tn


def fig_confusion_matrix(threshold=0.5):
    """The four ways a prediction can land, as a 2 by 2 grid with the beetle
    counts filled in. Rows are the truth, columns the model's call."""
    tp, fn, fp, tn = confusion_counts(threshold)
    cells = [
        (0, 1, tp, "True positive", MADRID, "white"),
        (1, 1, fn, "False negative\n(type II error)", "#E6E6EA", INK),
        (0, 0, fp, "False positive\n(type I error)", "#E6E6EA", INK),
        (1, 0, tn, "True negative", MADRID, "white"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for cx, cy, n, lab, face, txt in cells:
        ax.add_patch(plt.Rectangle((cx, cy), 1, 1, facecolor=face,
                                   edgecolor="white", lw=3))
        ax.text(cx + 0.5, cy + 0.60, "%d" % n, ha="center", va="center",
                fontsize=23, weight="bold", color=txt)
        ax.text(cx + 0.5, cy + 0.28, lab, ha="center", va="center",
                fontsize=10.5, color=txt)
    ax.text(0.5, 2.06, "predicted: dies", ha="center", fontsize=11.5,
            color=INK)
    ax.text(1.5, 2.06, "predicted: survives", ha="center", fontsize=11.5,
            color=INK)
    ax.text(-0.06, 1.5, "actually\ndies", ha="right", va="center",
            fontsize=11.5, color=INK)
    ax.text(-0.06, 0.5, "actually\nsurvives", ha="right", va="center",
            fontsize=11.5, color=INK)
    ax.set_xlim(-0.75, 2.05); ax.set_ylim(-0.08, 2.42)
    ax.axis("off")
    ax.set_title("The confusion matrix, threshold %.1f" % threshold,
                 fontsize=14, color=MADRID, weight="bold", pad=4)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_threshold_tradeoff():
    """Sensitivity and specificity as the threshold sweeps from 0 to 1. One
    rises where the other falls; the threshold chooses which mistake to make."""
    _, y, p = _bliss_fit()
    ts = np.linspace(0.01, 0.99, 99)
    sens = [(p[y == 1] >= t).mean() for t in ts]
    spec = [(p[y == 0] < t).mean() for t in ts]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(ts, sens, color=MADRID, lw=2.8, zorder=4,
            label="sensitivity  TP/(TP+FN)")
    ax.plot(ts, spec, color=ACCENT, lw=2.8, zorder=4,
            label="specificity  TN/(TN+FP)")
    ax.axvline(0.5, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=3)
    ax.text(0.505, 0.06, "default 0.5", fontsize=9.5, color=INK)
    ax.set_xlabel("classification threshold on $\\hat{p}$", fontsize=12)
    ax.text(0, 1.02, "rate", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("Move the threshold and the mistakes trade places",
                 fontsize=13.5, color=MADRID, weight="bold", pad=24)
    ax.legend(fontsize=10, framealpha=0.92, loc="center right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_roc_curve():
    """The ROC curve for the beetle logit: every threshold at once, true
    positive rate against false positive rate, AUC shaded underneath."""
    _, y, p = _bliss_fit()
    ts = np.unique(np.concatenate([[0], np.sort(p), [1.0001]]))[::-1]
    tpr = np.array([(p[y == 1] >= t).mean() for t in ts])
    fpr = np.array([(p[y == 0] >= t).mean() for t in ts])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
        else float(np.trapz(tpr, fpr))
    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.fill_between(fpr, 0, tpr, color=MADRID, alpha=0.12, zorder=2)
    ax.plot(fpr, tpr, color=MADRID, lw=2.8, zorder=4, label="logit on dose")
    ax.plot([0, 1], [0, 1], color=INK, lw=1.4, ls=(0, (4, 3)), zorder=3,
            label="coin flip")
    for t, mcol in ((0.3, ACCENT), (0.5, FITLINE), (0.7, _GREEN)):
        fx = (p[y == 0] >= t).mean()
        fy = (p[y == 1] >= t).mean()
        ax.scatter([fx], [fy], s=64, color=mcol, zorder=5,
                   edgecolors="white", lw=1.2)
        ax.annotate("threshold %.1f" % t, (fx, fy), xytext=(fx + 0.05, fy - 0.06),
                    fontsize=9.5, color=mcol)
    ax.text(0.62, 0.25, "AUC = %.2f" % auc, fontsize=13, color=MADRID,
            weight="bold")
    ax.set_xlabel("false positive rate  1 - specificity", fontsize=11.5)
    ax.text(0, 1.02, "true positive rate (sensitivity)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11.5, color=INK)
    ax.set_title("The ROC curve: every threshold at once", fontsize=13.5,
                 color=MADRID, weight="bold", pad=24)
    ax.legend(fontsize=10, framealpha=0.92, loc="lower right")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig



# ============================================================== regularization details
_TINY_TRAIN = (np.array([1.0, 2.0]), np.array([1.35, 3.05]))


def _tiny_test(seed=9, n=9):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.2, 3.8, n)
    y = 1.0 + 0.6 * x + rng.normal(0, 0.18, n)
    return x, y


def _ridge_line(lam):
    """Intercept unpenalized, slope shrunk: the 1D ridge closed form."""
    xt, yt = _TINY_TRAIN
    xm, ym = xt.mean(), yt.mean()
    b = ((xt - xm) @ (yt - ym)) / (((xt - xm) ** 2).sum() + lam)
    return ym - b * xm, b


def fig_reg_overfit():
    """Two training points fit perfectly by OLS, badly by the world. The ridge
    line gives up a little training error and buys back most of the test
    error. Overfitting and its cure in the smallest possible sample."""
    xt, yt = _TINY_TRAIN
    xg, yg = _tiny_test()
    g = np.linspace(0, 4, 50)
    a0, b0 = _ridge_line(0.0)
    a1, b1 = _ridge_line(1.0)
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.scatter(xg, yg, s=34, color=ACCENT, alpha=0.85, zorder=3,
               label="test data (unseen)")
    ax.scatter(xt, yt, s=110, color=MADRID, zorder=5, edgecolors="white",
               lw=1.4, label="training data (all two of it)")
    ax.plot(g, a0 + b0 * g, color=FITLINE, lw=2.6, zorder=4,
            label="OLS: training error zero")
    ax.plot(g, a1 + b1 * g, color=MADRID, lw=2.6, ls=(0, (6, 3)), zorder=4,
            label="ridge $\\lambda=1$: a little bias")
    for line, (a, b), yy in (("OLS", (a0, b0), 0.14), ("ridge", (a1, b1), 0.02)):
        mse = np.mean((yg - (a + b * xg)) ** 2)
        ax.text(0.99, yy, "%s test MSE %.2f" % (line, mse),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10.5, color=FITLINE if line == "OLS" else MADRID)
    ax.set_xlabel("x", fontsize=12)
    ax.text(0, 1.02, "y", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("A perfect fit to two points is a bad fit to the world",
                 fontsize=13.5, color=MADRID, weight="bold", pad=24)
    ax.legend(fontsize=9.5, framealpha=0.92, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_lambda_progression():
    """The same two point fit under a rising ridge penalty. The slope walks
    down smoothly toward zero; at huge lambda the line is the training mean."""
    xt, yt = _TINY_TRAIN
    xg, yg = _tiny_test()
    g = np.linspace(0, 4, 50)
    lams = [0, 1, 5, 25]
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(lams)))
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.scatter(xg, yg, s=30, color=ACCENT, alpha=0.7, zorder=3)
    ax.scatter(xt, yt, s=110, color=MADRID, zorder=5, edgecolors="white",
               lw=1.4)
    for lam, col in zip(lams, cols):
        a, b = _ridge_line(lam)
        ax.plot(g, a + b * g, color=col, lw=2.6, zorder=4,
                label="$\\lambda$ = %g,  slope %.2f" % (lam, b))
    ax.set_xlabel("x", fontsize=12)
    ax.text(0, 1.02, "y", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("Raising $\\lambda$ walks the slope toward zero, smoothly",
                 fontsize=13.5, color=MADRID, weight="bold", pad=24)
    ax.legend(fontsize=10, framealpha=0.92, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_penalty_geometry():
    """Why ridge shrinks and lasso selects, in one picture. Total cost against
    the slope. The squared penalty keeps a smooth bowl whose minimum slides
    toward zero but never lands on it. The absolute penalty folds a kink at
    zero, and once lambda is big enough the minimum sits exactly there."""
    b = np.linspace(-0.6, 1.6, 400)
    ssr = 10 * (b - 1) ** 2 + 2
    lams = [0, 10, 20, 40]
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(lams)))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), dpi=110, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, pen, ttl in ((axes[0], lambda l: l * b ** 2,
                          "Ridge: SSR + $\\lambda\\,$slope$^2$"),
                         (axes[1], lambda l: l * np.abs(b),
                          "Lasso: SSR + $\\lambda\\,$|slope|")):
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.axvline(0, color=INK, lw=1.0, zorder=2)
        for lam, col in zip(lams, cols):
            cost = ssr + pen(lam)
            ax.plot(b, cost, color=col, lw=2.4, zorder=3,
                    label="$\\lambda$ = %g" % lam)
            bmin = b[np.argmin(cost)]
            ax.scatter([bmin], [cost.min()], s=52, color=col, zorder=5,
                       edgecolors="white", lw=1.0)
        ax.set_xlabel("slope $b$", fontsize=11.5)
        ax.set_title(ttl, fontsize=12.5, color=MADRID, weight="bold", pad=22)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(0, 1.03, "total cost", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=11.5, color=INK)
    axes[0].legend(fontsize=9.5, framealpha=0.92, loc="upper right")
    axes[0].set_ylim(0, 40)
    fig.tight_layout(); plt.close(fig)
    return fig


# ============================================================== cross validation details
def fig_train_test_split():
    """The simplest honesty device: fit on one part, judge on the other."""
    n = 20
    n_tr = 15
    fig, ax = plt.subplots(figsize=(9.6, 2.6), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for i in range(n):
        col = MADRID if i < n_tr else ACCENT
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, facecolor=col,
                                   edgecolor="white", lw=1.5, alpha=0.9))
    ax.text(n_tr / 2, 1.28, "training set: fit the model here",
            ha="center", fontsize=12, color=MADRID, weight="bold")
    ax.text(n_tr + (n - n_tr) / 2, 1.28, "test set:\njudge it here",
            ha="center", fontsize=12, color=ACCENT, weight="bold")
    ax.text(n / 2, -0.42, "one row of data per block; the split is random,"
            " and the test rows stay untouched until the end",
            ha="center", fontsize=10.5, color=INK)
    ax.set_xlim(-0.3, n + 0.3); ax.set_ylim(-0.75, 2.05)
    ax.axis("off")
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_cv_fold_errors(seed=3):
    """What cross-validation actually averages: one error number per fold,
    each computed on rows the model never saw, then the mean."""
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(seed)
    n = 100
    x = rng.uniform(0, 4, n)
    y = 1 + 0.8 * x + 0.35 * np.sin(3 * x) + rng.normal(0, 0.4, n)
    X = np.column_stack([x ** k for k in range(1, 4)])
    errs = []
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=1).split(X):
        m = LinearRegression().fit(X[tr], y[tr])
        errs.append(np.mean((y[te] - m.predict(X[te])) ** 2))
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.bar(range(1, 6), errs, color=MADRID, alpha=0.85, zorder=3, width=0.62)
    ax.axhline(np.mean(errs), color=FITLINE, lw=2.4, ls=(0, (6, 3)), zorder=4)
    ax.text(5.42, np.mean(errs), "mean = the CV score", fontsize=10.5,
            color=FITLINE, va="bottom", ha="right")
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["fold %d" % k for k in range(1, 6)], fontsize=11)
    ax.text(0, 1.02, "MSE on the held-out fold", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11.5, color=INK)
    ax.set_title("Five folds, five honest errors, one average", fontsize=13,
                 color=MADRID, weight="bold", pad=24)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig


def fig_cv_variants():
    """Fold counts as a dial: 5-fold, 10-fold, leave one out. More folds mean
    more training data per fit and more fits to pay for."""
    fig, ax = plt.subplots(figsize=(9.8, 4.4), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    rows = [("5-fold", 5, "5 fits, each on 80% of the data"),
            ("10-fold", 10, "10 fits, each on 90% of the data"),
            ("leave one out", 30, "n fits, each on all but one row")]
    for r, (name, k, note) in enumerate(rows):
        y0 = 2 - r
        w = 8.6 / k
        for i in range(k):
            col = ACCENT if i == min(1, k - 1) else MADRID
            ax.add_patch(plt.Rectangle((1.6 + i * w, y0 + 0.08), w * 0.9,
                                       0.5, facecolor=col,
                                       edgecolor="white", lw=1.0, alpha=0.9))
        ax.text(1.45, y0 + 0.33, name, ha="right", va="center", fontsize=12,
                color=INK, weight="bold")
        ax.text(10.4, y0 + 0.33, note, ha="left", va="center", fontsize=10.5,
                color=INK)
    ax.text(1.6, 2.86, "orange = one held-out block per round; every block"
            " takes its turn", fontsize=10.5, color=INK)
    ax.set_xlim(-1.2, 14.6); ax.set_ylim(-0.25, 3.2)
    ax.axis("off")
    fig.tight_layout(); plt.close(fig)
    return fig



# ============================================================== gradient descent details
# ============================================================== decision tree details
def fig_line_vs_tree():
    """One straight line against four flat pieces, on the same nineteen
    patients. Nothing else, because nothing else is needed to ask the question."""
    x, y = _DT_DOSE, _DT_EFF
    g = np.linspace(0, 40.99, 600)
    b1, b0 = np.polyfit(x, y, 1)
    line = b0 + b1 * g
    line_ssr = ((y - (b0 + b1 * x)) ** 2).sum()

    edges = [0, 14.5, 23.5, 29, 41]
    step = np.empty_like(g)
    tree_ssr = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        step[(g >= lo) & (g < hi)] = y[m].mean()
        tree_ssr += ((y[m] - y[m].mean()) ** 2).sum()

    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=110)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.plot(g, line, color=FITLINE, lw=2.6, zorder=3,
            label="a straight line, SSR {:,.0f}".format(line_ssr))
    ax.plot(g, step, color=MADRID, lw=3.0, zorder=4,
            label="four flat pieces, SSR %.0f" % tree_ssr)
    ax.scatter(x, y, s=80, color=ACCENT, edgecolors=INK, lw=1.2, zorder=6)
    ax.set_xlim(0, 41); ax.set_ylim(-8, 112)
    ax.set_xlabel("Drug dose (mg)", fontsize=12)
    ax.text(0, 1.02, "Effectiveness (%)", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.legend(fontsize=10.5, framealpha=0.95, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); plt.close(fig)
    return fig

def multi_feature_trial(seed=17, n=140):
    """Simulated drug trial with three features. Dose only helps patients
    under fifty, and only in the middle of its range."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 75, n).round(0)
    dose = rng.uniform(0, 40, n).round(1)
    male = rng.integers(0, 2, n)
    eff = (70 * ((age < 50) & (dose > 10) & (dose < 30))
           + 8 + rng.normal(0, 5, n))
    # Weight is drawn last so every number the deck already prints is unchanged.
    # It tracks age at about 0.73, which is what the missing value slides use.
    weight = (120 + 0.6 * age + 8 * male + rng.normal(0, 8, n)).round(0)
    return pd.DataFrame(dict(age=age, dose=dose, male=male, weight=weight,
                             effectiveness=eff.round(1)))



# ============================================================== CNN details
def _conv2d_valid(img, filt):
    """Plain sliding multiply-and-add, no padding."""
    H = img.shape[0] - filt.shape[0] + 1
    W = img.shape[1] - filt.shape[1] + 1
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = (img[i:i + 3, j:j + 3] * filt).sum()
    return out


def fig_cnn_shift():
    """Shift tolerance made visible. Top row: the drawing, its feature map,
    and the max-pooled grid. Bottom row: the same drawing pushed one pixel
    right. The feature maps differ cell by cell; the pooled grids barely
    notice, because each pooled cell keeps only the strongest response in
    its block."""
    img = np.array(_IMG_O, float)
    shifted = np.zeros_like(img)
    shifted[:, 1:] = img[:, :-1]
    filt = np.array(_FILTER, float)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2), dpi=110)
    fig.patch.set_facecolor("white")
    for r, (im, rowlab) in enumerate(((img, "original"),
                                      (shifted, "shifted one pixel right"))):
        fm = np.maximum(_conv2d_valid(im, filt), 0)          # convolve + ReLU
        pooled = fm.reshape(2, 2, 2, 2).max(axis=(1, 3))     # 2x2 max pool
        for c, (mat, ttl, cmap) in enumerate((
                (im, "drawing", "Greys"),
                (fm, "feature map (ReLU)", "Blues"),
                (pooled, "max pooled", "Blues"))):
            ax = axes[r, c]
            ax.imshow(mat, cmap=cmap, vmin=0,
                      vmax=max(1.0, mat.max()))
            for (yy, xx), v in np.ndenumerate(mat):
                if c > 0:
                    ax.text(xx, yy, "%.0f" % v, ha="center", va="center",
                            fontsize=10 if c == 1 else 13,
                            color=INK if v < 0.6 * max(1.0, mat.max())
                            else "white")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(ttl, fontsize=12, color=MADRID, weight="bold",
                             pad=8)
        axes[r, 0].set_ylabel(rowlab, fontsize=11.5, color=INK)
    fig.suptitle("The feature map moves with the drawing; the pooled summary"
                 " stays put", fontsize=12.5, color=MADRID, weight="bold",
                 y=0.99)
    fig.tight_layout(); plt.close(fig)
    return fig



# ============================================================== CNN animation
def cnn_animation(hold=5, fps=3):
    """The filter gliding over the drawing while the feature map fills in,
    one embedded video instead of sixteen manual subslides. Left: the drawing
    with the filter's current 3x3 window outlined. Right: the feature map,
    each cell appearing as the filter reaches it."""
    from matplotlib.patches import Rectangle
    img = np.array(_IMG_O, float)
    filt = np.array(_FILTER, float)
    H = img.shape[0] - 2
    W = img.shape[1] - 2
    fm = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            fm[i, j] = (img[i:i + 3, j:j + 3] * filt).sum()

    positions = [(i, j) for i in range(H) for j in range(W)]
    frames = []
    for k, pos in enumerate(positions):
        reps = hold if (k == 0 or k == len(positions) - 1) else 2
        frames += [(k, pos)] * reps

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.6), dpi=100)
    fig.patch.set_facecolor("white")
    axL.imshow(img, cmap="Greys", vmin=0, vmax=1)
    axL.set_title("the drawing, filter sliding", fontsize=12, color=MADRID,
                  weight="bold", pad=10)
    rect = Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor=ACCENT, lw=3.5)
    axL.add_patch(rect)
    shown = np.full((H, W), np.nan)
    im = axR.imshow(shown, cmap="Blues", vmin=0, vmax=max(1.0, fm.max()))
    axR.set_title("feature map, filling in", fontsize=12, color=MADRID,
                  weight="bold", pad=10)
    texts = [[axR.text(j, i, "", ha="center", va="center", fontsize=13,
                       color=INK) for j in range(W)] for i in range(H)]
    for ax in (axL, axR):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    def update(frame):
        k, (i, j) = frame
        rect.set_xy((j - 0.5, i - 0.5))
        for kk in range(k + 1):
            ii, jj = positions[kk]
            if np.isnan(shown[ii, jj]):
                shown[ii, jj] = fm[ii, jj]
                texts[ii][jj].set_text("%.0f" % fm[ii, jj])
                texts[ii][jj].set_color(
                    "white" if fm[ii, jj] > 0.6 * max(1.0, fm.max()) else INK)
        im.set_data(shown)
        return [rect, im]

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False,
                                   interval=1000 / fps)
    from IPython.display import HTML
    out = HTML(anim.to_html5_video())
    plt.close(fig)
    return out


# ============================================================== one-neuron logit
def _logit_fit_std():
    """Bliss beetles, dose standardized: intercept and slope of the logit."""
    import pandas as pd
    import statsmodels.api as sm
    d = pd.read_csv(chh._DATA / "bliss_beetle_mortality.csv")
    xs, ys = [], []
    for _, r in d.iterrows():
        n, k = int(r["n_tested"]), int(r["n_killed"])
        xs += [r["log_dose"]] * n
        ys += [1] * k + [0] * (n - k)
    x = np.array(xs)
    z = (x - x.mean()) / x.std()
    m = sm.Logit(np.array(ys), sm.add_constant(z)).fit(disp=False)
    return float(m.params[0]), float(m.params[1])


def fig_one_neuron(step=1, dose_z=0.8):
    """The one-neuron network drawn Starmer-style: the activation function's
    curve shown inside the node. step 1: the bare network. step 2: one beetle's
    dose traced through, the dot riding the sigmoid. step 3: both vocabularies
    labeling the same three objects."""
    from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
    b0, b1 = _logit_fit_std()
    z = b0 + b1 * dose_z
    p = 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(9.8, 4.6), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # input node
    ax.add_patch(Circle((1.2, 2.5), 0.55, facecolor="white", edgecolor=INK,
                        lw=2))
    ax.text(1.2, 2.5, "dose\n$x$", ha="center", va="center", fontsize=12)
    # neuron with the sigmoid inside
    ax.add_patch(FancyBboxPatch((4.0, 1.4), 2.6, 2.2,
                                boxstyle="round,pad=0.12",
                                facecolor="white", edgecolor=MADRID, lw=2.4))
    inset = ax.inset_axes([0.44, 0.34, 0.20, 0.40])
    zz = np.linspace(-5, 5, 120)
    inset.plot(zz, 1 / (1 + np.exp(-zz)), color=MADRID, lw=2.4)
    inset.set_xticks([])
    inset.set_yticks([0, 1])
    inset.tick_params(labelsize=8)
    for s in ("top", "right"):
        inset.spines[s].set_visible(False)
    # output node
    ax.add_patch(Circle((8.6, 2.5), 0.62, facecolor="white", edgecolor=INK,
                        lw=2))
    ax.text(8.6, 2.5, "$P(y=1)$", ha="center", va="center", fontsize=11)
    # edges
    ax.add_patch(FancyArrowPatch((1.8, 2.5), (3.9, 2.5),
                                 arrowstyle="-|>", mutation_scale=18,
                                 color=INK, lw=2))
    ax.add_patch(FancyArrowPatch((6.75, 2.5), (7.95, 2.5),
                                 arrowstyle="-|>", mutation_scale=18,
                                 color=INK, lw=2))
    ax.text(2.85, 2.78, "$\\times\\ w$", ha="center", fontsize=13, color=INK)
    ax.text(5.3, 3.85, "$+\\ b$, then squash", ha="center", fontsize=11,
            color=INK)

    if step == 1:
        ax.set_title("Logistic regression drawn as a network: one neuron",
                     fontsize=13.5, color=MADRID, weight="bold")
    if step >= 2:
        inset.scatter([z], [p], s=60, color=FITLINE, zorder=5)
        inset.axvline(z, color=FITLINE, lw=1.0, ls=(0, (3, 2)), alpha=0.6)
        ax.text(2.85, 2.14, "$%.2f$" % b1, ha="center", fontsize=11,
                color=ACCENT)
        ax.text(5.3, 1.06, "$z = %.2f x %+.2f = %.2f$" % (b1, b0, z),
                ha="center", fontsize=11, color=INK)
        ax.text(8.6, 1.55, "$= %.2f$" % p, ha="center", fontsize=12,
                color=FITLINE, weight="bold")
        ax.set_title("One beetle's dose, traced through the neuron",
                     fontsize=13.5, color=MADRID, weight="bold")
    if step >= 3:
        ax.text(2.85, 3.35, "ML: weight\nEcon: slope $\\beta_1$", ha="center",
                fontsize=10, color=MADRID)
        ax.text(5.3, 0.35, "ML: bias + sigmoid activation\n"
                "Econ: intercept $\\beta_0$ + logistic link", ha="center",
                fontsize=10, color=MADRID)
        ax.text(8.6, 3.55, "ML: output\nEcon: fitted probability", ha="center",
                fontsize=10, color=MADRID)
        ax.set_title("Same three objects, two vocabularies", fontsize=13.5,
                     color=MADRID, weight="bold")
    plt.close(fig)
    return fig



# ============================================================================
#  StatQuest chapter 5, Gradient Descent: the figures, one per step.
#  Josh's running example and his numbers throughout:
#      Weight = [0.5, 2.3, 2.9]   Height = [1.4, 1.9, 3.2]
#      slope fixed at 0.64 for the one-parameter walk, learning rate 0.1
#      two-parameter walk starts at (0, 0.5), learning rate 0.01
#      closed form answer: intercept 0.95, slope 0.64
#  Concepts re-created in the course palette; none of his art is copied.
# ============================================================================
_GD_M = 0.64            # the slope he plugs in for the one-parameter example
_GD_CURVE = FITLINE     # the SSR curve, red the way he draws it
_GD_TAN = MADRID        # tangent lines, blue the way he draws it
_GD_RING = "#1F7A47"    # the open green circles he puts on the curve
_GD_GHOST = "#AFC0E8"   # the faded earlier lines he leaves behind


def _gd_axes(ax, xlim=(0, 3.4), ylim=(0, 3.8), labels=True):
    """The Height against Weight panel he redraws on almost every page."""
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if labels:
        ax.set_xlabel("Weight", fontsize=12)
        ax.text(0, 1.02, "Height", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=12, color=INK)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _gd_points(ax, s=110):
    ax.scatter(_W, _H, s=s, color=ACCENT, edgecolors=INK, lw=1.3, zorder=6)


def _gd_line(ax, b, m=_GD_M, color=MADRID, lw=2.8, zorder=3, alpha=1.0):
    xs = np.array([0.0, 3.4])
    ax.plot(xs, b + m * xs, color=color, lw=lw, zorder=zorder, alpha=alpha)


def _gd_stems(ax, b, m=_GD_M, lw=2.4):
    for wi, hi in zip(_W, _H):
        ax.plot([wi, wi], [b + m * wi, hi], color=_STEM, lw=lw, zorder=4)


def _gd_ssr(b, m=_GD_M):
    return float(np.sum((_H - (b + m * _W)) ** 2))


def _gd_deriv(b, m=_GD_M):
    """d SSR / d intercept at this intercept."""
    return float(-2 * np.sum(_H - (b + m * _W)))


def gd_path_one(lr=0.1, n_steps=7, b0=0.0):
    """His one-parameter walk: 0 -> 0.57 -> 0.80 -> ... -> 0.95 in 7 steps."""
    b, out = b0, [b0]
    for _ in range(n_steps):
        b = b - lr * _gd_deriv(b)
        out.append(b)
    return out


def gd_path_two(lr=0.01, n_steps=475, b0=0.0, m0=0.5):
    """His two-parameter walk: (0, 0.5) -> (0.073, 0.648) -> ... -> (0.95, 0.64)."""
    b, m, out = b0, m0, [(b0, m0)]
    for _ in range(n_steps):
        r = _H - (b + m * _W)
        b -= lr * (-2 * np.sum(r))
        m -= lr * (-2 * np.sum(_W * r))
        out.append((b, m))
    return out


# ------------------------------------------------- Main Ideas: no closed form
def fig_gd_logistic_squiggle():
    """His first example of a model with no formula for the answer: logistic
    regression fits an s-shaped curve to data that is either 0 or 1."""
    x0 = np.array([0.4, 0.7, 0.9, 1.9, 2.9])
    x1 = np.array([4.1, 5.0, 6.2, 6.6, 7.0])
    grid = np.linspace(0, 7.6, 300)
    curve = 1 / (1 + np.exp(-1.9 * (grid - 3.5)))

    fig, ax = _fig1(6.6, 4.2)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(grid, curve, color=MADRID, lw=3.0, zorder=3)
    ax.scatter(x0, np.zeros_like(x0), s=120, color=ACCENT, edgecolors=INK,
               lw=1.3, zorder=5)
    ax.scatter(x1, np.ones_like(x1), s=120, color=ACCENT, edgecolors=INK,
               lw=1.3, zorder=5)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predictor", fontsize=12)
    ax.text(0, 1.02, "Outcome", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("Logistic regression fits an s-shaped curve",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def fig_gd_fancy_squiggle():
    """His second example: a neural network fits a far wilder curve to the
    same kind of data. Still no formula for the answer."""
    x0 = np.array([0.4, 0.7, 3.6, 4.0, 4.3])
    x1 = np.array([1.7, 2.1, 2.4, 6.2, 6.6, 7.0])
    grid = np.linspace(0, 7.6, 500)
    curve = (1 / (1 + np.exp(-6 * (grid - 1.2)))
             - 1 / (1 + np.exp(-6 * (grid - 2.9)))
             + 1 / (1 + np.exp(-6 * (grid - 5.4))))

    fig, ax = _fig1(6.6, 4.2)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(grid, curve, color=MADRID, lw=3.0, zorder=3)
    ax.scatter(x0, np.zeros_like(x0), s=120, color=ACCENT, edgecolors=INK,
               lw=1.3, zorder=5)
    ax.scatter(x1, np.ones_like(x1), s=120, color=ACCENT, edgecolors=INK,
               lw=1.3, zorder=5)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predictor", fontsize=12)
    ax.text(0, 1.02, "Outcome", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=12, color=INK)
    ax.set_title("A neural network fits a much wilder curve",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def fig_gd_network():
    """The network he draws beside the wild curve: two inputs, two hidden
    layers, one output, and no formula for any of the weights."""
    layers = [2, 3, 3, 1]
    xs = [0.0, 1.0, 2.0, 3.0]
    cols = [MADRID, ACCENT, ACCENT, "#9AA0B5"]

    fig, ax = _fig1(6.8, 4.2)
    ax.set_facecolor("white")
    pos = []
    for xi, n in zip(xs, layers):
        ys = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
        pos.append([(xi, y) for y in ys])
    for a, b in zip(pos[:-1], pos[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.annotate("", xy=(x2 - 0.13, y2), xytext=(x1 + 0.13, y1),
                            arrowprops=dict(arrowstyle="-|>", color="#7A7A85",
                                            lw=1.0, shrinkA=0, shrinkB=0))
    for layer, col in zip(pos, cols):
        for (x, y) in layer:
            ax.add_patch(plt.Circle((x, y), 0.13, facecolor=col,
                                    edgecolor=INK, lw=1.4, zorder=5))
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    for x, lab in zip(xs, ["inputs", "hidden", "hidden", "output"]):
        ax.text(x, -1.35, lab, ha="center", fontsize=11, color=INK)
    ax.set_title("Every arrow carries a weight with no closed form",
                 fontsize=13.5, color=MADRID, weight="bold", pad=10)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------- Main Ideas: guess, then improve
_GD_GUESS_BOTH = [(0.0, 0.15), (0.45, 0.42), (0.95, 0.64)]
_GD_GUESS_ONE = [(0.0, _GD_M), (0.57, _GD_M), (0.95, _GD_M)]


def fig_gd_guess_at(k, mode="both"):
    """His three-panel story, one panel at a time: start with a guess, improve
    it one step at a time, stop at the best fit. Earlier lines stay behind in
    pale blue, the way he leaves them on the page."""
    seq = _GD_GUESS_BOTH if mode == "both" else _GD_GUESS_ONE
    b, m = seq[k]
    fig, ax = _fig1(6.4, 4.2)
    _gd_axes(ax)
    for (pb, pm) in seq[:k]:
        _gd_line(ax, pb, pm, color=_GD_GHOST, lw=2.4, zorder=2)
    _gd_line(ax, b, m)
    _gd_stems(ax, b, m)
    _gd_points(ax)
    heads = ["the initial guess", "one step better", "the best fit"]
    ax.set_title("%s    SSR = %.2f" % (heads[k], _gd_ssr(b, m)),
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ------------------------------------------------------- Details 1 and 2
def fig_gd_raw_data():
    """The three Height and Weight measurements, on their own. He opens with
    this plot and comes back to it before every new walk."""
    fig, ax = _fig1(6.4, 4.2)
    _gd_axes(ax)
    _gd_points(ax, s=150)
    ax.set_title("Three measurements", fontsize=13.5, color=MADRID,
                 weight="bold", pad=26)
    fig.tight_layout()
    plt.close(fig)
    return fig


def fig_gd_fitted_line():
    """The line he is asking gradient descent to find: Height = 0.95 + 0.64
    times Weight."""
    fig, ax = _fig1(6.4, 4.2)
    _gd_axes(ax)
    _gd_line(ax, 0.95)
    _gd_points(ax)
    ax.text(0.24, 3.42, "Height = intercept + slope $\\times$ Weight",
            fontsize=13, color=MADRID, weight="bold")
    ax.set_title("Pick the intercept and slope that minimize the SSR",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    fig.tight_layout()
    plt.close(fig)
    return fig


def fig_gd_predicted_marks(b=0.0):
    """His green X figure: the predicted height is where the line sits above
    each weight, and the residual is the gap up to the measured point."""
    fig, ax = _fig1(6.4, 4.2)
    _gd_axes(ax)
    _gd_line(ax, b)
    _gd_stems(ax, b)
    ax.scatter(_W, b + _GD_M * _W, marker="X", s=190, color=_STEM,
               edgecolors=INK, lw=1.2, zorder=7)
    _gd_points(ax)
    ax.text(1.35, 3.35, "X  =  predicted height", fontsize=12.5,
            color=_STEM, weight="bold")
    ax.set_title("Residual = observed height $-$ predicted height",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ------------------------------------------------------------- Details 3
def fig_gd_line_at(b, slope=_GD_M, show_resid=True, label_resid=False):
    """The fitted line at one intercept, with its residual stems. Setting
    label_resid prints each residual beside its stem, which is how he gets to
    1.1 squared plus 0.4 squared plus 1.3 squared."""
    fig, ax = _fig1(6.4, 4.2)
    _gd_axes(ax)
    _gd_line(ax, b, slope)
    if show_resid:
        _gd_stems(ax, b, slope)
    if label_resid:
        for wi, hi in zip(_W, _H):
            r = hi - (b + slope * wi)
            ax.text(wi + 0.09, (hi + b + slope * wi) / 2.0, "%.1f" % r,
                    fontsize=13, color=_STEM, weight="bold", va="center")
    _gd_points(ax)
    ax.set_title("intercept = %.2f    SSR = %.2f" % (b, _gd_ssr(b, slope)),
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    fig.tight_layout()
    plt.close(fig)
    return fig


def fig_gd_ssr_arithmetic():
    """His step 9: square the three residuals, add them up, and the SSR at an
    intercept of 0 is 3.1."""
    r = _H - _GD_M * _W
    fig, ax = _fig1(6.6, 4.4)
    _gd_axes(ax)
    _gd_line(ax, 0.0)
    _gd_stems(ax, 0.0)
    for wi, hi in zip(_W, _H):
        ax.text(wi + 0.09, (hi + _GD_M * wi) / 2.0,
                "%.1f" % (hi - _GD_M * wi), fontsize=13, color=_STEM,
                weight="bold", va="center")
    _gd_points(ax)
    rr = np.round(r, 1)
    ax.text(0.12, 3.35,
            "$%.1f^2 + %.1f^2 + %.1f^2 = %.1f$"
            % (rr[0], rr[1], rr[2], round(float(np.sum(rr ** 2)), 1)),
            fontsize=16, color=FITLINE, weight="bold")
    ax.set_title("The SSR when the intercept is 0", fontsize=13.5,
                 color=MADRID, weight="bold", pad=26)
    fig.tight_layout()
    plt.close(fig)
    return fig


# --------------------------------------------------- Details 4: the curve
_GD_BLO, _GD_BHI = -0.6, 2.5


def fig_gd_ssr_curve(marks=(), tangent_at=None, arrow=False, title=None,
                     annotate_slope=True, show_curve=True):
    """The master SSR picture: loss on the vertical axis, intercept on the
    horizontal. Green rings mark intercepts already tried, a blue tangent
    shows the derivative, and the arrow says which way the next step goes.

    show_curve=False plots the rings on bare axes with no curve drawn, which
    is how the video builds it: plug and chug a few intercepts first, and only
    later reveal that they lie on a curve.
    """
    grid = np.linspace(_GD_BLO, _GD_BHI, 300)
    loss = np.array([_gd_ssr(x) for x in grid])

    fig, ax = _fig1(6.6, 4.3)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    if show_curve:
        ax.plot(grid, loss, color=_GD_CURVE, lw=3.0, zorder=3)
    else:
        # keep the frame identical so the curve appears into an unchanged axis
        ax.set_ylim(loss.min() - 0.45, loss.max() + 0.35)
    if tangent_at is not None:
        g = _gd_deriv(tangent_at)
        span = 0.62
        xs = np.array([tangent_at - span, tangent_at + span])
        ax.plot(xs, _gd_ssr(tangent_at) + g * (xs - tangent_at),
                color=_GD_TAN, lw=3.0, zorder=4)
        if annotate_slope:
            ax.text(tangent_at + span + 0.06, _gd_ssr(tangent_at),
                    "slope = %.1f" % g, fontsize=12.5, color=_GD_TAN,
                    weight="bold", va="center")
    for mb in marks:
        ax.scatter([mb], [_gd_ssr(mb)], s=175, facecolors="none",
                   edgecolors=_GD_RING, lw=2.6, zorder=6)
    if arrow and tangent_at is not None:
        g = _gd_deriv(tangent_at)
        direction = 0.55 if g < 0 else -0.55
        y = -0.95
        ax.annotate("", xy=(tangent_at + direction, y),
                    xytext=(tangent_at, y),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=3.0))
        ax.text(tangent_at + direction / 2.0, y - 0.60,
                "step this way", fontsize=11.5, color=ACCENT, weight="bold",
                ha="center")
    ax.set_xlim(_GD_BLO, _GD_BHI)
    ax.set_ylim(-1.95 if arrow else -0.4, loss.max() + 0.6)
    ax.set_xlabel("Intercept", fontsize=12)
    ax.text(0, 1.02, "SSR", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title(title or "SSR as a function of the intercept",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def fig_gd_loss_at(b, slope=_GD_M, tangent=True):
    """Kept for the paired line-and-loss slides: the curve with the current
    intercept ringed, and its tangent."""
    return fig_gd_ssr_curve(marks=[b], tangent_at=b if tangent else None,
                            title="intercept = %.2f" % b)


def fig_gd_three_lines_to_ssr(show_curve=True):
    """His steps 10 and 11: three intercepts, three lines, three rings.

    His step 10 plots the three rings on bare SSR-versus-intercept axes with
    NO curve drawn; the curve is withheld until step 11, whose sentence is
    "corresponds to this curve". Pass show_curve=False for the step 10 state.
    """
    from matplotlib import gridspec
    tried = [0.0, 0.95, 1.95]

    fig = plt.figure(figsize=(11.2, 6.0), dpi=110)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.0, 1.35], wspace=0.34,
                           hspace=0.62, left=0.04, right=0.97, top=0.88,
                           bottom=0.09)
    small = []
    for k, b in enumerate(tried):
        ax = fig.add_subplot(gs[k, 0])
        _gd_axes(ax, labels=False)
        ax.tick_params(length=0)
        _gd_line(ax, b, lw=2.4)
        _gd_stems(ax, b, lw=2.0)
        _gd_points(ax, s=60)
        ax.set_title("intercept = %.2f    SSR = %.2f" % (b, _gd_ssr(b)),
                     fontsize=11.5, color=MADRID, weight="bold", pad=5)
        small.append(ax)

    grid = np.linspace(_GD_BLO, _GD_BHI, 300)
    axc = fig.add_subplot(gs[:, 1])
    axc.set_facecolor("white")
    axc.grid(True, color=GRID, lw=0.8, zorder=0)
    axc.set_axisbelow(True)
    if show_curve:
        axc.plot(grid, [_gd_ssr(x) for x in grid], color=_GD_CURVE, lw=3.0,
                 zorder=3)
    for b in tried:
        axc.scatter([b], [_gd_ssr(b)], s=195, facecolors="none",
                    edgecolors=_GD_RING, lw=2.8, zorder=6)
    axc.set_xlim(_GD_BLO, _GD_BHI)
    # Both states share the curve's frame, so the axes do not jump when the
    # curve arrives at his step 11 and the ring labels sit in the same places.
    _ys = [_gd_ssr(x) for x in grid]
    axc.set_ylim(min(_ys) - 0.45, max(_ys) + 0.35)
    axc.set_xlabel("Intercept", fontsize=12)
    axc.text(0, 1.02, "SSR", transform=axc.transAxes, ha="left",
             va="bottom", fontsize=12, color=INK)
    axc.set_title("Each line is one dot on this graph" if not show_curve
                  else "Each line is one dot on this curve", fontsize=13.5,
                  color=MADRID, weight="bold", pad=24)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)

    for b, dx, dy, ha in ((tried[0], 0.0, 0.50, "center"),
                          (tried[1], 0.45, 0.30, "left"),
                          (tried[2], 0.0, 0.50, "center")):
        axc.annotate("intercept = %.2f" % b, (b, _gd_ssr(b)),
                     xytext=(b + dx, _gd_ssr(b) + dy), ha=ha, fontsize=11,
                     color=_GD_RING, weight="bold")
    plt.close(fig)
    return fig


# ------------------------------------------- Details 5: what the tangent says
def fig_gd_curve_with_tangents():
    """His step 13: the derivative is the slope of the tangent line, and it is
    different at every point along the curve."""
    grid = np.linspace(_GD_BLO, _GD_BHI, 300)
    fig, ax = _fig1(6.8, 4.4)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(grid, [_gd_ssr(x) for x in grid], color=_GD_CURVE, lw=3.0,
            zorder=3)
    for b in (-0.35, 0.35, 0.95, 1.6, 2.25):
        g = _gd_deriv(b)
        xs = np.array([b - 0.42, b + 0.42])
        ax.plot(xs, _gd_ssr(b) + g * (xs - b), color=_GD_TAN, lw=2.6,
                zorder=4)
        ax.scatter([b], [_gd_ssr(b)], s=150, facecolors="none",
                   edgecolors=_GD_RING, lw=2.4, zorder=6)
    ax.set_xlim(_GD_BLO, _GD_BHI)
    ax.set_ylim(-0.4, 8.4)
    ax.set_xlabel("Intercept", fontsize=12)
    ax.text(0, 1.02, "SSR", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("The derivative is the slope of the tangent line",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def fig_gd_tangent_step(b, title=None):
    """His steps 14 and 15: a steep tangent far from the bottom earns a big
    step, a shallow one near the bottom earns a small one, and the sign says
    which way to go."""
    g = _gd_deriv(b)
    size = "large" if abs(g) > 2.0 else "small"
    sign = "negative, so step right" if g < 0 else "positive, so step left"
    return fig_gd_ssr_curve(marks=[b], tangent_at=b, arrow=True,
                            title=title or "derivative %.1f: %s step, %s"
                            % (g, size, sign))


# -------------------------------------------------- Details 7: the loop
def _cycle_diagram(labels, title, radius=1.0):
    """The green round-and-round diagram he draws for the repeating steps."""
    from matplotlib.patches import FancyArrowPatch
    n = len(labels)
    fig, ax = _fig1(9.2, 3.4)
    ax.set_facecolor("white")
    gap = 2.6
    xs = np.arange(n) * gap
    for x, (tag, text) in zip(xs, labels):
        ax.add_patch(plt.Circle((x, 0), 0.30, facecolor="white",
                                edgecolor=INK, lw=2.0, zorder=5))
        ax.text(x, 0, tag, ha="center", va="center", fontsize=14,
                weight="bold", color=INK, zorder=6)
        ax.text(x, -0.62, text, ha="center", va="top", fontsize=11.5,
                color=INK, zorder=6)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((x0 + 0.38, 0.12), (x1 - 0.38, 0.12),
                                     connectionstyle="arc3,rad=-0.42",
                                     arrowstyle="-|>", mutation_scale=20,
                                     color=_GD_RING, lw=2.4, zorder=4))
    ax.add_patch(FancyArrowPatch((xs[-1], -2.05), (xs[0], -2.05),
                                 connectionstyle="arc3,rad=-0.26",
                                 arrowstyle="-|>", mutation_scale=20,
                                 color=_GD_RING, lw=2.4, zorder=4))
    ax.set_xlim(-1.2, xs[-1] + 1.2)
    ax.set_ylim(-3.6, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13.5, color=MADRID, weight="bold", pad=6)
    fig.tight_layout()
    plt.close(fig)
    return fig


def fig_gd_cycle_abc():
    """His three-step loop for plain gradient descent."""
    return _cycle_diagram(
        [("a", "evaluate the\nderivative"),
         ("b", "calculate the\nStep Size"),
         ("c", "calculate the\nnew value")],
        "Repeat until the Step Size is close to 0")


def fig_gd_cycle_abcd():
    """His four-step loop for stochastic gradient descent, with the random
    draw added at the front."""
    return _cycle_diagram(
        [("a", "pick a random\npoint"),
         ("b", "evaluate the\nderivatives"),
         ("c", "calculate the\nStep Sizes"),
         ("d", "calculate the\nnew values")],
        "Repeat until the Step Sizes are close to 0")


def fig_gd_loop_collage():
    """His step 18: the walk seen as a circuit. A line gives a dot on the
    curve, the curve gives the next intercept, the next intercept gives a
    better line."""
    from matplotlib.patches import FancyArrowPatch
    walk = gd_path_one()

    fig = plt.figure(figsize=(10.4, 6.2), dpi=110)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.05, right=0.96, top=0.84, bottom=0.07,
                        wspace=0.30, hspace=0.52)
    axes = [fig.add_subplot(2, 2, k) for k in (1, 2, 3, 4)]
    for ax, b in zip((axes[0], axes[3], axes[2]), (walk[0], walk[1], walk[7])):
        _gd_axes(ax, labels=False)
        ax.tick_params(length=0)
        _gd_line(ax, b, lw=2.4)
        _gd_stems(ax, b, lw=2.0)
        _gd_points(ax, s=62)
        ax.set_title("intercept = %.2f" % b, fontsize=12, color=MADRID,
                     weight="bold", pad=6)

    grid = np.linspace(_GD_BLO, _GD_BHI, 300)
    axc = axes[1]
    axc.set_facecolor("white")
    axc.grid(True, color=GRID, lw=0.8, zorder=0)
    axc.set_axisbelow(True)
    axc.plot(grid, [_gd_ssr(x) for x in grid], color=_GD_CURVE, lw=2.8,
             zorder=3)
    for b in walk:
        axc.scatter([b], [_gd_ssr(b)], s=125, facecolors="none",
                    edgecolors=_GD_RING, lw=2.2, zorder=6)
    axc.set_xlim(_GD_BLO, _GD_BHI)
    axc.set_xlabel("Intercept", fontsize=11)
    axc.text(0, 1.02, "SSR", transform=axc.transAxes, ha="left", va="bottom",
             fontsize=11, color=INK)
    axc.set_title("every intercept tried", fontsize=12, color=MADRID,
                  weight="bold", pad=6)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)

    box = [ax.get_position() for ax in axes]
    arrows = [((box[0].x1 + 0.008, box[0].y0 + box[0].height * 0.5),
               (box[1].x0 - 0.008, box[1].y0 + box[1].height * 0.5), -0.22),
              ((box[1].x0 + box[1].width * 0.14, box[1].y0 - 0.008),
               (box[3].x0 + box[3].width * 0.14, box[3].y1 + 0.006), -0.22),
              ((box[3].x0 - 0.008, box[3].y0 + box[3].height * 0.5),
               (box[2].x1 + 0.008, box[2].y0 + box[2].height * 0.5), -0.22),
              ((box[2].x0 + box[2].width * 0.14, box[2].y1 + 0.006),
               (box[0].x0 + box[0].width * 0.14, box[0].y0 - 0.008), -0.22)]
    for a, b, rad in arrows:
        fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure,
                                       connectionstyle="arc3,rad=%.2f" % rad,
                                       arrowstyle="-|>", mutation_scale=22,
                                       color=_GD_RING, lw=2.4))
    fig.suptitle("Line, curve, better line, and back again",
                 fontsize=14.5, color=MADRID, weight="bold", y=0.955)
    plt.close(fig)
    return fig


# ------------------------------------------- two parameters at the same time
def fig_gd_line2_at(b, m, show_resid=True, note=None):
    """The line when both the intercept and the slope are in play."""
    fig, ax = _fig1(6.4, 4.2)
    _gd_axes(ax)
    _gd_line(ax, b, m)
    if show_resid:
        _gd_stems(ax, b, m)
    _gd_points(ax)
    ax.set_title("intercept = %.3f    slope = %.3f    SSR = %.2f"
                 % (b, m, _gd_ssr(b, m)), fontsize=13, color=MADRID,
                 weight="bold", pad=26)
    if note:
        ax.text(0.12, 3.4, note, fontsize=12.5, color=FITLINE, weight="bold")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ------------------------------------------------ stochastic gradient descent
def fig_sgd_pick(i, b=0.0, m=0.5):
    """His step 3: one point picked at random, the rest faded out. Only the
    picked point goes into the derivatives on this step."""
    fig, ax = _fig1(6.4, 4.2)
    _gd_axes(ax)
    _gd_line(ax, b, m)
    keep = np.zeros(len(_W), dtype=bool)
    keep[i] = True
    ax.scatter(_W[~keep], _H[~keep], s=110, color=ACCENT, edgecolors=INK,
               lw=1.3, zorder=5, alpha=0.32)
    ax.plot([_W[i], _W[i]], [b + m * _W[i], _H[i]], color=_STEM, lw=2.4,
            zorder=4)
    ax.scatter(_W[keep], _H[keep], s=150, color=ACCENT, edgecolors=INK,
               lw=1.6, zorder=6)
    ax.set_title("One point picked at random, one term per derivative",
                 fontsize=13, color=MADRID, weight="bold", pad=26)
    fig.tight_layout()
    plt.close(fig)
    return fig


_SGD_X = np.array([0.35, 0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95,
                   2.15, 2.35, 2.55, 2.75, 2.95, 3.15, 3.35])
_SGD_Y = np.array([0.55, 1.05, 1.35, 1.00, 1.35, 1.75, 1.55, 1.85, 1.60,
                   2.35, 2.10, 2.85, 2.45, 2.20, 2.65, 2.40])


def fig_sgd_minibatch(picked=None):
    """His step 9: with a bigger dataset, a mini-batch takes a handful of
    points per step instead of one, or of all of them."""
    fig, ax = _fig1(6.6, 4.2)
    _gd_axes(ax, xlim=(0, 3.7), ylim=(0, 3.3))
    if picked is None:
        ax.scatter(_SGD_X, _SGD_Y, s=120, color=ACCENT, edgecolors=INK,
                   lw=1.3, zorder=5)
        head = "The whole dataset: every point in every derivative"
    else:
        keep = np.zeros(len(_SGD_X), dtype=bool)
        keep[list(picked)] = True
        ax.scatter(_SGD_X[~keep], _SGD_Y[~keep], s=120, color=ACCENT,
                   edgecolors=INK, lw=1.3, zorder=4, alpha=0.32)
        ax.scatter(_SGD_X[keep], _SGD_Y[keep], s=150, color=ACCENT,
                   edgecolors=INK, lw=1.6, zorder=6)
        head = "A mini-batch of %d, drawn fresh each step" % len(picked)
    ax.set_title(head, fontsize=13, color=MADRID, weight="bold", pad=26)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ----------------------------------------------------------------- the FAQ
def fig_gd_local_global():
    """His FAQ picture: a loss with a shallow dent beside the real bottom.
    Downhill from the wrong side parks in the dent."""
    x = np.linspace(0, 10, 800)
    y = (3.5 + 0.10 * (x - 1.0)
         - 1.90 * np.exp(-((x - 3.0) ** 2) / 1.0)
         - 1.00 * np.exp(-((x - 7.0) ** 2) / 1.0))

    fig, ax = _fig1(7.6, 4.4)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(x, y, color=_GD_CURVE, lw=3.0, zorder=3)
    gi = int(np.argmin(np.where(x < 5.0, y, 99)))    # the deep valley
    li = int(np.argmin(np.where(x > 5.0, y, 99)))    # the shallow one
    for idx, lab, dy in ((gi, "global minimum", -0.75),
                         (li, "local minimum", -0.75)):
        ax.scatter([x[idx]], [y[idx]], s=170, facecolors="none",
                   edgecolors=_GD_RING, lw=2.6, zorder=6)
        ax.annotate(lab, (x[idx], y[idx]),
                    xytext=(x[idx], y[idx] + dy), ha="center", fontsize=12,
                    color=INK, weight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.set_ylim(0, y.max() + 0.6)
    ax.set_xlabel("parameter of interest", fontsize=12)
    ax.text(0, 1.02, "SSR", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=INK)
    ax.set_title("Gradient descent does not promise the best bottom",
                 fontsize=13.5, color=MADRID, weight="bold", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ================================================================== SGD video
#  "Stochastic Gradient Descent, Clearly Explained!!!" (vMh0zPT0tLI).
#  He uses a DIFFERENT dataset in that video: 12 measurements that clump into
#  three clusters, which is what carries his redundancy argument. Built so that
#  least squares returns exactly his gold standard, intercept 0.87, slope 0.68.
_SGDC_W = np.array([0.85, 1.00, 1.10, 1.20, 1.90, 2.00,
                    2.10, 2.20, 2.85, 3.00, 3.05, 3.15])
_SGDC_H = np.array([1.54, 1.32, 1.91, 1.52, 2.40, 1.93,
                    2.42, 2.29, 2.63, 3.31, 2.69, 3.07])


def fig_sgd_clusters(b=None, m=None, picked=None, extra=None, title=None):
    """His SGD video dataset: 12 points falling into three clusters.

    b, m    draw the line with this intercept and slope
    picked  indices entering this step; everything else fades back
    extra   (weight, height) of a brand new sample, drawn as a diamond
    """
    fig, ax = _fig1(6.8, 4.3)
    _gd_axes(ax, xlim=(0, 3.6), ylim=(0, 4.0 if extra else 3.7))
    if b is not None:
        xs = np.array([0.0, 3.5])
        ax.plot(xs, b + m * xs, color=MADRID, lw=2.8, zorder=4)
    if picked is None:
        ax.scatter(_SGDC_W, _SGDC_H, s=125, color=ACCENT, edgecolors=INK,
                   lw=1.3, zorder=5)
    else:
        keep = np.zeros(len(_SGDC_W), dtype=bool)
        keep[list(picked)] = True
        ax.scatter(_SGDC_W[~keep], _SGDC_H[~keep], s=125, color=ACCENT,
                   edgecolors=INK, lw=1.3, zorder=4, alpha=0.32)
        ax.scatter(_SGDC_W[keep], _SGDC_H[keep], s=165, color=ACCENT,
                   edgecolors=INK, lw=1.8, zorder=6)
    if extra:
        ax.scatter([extra[0]], [extra[1]], s=200, color=_GD_RING,
                   edgecolors=INK, lw=1.8, marker="D", zorder=7)
        ax.annotate("new sample", extra,
                    xytext=(extra[0] + 0.14, extra[1] + 0.36),
                    fontsize=11.5, color=_GD_RING, weight="bold")
    if title:
        ax.set_title(title, fontsize=13, color=MADRID, weight="bold", pad=26)
    fig.tight_layout()
    plt.close(fig)
    return fig
