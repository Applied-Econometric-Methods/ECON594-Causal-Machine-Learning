# -*- coding: utf-8 -*-
"""Figures for the FWL-by-projection test deck.

Observation space with n = 3, rotated into the column space's own frame so the
plane is the horizontal z = 0 plane and y sticks straight up. Every 3D step is
a full 360 degree orbit, and every step carries a coordinate key showing the
actual data triples. The example keeps the fit clean while making both
coefficients visibly different from 1:

    1 = (1, 1, 1)   x = (-1, 1, 3)   y = (2, 0, 4)
    beta0 = 1.5   beta1 = 0.5   y-hat = (1, 2, 3)   residual = (1, -2, 1)
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import animation

MADRID = "#2A3B8F"
INK = "#1A1A1A"
ACCENT = "#E69F00"      # x
BLUE = "#0072B2"        # y-hat
RED = "#B3121F"         # the 1 vector
TEAL = "#009E8E"        # y
GREY = "#7A7A85"
GRIDCOL = "#C4C4D0"

matplotlib.rcParams["animation.embed_limit"] = 100
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

import os
from mpl_toolkits.mplot3d import proj3d
try:
    from PIL import Image as _PILImage
    _FLASH_IMG = _PILImage.open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "fwl_flashlight.png")).convert("RGBA")
except Exception:
    _FLASH_IMG = None

_ONE = np.array([1.0, 1.0, 1.0])
_X = np.array([-1.0, 1.0, 3.0])
_Y = np.array([2.0, 0.0, 4.0])       # gives beta0 = 1.5, beta1 = 0.5


def coefs():
    X = np.column_stack([_ONE, _X])
    b0, b1 = np.linalg.lstsq(X, _Y, rcond=None)[0]
    return float(b0), float(b1)


def _raw():
    b0, b1 = coefs()
    yhat = b0 * _ONE + b1 * _X
    resid = _Y - yhat
    return yhat, resid


def _fmt(v):
    return "(%g, %g, %g)" % tuple(round(float(c), 2) for c in v)


def _video(anim):
    # let the video grow to fill the cell/slide width instead of sitting at its
    # native pixel size with dead space to the right; cap it so the wide edit
    # view does not blow it up
    from IPython.display import HTML
    # size by HEIGHT so the video always fits under the slide bullets (filling the
    # width instead caused the bottom to run off the slide); width follows the
    # aspect, centered so any leftover space is balanced
    html = anim.to_html5_video().replace(
        "<video",
        '<video style="display:block;margin:0 auto;height:58vh;'
        'width:auto;max-width:100%;"', 1)
    return HTML(html)


def _frame():
    u1 = _ONE / np.linalg.norm(_ONE)
    w = _X - (_X @ u1) * u1
    u2 = w / np.linalg.norm(w)
    n = np.cross(u1, u2)
    n = n / np.linalg.norm(n)
    return u1, u2, n


def _c(v):
    u1, u2, n = _frame()
    return np.array([v @ u1, v @ u2, v @ n])


def _pts():
    one = _c(_ONE)
    x = _c(_X)
    y = _c(_Y)
    yhat = y.copy()
    yhat[2] = 0.0
    return one, x, y, yhat


def _setup(ax, azim, elev=20, pos=(-0.02, -0.06, 1.04, 1.16)):
    ax.set_axis_off()
    ax.set_xlim(-1.6, 5.6)
    ax.set_ylim(-3.0, 3.8)
    ax.set_zlim(-0.3, 2.7)
    try:
        ax.set_box_aspect((7.2, 6.8, 3.2))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_position(list(pos))


# the spin/still animations shift the 3D box left and blow it up, leaving the
# right margin free for the coordinate key (see _key)
_ANIM_POS = (-0.30, -0.24, 1.30, 1.60)


def _plane(ax, alpha=0.22):
    corners = np.array([[-1.6, -3.0, 0.0], [5.3, -3.0, 0.0],
                        [5.3, 3.6, 0.0], [-1.6, 3.6, 0.0]])
    poly = Poly3DCollection([corners], facecolor=TEAL, alpha=alpha,
                            edgecolor=MADRID, lw=1.0)
    ax.add_collection3d(poly)


def _arrow(ax, tip, color, label, lw=3.4, tail=None, ls="-", lab=1.10,
           dz=0.08):
    tail = np.zeros(3) if tail is None else np.asarray(tail, float)
    tip = np.asarray(tip, float)
    d = tip - tail
    ax.quiver(tail[0], tail[1], tail[2], d[0], d[1], d[2], color=color,
              lw=lw, arrow_length_ratio=0.11, linestyle=ls)
    if label:
        p = tail + (tip - tail) * lab
        ax.text(p[0], p[1], p[2] + dz, label, color=color, fontsize=15,
                weight="bold")


def _key(ax, lines):
    # numbers live in the right margin, clear of the enlarged 3D box, stacked
    # from the middle so every line shows (nothing clips off the top)
    fig = ax.figure
    y0 = 0.55 + 0.036 * (len(lines) - 1)
    for i, (text, col) in enumerate(lines):
        fig.text(0.645, y0 - i * 0.072, text, color=col, fontsize=11,
                 weight="bold", va="top", ha="left")


# ---- per-step scene drawers (draw the whole scene onto an existing 3D ax) ----
def _draw_plane(ax):
    one, x, y, yhat = _pts()
    _plane(ax)
    _arrow(ax, one, RED, "$\\mathbf{1}$")
    _arrow(ax, x, ACCENT, "$\\mathbf{x}$")
    _key(ax, [("$\\mathbf{1} = %s$" % _fmt(_ONE), RED),
              ("$\\mathbf{x} = %s$" % _fmt(_X), ACCENT)])


def _draw_off(ax):
    one, x, y, yhat = _pts()
    _plane(ax)
    _arrow(ax, one, RED, "$\\mathbf{1}$")
    _arrow(ax, x, ACCENT, "$\\mathbf{x}$")
    _arrow(ax, y, TEAL, "$\\mathbf{y}$")
    _key(ax, [("$\\mathbf{1} = %s$" % _fmt(_ONE), RED),
              ("$\\mathbf{x} = %s$" % _fmt(_X), ACCENT),
              ("$\\mathbf{y} = %s$" % _fmt(_Y), TEAL)])


def _draw_proj(ax):
    one, x, y, yhat = _pts()
    yhat_r, resid_r = _raw()
    _plane(ax)
    _arrow(ax, one, RED, "$\\mathbf{1}$")
    _arrow(ax, x, ACCENT, "$\\mathbf{x}$")
    _arrow(ax, y, TEAL, "$\\mathbf{y}$")
    _arrow(ax, yhat, BLUE, "$\\hat{\\mathbf{y}}$")
    ax.plot([yhat[0], y[0]], [yhat[1], y[1]], [yhat[2], y[2]], color=INK,
            lw=2.0, ls=(0, (4, 3)))
    mid = 0.5 * (yhat + y)
    ax.text(mid[0] + 0.15, mid[1], mid[2], "$M\\mathbf{y}$", color=INK,
            fontsize=12)
    _key(ax, [("$\\mathbf{1} = %s$" % _fmt(_ONE), RED),
              ("$\\mathbf{x} = %s$" % _fmt(_X), ACCENT),
              ("$\\mathbf{y} = %s$" % _fmt(_Y), TEAL),
              ("$\\hat{\\mathbf{y}} = %s$" % _fmt(yhat_r), BLUE),
              ("$M\\mathbf{y} = %s$" % _fmt(resid_r), INK)])


def _draw_decomp(ax):
    one, x, y, yhat = _pts()
    b0, b1 = coefs()
    yhat_r, resid_r = _raw()
    p0 = b0 * one
    _plane(ax)
    _arrow(ax, p0, RED, "$\\beta_0\\mathbf{1}$", lw=3.6, lab=0.98, dz=-0.32)
    _arrow(ax, yhat, ACCENT, "$\\beta_1\\mathbf{x}$", tail=p0, lw=3.6,
           lab=0.95, dz=0.16)
    _arrow(ax, yhat, BLUE, "$\\hat{\\mathbf{y}}$", lw=2.4, lab=1.05)
    # residual is the third leg of the running sum, so it carries an arrowhead
    _arrow(ax, y, INK, "$M\\mathbf{y}$", tail=yhat, lw=2.2, ls=(0, (4, 3)),
           lab=0.5, dz=0.16)
    _arrow(ax, y, TEAL, "$\\mathbf{y}$", lw=2.2)
    _key(ax, [("$\\beta_0\\mathbf{1} = %s$" % _fmt(b0 * _ONE), RED),
              ("$\\beta_1\\mathbf{x} = %s$" % _fmt(b1 * _X), ACCENT),
              ("$\\hat{\\mathbf{y}} = %s$" % _fmt(yhat_r), BLUE),
              ("$M\\mathbf{y} = %s$" % _fmt(resid_r), INK),
              ("sum $= \\mathbf{y} = %s$" % _fmt(_Y), TEAL)])


def _draw_partial(ax):
    one, x, y, yhat = _pts()
    xt = np.array([0.0, x[1], 0.0])
    xtilde = _X - _X.mean() * _ONE
    _plane(ax)
    _arrow(ax, one, RED, "$\\mathbf{1}$", lw=2.6)
    _arrow(ax, x, ACCENT, "$\\mathbf{x}$", lw=2.2)
    ax.plot([x[0], xt[0]], [x[1], xt[1]], [x[2], xt[2]], color=GREY,
            lw=1.4, ls=(0, (3, 3)))
    _arrow(ax, xt, ACCENT, "$\\tilde{\\mathbf{x}}$", lw=3.8)
    _key(ax, [("$\\mathbf{1} = %s$" % _fmt(_ONE), RED),
              ("$\\mathbf{x} = %s$" % _fmt(_X), ACCENT),
              ("mean of $\\mathbf{x} = %g$" % _X.mean(), GREY),
              ("$\\tilde{\\mathbf{x}} = \\mathbf{x} - \\bar{x}\\mathbf{1} = %s$"
               % _fmt(xtilde), ACCENT)])


def _draw_partial_y(ax):
    # the exact mirror of _draw_partial, now taking the average out of y.
    # demeaning zeroes the ones-direction component, so y-tilde sits at
    # (0, y[1], y[2]) in the rotated frame, just as x-tilde sits at (0, x[1], 0).
    one, x, y, yhat = _pts()
    yt = np.array([0.0, y[1], y[2]])
    ytilde = _Y - _Y.mean() * _ONE
    _plane(ax)
    _arrow(ax, one, RED, "$\\mathbf{1}$", lw=2.6)
    _arrow(ax, y, TEAL, "$\\mathbf{y}$", lw=2.2)
    ax.plot([y[0], yt[0]], [y[1], yt[1]], [y[2], yt[2]], color=GREY,
            lw=1.4, ls=(0, (3, 3)))
    _arrow(ax, yt, TEAL, "$\\tilde{\\mathbf{y}}$", lw=3.8)
    _key(ax, [("$\\mathbf{1} = %s$" % _fmt(_ONE), RED),
              ("$\\mathbf{y} = %s$" % _fmt(_Y), TEAL),
              ("mean of $\\mathbf{y} = %g$" % _Y.mean(), GREY),
              ("$\\tilde{\\mathbf{y}} = \\mathbf{y} - \\bar{y}\\mathbf{1} = %s$"
               % _fmt(ytilde), TEAL)])


def _rightangle(ax, a, b, s=0.55):
    # a small square corner marking the right angle between directions a and b
    ua = a / np.linalg.norm(a)
    ub = b / np.linalg.norm(b)
    p1, p2, p3 = s * ua, s * (ua + ub), s * ub
    ax.add_collection3d(Poly3DCollection(
        [[np.zeros(3), p1, p2, p3]], facecolor="#F4A300", alpha=0.20,
        edgecolor="none", zorder=1))
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=INK,
            lw=2.2, zorder=8)
    ax.plot([p3[0], p2[0]], [p3[1], p2[1]], [p3[2], p2[2]], color=INK,
            lw=2.2, zorder=8)


def _draw_together(ax):
    # x-tilde and y-tilde from the two steps on one scene, each springing from
    # the origin at a right angle to the ones vector.
    one, x, y, yhat = _pts()
    xt = np.array([0.0, x[1], 0.0])
    yt = np.array([0.0, y[1], y[2]])
    _plane(ax)
    _arrow(ax, x, ACCENT, "$\\mathbf{x}$", lw=1.5, lab=1.07)
    _arrow(ax, y, TEAL, "$\\mathbf{y}$", lw=1.5, lab=1.07)
    ax.plot([x[0], xt[0]], [x[1], xt[1]], [x[2], xt[2]], color=GREY,
            lw=1.2, ls=(0, (3, 3)))
    ax.plot([y[0], yt[0]], [y[1], yt[1]], [y[2], yt[2]], color=GREY,
            lw=1.2, ls=(0, (3, 3)))
    _arrow(ax, one, RED, "$\\mathbf{1}$", lw=3.4)
    _arrow(ax, xt, ACCENT, "$\\tilde{\\mathbf{x}}$", lw=3.8)
    _arrow(ax, yt, TEAL, "$\\tilde{\\mathbf{y}}$", lw=3.8)
    _rightangle(ax, one, xt)
    _rightangle(ax, one, yt)
    _key(ax, [("$\\mathbf{1} = %s$" % _fmt(_ONE), RED),
              ("$\\mathbf{x} = %s$" % _fmt(_X), ACCENT),
              ("$\\tilde{\\mathbf{x}} = %s$" % _fmt(_X - _X.mean() * _ONE), ACCENT),
              ("$\\mathbf{y} = %s$" % _fmt(_Y), TEAL),
              ("$\\tilde{\\mathbf{y}} = %s$" % _fmt(_Y - _Y.mean() * _ONE), TEAL)])


def _draw_paral(ax):
    # a clean parallelogram: both components rooted at the origin, y-hat is the
    # diagonal, dashed lines complete the two remaining sides. No y, no residual
    # here (those belong to the head-to-tail slide), so nothing extra crosses.
    one, x, y, yhat = _pts()
    b0, b1 = coefs()
    yhat_r, resid_r = _raw()
    p0 = b0 * one          # beta0 * 1
    p1 = b1 * x            # beta1 * x
    yh = yhat              # diagonal corner = p0 + p1
    _plane(ax)
    _arrow(ax, p0, RED, "$\\beta_0\\mathbf{1}$", lw=3.8, lab=0.98, dz=-0.30)
    _arrow(ax, p1, ACCENT, "$\\beta_1\\mathbf{x}$", lw=3.8, lab=0.98, dz=0.18)
    # dashed sides that close the parallelogram back to the diagonal corner
    ax.plot([p0[0], yh[0]], [p0[1], yh[1]], [p0[2], yh[2]], color=GREY,
            lw=1.5, ls=(0, (3, 3)))
    ax.plot([p1[0], yh[0]], [p1[1], yh[1]], [p1[2], yh[2]], color=GREY,
            lw=1.5, ls=(0, (3, 3)))
    _arrow(ax, yh, MADRID, "$\\hat{\\mathbf{y}}$", lw=3.0, lab=1.07)
    _key(ax, [("$\\beta_0\\mathbf{1} = %s$" % _fmt(b0 * _ONE), RED),
              ("$\\beta_1\\mathbf{x} = %s$" % _fmt(b1 * _X), ACCENT),
              ("$\\hat{\\mathbf{y}} = \\beta_0\\mathbf{1}+\\beta_1\\mathbf{x} = %s$"
               % _fmt(yhat_r), MADRID)])


_DRAW = {"plane": _draw_plane, "off": _draw_off, "proj": _draw_proj,
         "paral": _draw_paral, "decomp": _draw_decomp, "partial": _draw_partial,
         "partialy": _draw_partial_y, "together": _draw_together}


def _still(key, azim=-62):
    fig = plt.figure(figsize=(11.5, 5.0), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    _setup(ax, azim, pos=_ANIM_POS)
    _DRAW[key](ax)
    plt.close(fig)
    return fig


def _spin(key, azim0=-62, frames=60, fps=12):
    fig = plt.figure(figsize=(11.5, 5.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    draw = _DRAW[key]

    def update(k):
        ax.clear()
        _setup(ax, azim0 + 360.0 * k / frames, pos=_ANIM_POS)
        draw(ax)
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=1000 / fps)
    out = _video(anim)
    plt.close(fig)
    return out


def fig_plane():
    return _still("plane")


def fig_off_plane():
    return _still("off")


def fig_projection():
    return _still("proj")


def fig_decompose():
    return _still("decomp")


def fig_partial():
    return _still("partial")


def anim_plane():
    return _spin("plane")


def anim_off():
    return _spin("off")


def anim_proj():
    return _spin("proj")


def anim_paral():
    return _spin("paral")


def anim_decomp():
    return _spin("decomp")


def anim_partial():
    return _spin("partial")


def fig_partial_y():
    return _still("partialy")


def anim_partial_y():
    return _spin("partialy")


def fig_together():
    return _still("together")


def anim_together():
    return _spin("together")


def fig_ratio():
    u1, u2, n = _frame()
    xt = float(_X @ u2)
    yt_h = float(_Y @ u2)
    yt_v = abs(float(_Y @ n))
    b0, b1 = coefs()
    xtilde = _X - _X.mean() * _ONE
    ytilde = _Y - _Y.mean() * _ONE
    xty = int(round(xtilde @ ytilde))
    xtx = int(round(xtilde @ xtilde))
    fig, ax = plt.subplots(figsize=(8.4, 4.7), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.annotate("", xy=(xt, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=3.2, color=ACCENT))
    ax.text(xt * 1.02, -0.3, "$\\tilde{\\mathbf{x}}$", color=ACCENT,
            fontsize=16, weight="bold")
    ax.annotate("", xy=(yt_h, yt_v), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=3.2, color=TEAL))
    ax.text(yt_h * 1.02, yt_v + 0.06, "$\\tilde{\\mathbf{y}}$", color=TEAL,
            fontsize=16, weight="bold")
    ax.plot([yt_h, yt_h], [yt_v, 0], color=INK, lw=1.8, ls=(0, (4, 3)))
    ax.scatter([yt_h], [0], s=55, color=BLUE, zorder=5)
    ax.text(yt_h, -0.34, "$\\beta_1\\tilde{\\mathbf{x}}$", color=BLUE,
            fontsize=13, ha="center")
    ax.axhline(0, color=INK, lw=1.0)
    ax.text(0.03, 0.94, "$\\tilde{\\mathbf{x}} = %s$" % _fmt(xtilde),
            transform=ax.transAxes, color=ACCENT, fontsize=12.5, weight="bold",
            va="top")
    ax.text(0.03, 0.86, "$\\tilde{\\mathbf{y}} = %s$" % _fmt(ytilde),
            transform=ax.transAxes, color=TEAL, fontsize=12.5, weight="bold",
            va="top")
    ax.text(0.03, 0.74,
            "$\\beta_1 = \\dfrac{\\tilde{\\mathbf{x}}'\\tilde{\\mathbf{y}}}"
            "{\\tilde{\\mathbf{x}}'\\tilde{\\mathbf{x}}} = \\dfrac{%d}{%d} = %g$"
            % (xty, xtx, b1), transform=ax.transAxes, fontsize=14, color=MADRID,
            va="top")
    ax.set_xlim(-0.4, xt + 0.8)
    ax.set_ylim(-0.75, yt_v + 0.6)
    ax.set_xlabel("along $\\tilde{\\mathbf{x}}$  (demeaned x)", fontsize=12)
    ax.text(0, 1.02, "leftover, perpendicular to x", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11.5, color=INK)
    ax.set_title("After demeaning, $\\beta_1$ is one ratio", fontsize=13.5,
                 color=MADRID, weight="bold", pad=20)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def flashlight_animation(frames=54, fps=10):
    """The book's Shadow on the Wall. A beam of light tilts from oblique to
    straight down; the shadow of y on the plane slides toward y-hat, and the
    gap from y to the shadow shrinks to its minimum exactly when the light is
    perpendicular. That least gap is the residual, and the right angle is the
    orthogonality that puts the ordinary in ordinary least squares."""
    one, x, y, yhat = _pts()
    yh = float(y[2])                       # height of y above the plane
    hold = 12
    tilt = np.concatenate([np.linspace(0.95, 0.0, frames - hold),
                           np.zeros(hold)])
    fig = plt.figure(figsize=(8.8, 5.1), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")

    def update(k):
        ax.clear()
        _setup(ax, azim=-58)
        _plane(ax)
        dx = tilt[k]
        d = np.array([dx, 0.0, -1.0])              # ray travel direction
        S = np.array([y[0] + yh * dx, y[1], 0.0])  # shadow of y on the plane
        top = y + 1.5 * np.array([-dx, 0.0, 1.0])   # keep the body in frame
        # every ray leaves the lens and fans out to the plane, so the light has
        # one point of origin instead of sprouting from the sides of the barrel
        for ou1, ou2 in [(0, 0), (0, 0.7), (0, -0.7), (0.6, 0), (-0.6, 0)]:
            end = S + np.array([ou1, ou2, 0.0])
            ax.plot([top[0], end[0]], [top[1], end[1]], [top[2], end[2]],
                    color="#F2C200", lw=1.1, alpha=0.55)
        # a real flashlight image at the beam head, rotated to point down the beam
        if _FLASH_IMG is not None:
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            du = d / np.linalg.norm(d)
            fc = top - 0.90 * du            # sit the body above the beam origin,
            #                                 so the lens (front) is at the beam
            fx, fy, _ = proj3d.proj_transform(fc[0], fc[1], fc[2], ax.get_proj())
            x2, y2, _ = proj3d.proj_transform(top[0], top[1], top[2], ax.get_proj())
            sx, sy, _ = proj3d.proj_transform(S[0], S[1], S[2], ax.get_proj())
            ang = np.degrees(np.arctan2(sy - y2, sx - x2))   # down-beam, screen
            rot = _FLASH_IMG.rotate(ang - 180.0, expand=True,
                                    resample=_PILImage.BICUBIC)
            oi = OffsetImage(np.asarray(rot), zoom=0.115)
            ab = AnnotationBbox(oi, (fx, fy), frameon=False,
                                xycoords=ax.transData, box_alignment=(0.5, 0.5),
                                zorder=9)
            ax.add_artist(ab)
        else:
            du = d / np.linalg.norm(d)
            lens = top
            ax.plot([top[0] - 0.5 * du[0], lens[0]], [top[1], lens[1]],
                    [top[2] + 0.5, lens[2]], color="#8A8A8A", lw=16,
                    solid_capstyle="round", zorder=6)
            ax.scatter([lens[0]], [lens[1]], [lens[2]], color="#FFD21A", s=90,
                       zorder=7)
        _arrow(ax, y, TEAL, "$\\mathbf{y}$", lw=2.8)
        _arrow(ax, S, BLUE, "shadow", lw=2.8, lab=1.12)
        ax.plot([y[0], S[0]], [y[1], S[1]], [y[2], S[2]], color=RED, lw=2.2,
                ls=(0, (4, 3)))
        gap = float(np.linalg.norm(y - S))
        ax.text2D(0.02, 0.97, "gap from $\\mathbf{y}$ to its shadow: %.2f" % gap,
                  transform=ax.transAxes, fontsize=12.5, color=INK, va="top")
        if dx < 1e-6:
            ax.text2D(0.02, 0.16, "straight down: the shadow is "
                      "$\\hat{\\mathbf{y}}$, the closest point",
                      transform=ax.transAxes, fontsize=12, color=MADRID,
                      weight="bold", va="bottom")
            ax.text2D(0.02, 0.10, "the gap is now perpendicular: orthogonal",
                      transform=ax.transAxes, fontsize=12, color=RED, va="bottom")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(tilt),
                                   interval=1000 / fps)
    out = _video(anim)
    plt.close(fig)
    return out


def fig_mean(z=None):
    """Projecting a vector z onto the single vector 1 returns its mean, spread
    across every row. Shown in the plane spanned by 1 and z: the horizontal
    axis is the 1 direction, the foot of the perpendicular is the projection
    z-bar * 1, and the perpendicular leftover is the demeaned z."""
    if z is None:
        z = _X
    z = np.asarray(z, float)
    n = len(z)
    s = float(z.sum())
    m = s / n
    along = m * np.linalg.norm(_ONE)
    ztil = z - m * _ONE
    perp = float(np.linalg.norm(ztil))
    fig, ax = plt.subplots(figsize=(8.4, 4.7), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color=GRIDCOL, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color=INK, lw=1.0)
    ax.annotate("", xy=(np.linalg.norm(_ONE), 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=3.0, color=RED))
    ax.text(np.linalg.norm(_ONE), -0.34, "$\\mathbf{1}$", color=RED,
            fontsize=15, weight="bold", ha="center")
    ax.annotate("", xy=(along, perp), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=3.0, color=ACCENT))
    ax.text(along * 0.5 - 0.15, perp * 0.6, "$\\mathbf{z}$", color=ACCENT,
            fontsize=15, weight="bold")
    ax.plot([along, along], [perp, 0], color=INK, lw=1.8, ls=(0, (4, 3)))
    ax.scatter([along], [0], s=60, color=MADRID, zorder=5)
    ax.text(along, -0.34, "$\\bar{z}\\,\\mathbf{1}$", color=MADRID,
            fontsize=13, ha="center")
    ax.text(along + 0.12, perp * 0.5, "$\\tilde{\\mathbf{z}}$", color=GREY,
            fontsize=13)
    ax.text(0.03, 0.93, "$\\mathrm{proj}_{\\mathbf{1}}\\,\\mathbf{z} = "
            "\\dfrac{\\mathbf{1}'\\mathbf{z}}{\\mathbf{1}'\\mathbf{1}}\\,\\mathbf{1}"
            " = \\dfrac{%g}{%d}\\,\\mathbf{1} = %g\\,\\mathbf{1} = %s$"
            % (s, n, m, _fmt(m * _ONE)), transform=ax.transAxes,
            fontsize=13.5, color=MADRID, va="top")
    ax.set_xlim(-0.4, along + 1.0)
    ax.set_ylim(-0.75, perp + 0.7)
    ax.set_xlabel("along $\\mathbf{1}$", fontsize=12)
    ax.text(0, 1.02, "perpendicular to $\\mathbf{1}$", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11.5, color=INK)
    ax.set_title("Project onto $\\mathbf{1}$ and you land on the mean",
                 fontsize=13.5, color=MADRID, weight="bold", pad=20)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def commute_animation(hold=9, fps=6):
    """Head-to-tail addition is commutative: the same three legs, beta0*1,
    beta1*x, and the residual My, chained in every one of the six possible
    orders, all land on the same y. The legs keep their identity colors so the
    viewer sees the same three arrows rearranged, and a star marks the shared
    destination that never moves."""
    import itertools
    one, x, y, yhat = _pts()
    b0, b1 = coefs()
    legs = [("$\\beta_0\\mathbf{1}$", b0 * one, RED),
            ("$\\beta_1\\mathbf{x}$", b1 * x, ACCENT),
            ("$M\\mathbf{y}$", y - yhat, INK)]
    perms = list(itertools.permutations(range(3)))
    fig = plt.figure(figsize=(8.8, 5.1), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")

    def update(f):
        pi = (f // hold) % len(perms)
        perm = perms[pi]
        ax.clear()
        _setup(ax, azim=-58)
        _plane(ax)
        pt = np.zeros(3)
        names = []
        for j in perm:
            lab, vec, col = legs[j]
            nxt = pt + vec
            ls = (0, (4, 3)) if col is INK else "-"
            _arrow(ax, nxt, col, lab, tail=pt, lw=3.2, ls=ls, lab=0.5, dz=0.14)
            pt = nxt
            names.append(lab)
        ax.scatter([y[0]], [y[1]], [y[2]], color=TEAL, s=130, marker="*",
                   zorder=8, edgecolors="white", lw=0.6)
        ax.text(y[0], y[1], y[2] + 0.2, "$\\mathbf{y}$", color=TEAL,
                fontsize=15, weight="bold")
        ax.text2D(0.02, 0.97, "order %d of 6:   %s" % (pi + 1, "  +  ".join(names)),
                  transform=ax.transAxes, fontsize=12.5, color=INK, va="top")
        ax.text2D(0.02, 0.90, "every order lands on the same $\\mathbf{y}$",
                  transform=ax.transAxes, fontsize=12, color=MADRID,
                  weight="bold", va="top")
        return []

    anim = animation.FuncAnimation(fig, update, frames=hold * len(perms),
                                   interval=1000 / fps)
    out = _video(anim)
    plt.close(fig)
    return out



def fig_corr_dial():
    """A trig-free chart. As the second variable's arrow opens away from the
    first, from 0 to 180 degrees, their correlation slides from +1 (same
    direction) through 0 (a right angle) down to -1 (opposite). Read the angle,
    read the correlation, no cosine required."""
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")
    R = 1.0
    # the first variable's direction: a faint baseline pointing right
    ax.plot([0, 1.34], [0, 0], color=GREY, lw=1.3, ls=(0, (5, 4)), zorder=1)
    ax.text(0.66, -0.14, "direction of the first variable", color=GREY,
            fontsize=10, ha="center", va="center")
    # faint arc through the arrow tips
    aa = np.radians(np.linspace(0, 180, 200))
    ax.plot(R * np.cos(aa), R * np.sin(aa), color=GRIDCOL, lw=1.2, zorder=1)
    cases = [(0, "+1", TEAL, 0.12, "0° = $0$"),
             (45, "+0.7", "#4FB0A3", 0.0, r"45° = $\pi/4$"),
             (90, "0", GREY, 0.06, r"90° = $\pi/2$"),
             (135, "−0.7", "#CF7F63", 0.0, r"135° = $3\pi/4$"),
             (180, "−1", RED, 0.12, r"180° = $\pi$")]
    for deg, lab, col, dyl, ang in cases:
        th = np.radians(deg)
        vx, vy = R * np.cos(th), R * np.sin(th)
        ax.annotate("", xy=(vx, vy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=3.0, color=col),
                    zorder=4)
        lx, ly = 1.17 * np.cos(th), 1.17 * np.sin(th) + dyl
        ha = "left" if vx > 0.05 else ("right" if vx < -0.05 else "center")
        ax.text(lx, ly, "corr " + lab, color=col, fontsize=13, weight="bold",
                ha=ha, va="center")
        # on top of the arrow, on a white patch, so the arrow does not hide it
        ax.text(0.56 * np.cos(th), 0.56 * np.sin(th), ang, color=INK,
                fontsize=10, ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="none", alpha=0.95))
    # right-angle mark between the baseline and the vertical (90 degrees) arrow
    s = 0.12
    ax.plot([s, s, 0], [0, s, s], color=INK, lw=1.6, zorder=5)
    ax.text(0.0, -0.42,
            "same direction: they move together     square: unrelated     "
            "opposite: mirror image",
            color=INK, fontsize=11.5, ha="center", va="center")
    ax.set_xlim(-1.7, 1.95)
    ax.set_ylim(-0.62, 1.4)
    plt.close(fig)
    return fig
