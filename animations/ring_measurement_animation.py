"""
ring_measurement_animation.py
=============================
Анимация к ring-resonator-measurement (SOI, all-pass кольцо, R = 10 мкм):
  1) развёртка перестраиваемого лазера по спектру + свечение кольца на резонансах
  2) зум на резонанс вблизи 1550 нм
  3) удаление базовой линии (детренд)
  4) фит all-pass формой и вывод Q, ER, t, A в сравнении с истинными значениями

Запуск из корня репозитория:  python animations/ring_measurement_animation.py
Результат: images/ring_measurement.mp4 и images/ring_measurement.gif
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection

REPO = os.environ.get("MEAS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from spectrum_fit import (characterize, fit_one_resonance, true_values,
                          phase_full, T_TRUE, A_TRUE)

FPS = 20
CSV = os.path.join(REPO, "measured_spectrum.csv")

# ---------------- данные и обработка ----------------
data = np.loadtxt(CSV, delimiter=",", skiprows=1)
lam, T = data[:, 0], data[:, 1]
rec = characterize(lam, T)
Tn, base = rec["Tn"], rec["base"]
fit = min(rec["fits"], key=lambda f: abs(f["lam0"] - 1550))
lam0, fsr = fit["lam0"], rec["FSR"]
tru = true_values()
t_fit, A_fit = fit["pair"]

# внутрирезонаторная мощность (all-pass): |b_ring/a_in|^2
def ring_power(l):
    phi = phase_full(l)
    t, A = T_TRUE, A_TRUE
    return (1 - t**2) * A / (1 - 2*t*A*np.cos(phi) + (t*A)**2)
P_MAX = (1 - T_TRUE**2) * A_TRUE / (1 - T_TRUE*A_TRUE)**2

# ---------------- расписание кадров ----------------
# фаза 1: развёртка, замедляемся на резонансах (иначе узкие провалы "проскакивают")
w = 1 + 60 * ring_power(lam) / P_MAX
cdf = np.cumsum(w); cdf /= cdf[-1]
N1 = 170
sweep_idx = np.searchsorted(cdf, np.linspace(0, 1, N1)).clip(0, lam.size - 1)
N_HOLD1, N2, N3, N4, N_HOLD = 15, 40, 35, 60, 50
ZOOM_HALF = 0.6                      # нм, полуокно зума
xw, yw = fit["window"]
x_dense = np.linspace(lam0 - ZOOM_HALF, lam0 + ZOOM_HALF, 800)
y_fit = fit["model"](x_dense, *(fit["pair"][::-1]), lam0) if False else None
# модель: model(lam, t, A, lam0) — |T| симметрична по t<->A
y_fit = fit["model"](x_dense, t_fit, A_fit, lam0)

phases = ([("sweep", i) for i in range(N1)] + [("hold1", i) for i in range(N_HOLD1)] +
          [("zoom", i) for i in range(N2)] + [("detrend", i) for i in range(N3)] +
          [("fit", i) for i in range(N4)] + [("hold", i) for i in range(N_HOLD)])

ease = lambda s: 0.5 - 0.5*np.cos(np.pi*np.clip(s, 0, 1))

# ---------------- оформление ----------------
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
fig = plt.figure(figsize=(12.8, 5.6), dpi=100)
axr = fig.add_axes([0.02, 0.10, 0.30, 0.78])
axs = fig.add_axes([0.40, 0.14, 0.57, 0.72])
title = fig.suptitle("", fontsize=14, fontweight="bold", x=0.5, y=0.97)
cmap = plt.get_cmap("inferno")

# --- кольцо ---
axr.set_xlim(-1.6, 1.6); axr.set_ylim(-1.75, 1.45); axr.set_aspect("equal"); axr.axis("off")
th = np.linspace(0, 2*np.pi, 300)
rx, ry = np.cos(th), np.sin(th)
glows = [axr.plot(rx, ry, lw=lw, color="orange", alpha=0, solid_capstyle="round")[0]
         for lw in (22, 14, 8)]
axr.plot(rx, ry, lw=4, color="0.35")
ring_core, = axr.plot(rx, ry, lw=2.5, color=cmap(0.1))
dots = axr.scatter(np.zeros(12), np.zeros(12), s=18, color="yellow", zorder=5)
# шина: вход слева (постоянно), выход справа ~ T
axr.plot([-1.6, 1.6], [-1.22, -1.22], lw=4, color="0.35")
bus_in, = axr.plot([-1.6, 0], [-1.22, -1.22], lw=2.5, color="tab:red")
bus_out, = axr.plot([0, 1.6], [-1.22, -1.22], lw=2.5, color="tab:red")
axr.text(-1.55, -1.5, "вход", fontsize=10); axr.text(1.05, -1.5, "выход", fontsize=10)
axr.text(0, 0.0, "SOI\nR = 10 мкм", ha="center", va="center", fontsize=10, color="0.3")
lam_txt = axr.text(0, 1.28, "", ha="center", fontsize=12, family="monospace")
pw_txt = axr.text(0, -1.72, "", ha="center", fontsize=10, color="0.3")
angle = [0.0]

# --- спектр ---
axs.set_xlim(lam[0], lam[-1]); axs.set_ylim(-0.02, 1.08)
axs.set_xlabel("Длина волны, нм"); axs.set_ylabel("Пропускание")
raw_line, = axs.plot([], [], color="tab:blue", lw=0.8, label="измерение")
base_line, = axs.plot([], [], color="0.5", lw=1.5, ls="--", label="базовая линия")
win_line, = axs.plot([], [], color="tab:blue", lw=1.2)
fit_line, = axs.plot([], [], color="crimson", lw=2.2, label="фит all-pass")
marker = axs.axvline(lam[0], color="orange", lw=1.5, alpha=0.8)
info = axs.text(0.98, 0.04, "", transform=axs.transAxes, fontsize=10.5, ha="right", multialignment="left",
                family="monospace", va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.95))
fwhm_ann = axs.annotate("", xy=(0, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.5))
axs.legend(loc="lower left", frameon=False, fontsize=9)

sel_z = (lam > lam0 - ZOOM_HALF) & (lam < lam0 + ZOOM_HALF)
lz, Tz, Bz, Tnz = lam[sel_z], T[sel_z], base[sel_z], Tn[sel_z]


def draw_ring(l):
    p = ring_power(l) / P_MAX                     # 0..1
    for g, a in zip(glows, (0.12, 0.2, 0.35)):
        g.set_alpha(a * p)
    ring_core.set_color(cmap(0.15 + 0.8*p))
    tr = float(np.interp(l, lam, T / base))
    bus_out.set_alpha(max(0.08, min(1, tr)))
    angle[0] += 0.12 + 0.25*p
    a = angle[0] + np.linspace(0, 2*np.pi, 12, endpoint=False)
    dots.set_offsets(np.column_stack([np.cos(a), np.sin(a)]))
    dots.set_alpha(p)
    lam_txt.set_text(f"λ = {l:8.3f} нм")
    pw_txt.set_text(f"мощность в кольце ×{ring_power(l):5.1f}")


def update(k):
    ph, i = phases[k]
    if ph == "sweep":
        if i == 0:   # сброс состояния (важно при повторном сохранении)
            axs.set_xlim(lam[0], lam[-1]); raw_line.set_linewidth(0.8)
            base_line.set_data([], []); base_line.set_alpha(1)
            fit_line.set_data([], []); info.set_text("")
            fwhm_ann.set_position((0, 0)); fwhm_ann.xy = (0, 0)
        j = sweep_idx[i]
        title.set_text("1. Развёртка перестраиваемого лазера")
        raw_line.set_data(lam[:j+1], T[:j+1])
        marker.set_xdata([lam[j]]); draw_ring(lam[j])
    elif ph == "hold1":
        raw_line.set_data(lam, T)
        base_line.set_data(lam, base)
        title.set_text(f"Резонансы через FSR ≈ {fsr:.2f} нм; оценка базовой линии")
        marker.set_xdata([lam0]); draw_ring(lam0 + 0.3)
    elif ph == "zoom":
        s = ease(i / (N2 - 1))
        axs.set_xlim(lam[0] + s*(lam0 - ZOOM_HALF - lam[0]),
                     lam[-1] + s*(lam0 + ZOOM_HALF - lam[-1]))
        raw_line.set_linewidth(0.8 + 1.0*s)
        title.set_text(f"2. Зум на резонанс вблизи {lam0:.2f} нм")
        draw_ring(lam0 + 0.3*(1 - s) + 0.3*s)
    elif ph == "detrend":
        s = ease(i / (N3 - 1))
        raw_line.set_data(lz, (1-s)*Tz + s*Tnz)
        base_line.set_data(lz, (1-s)*Bz + s*1.0)
        title.set_text("3. Нормировка на базовую линию: T / T_base")
        draw_ring(lam0 + 0.3)
    elif ph == "fit":
        s = i / (N4 - 1)
        n = int(ease(s) * x_dense.size)
        base_line.set_alpha(0.4)
        fit_line.set_data(x_dense[:n], y_fit[:n])
        lcur = x_dense[max(n-1, 0)]
        marker.set_xdata([lcur]); draw_ring(lcur)
        title.set_text("4. Фит формой all-pass: T(φ; t, A)")
    else:
        fit_line.set_data(x_dense, y_fit)
        marker.set_xdata([lam0]); draw_ring(lam0)
        hw = fit["fwhm"] / 2; yh = (1 + fit["Tmin"]) / 2
        fwhm_ann.set_position((lam0 - hw, yh)); fwhm_ann.xy = (lam0 + hw, yh)
        err = lambda r, t_: f"{abs(r - t_)/t_*100:4.1f}%"
        info.set_text(
            f"{'':6}{'фит':>9}{'истина':>9}{'ошибка':>8}\n"
            f"{'Q_L':6}{fit['Q']:9.0f}{tru['Q']:9.0f}{err(fit['Q'], tru['Q']):>8}\n"
            f"{'ER,дБ':6}{fit['ER']:9.1f}{tru['ER']:9.1f}{err(fit['ER'], tru['ER']):>8}\n"
            f"{'t':6}{t_fit:9.4f}{tru['t']:9.4f}{err(t_fit, tru['t']):>8}\n"
            f"{'A':6}{A_fit:9.4f}{tru['A']:9.4f}{err(A_fit, tru['A']):>8}\n"
            f"FWHM = {fit['fwhm']*1000:.0f} пм\n"
            f"t↔A по |T| неразличимы")
        title.set_text(f"Результат: Q_L ≈ {fit['Q']:.0f}, пересвязь (t < A принято из дизайна)")
    return []


anim = FuncAnimation(fig, update, frames=len(phases), interval=1000/FPS)
if __name__ == "__main__":
    out = os.environ.get("OUT_DIR", os.path.join(REPO, "images"))
    os.makedirs(out, exist_ok=True)
    # H.264 Main + yuv420p + faststart: открывается в QuickTime/Keynote/PowerPoint
    writer = FFMpegWriter(fps=FPS, codec="libx264",
                          extra_args=["-profile:v", "main", "-pix_fmt", "yuv420p",
                                      "-crf", "20", "-movflags", "+faststart",
                                      "-tag:v", "avc1"])
    anim.save(os.path.join(out, "ring_measurement.mp4"), writer=writer)
    print("mp4 готов")
    # лёгкий GIF для README (800 px, 12 fps, оптимальная палитра)
    import subprocess
    mp4 = os.path.join(out, "ring_measurement.mp4")
    gif = os.path.join(out, "ring_measurement.gif")
    vf = "fps=12,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp4, "-vf", vf, gif], check=True)
    print("gif готов")
