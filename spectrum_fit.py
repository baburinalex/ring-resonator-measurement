"""
spectrum_fit.py
===============
Characterization-инструмент к ring-resonator-tutorial.

Задача: по ИЗМЕРЕННОМУ спектру пропускания all-pass кольца (как с tunable laser
+ фотодетектора) восстановить физические параметры устройства и сравнить их
с расчётными:
    FSR, групповой индекс n_g, нагруженная добротность Q, extinction ratio ER,
    коэффициенты t и A, потери alpha, коэффициент связи kappa.

Чтобы было что обрабатывать, мы сначала "придумываем" реалистичный спектр
(известные истинные параметры + базовая линия + шум), сохраняем его в CSV,
а затем восстанавливаем параметры так, будто истинных значений мы не знаем.

ВАЖНЫЙ ФИЗИЧЕСКИЙ НЮАНС (вырождение t <-> A):
форма резонанса по МОДУЛЮ пропускания симметрична относительно перестановки
t и A. Поэтому фит по одному только |T| восстанавливает ПАРУ {t, A}, но не может
сам сказать, что из них что (overcoupled или undercoupled). Чтобы разрешить
неоднозначность, нужна доп. информация (фазовый отклик, или серия с разной связью).
Здесь мы это честно отмечаем и для сравнения принимаем режим из замысла дизайна.

Запуск:  python spectrum_fit.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import percentile_filter

# ----------------------------------------------------------------------
# "Истинное" устройство (фит его НЕ знает) и геометрия (её знаем — это маска)
# ----------------------------------------------------------------------
R_KNOWN   = 10.0        # радиус, мкм — известен из топологии
T_TRUE    = 0.978       # self-coupling t (overcoupled: t < A)
A_TRUE    = 0.985       # выживание амплитуды за оборот
NG_TRUE   = 4.18        # групповой индекс реального устройства
NEFF0     = 2.45        # n_eff на 1550 нм
LAMBDA0   = 1550.0      # нм

L_UM = 2.0 * np.pi * R_KNOWN                 # длина оборота, мкм
DNEFF_DLAM = (NEFF0 - NG_TRUE) / (LAMBDA0/1000.0)  # дисперсия (1/мкм), λ в мкм


# ----------------------------------------------------------------------
# Форма all-pass резонанса (одна функция — и для генерации, и для фита)
# ----------------------------------------------------------------------
def allpass_T(phi, t, A):
    """T(phi) = |b_out/b_in|^2 для all-pass кольца."""
    return (t**2 - 2*t*A*np.cos(phi) + A**2) / (1 - 2*t*A*np.cos(phi) + (t*A)**2)


def phase_full(lam_nm):
    """Полная round-trip фаза phi(lambda) с дисперсией (для генерации спектра)."""
    lam_um = lam_nm / 1000.0
    n_eff = NEFF0 + DNEFF_DLAM * (lam_um - LAMBDA0/1000.0)
    return 2*np.pi * n_eff * L_UM / lam_um


# ----------------------------------------------------------------------
# 1. Генерация реалистичного "измеренного" спектра
# ----------------------------------------------------------------------
def make_synthetic_spectrum(path="measured_spectrum.csv", seed=42):
    rng = np.random.default_rng(seed)
    lam = np.arange(1535.0, 1565.0 + 1e-9, 0.001)        # 1 пм шаг
    T_ideal = allpass_T(phase_full(lam), T_TRUE, A_TRUE)

    # базовая линия: вносимые потери + наклон + слабая Fabry-Perot рябь от граней
    baseline = (0.95
                - 0.0012 * (lam - LAMBDA0)
                + 0.015 * np.sin(2*np.pi*(lam - 1535.0)/6.0 + 0.7))
    noise = rng.normal(0.0, 0.004, size=lam.size)        # шум детектора
    T_meas = T_ideal * baseline + noise
    T_meas = np.clip(T_meas, 1e-4, None)

    np.savetxt(path, np.column_stack([lam, T_meas]),
               delimiter=",", header="wavelength_nm,transmission", comments="")
    return lam, T_meas


# ----------------------------------------------------------------------
# 2. Обработка: детренд базовой линии -> поиск резонансов -> фит каждого
# ----------------------------------------------------------------------
def detrend_baseline(lam, T, fsr_guess_nm):
    """Оценить базовую линию как верхнюю огибающую (перцентильный фильтр) и убрать её."""
    dlam = np.mean(np.diff(lam))
    win = int(fsr_guess_nm / dlam)                # окно ~ один FSR
    win = max(101, win | 1)                       # нечётное, не слишком маленькое
    base = percentile_filter(T, percentile=90, size=win, mode="nearest")
    return T / base, base


def rough_resonances(lam, T):
    """Грубо найти резонансы на сыром спектре, оценить FSR."""
    depth = 1.0 - T / np.median(T)
    dlam = np.mean(np.diff(lam))
    peaks, _ = find_peaks(depth, prominence=0.15, distance=int(1.0/dlam))
    fsr = np.median(np.diff(lam[peaks])) if len(peaks) > 1 else 5.0
    return peaks, fsr


def fit_one_resonance(lam, T, center_nm, fsr_nm):
    """Фит одного резонанса all-pass формой. Возвращает параметры."""
    # оценим полуширину по данным, возьмём окно ~6*FWHM
    win0 = slice(*np.searchsorted(lam, [center_nm - fsr_nm*0.25,
                                        center_nm + fsr_nm*0.25]))
    lo = lam[win0]; To = T[win0]
    Tmin0 = To.min()
    half = (1 + Tmin0) / 2
    width = lo[To <= half]
    fwhm0 = (width[-1] - width[0]) if len(width) > 2 else fsr_nm * 0.02
    w = max(fwhm0 * 6, fsr_nm * 0.03)
    sel = slice(*np.searchsorted(lam, [center_nm - w, center_nm + w]))
    x, y = lam[sel], T[sel]

    def model(lam_nm, t, A, lam0):
        phi = 2*np.pi * (lam_nm - lam0) / fsr_nm
        return allpass_T(phi, t, A)

    # старт: t < A (overcoupled-ветвь), оба близки к 1
    p0 = [0.97, 0.99, center_nm]
    bounds = ([0.0, 0.0, center_nm - fwhm0], [1.0, 1.0, center_nm + fwhm0])
    try:
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return None
    t_fit, A_fit, lam0 = popt
    pair = tuple(sorted([t_fit, A_fit]))          # неупорядоченная пара {min, max}
    Tmin = allpass_T(0.0, t_fit, A_fit)
    # добротность через финесс: F = pi*sqrt(tA)/(1-tA), FWHM = FSR/F, Q = lam/FWHM
    tA = t_fit * A_fit
    finesse = np.pi * np.sqrt(tA) / (1 - tA)
    fwhm = fsr_nm / finesse
    Q = lam0 / fwhm
    ER = -10 * np.log10(max(Tmin, 1e-6))
    return dict(lam0=lam0, pair=pair, Tmin=Tmin, Q=Q, ER=ER, fwhm=fwhm,
                model=model, window=(x, y))


def characterize(lam, T):
    """Полный разбор спектра -> словарь восстановленных величин."""
    _, fsr_guess = rough_resonances(lam, T)
    Tn, base = detrend_baseline(lam, T, fsr_guess)
    depth = 1.0 - Tn
    dlam = np.mean(np.diff(lam))
    peaks, _ = find_peaks(depth, prominence=0.1, distance=int(fsr_guess*0.5/dlam))
    centers = lam[peaks]

    fits = [fit_one_resonance(lam, Tn, c, fsr_guess) for c in centers]
    fits = [f for f in fits if f is not None]

    lam0s = np.array([f["lam0"] for f in fits])
    fsr = np.mean(np.diff(np.sort(lam0s))) if len(lam0s) > 1 else fsr_guess
    Q = np.mean([f["Q"] for f in fits])
    ER = np.mean([f["ER"] for f in fits])
    pair = np.mean([f["pair"] for f in fits], axis=0)   # {min, max} усреднённые

    lam_c = np.mean(lam0s) / 1000.0                     # мкм
    n_g = lam_c**2 / (fsr/1000.0 * L_UM)                # из FSR и известного R

    return dict(Tn=Tn, base=base, peaks=peaks, fits=fits,
                FSR=fsr, n_g=n_g, Q=Q, ER=ER, pair=pair)


# ----------------------------------------------------------------------
# 3. Расчётные ("истинные") величины для сравнения
# ----------------------------------------------------------------------
def true_values():
    lam_um = LAMBDA0 / 1000.0
    tA = T_TRUE * A_TRUE
    finesse = np.pi * np.sqrt(tA) / (1 - tA)
    fsr = lam_um**2 / (NG_TRUE * L_UM) * 1000.0         # нм
    fwhm = fsr / finesse
    Q = LAMBDA0 / fwhm
    Tmin = allpass_T(0.0, T_TRUE, A_TRUE)
    ER = -10*np.log10(Tmin)
    alpha = -2*np.log(A_TRUE) / L_UM                    # 1/мкм
    kappa = np.sqrt(1 - T_TRUE**2)
    return dict(FSR=fsr, n_g=NG_TRUE, Q=Q, ER=ER, t=T_TRUE, A=A_TRUE,
                alpha=alpha, kappa=kappa)


# ----------------------------------------------------------------------
# Печать сравнения + рисунки
# ----------------------------------------------------------------------
def print_comparison(true, rec):
    # разрешаем вырождение t<->A в пользу режима из дизайна (overcoupled: t<A)
    t_rec, A_rec = rec["pair"][0], rec["pair"][1]
    alpha_rec = -2*np.log(A_rec) / L_UM
    kappa_rec = np.sqrt(1 - t_rec**2)
    rows = [
        ("FSR, нм",        true["FSR"],   rec["FSR"]),
        ("n_g",            true["n_g"],   rec["n_g"]),
        ("Q (loaded)",     true["Q"],     rec["Q"]),
        ("ER, дБ",         true["ER"],    rec["ER"]),
        ("t",              true["t"],     t_rec),
        ("A",              true["A"],     A_rec),
        ("alpha, 1/мкм",   true["alpha"], alpha_rec),
        ("kappa",          true["kappa"], kappa_rec),
    ]
    print(f"\n{'Величина':<16}{'Расчёт':>12}{'Восстановл.':>14}{'Ошибка':>10}")
    print("-" * 52)
    for name, tv, rv in rows:
        err = abs(rv - tv) / abs(tv) * 100
        print(f"{name:<16}{tv:>12.4g}{rv:>14.4g}{err:>9.1f}%")
    print("\nПримечание: режим связи (over/under) принят из дизайна — фит по |T|")
    print("его не различает (вырождение t<->A).")


def make_figures(lam, T, rec, true, outdir="images"):
    os.makedirs(outdir, exist_ok=True)
    f = rec["fits"][len(rec["fits"]) // 2]              # центральный резонанс

    # (1) обзор спектра
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(lam, T, lw=0.7, color="#718096", label="измерено (с шумом)")
    ax.plot(lam, rec["base"], lw=1.5, color="#dd6b20", ls="--", label="базовая линия")
    ax.plot(lam[rec["peaks"]], T[rec["peaks"]], "v", color="#c53030",
            ms=9, label="найденные резонансы")
    ax.set_xlabel("Длина волны, нм"); ax.set_ylabel("Пропускание (сырое)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title("Шаг 1. Спектр, базовая линия, детектирование резонансов")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig_spectrum_overview.png", dpi=130)
    plt.close(fig)

    # (2) фит одного резонанса
    x, y = f["window"]
    xx = np.linspace(x[0], x[-1], 2000)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot((x - f["lam0"])*1000, y, ".", ms=3, color="#718096", label="данные")
    ax.plot((xx - f["lam0"])*1000, f["model"](xx, f["pair"][0], f["pair"][1], f["lam0"]),
            lw=2, color="#2b6cb0", label="фит all-pass")
    ax.axhline((1 + f["Tmin"])/2, color="#c53030", ls=":", lw=1)
    ax.set_xlabel(f"Отстройка от {f['lam0']:.2f} нм, пм")
    ax.set_ylabel("Пропускание (норм.)")
    ax.set_title(f"Шаг 2. Фит резонанса: Q≈{f['Q']:,.0f}, ER≈{f['ER']:.1f} дБ"
                 .replace(",", " "))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig_resonance_fit.png", dpi=130)
    plt.close(fig)

    # (3) сравнение восстановленного и расчётного
    t_rec, A_rec = rec["pair"][0], rec["pair"][1]
    metrics = ["FSR", "n_g", "Q", "ER"]
    tv = [true["FSR"], true["n_g"], true["Q"], true["ER"]]
    rv = [rec["FSR"],  rec["n_g"], rec["Q"],  rec["ER"]]
    # нормируем к расчёту, чтобы показать в одних осях
    tv_n = [1.0]*4
    rv_n = [rv[i]/tv[i] for i in range(4)]
    xpos = np.arange(4); wbar = 0.35
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(xpos - wbar/2, tv_n, wbar, color="#2f855a", label="расчёт")
    ax.bar(xpos + wbar/2, rv_n, wbar, color="#2b6cb0", label="восстановлено")
    for i in range(4):
        ax.text(xpos[i]+wbar/2, rv_n[i]+0.01, f"{rv[i]:.3g}", ha="center", fontsize=8)
        ax.text(xpos[i]-wbar/2, tv_n[i]+0.01, f"{tv[i]:.3g}", ha="center", fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels(["FSR", "n_g", "Q", "ER"])
    ax.set_ylabel("отношение к расчёту"); ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Шаг 3. Восстановлено vs расчёт")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig_comparison.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    lam, T = make_synthetic_spectrum()
    rec = characterize(lam, T)
    true = true_values()
    print_comparison(true, rec)
    print(f"\nНайдено резонансов: {len(rec['fits'])}")
    make_figures(lam, T, rec, true)
    print("Рисунки сохранены в images/: fig_spectrum_overview, fig_resonance_fit, fig_comparison")
