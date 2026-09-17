"""
adddrop_fit.py
==============
Characterization-инструмент для РЕАЛЬНЫХ спектров add-drop (drop-through) колец.

Отличие от spectrum_fit.py: тот работает с синтетическим all-pass спектром и
фитирует резонансы по одному. Здесь обрабатывается измерение с перестраиваемого
лазера (сотни тысяч–миллионы точек, детектор в вольтах, без нормировки), и фит
идёт СРАЗУ ПО ОКНУ из нескольких периодов. Так надёжнее, когда финесса низкая
и отдельный резонанс лоренцианом уже не описывается.

Модель through-порта add-drop кольца с симметричными ответвителями (t1 = t2 = t):

    T(lambda) = env(lambda) * (t^2 - 2 t^2 a cos(phi) + t^2 a^2)
                            / (1 -   2 t^2 a cos(phi) + t^4 a^2) + offset
    phi = 2 pi n_g L / lambda

Свободные параметры на окно: t, a, n_g*L, начальная фаза и квадратичная
огибающая env (грейтинги/юстировка). offset — тёмновой уровень детектора,
задаётся измеренным значением (DARK), иначе он коррелирует с a.

Что считается из фита:
    FSR = lambda^2 / (n_g L)
    F   = pi sqrt(t^2 a) / (1 - t^2 a)          финесса
    Q_L = F lambda / FSR                        нагруженная добротность
    Q_i = pi n_g L sqrt(a) / (lambda (1 - a))   внутренняя добротность
    kappa^2 = 1 - t^2                           связь одного ответвителя
    потери за обход [дБ] = -20 lg(a)

ВАЖНО (граница применимости): при сильной связи (kappa^2 больше ~0.2) величина
a определяется почти только глубиной минимума пропускания, и точность падает
катастрофически. Признак: a "плывёт" от окна к окну на 0.1 и больше, а
повторные копии одного кольца дают разные a. В этом режиме честный результат —
Q_L и kappa^2; Q_i и потери волновода из такого спектра НЕ извлекаются, и
скрипт печатает предупреждение. См. README, раздел про слабую связь.

Запуск:
    python adddrop_fit.py data/ring_through_example.csv
    python adddrop_fit.py my.csv --sep ';' --dark 0.0 --ngl0 1200
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import find_peaks

# ----------------------------------------------------------------------
# Загрузка и подготовка
# ----------------------------------------------------------------------
def load_sweep(path, sep=",", bin_pm=1.0, smooth_pm=60.0):
    """Читает CSV (две колонки: длина волны, сигнал), биннингует на равномерную
    сетку и сглаживает. Биннинг убирает квантование показаний лазера, когда на
    одно значение длины волны приходится несколько отсчётов."""
    raw = np.genfromtxt(path, delimiter=sep, skip_header=1, usecols=(0, 1))
    raw = raw[np.isfinite(raw).all(axis=1)]
    wl, v = raw[:, 0], raw[:, 1]

    step = bin_pm * 1e-3
    grid = np.arange(wl.min(), wl.max(), step)
    idx = np.digitize(wl, grid)
    sums = np.bincount(idx, weights=v, minlength=len(grid) + 1)[1:len(grid) + 1]
    cnt = np.bincount(idx, minlength=len(grid) + 1)[1:len(grid) + 1]
    ok = cnt > 0
    lam, T = grid[ok] + step / 2, sums[ok] / cnt[ok]

    k = max(1, int(round(smooth_pm / bin_pm)))
    if k > 1:
        T = np.convolve(T, np.ones(k) / k, "same")[k:-k]
        lam = lam[k:-k]
    return lam, T


def estimate_ngL(lam, T, ngL_min=100.0, ngL_max=5000.0):
    """Оценка n_g*L по спектру: пик преобразования Фурье сигнала, равномерно
    пересчитанного по частоте. Даёт стартовое приближение для фита.

    Диапазон поиска ограничен сверху: паразитные полости в тракте (волокно,
    торец чипа) дают n_g*L порядка десятков миллиметров и легко перебивают
    кольцо по амплитуде. Если кольцо длиннее ngL_max, поднимите границу."""
    f = 299792458.0 / (lam * 1e-9) / 1e9          # ГГц
    fu = np.linspace(f.min(), f.max(), len(f))
    vu = np.interp(fu, f[::-1], T[::-1])
    vu = vu - vu.mean()
    sp = np.abs(np.fft.rfft(vu * np.hanning(len(vu))))
    tau = np.fft.rfftfreq(len(vu), fu[1] - fu[0])  # нс
    ngL = tau * 1e-9 * 299792458 * 1e6             # мкм
    m = (ngL > ngL_min) & (ngL < ngL_max)
    g, s_ = ngL[m], sp[m]

    # Узкие резонансы дают богатый набор гармоник, и вторая-третья могут быть
    # выше первой. Поэтому берём не абсолютный максимум, а САМЫЙ КОРОТКИЙ
    # период среди заметных пиков — гармоники всегда длиннее основного.
    idx, _ = find_peaks(s_, height=0.3 * s_.max())
    return float(g[idx[0]]) if len(idx) else float(g[np.argmax(s_)])


# ----------------------------------------------------------------------
# Модель и фит
# ----------------------------------------------------------------------
def through_model(p, lam, lam0, dark=None):
    t, a, ngL, phi0, c0, c1, c2 = p
    x = lam - lam0
    phi = phi0 + 2 * np.pi * ngL * 1e3 * (1 / lam - 1 / lam0)
    num = t**2 - 2 * t * t * a * np.cos(phi) + (t * a) ** 2
    den = 1 - 2 * t * t * a * np.cos(phi) + (t * t * a) ** 2
    return (c0 + c1 * x + c2 * x**2) * num / den + (dark or 0.0)


def fit_window(lam, T, lam1, lam2, ngL0, dark=None):
    """Фит одного окна. Начальная фаза многозначна, поэтому стартуем из
    восьми значений phi0 и берём лучшее решение."""
    m = (lam > lam1) & (lam < lam2)
    L, y = lam[m], T[m]
    lam0 = 0.5 * (lam1 + lam2)
    best = None
    for phi0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        p0 = [0.5, 0.8, ngL0, phi0, y.max(), 0, 0]
        lo = [0, 0, 0.5 * ngL0, -10, 0, -1, -1]
        hi = [1, 1, 1.5 * ngL0, 20, 5 * max(y.max(), 1e-9), 1, 1]
        r = least_squares(lambda p: through_model(p, L, lam0, dark) - y,
                          p0, bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    return best, L, y, lam0


def figures_of_merit(t, a, ngL, lam0):
    x = t * t * a
    FSR = lam0**2 / (ngL * 1e3)
    F = np.pi * np.sqrt(x) / (1 - x)
    return dict(
        kappa2=1 - t * t,
        a=a,
        ngL=ngL,
        FSR=FSR,
        finesse=F,
        Q_L=F * lam0 / FSR,
        Q_i=np.pi * ngL * 1e3 * np.sqrt(a) / (lam0 * (1 - a)),
        loss_dB=-20 * np.log10(max(a, 1e-12)),
    )


def characterize(lam, T, windows=None, ngL0=None, dark=None, width_nm=15.0):
    """Фитирует набор окон и возвращает список словарей с параметрами."""
    if ngL0 is None:
        ngL0 = estimate_ngL(lam, T)
    if windows is None:
        edges = np.arange(lam.min() + 2, lam.max() - width_nm - 2, width_nm * 2)
        windows = [(e, e + width_nm) for e in edges]

    out = []
    for lam1, lam2 in windows:
        res, L, y, lam0 = fit_window(lam, T, lam1, lam2, ngL0, dark)
        t, a, ngL = res.x[:3]
        row = figures_of_merit(t, a, ngL, lam0)
        row.update(lam0=lam0, window=(lam1, lam2), params=res.x,
                   rms=float(np.sqrt(np.mean(res.fun**2)) / max(y.max(), 1e-12)))
        out.append(row)
    return out


# ----------------------------------------------------------------------
# Вывод
# ----------------------------------------------------------------------
def print_table(rows):
    print(f"{'окно, нм':>14} {'kappa^2':>8} {'a':>7} {'ngL,мкм':>9} "
          f"{'FSR,нм':>8} {'F':>6} {'Q_L':>8} {'Q_i':>10} {'rms':>6}")
    for r in rows:
        w = f"{r['window'][0]:.0f}-{r['window'][1]:.0f}"
        print(f"{w:>14} {r['kappa2']:8.3f} {r['a']:7.3f} {r['ngL']:9.1f} "
              f"{r['FSR']:8.3f} {r['finesse']:6.2f} {r['Q_L']:8.0f} "
              f"{r['Q_i']:10.3g} {r['rms']:6.3f}")

    k2 = np.array([r["kappa2"] for r in rows])
    a = np.array([r["a"] for r in rows])
    print(f"\nсреднее: kappa^2 = {k2.mean():.3f}, a = {a.mean():.3f} "
          f"(разброс по окнам {a.std():.3f}), потери за обход "
          f"{-20 * np.log10(a.mean()):.2f} дБ")
    if k2.mean() > 0.2 or a.std() > 0.05:
        print("\nВНИМАНИЕ: кольцо перевязано (kappa^2 велика) и/или a нестабильна "
              "по окнам.\nQ_L и kappa^2 достоверны; Q_i и потери волновода из "
              "этого спектра не извлекаются —\nсм. README, раздел \"Когда Q_i "
              "извлечь нельзя\".")


def make_figure(lam, T, rows, index=None, path="images/fig_adddrop_fit.png"):
    if index is None:
        index = len(rows) // 2
    r = rows[index]
    lam1, lam2 = r["window"]
    m = (lam > lam1) & (lam < lam2)
    fit = through_model(r["params"], lam[m], r["lam0"])

    fig, ax = plt.subplots(2, 1, figsize=(9, 5.4),
                           gridspec_kw=dict(height_ratios=[1, 1.2]))
    ax[0].plot(lam, T, lw=0.5, color="#21476E")
    ax[0].axvspan(lam1, lam2, color="#04D6AE", alpha=0.25)
    ax[0].set_ylabel("сигнал детектора")
    ax[0].set_title("Измеренный спектр through-порта", loc="left")
    ax[1].plot(lam[m], T[m], lw=1, color="#21476E", label="измерение")
    ax[1].plot(lam[m], fit, "--", lw=2, color="#C8742B", label="фит add-drop")
    ax[1].set_xlabel("λ, нм")
    ax[1].set_ylabel("сигнал детектора")
    ax[1].legend(frameon=False)
    ax[1].text(0.02, 0.08,
               f"κ² = {r['kappa2']:.2f}   a = {r['a']:.2f}   "
               f"FSR = {r['FSR']:.2f} нм   F = {r['finesse']:.1f}   "
               f"Q_L = {r['Q_L']:.0f}",
               transform=ax[1].transAxes, color="#5A5A5A")
    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"\nрисунок: {path}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("csv", nargs="?", default="data/ring_through_example.csv")
    ap.add_argument("--sep", default=",", help="разделитель CSV (например ';')")
    ap.add_argument("--dark", type=float, default=0.0,
                    help="тёмновой уровень детектора; None -> фитируется")
    ap.add_argument("--ngL0", type=float, default=None,
                    help="стартовое n_g*L, мкм (по умолчанию из ПФ)")
    ap.add_argument("--width", type=float, default=15.0, help="ширина окна, нм")
    ap.add_argument("--bin", type=float, default=1.0, help="биннинг, пм")
    ap.add_argument("--smooth", type=float, default=60.0,
                    help="сглаживание, пм; должно быть заметно уже резонанса — "
                         "при финессе 100+ ставьте единицы пм")
    ap.add_argument("--ngLmax", type=float, default=5000.0,
                    help="верхняя граница поиска n_g*L в ПФ, мкм")
    args = ap.parse_args()

    lam, T = load_sweep(args.csv, sep=args.sep,
                        bin_pm=args.bin, smooth_pm=args.smooth)
    ngL0 = args.ngL0 or estimate_ngL(lam, T, ngL_max=args.ngLmax)
    print(f"файл: {args.csv}")
    print(f"диапазон: {lam.min():.1f}–{lam.max():.1f} нм, точек после биннинга: {len(lam)}")
    print(f"стартовое n_g*L из ПФ: {ngL0:.0f} мкм\n")

    rows = characterize(lam, T, ngL0=ngL0, dark=args.dark, width_nm=args.width)
    print_table(rows)
    make_figure(lam, T, rows)


if __name__ == "__main__":
    main()
