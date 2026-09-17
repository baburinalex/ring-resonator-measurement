"""
coupler_supermodes.py
=====================
Расчёт коэффициента связи kappa^2 направленного ответвителя из геометрии —
чтобы было с чем сравнить kappa^2, восстановленный из спектра в adddrop_fit.py.

Метод: скалярное уравнение Гельмгольца на двумерном сечении, конечные разности,
поиск двух старших собственных мод пары волноводов (чётной и нечётной).
Разность их эффективных индексов задаёт длину перекачки:

    L_pi = lambda / (2 * (n_even - n_odd))
    kappa^2 = sin^2(pi * delta_n * L_eff / lambda)

L_eff — эффективная длина связи: прямой участок плюс вклад S-изгибов, где
волноводы ещё близко. Обычно это единственный подгоночный параметр при
сравнении с измерением: из спектра определяется произведение delta_n * L_eff.

Скалярное приближение занижает delta_n для сильно контрастных волноводов на
единицы–десятки процентов. Для окончательных чисел стоит пересчитать в
векторном решателе (Lumerical MODE, femwell); здесь — быстрая оценка и
правильный порядок при выборе зазора.

Запуск:
    python coupler_supermodes.py                     # таблица по зазорам
    python coupler_supermodes.py --gap 0.6 --target 0.02
"""

import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

# Платформа по умолчанию: SiN 220 x 1200 нм в SiO2
N_CORE = 1.98
N_CLAD = 1.444
H_CORE = 0.22      # мкм
W_CORE = 1.20      # мкм


def solve_modes(lam, gap=None, w=W_CORE, h=H_CORE, nco=N_CORE, ncl=N_CLAD,
                dx=0.02, dy=0.02, X=4.0, Y=1.6, nmodes=2):
    """Скалярный FD-решатель. gap=None — одиночный волновод (для n_eff, n_g)."""
    x = np.arange(-X, X, dx)
    y = np.arange(-Y, Y, dy)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    n = np.full(XX.shape, ncl)
    cores = [0.0] if gap is None else [-(gap + w) / 2, (gap + w) / 2]
    for xc in cores:
        n[(np.abs(XX - xc) <= w / 2) & (np.abs(YY) <= h / 2)] = nco

    k0 = 2 * np.pi / lam
    Nx, Ny = n.shape
    N = Nx * Ny
    diag = -2 / dx**2 - 2 / dy**2 + (k0 * n.ravel()) ** 2
    offx = np.ones(N - Ny) / dx**2
    offy = np.ones(N - 1) / dy**2
    offy[np.arange(1, N) % Ny == 0] = 0          # граница по y
    A = (sp.diags([diag], [0])
         + sp.diags([offx, offx], [Ny, -Ny])
         + sp.diags([offy, offy], [1, -1]))

    vals, vecs = spl.eigsh(A.tocsc(), k=nmodes, sigma=(k0 * nco) ** 2, which="LM")
    neff = np.sqrt(vals) / k0
    order = np.argsort(-neff)
    return neff[order]


def delta_n(lam, gap, **kw):
    ne = solve_modes(lam, gap=gap, nmodes=2, **kw)
    return float(ne[0] - ne[1])


def kappa2(lam, gap, L_eff, **kw):
    """Доля мощности, перешедшая в соседний волновод."""
    return float(np.sin(np.pi * delta_n(lam, gap, **kw) * L_eff / lam) ** 2)


def L_eff_for(lam, gap, target_k2, **kw):
    """Длина связи, дающая нужный kappa^2 (первая ветвь)."""
    dn = delta_n(lam, gap, **kw)
    return float(np.arcsin(np.sqrt(target_k2)) * lam / (np.pi * dn))


def group_index(lam0=1.55, dl=0.02, **kw):
    """n_g одиночного волновода численным дифференцированием n_eff(lambda)."""
    n_m = solve_modes(lam0 - dl, gap=None, nmodes=1, **kw)[0]
    n_0 = solve_modes(lam0, gap=None, nmodes=1, **kw)[0]
    n_p = solve_modes(lam0 + dl, gap=None, nmodes=1, **kw)[0]
    return float(n_0 - lam0 * (n_p - n_m) / (2 * dl))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--lam", type=float, default=1.55, help="длина волны, мкм")
    ap.add_argument("--gap", type=float, default=None,
                    help="один зазор, мкм (иначе таблица)")
    ap.add_argument("--Leff", type=float, default=45.0,
                    help="эффективная длина связи, мкм")
    ap.add_argument("--target", type=float, default=0.02,
                    help="целевой kappa^2 для колонки L_eff")
    args = ap.parse_args()

    print(f"SiN {W_CORE*1e3:.0f} x {H_CORE*1e3:.0f} нм, n = {N_CORE} в {N_CLAD}, "
          f"λ = {args.lam*1e3:.0f} нм")
    print(f"n_g одиночного волновода: {group_index(args.lam):.3f}\n")

    gaps = [args.gap] if args.gap else [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    print(f"{'зазор,мкм':>10} {'Δn':>9} {'L_pi,мкм':>10} "
          f"{'κ² при L_eff':>14} {'L_eff для κ²=%.0f%%' % (args.target*100):>18}")
    for g in gaps:
        dn = delta_n(args.lam, g)
        print(f"{g:10.2f} {dn:9.5f} {args.lam/(2*dn):10.0f} "
              f"{kappa2(args.lam, g, args.Leff):14.3f} "
              f"{L_eff_for(args.lam, g, args.target):18.1f}")


if __name__ == "__main__":
    main()
