"""
make_example_sweep.py
=====================
Генератор примерного измерения для adddrop_fit.py.

Строит правдоподобный свип перестраиваемого лазера через through-порт add-drop
кольца — с тем набором артефактов, из-за которых обработка реальных данных
отличается от обработки идеальной кривой:

  * огибающая ввода-вывода (грейтинги/торцы) — спектр не нормирован, в вольтах;
  * паразитная полость Фабри–Перо в тракте (волокно, торец чипа) — мелкая рябь;
  * дробовой и тепловой шум детектора;
  * тёмновой уровень (постоянная добавка);
  * паразитная засветка: часть света обходит кольцо (подложечные моды,
    кросс-поляризация) и подмешивается в детектор — именно она заполняет
    минимумы пропускания и делает a плохо определённой величиной;
  * дисперсия связи: kappa^2 растёт с длиной волны, как у реального
    направленного ответвителя (delta_n растёт с lambda).

Истинные параметры печатаются при запуске — с ними можно сверить то, что
восстановит adddrop_fit.py.

Два режима, ради которых всё и затевалось:

    python make_example_sweep.py                      # kappa^2 ~ 0.8: перевязанное
                                                      # кольцо, финесса ~1.5, a почти
                                                      # не определяется
    python make_example_sweep.py --kappa2 0.02 \\
        --out data/ring_through_weak.csv              # kappa^2 = 2 %: финесса ~150,
                                                      # a восстанавливается точно

Сравнение вывода adddrop_fit.py на этих двух файлах — самая короткая
иллюстрация того, почему резонатор для измерения потерь проектируют слабо
связанным.
"""

import argparse
import numpy as np

C = 299792458.0


def through_port(lam, t, a, ngL, phi0=0.0):
    """Пропускание through-порта add-drop кольца, t1 = t2 = t."""
    phi = phi0 + 2 * np.pi * ngL * 1e3 / lam
    num = t**2 - 2 * t * t * a * np.cos(phi) + (t * a) ** 2
    den = 1 - 2 * t * t * a * np.cos(phi) + (t * t * a) ** 2
    return num / den


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--out", default="data/ring_through_example.csv")
    ap.add_argument("--lam1", type=float, default=1520.0, help="начало, нм")
    ap.add_argument("--lam2", type=float, default=1620.0, help="конец, нм")
    ap.add_argument("--step", type=float, default=1.0, help="шаг, пм")
    ap.add_argument("--ngL", type=float, default=1150.0,
                    help="оптический путь за обход n_g*L, мкм")
    ap.add_argument("--kappa2", type=float, default=0.80,
                    help="kappa^2 одного ответвителя на 1550 нм")
    ap.add_argument("--loss", type=float, default=2.0,
                    help="потери за обход, дБ (задают a)")
    ap.add_argument("--ripple", type=float, default=0.08,
                    help="амплитуда паразитной ряби, доля")
    ap.add_argument("--noise", type=float, default=0.012,
                    help="шум детектора, В")
    ap.add_argument("--dark", type=float, default=0.006,
                    help="тёмновой уровень, В")
    ap.add_argument("--stray", type=float, default=0.03,
                    help="паразитная засветка мимо кольца, доля от огибающей")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lam = np.arange(args.lam1, args.lam2, args.step * 1e-3)
    a = 10 ** (-args.loss / 20.0)

    # Связь: kappa^2 = sin^2(pi*delta_n*L_eff/lambda), delta_n растёт с lambda.
    # Параметры подобраны так, чтобы на 1550 нм получилось заданное kappa^2.
    # delta_n(lambda) ~ lambda^3 для слабо связанных волноводов, theta = pi*delta_n*L_eff/lambda
    theta0 = np.arcsin(np.sqrt(np.clip(args.kappa2, 0, 1)))
    theta = theta0 * (lam / 1550.0) ** 2
    k2 = np.sin(theta) ** 2
    t = np.sqrt(np.clip(1 - k2, 1e-9, 1))

    T = through_port(lam, t, a, args.ngL, phi0=0.7)

    # Огибающая ввода-вывода: широкий гауссов профиль грейтингов + наклон.
    env = 1.35 * np.exp(-0.5 * ((lam - 1528.0) / 62.0) ** 2) * (1 - 0.0012 * (lam - 1520.0))

    # Паразитные полости в тракте (волокно, торец чипа): две несоизмеримые
    # полости с медленно уходящей фазой — так рябь не вычитается одной
    # синусоидой и ведёт себя как в реальном измерении.
    drift = np.cumsum(rng.normal(0, 0.02, lam.size))
    ripple = (1
              + args.ripple * np.cos(2 * np.pi * 16.0e3 * 1e3 / lam + 1.1 + 0.3 * drift)
              + 0.4 * args.ripple * np.cos(2 * np.pi * 3.7e3 * 1e3 / lam - 0.4))

    # Медленный дрейф юстировки поверх огибающей.
    slow = 1 + 0.03 * np.sin(2 * np.pi * (lam - args.lam1) / 37.0 + 0.9)

    v = env * slow * (ripple * T + args.stray) + args.dark
    v = v + rng.normal(0, args.noise, lam.size) * np.sqrt(np.maximum(v, 1e-3))
    v = np.maximum(v, 0.0)

    np.savetxt(args.out, np.c_[lam, v], delimiter=",",
               header="wavelength_nm,detector_V", comments="",
               fmt=["%.4f", "%.6f"])

    x = (1 - args.kappa2) * a
    fsr = 1550.0**2 / (args.ngL * 1e3)
    finesse = np.pi * np.sqrt(x) / (1 - x)
    print(f"записано: {args.out}  ({lam.size} точек, {args.lam1}-{args.lam2} нм)")
    print("истинные параметры на 1550 нм:")
    print(f"  kappa^2 = {args.kappa2:.3f}   a = {a:.4f} ({args.loss} дБ за обход)")
    print(f"  n_g*L   = {args.ngL:.0f} мкм   FSR = {fsr:.3f} нм")
    print(f"  финесса = {finesse:.1f}   Q_L = {finesse*1550.0/fsr:.0f}   "
          f"Q_i = {np.pi*args.ngL*1e3*np.sqrt(a)/(1550.0*(1-a)):.3g}")
    print(f"  паразитная засветка {args.stray*100:.0f} % — она смещает a вверх при фите")
    print(f"  тёмновой уровень {args.dark} В — передайте его как --dark в adddrop_fit.py")


if __name__ == "__main__":
    main()
