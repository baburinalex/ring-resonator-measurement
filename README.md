# Ring Resonator Spectrum Characterization

A small tool that recovers the **physical parameters of an all-pass microring resonator from its measured transmission spectrum** — the kind of trace you get from a tunable-laser sweep into a photodetector — and compares the recovered values against the calculated ones.

It is the experimental counterpart to the [`ring-resonator-tutorial`](https://github.com/baburinalex/ring-resonator-tutorial) project: that one goes *geometry → spectrum*, this one goes *measured spectrum → physical parameters*.

![Spectrum overview](images/fig_spectrum_overview.png)

## What it does

To have something to process, the script first **synthesizes a realistic measurement** — the ideal all-pass lineshape plus a Fabry–Pérot baseline ripple and detector noise — and saves it to `measured_spectrum.csv`. It then characterizes that trace as if the true values were unknown:

1. **Detrend** the baseline with an upper-envelope percentile filter.
2. **Detect** resonances (`scipy.signal.find_peaks`).
3. **Fit** each resonance to the all-pass transmission lineshape (`scipy.optimize.curve_fit`) → self-coupling `t`, round-trip amplitude `A`, resonance wavelength.
4. **Derive** the figures of merit: FSR (resonance spacing), loaded Q (via finesse), extinction ratio ER, group index `n_g` (from FSR and the known radius), propagation loss `α`, coupling `κ`.
5. **Compare** recovered vs calculated in a table and plots.

![Resonance fit](images/fig_resonance_fit.png)
![Recovered vs calculated](images/fig_comparison.png)

Typical recovery on the synthetic data: FSR and `n_g` within ~0.5%, `t`/`A` within ~0.2%, and the noise-sensitive Q/ER within ~10%.

## The `t ↔ A` degeneracy (important)

The transmission **magnitude** lineshape is symmetric under swapping the self-coupling `t` and the round-trip amplitude `A`. A fit to `|T|` alone therefore recovers the *unordered pair* `{t, A}` but **cannot** distinguish an undercoupled device from an overcoupled one. Resolving it requires extra information — the phase response, or a sweep over coupling strength. The tool reports this explicitly instead of silently guessing, and assigns the regime from design intent only for the final comparison.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python spectrum_fit.py
```

This prints the comparison table and writes three figures into `images/`.

## Using your own data

Replace `measured_spectrum.csv` with your measurement — two columns, `wavelength_nm, transmission` — and call:

```python
from spectrum_fit import characterize
import numpy as np

data = np.loadtxt("measured_spectrum.csv", delimiter=",", skiprows=1)
lam, T = data[:, 0], data[:, 1]
result = characterize(lam, T)
print(result["FSR"], result["Q"], result["ER"], result["n_g"])
```

The known device radius is set near the top of `spectrum_fit.py` (`R_KNOWN`) and is used to convert the measured FSR into a group index.

## Repository layout

```
ring-resonator-measurement/
├── README.md
├── LICENSE
├── requirements.txt
├── spectrum_fit.py          # the characterization pipeline
├── measured_spectrum.csv    # synthetic example measurement (replace with real data)
└── images/                  # figures produced by the script
```

## License

MIT — see [`LICENSE`](LICENSE).
