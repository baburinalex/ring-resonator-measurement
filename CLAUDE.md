# ring-resonator-measurement

Recovers microring resonator parameters (FSR, Q, ER, coupling, loss) from
measured transmission spectra. Pure Python (numpy/scipy/matplotlib), no
Lumerical or other licensed software.

## Layout
- spectrum_fit.py        all-pass pipeline (`characterize(lam, T)`), synthetic trace
- adddrop_fit.py         add-drop through-port pipeline for raw laser sweeps
- make_example_sweep.py  synthesizes sweeps with realistic artefacts (truth is printed)
- coupler_supermodes.py  scalar FD supermode solve -> kappa^2 from geometry
- data/, images/, measured_spectrum.csv  committed example data and figures

## Commands
- Setup: `pip install -r requirements.txt`
- Run: `python spectrum_fit.py`, `python adddrop_fit.py --dark 0.006`
- Use the `Agg` matplotlib backend in headless runs.

## Rules
- Do not overwrite committed data/ or images/ files; write outputs to a temp dir in tests.
- Do not change numerical behaviour of the fitting code unless asked. If you
  find a bug, report it in the PR description instead of silently fixing it.
- Code, comments and commit messages in English.

## Physics constraints (do not "fix" these)
- |T| of an all-pass ring is symmetric under t <-> A: the fit recovers only
  the unordered pair {t, A}. The tool reports this on purpose.
- In the overcoupled add-drop regime (kappa^2 ~ 0.8) round-trip amplitude `a`
  is not recoverable from realistic data; the warnings printed by
  adddrop_fit.py are intended behaviour.
- Expected accuracy on synthetic data: FSR and n_g ~0.5 %, t/A ~0.2 %,
  Q/ER ~10 %.
