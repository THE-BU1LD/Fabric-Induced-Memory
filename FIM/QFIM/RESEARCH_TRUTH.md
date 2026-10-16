# QFIM / FIM Research Truth

Status audited: 2026-09-17

The canonical detailed experiment record currently lives in [`docs/experiments.md`](docs/experiments.md). This root file exists so the scientific verdict is visible without inferring it from prototype language, figures, or historical paper assets.

## Current verified evidence

A fresh current-code component-ablation matrix was executed externally across:

- 2 benchmarks: `lorenz96`, `delayed_recall`;
- 3 seeds: `11`, `23`, `37`;
- 4 variants: `full`, `no_memory`, `no_retrieval`, `no_salience_gating`;
- 24 expected cells and 24 validated cells.

The retained receipt in `docs/experiments.md` identifies GitHub Actions run `33257399564`, executed checkout `d4fdca0189f071ba4ce38f874301ae8be8e586ba`, artifact ID `9716689517`, and artifact ZIP SHA-256 `81bb5c79f751cfc9fea9b4cf6e90879e1b7fbe81b48629657bdc30b8d492fd13`.

## Scientific verdict

**MIXED / BROAD FIM COMPONENT SUPERIORITY NOT ESTABLISHED.**

### Lorenz-96

Relative to `no_memory` / `no_retrieval`, the full condition has:

- mean rollout MSE **higher by 0.021585** (~**4.22% worse**);
- mean rollout MAE **higher by 0.010639** (~**3.04% worse**).

Relative to `no_salience_gating`, the full condition has mean MSE higher by `0.012399` (~2.38% worse) and mean MAE nearly tied but slightly adverse (`+0.000517`).

### Delayed recall

The full condition is numerically better than `no_memory` / `no_retrieval` across the three retained seeds, but the mean differences are near zero:

- MSE delta: **-0.000154** (~**0.0184% better**);
- MAE delta: **-0.000049** (~**0.0071% better**).

The salience-gating comparison is even closer to zero.

`no_memory` and `no_retrieval` are exactly identical on retained MSE/MAE values across all six benchmark/seed combinations, so the current matrix does not separately identify those two conditions on the reported metrics.

Only three paired seeds exist per benchmark. No significance, universal mechanism-necessity, or broad superiority claim is supported.

## Historical-paper boundary

Historical paper/reference values and images are not automatically current-code reproduction evidence. Earlier evaluation paths could disable FIM memory during evaluation; maintained memory-aware evaluation was repaired in the repository history. Current paper-facing claims must be tied to the repaired current-code matrix or another explicitly retained current-source run.

## Claims currently supported

- A complete bounded current-code 24-cell component matrix exists with retained provenance.
- The current matrix is benchmark-dependent and mixed.
- Full FIM is adverse on Lorenz-96 under the retained primary metrics.
- Full FIM is only near-null numerically better on delayed recall.

## Claims not supported

- FIM consistently outperforms its ablations.
- Memory/retrieval/salience gating is established as necessary.
- The full architecture is superior across dynamical systems.
- Historical paper-reported results were freshly reproduced by the current matrix.
- The repository is submission-ready solely because figures or a PDF exist.

Negative, mixed, baseline-winning, and mechanism-falsifying outcomes are valid endpoints. New positive claims require prospectively frozen benchmarks, seeds, metrics, interventions, and retained current-source evidence.
