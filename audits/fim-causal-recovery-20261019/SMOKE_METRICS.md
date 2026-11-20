# Generated smoke metrics

Generated from `runs_02/cpu-smoke/evaluation/metrics.json`; one seed, two training epochs, synthetic engineering only.

| Variant | Validation-selected alpha | Validation MSE | Test MSE | Test retrieval hits |
|---|---:|---:|---:|---:|
| base | 0 | 0.124354654293 | 0.0725440009565 | 0 / 18 |
| memory | 1 | 0.017695708507 | 0.00835317594033 | 16 / 18 |
| erased | 0 | 0.124354654293 | 0.0725440009565 | 16 / 18 |
| spatial_mean | 1 | 0.12430143939 | 0.0725251487591 | 16 / 18 |
| last_residual | 1 | 0.0151434358209 | 0.00806379866561 | 16 / 18 |
| persistence | — | 0.00116534428717 | 0.000337322700398 | no memory |

## Post hoc cold/warm decomposition

Generated independently from retained predictions by `analyze_retained_fixture.py`. Overall MSE remains the primary diagnostic. Two first-owner forecasts are cold; the remaining sixteen are warm. These rows are not independent experimental repetitions.

| Variant | Overall MSE | Cold-start MSE | Warm-memory MSE |
|---|---:|---:|---:|---:|
| base | 0.0725440009565 | 0.0699819171845 | 0.072864261428 |
| memory | 0.00835317594033 | 0.0699819171845 | 0.00064958328481 |
| last_residual | 0.00806379866561 | 0.0699819171845 | 0.000324033850743 |
| persistence | 0.000337322700398 | 0.000439808882938 | 0.00032451192758 |

Cold starts account for 93.087571% of full-memory squared error.

The nine-step autonomous base diagnostic is separate: MSE 0.0815668353758, with all predictions finite. It does not measure an autonomous memory advantage.
