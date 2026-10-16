# Checkpoint evaluation weights

The experiment trainer can measure validation loss using an exponential moving
average (EMA) of model parameters. Its checkpoints also preserve the raw model
and optimizer state so training can resume from the actual optimization state.
Those two predictors can produce different predictions.

New trainer checkpoints record `validation_weight_source` as `raw` or `ema`
when a validation pass executes, or `null` when no source is declared. Both the
last checkpoint and the selected best checkpoint carry this metadata. The
`model`, `optimizer`, and `ema` payloads retain their existing meanings.

## Evaluate the recorded predictor

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint results/my_run/checkpoints/best_fim_checkpoint.pt \
  --config results/my_run/configs/resolved_config.yaml \
  --weights auto
```

`auto` is the default. It uses the checkpoint's explicit validation-weight
declaration. Historical checkpoints and `final_state.pt` files without that
declaration continue to use raw weights; the presence of an EMA mapping or a
training configuration does not establish which historical validation pass
used it. To evaluate an available EMA explicitly, pass `--weights ema`. To
evaluate raw optimization weights explicitly, pass `--weights raw`.

The evaluation provenance records the requested source, selected source,
selection reason, and saved validation-weight declaration. A deliberate
override is therefore visible in the output.

EMA contains trainable named parameters, rather than every entry in a model's
state dictionary. Evaluation first restores the complete raw model state, then
copies the admitted EMA parameters. Frozen parameters and persistent buffers
keep their saved raw values, and tied parameter aliases remain tied. Missing,
extra, wrongly shaped, or differently typed EMA entries fail before evaluation;
selecting EMA never silently falls back to raw weights.

This fixes the identity of the predictor loaded by the checkpoint evaluator.
It does not make a fresh rollout reproduce an earlier validation loss unless
the data, trajectory semantics, endpoint, random streams, and evaluation settings
also match. Validation fixtures exercise checkpoint behavior only. No retained
scientific result or protocol is rewritten by this change.
