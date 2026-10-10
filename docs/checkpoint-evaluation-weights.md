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

## Identify checkpoint and configuration contents

`evaluation_provenance.json` also records `checkpoint_sha256` and
`checkpoint_size_bytes`. The evaluator copies the input into a temporary
snapshot while hashing it, then loads that same snapshot. Replacing or
truncating the original path after the copy cannot make the receipt identify
different bytes from those deserialized. A checkpoint rewritten in place
during the copy can still produce a mixed or unreadable snapshot; its hash
describes the bytes consumed, not proof that a concurrent writer completed a
valid checkpoint. Use a completed checkpoint from a single writer.

The snapshot retains up to 8 MiB before spilling to the system temporary
directory. Copying uses 1 MiB blocks and adds one checkpoint-sized copy plus
hashing work. Large checkpoints require temporary disk space approximately
equal to the checkpoint size. The snapshot is closed after loading, including
on catchable load failure; this is not a crash-recovery or persistent backup
mechanism. The existing two-argument `_load_checkpoint` helper remains
compatible; snapshot provenance is enabled by the CLI's additional argument.

`config_source` identifies `explicit_yaml`, `checkpoint_embedded`, or
`sibling_yaml`, preserving that priority order. `config_source_path` contains
the actual YAML path, or `null` for the checkpoint's embedded `config` mapping.
`resolved_config_sha256` identifies the parsed mapping used by the evaluator,
not the YAML file's formatting. It hashes UTF-8 bytes from Python
`json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False)`, with string mapping keys required at every nesting level.
Non-finite numbers, non-string keys, and values that cannot be represented in
JSON are refused before benchmark creation or output publication. YAML
comments, whitespace, and mapping order therefore do not change this digest.
The legacy `config` receipt field is retained for existing consumers.

The receipt continues to record device, seed, rollout length, batch size, and
raw/EMA selection separately. These content hashes do not establish dataset
identity, source-code identity, protocol admission, or that a reported
scientific run occurred. Existing output paths, including empty directories,
remain refused; a retry must use a new `--output_dir`.
