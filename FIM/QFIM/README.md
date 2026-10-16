Fabric-Induced Memory (FIM)

FIM is an experiment in giving neural networks a real memory system instead of just a bigger context window.

Most modern sequence models either:

compress everything into a hidden state,
or repeatedly scan huge amounts of past information.

FIM tries something different.

Instead of treating memory like a giant list of tokens, it treats memory like a living latent fabric:

information spreads locally,
important events leave traces,
old structure slowly fades instead of exploding,
and retrieval only pulls back the pieces that still matter.

The result is a research framework focused on:

long-horizon dynamics,
scientific forecasting,
sparse memory,
latent geometry,
and stable temporal reasoning.

This repository contains the full research codebase, compact benchmarks, experiment runners, baselines, evaluation tools, and paper assets used for the current FIM prototype.