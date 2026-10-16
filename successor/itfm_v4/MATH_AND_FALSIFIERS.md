# Mathematical requirements and falsifiers for FIM successors

The proposed model cannot be proved perfect or universally better. Correctness properties can be proved under explicit contracts; prediction benefits require assumptions and empirical evidence. These derivations delimit the claim. They are elementary identities and standard conditional-expectation arguments, not claims of new theorems.

## When a residual correction helps

Let \(Y\) be a square-integrable target vector, \(f\) a base prediction, \(e=Y-f\), and \(c\) a correction available at prediction time. For a fixed scalar \(a\),

\[
R(f)-R(f+ac)=2a\,\mathbb E\langle e,c\rangle-a^2\mathbb E\|c\|^2.
\]

Proof: expand \(\|e-ac\|^2=\|e\|^2-2a\langle e,c\rangle+a^2\|c\|^2\) and take expectations. For \(a>0\), improvement holds exactly when \(2\mathbb E\langle e,c\rangle>a\mathbb E\|c\|^2\). If the correction has zero energy it cannot improve. Otherwise the unconstrained optimal scalar is \(a^*=\mathbb E\langle e,c\rangle/\mathbb E\|c\|^2\); under \(a\in[0,1]\), project this value onto the interval.

A validation estimate does not guarantee improvement after distribution shift. With deterministic residual \(e=1\) and correction \(c=-1\), using \(a=1\) changes squared error from 1 to 4. A base that already predicts perfectly has error zero, so every nonzero correction harms it. Scalar calibration and this risk decomposition are already closely related to [FreshCast](https://arxiv.org/html/2610.07834v1); they cannot serve as FIM's sole novelty claim.

The required diagnostic is held-out residual-correction alignment, correction energy, gain and harm after the same information has been supplied to each control. A new mechanism is rejected if standard retrieval captures the same useful alignment with lower cost.

## Information and irreducible error

Let \(\mathcal F_t\) contain all legally available information at prediction time. The conditional mean \(m=\mathbb E[Y\mid\mathcal F_t]\) minimizes squared error over measurable predictors. For any such predictor \(g\),

\[
\mathbb E\|Y-g\|^2
=\mathbb E\|Y-m\|^2+\mathbb E\|m-g\|^2.
\]

The cross term vanishes because \(\mathbb E[Y-m\mid\mathcal F_t]=0\). Thus memory cannot improve the Bayes optimum when it supplies no additional information. A practical finite-window base need not attain that optimum, so this is not evidence that memory is universally useless.

If \(Y=\mu+\epsilon\), with independent zero-mean observation noise having per-coordinate variance \(\sigma^2\), any causal estimator has expected per-coordinate squared error at least \(\sigma^2\), provided the noise is unobserved at prediction time. The v3 generator's stated standard deviation 0.03 gives 0.0009 for that noise component. This is an unnormalized population bound, not a bound for every finite sample or every reported normalized score.

## Visibility and state noninterference

A correct deployment interface carries only the observed event
\[
O=(\text{owner},\text{forecast ID},t_{\rm arrival},J,\{Y_j:j\in J\}).
\]
Scorer truth for coordinates outside \(J\) must be inaccessible to prediction, writing, normalization, calibration and serialization.

Suppose two executions start with identical state and random stream and receive identical visible events. If each transition depends only on prior state and these events, their next states and predictions are identical by induction. Consequently, altering unavailable targets cannot change the model. This proves an interface invariant, not forecasting improvement. An interface that passes the full target tensor must enforce the same property by construction and adversarial tests; merely multiplying hidden values by zero is insufficient because \(0\cdot{\rm NaN}\) is NaN.

For a saved base forecast \(f_i\), observed residual entries are \(r_{ij}=Y_{ij}-f_{ij}\) only for arrived, observed coordinates. They use the original pre-observation forecast, not a recomputation after revealing the answer. For nonnegative retrieval weights \(w_i\) and visible masks \(m_{ij}\), a basic reference is
\[
c_j=\frac{\sum_i w_i m_{ij}r_{ij}}{\sum_i w_i m_{ij}}
\]
when the denominator is positive, and zero correction otherwise. Gather visible values before arithmetic. Missing entries are not measured zeroes. The formula is a classical masked weighted estimator, not an architectural novelty.

## Missingness and identification

Mask normalization removes dilution from zero-filled missing entries; it does not guarantee an unbiased full-population estimate. For a simple example, let a residual be equally likely to be +1 or -1, but reveal only +1 outcomes. The visible mean is +1 while the true mean is zero. No mask-normalized estimator can recover the missing negative outcomes without further assumptions or information.

A proposed inverse-probability correction needs a correctly specified observation model or suitable robustness conditions, strictly positive observation probabilities, and an accounting of increased variance. Missing-outcome kernel methods already address this class of problem under explicit assumptions: [Wang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3491912/). No arbitrary-MNAR guarantee follows from adding a mask.

## Metric and inference requirements

For per-sample relative error \(q_i=\sqrt{\sum_j(\hat Y_{ij}-Y_{ij})^2/\sum_jY_{ij}^2}\), the sample mean is generally different from the square root of the pooled energy ratio. With targets [1] and [10] and predictions [0] and [10], the former is 0.5 and the latter is \(\sqrt{1/101}\approx0.099504\). Both can be defined metrics; using them interchangeably is invalid. Zero target energy needs a frozen explicit policy, including a scale floor or an undefined-result disposition.

Equal-cell macro averaging first forms each cell's declared statistic, then averages cells. One perfect trajectory in cell A and nine unimproved trajectories in cell B give a macro relative gain of 0.5, while pooled trajectory weighting gives 0.1. Dependence-preserving bootstrap samples must recompute the declared estimand, not silently change its weights.

For five independent nonzero paired differences, a two-sided exact sign-flip test has minimum attainable probability \(2/2^5=0.0625\). Larger repetitions need a prospective precision design. Repeated time steps and optimization seeds cannot automatically be treated as independent datasets.

## Causal intervention limits

For an additive model \(f+c\), erasing the correction returns \(f\). If there was a gain \(G=R(f)-R(f+c)>0\), the fraction removed by erasure is
\[
\frac{R(f)-R(f+c)}{G}=1.
\]
That equality checks wiring. It cannot independently prove that the stored historical information was meaningful. Relevant versus irrelevant deletion, matched-scale value replacement, key/value mispairing, coverage and provenance are needed to discriminate explanations.

## What has actually been checked

The audit tool evaluates 27 exact rational instances of the first algebraic identity and executes the existing scalar verifier and statistics functions on small adversarial fixtures. The general proof is the algebra above. Four audit controls also confirm that ordinary metric mismatches, incomplete matrices and explicitly absent independence are rejected, while benign summary fixtures follow the expected path. These checks involve no trained model, no public scientific dataset and no positive performance result. The receipt is [SOURCE_DEFECTS_REFINED.json](audit/SOURCE_DEFECTS_REFINED.json).

