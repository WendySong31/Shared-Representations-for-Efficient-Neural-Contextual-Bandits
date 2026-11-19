# Reproduction protocol and departures

Source: Allesiardo, Féraud, Bouneffouf, *A Neural Networks Committee for the Contextual Bandit Problem*, https://arxiv.org/abs/1409.8191 . Sections 3–5 and Algorithms 1–3 guide this implementation. This is an independent reconstruction, not author-provided code.

## Architecture and experiment

- Each configuration has seven independent single-hidden-layer sigmoid MLPs. No weights are shared across actions or configurations.
- Half squared error and plain online SGD, uniform initialization in (-0.5, 0.5), and inverse action-probability weighting. No clipping, Adam, minibatches, replay, or pretraining.
- No bias terms: this follows the paper's connection count d*C+C. Bias conventions are not otherwise specified.
- Committee grid: hidden sizes 1, 5, 25, 50, 100 crossed with learning rates .01, .1, 1 (15 configurations, 105 networks).
- NeuralBandit1 is an additional fixed-configuration control, default hidden=25/rate=.1. The paper's main figure emphasizes the committees, not this arbitrarily selected control.
- Action exploration .005; configuration exploration .1.
- All configurations' predictors for the played action are trained. Unplayed-action networks remain untouched.
- EXP3 is evaluated in log space for numerical stability, without changing its probabilities.
- UCI Covertype: 581,012 rows; ten continuous features encoded into five one-hot quantile bins each, with 44 existing binary features retained, producing 94 features. Ties go to the upper bin, so tied features need not produce equally populated bins. Binning uses the full original dataset without labels, consistent with a repeated-stream reconstruction rather than held-out generalization.
- Per-seed shuffle and random starting position; circular repetition. Identical seed gives identical ordering/start across algorithms and conditions. Original shuffle seeds are unavailable.
- Drift: class index becomes (original index + floor(t/500000)) modulo seven, with first change after 500,000 observations. The learner sees only selected-action reward.
- Ten seeds for the main run. Default horizon is 2,000,000, a configurable reconstruction choice; do not call the run an exact numerical replication of the figure.

## Important ambiguities

1. Equation (1) uses the greedy-action indicator, while Algorithm 1 uses a played-action indicator. We use the played action: only its reward is available. Gradients are formed from pre-update parameters.
2. NeuralBandit2 trains all configurations but does not specify clearly whether the denominator is the selected configuration's action probability or a mixture probability. Default `--nb2-propensity conditional` uses the actual selected configuration's probability for every trained network. `marginal` uses the mixture over EXP3 configurations as a sensitivity check. These choices should not be silently combined in a results table.
3. Algorithm 3 says update "each" EXP3 vector but provides only the played action's reward. Default `--nb3-feedback all` applies that observed global reward to the configuration sampled by every action selector, a literal global-credit reading of the pseudocode; it does NOT invent unobserved rewards. `selected` updates only the played action's selector. Neither interpretation should be presented as author-confirmed. Compare them if committee conclusions depend on this choice.
4. Half squared error vs. unscaled squared error differs by a factor of two in effective learning rate; we explicitly choose half squared error.
5. Baseline tuning settings are not reported in the paper. We implement standard disjoint ridge LinUCB (alpha=1, ridge=1), independent Gaussian linear Thompson sampling (posterior score scale=.1, ridge=1), and Banditron (.005 exploration). CTS samples the scalar posterior score, equivalent for action selection to sampling each independent arm's coefficient vector. These are documented reconstructions, not exact reproductions of undisclosed baseline settings. Tune only in separately identified sensitivity runs.

## Metrics

Predictions/rewards are recorded before learning. Report final-100,000-round accuracy, all-stream accuracy, per-seed variation, runtime, and learning curves. Shading is an approximate 95% normal interval for the across-seed mean, not variation across examples.

The paper compares cumulative reward with an offline reference described as 93% accurate, but does not supply its per-example predictions. Our paper-reference curve is **0.93*t - cumulative reward**, an expected-reference proxy, not exact reconstructed regret. We separately report mistakes **t - cumulative reward**, corresponding to the known correct-label oracle. Never describe these two references interchangeably.

## Reproduction status

Passing tests and short runs validate implementation mechanics; they do not reproduce the paper's main findings. Main-result claims require completed long runs, seed averaging, and examination of protocol sensitivities. Saved metadata distinguishes those runs. Original reported outcomes are not inserted as our experimental results.
