# Reproduction of the Neural Networks Committee

This folder independently implements **NeuralBandit1, NeuralBandit2, and NeuralBandit3** from:

> Allesiardo, Féraud, and Bouneffouf. *A Neural Networks Committee for the Contextual Bandit Problem* (2014). [Paper](https://arxiv.org/abs/1409.8191).

This folder is separate from the proposed shared-representation model. It contains no shared encoder and no LLM routing. The implementation follows the paper's equations and pseudocode; it is not author-provided code. **Working software checks do not establish that the published experimental results have been reproduced.**

## Files

| File | Purpose |
| --- | --- |
| `prepare_data.py` | Downloads and verifies UCI Covertype; creates 94 binary features and seven class-actions. |
| `bandits.py` | Independent reward MLPs, EXP3, NeuralBandit1/2/3, Banditron, disjoint LinUCB, and linear contextual Thompson sampling. |
| `experiment.py` | Sequential online experiments with fixed seeds, optional label drift, and saved rewards/settings. |
| `summarize.py` | Produces aggregate CSV metrics and accuracy/reference-regret figures from completed runs. |
| `test_bandits.py` | Checks EXP3 probabilities, gradients, selected-action feedback isolation, encoding, and deterministic short streams. |
| `REPRODUCTION.md` | Paper settings, unspecified details, implementation interpretations, and metric definitions. |
| `requirements.txt` | Versions of the numerical, compilation, and plotting libraries. |
| `references.bib` | Citation for the original paper. |

## Installation

Use Python 3.12. From the repository root:

```bash
cd reproduce
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Verify the implementation

Run the short unit checks first; these do not download data or start long experiments:

```bash
python -m unittest discover -s . -p "test_*.py" -v
```

Prepare the public [UCI Forest Cover Type dataset](https://archive.ics.uci.edu/dataset/31/covertype):

```bash
python prepare_data.py
```

The script retains the 44 binary indicators and converts each of the ten numerical features into five one-hot quantile bins. It verifies the downloaded archive's SHA-256 checksum and saves `data/covertype.npz`. The source archive contains 581,012 examples with seven labels. Data is not committed to Git.

Run this **short functional check**, explicitly limiting the rounds and seeds:

```bash
python experiment.py --rounds 10000 --seeds 0 --drift-every 5000 --output results/smoke
python summarize.py --input results/smoke --window 1000 --stride 100
```

This checks all six algorithms in stationary and drifting conditions. The accelerated 5,000-round drift is for software verification, **not the original experimental protocol**. Numba compiles the loops on the first call, so the first run includes compilation overhead.

## Original-method settings

- One independent single-hidden-layer sigmoid MLP per action per configuration.
- Half squared error, online SGD, and inverse action-selection-probability weighting.
- Committee grid: hidden sizes `{1, 5, 25, 50, 100}` × learning rates `{0.01, 0.1, 1}`.
- Action exploration `0.005`; EXP3 configuration exploration `0.1`.
- Stationary and non-stationary repeated streams; the paper's label rotation occurs every 500,000 rounds.

Read `REPRODUCTION.md` before interpreting results. In particular, the paper leaves ambiguity in NeuralBandit3's EXP3 credit assignment and NeuralBandit2's update probabilities. Alternative interpretations are exposed as `--nb3-feedback` and `--nb2-propensity`, and baseline tuning defaults are documented rather than claimed to be author settings.

## Long experiments — run manually only when ready

The following command is provided for later use; it is not part of the short verification:

```bash
python experiment.py --rounds 2000000 --seeds 0 1 2 3 4 5 6 7 8 9 --drift-every 500000 --workers 2 --output results/paper
python summarize.py --input results/paper
```

Two million rounds is a reconstruction horizon, not a verified exact match to the original figure. An invocation without explicit arguments also selects a long experiment; use the short command above while developing. Completed runs are saved individually and matching completed jobs are skipped on rerun. An interrupted individual run restarts from its beginning.

## Outputs and interpretation

- `results/<experiment>/<condition>/<algorithm>/seed_N.npz`: per-round rewards, settings, dataset checksum, selector state, and runtime.
- `summary.csv`: stream accuracy, final-window accuracy, and seed variation.
- `accuracy.png`: trailing-window accuracy.
- `regret.png`: `0.93 × rounds − cumulative reward`, explicitly a proxy for the paper's offline reference rather than its unavailable per-example reference predictions.

Downloaded data, generated results, environments, and compiled caches are excluded by this folder's `.gitignore`. No long-run results are included in this code upload.
