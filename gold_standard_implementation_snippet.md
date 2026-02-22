## Implementation Strategy (Gold-Standard Weights)

**Option A: Run scipy L-BFGS-B as the production solver in `make tmd_files`**
- Replace PyTorch L-BFGS entirely
- ~7 min runtime, no GPU needed
- Cross-machine reproducible

**Option B: Generate reference weights once, commit them**
- Run scipy L-BFGS-B once to produce "gold standard" weights
- Commit weights to repo
- Other machines verify against committed weights
- Faster builds (skip reweighting), guaranteed reproducibility

### Why this works

Because the objective is a convex QP with a positive definite Hessian, it has a **unique global minimum**. Any solver that converges tightly enough — on any machine — must find the same answer. This means:

- Option A produces reproducible weights across machines (same unique minimum)
- Option B lets us verify that any machine's solver finds the same unique minimum
- Either way, we get a single canonical set of weights that all contributors can agree on
