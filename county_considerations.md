**Read `repo_conventions_session_notes.md` first.**

# Session Notes: County Considerations
*Created: 2026-03-21*

## Purpose

Assess the feasibility of county-level area targets and weights, to inform pipeline design alongside congressional district work.

## Data Source

- **File:** `county2022.zip` from IRS SOI, containing:
  - `22incyallagi.csv` — county data with AGI breakdown (stubs 1–8), 34MB
  - `22incyallnoagi.csv` — county totals without AGI breakdown (stub 0 only), 5MB
  - `22incbsa.csv` — CBSA/metro area data (bonus, not needed for counties)
  - `22incydocguide.docx` — documentation
  - Per-state Excel files (not needed, not committed)
- **Encoding:** latin-1 (not UTF-8) due to county name characters (e.g., Doña Ana)

## Exploration Findings (2026-03-21)

### Data Structure
- **8 AGI stubs** (1–8), fewer than CDs (9) and states (10).
- **No total row** in the with-AGI file; totals are in a separate no-AGI file. Pipeline will need to merge these.
- **166 columns** (including 5 ID columns: STATEFIPS, STATE, COUNTYFIPS, COUNTYNAME, agi_stub).
- **161 data columns — identical variable set to CDs.** No CD-only or county-only variables.

### Coverage
- **All 3,143 counties present** in both files — 100% coverage.
- Counties sum almost exactly to state aggregates within the file (ratio ~1.0000 for all 51 states).
- County national sum is ~1.5% higher than CD file's US total (different SOI products).
- **No crosswalk needed** — county boundaries are stable (unlike CDs which need 117th→118th Congress crosswalk).

### County vs CD vs State Comparison

| Attribute | States | CDs | Counties |
|-----------|--------|-----|----------|
| Count | 51 | 436 | 3,143 |
| AGI stubs | 10 | 9 | 8 |
| Total row in data | Yes (stub 0) | Yes (stub 0) | Separate file |
| Crosswalk needed | No | Yes (117th→118th) | No |
| Coverage | 100% | 100% (with at-large recode) | 100% |
| Variables | 161 data cols | 161 data cols | 161 data cols |
| Smallest area (returns) | WY ~281K | ~70K | Loving TX: 40 |

### Scale and Solver Challenges

**Size distribution of counties by number of returns (N1):**
- `<100`: 2 counties
- `100–500`: 38 counties
- `500–1K`: 75 counties
- `1K–5K`: 728 counties
- `5K–10K`: 614 counties
- `10K–50K`: 1,113 counties
- `50K–100K`: 247 counties
- `100K–500K`: 281 counties
- `500K+`: 45 counties

**Key concerns:**
- 115 counties have fewer than 1,000 returns. With 8 AGI bins, that's <125 returns per bin — QP solver may be over-constrained.
- 3,143 areas is ~7x CDs and ~62x states. Batch solve time scales linearly with area count (assuming sufficient workers).
- Recipe must be very conservative for small counties — fewer target variables, possibly fewer AGI bins.
- May need to consider grouping very small counties or accepting higher tolerances.

### Pipeline Design Implications
- Pipeline should be parameterized by area type (state/CD/county) with different AGI cuts, recipes, and tolerances.
- County data has separate with-AGI and no-AGI files that must be merged — different from CD/state single-file structure.
- Speed optimizations identified for CDs should be implemented first and will benefit counties even more.
- Consider a tiered recipe approach: full recipe for large counties, reduced recipe for small ones.

## Additional County Challenges

### Solver Feasibility
- The QP solver minimizes deviations from initial weights subject to target constraints. With 40–500 returns in the smallest counties and 8 AGI bins, some bins may have fewer than 10 records. The constraint matrix becomes nearly singular — the solver either fails or produces extreme multipliers (weights far from 1.0).
- For states and CDs, the smallest area has ~70K+ returns, so even the thinnest AGI bin has thousands of records. Counties break this assumption.
- **Possible mitigations:** (a) Collapse AGI bins for small counties (e.g., merge stubs 7+8 into "$200K+"), (b) Use fewer target variables, (c) Raise constraint tolerance, (d) Set tighter multiplier bounds, (e) Skip the smallest counties entirely and use state-level weights as a fallback.

### Data Quality and Suppression
- SOI suppresses cells with very few returns for confidentiality. Small counties likely have many suppressed cells, which will appear as zeros or NaN. This complicates target construction — a zero target for "number of returns with capital gains in the $100K–$200K bin" may mean "suppressed" rather than "truly zero."
- Need to distinguish genuine zeros from suppressed values. The CD data doesn't have this problem because CDs are large enough to avoid most suppression.

### Adding Up to State/National Totals
- County sums match state aggregates almost exactly (ratio ~1.0000). This is better than CDs (~0.984) and means no rescaling is needed. However, this also means there's no "slack" — every return is assigned to a county, so county weights are more tightly constrained than CD weights in aggregate.

### Computational Scale
- 3,143 QP solves vs 436 for CDs. Each solve takes ~26 seconds (dominated by Clarabel QP solver).
- **Benchmarked on Ryzen 9 9950X (16C/32T):** 16 workers is the sweet spot. 28 workers is slower due to memory contention (each worker holds ~0.55 GB TMD data copy).
- Estimated wall time at 16 workers: ~1.5 hours. With tolerance relaxation (option C), possibly ~1 hour.
- With conservative recipe (fewer targets → smaller QP), per-area solve time may drop to 15-18s.
- Memory: each worker loads full TMD dataset (~0.55 GB). 16 workers = ~9 GB. Manageable on most systems.
- Log/output file count: 3,143 weight files + 3,143 logs. File I/O and directory operations may become a factor.

### Recipe Design
- A one-size-fits-all recipe won't work. Consider tiered approaches:
  - **Large counties (50K+ returns, 573 counties):** Full recipe similar to CDs.
  - **Medium counties (5K–50K returns, 1,727 counties):** Reduced variable set, possibly fewer AGI bins.
  - **Small counties (<5K returns, 843 counties):** Minimal targets (returns and AGI only), wider tolerances, possibly merged AGI bins.
  - **Tiny counties (<500 returns, 40 counties):** May need state-level weights as fallback, or very few targets with high tolerance.

### FIPS Code Handling
- County FIPS codes are 3-digit within state (e.g., `037` for Los Angeles). Combined with 2-digit state FIPS, the full code is 5 digits. Need consistent zero-padding throughout the pipeline.
- Some "counties" are independent cities (VA), parishes (LA), boroughs (AK), or census areas. The SOI data handles this, but naming/coding conventions vary.

## Lessons from CD Pipeline (2026-03-23)

The CD implementation revealed several patterns directly applicable to counties:

### Developer mode is essential
- With 436 CDs, manual tuning of solver parameters is impractical. With 3,143 counties it's impossible.
- The developer mode auto-relaxation cascade (try default → drop unreachable → reduce slack → drop targets → raise cap → raise tolerance) finds per-area overrides automatically.
- For CDs: most areas solve at level 0 (default params). Only a handful of extreme areas (NY-12/Manhattan) need targets dropped. Counties will likely have more problem areas due to smaller sizes.
- The override YAML file is committed to the repo. Production solve reads it for guaranteed first-pass success.

### Flat CSV target spec works well
- One row per target, no crossing/exclude logic. WYSIWYG.
- Easy to add/remove targets incrementally.
- For counties: tiered specs could be separate CSV files per size tier, or one spec with a "min_returns" column indicating minimum county size for each target.

### Pre-computed shares separate stable from volatile
- SOI geographic shares change rarely (new SOI vintage). TMD national sums change with every rebuild.
- Shares file computed once per SOI year. Targets recomputed cheaply.
- For counties: shares file will be ~3,143 × 100+ rows = 300K+ rows. Still manageable.

### Extended targets: start with totals, then add bins
- Total-only targets (one per variable, all bins) are nearly risk-free — they add one constraint each.
- Per-bin targets for selected variables and stubs can then be added incrementally.
- Developer mode identifies which per-bin targets cause infeasibility in which areas.
- For counties: may need total-only targets only for small counties, per-bin for large ones.

### Bystander analysis reveals untargeted distortion
- The quality report's per-bin bystander check shows which variable-bin combinations drift even when not targeted.
- For CDs: capital gains showed -71% distortion in one bin before being targeted. Adding a cap gains total target constrained it.
- For counties: expect worse bystander distortion due to smaller areas and fewer targets.

### One-to-many SOI-to-TMD variable mapping
- Multiple TMD variables can share the same SOI geographic distribution (e.g., e01500 total pensions and e01700 taxable pensions both use SOI A01700).
- The shares file handles this by producing separate rows for each TMD variable.
- Counties use the same 161 SOI columns as CDs, so the same mappings work.

### Solve time scales linearly
- CDs (436 areas, 107 targets): ~52 min at 16 workers.
- Counties (3,143 areas): estimated ~6 hours with same recipe. With conservative recipe (fewer targets), ~3-4 hours.
- Developer mode adds 2-5x multiplier for problem areas (multiple solve iterations). But most areas solve first try.

### County-specific considerations not present for CDs
- **Tiered specs:** CDs are all similar size (~70K+ returns). Counties range from 40 to 5M+ returns. Need different target counts per tier.
- **Data suppression:** SOI suppresses small cells for confidentiality. CDs are large enough to avoid this. Small counties will have many suppressed cells that look like zeros.
- **Developer mode tiers:** Could run developer mode separately per size tier with different starting parameters.

## Cloud Solve via EC2 (2026-03-26)

### Motivation
- Local solve times: 5 min (51 states), 55 min (436 CDs), estimated 7+ hours (3,143 counties).
- A 64-vCPU EC2 spot instance could solve counties in ~8 minutes for ~$0.10.
- Frees up local machine; makes county solves routine rather than burdensome.

### Design Requirements
1. **Local/cloud toggle:** The solve must work identically locally (`--workers 16`) or on cloud (`--workers 64`). No cloud-specific code in the solver itself. Cloud is just a bigger machine.
2. **External user friendly:** Any user who can `make data` and has an AWS account should be able to use the cloud option. Setup instructions and a wrapper script should live in the repo.
3. **Security:** Even though TMD microdata is synthetic (not confidential PUF), we should treat it as sensitive:
   - EC2 instance runs in a **private VPC subnet** (no public IP) or with security group restricting SSH to the user's IP only.
   - Data transfers via **SCP over SSH** (encrypted in transit).
   - Instance uses an **IAM role** with minimal permissions (no S3 access unless explicitly needed).
   - **EBS volumes encrypted at rest** (AWS default encryption or customer-managed KMS key).
   - Instance is **terminated** (not just stopped) after solve completes — no data persists.
   - If S3 is used for staging: bucket policy restricts access to the user's IAM identity, with server-side encryption enabled.
   - These controls are more than sufficient for synthetic data and would also satisfy concerns if the pipeline were later adapted for PUF-based work.

### Proposed Architecture
```
User's machine                          EC2 spot instance (c7a.16xlarge)
─────────────                          ─────────────────────────────────
cloud_solve.sh counties
  ├─ Launch instance from saved AMI     AMI has: Python, repo deps, make
  ├─ SCP: tmd_weights.csv.gz + targets
  ├─ SSH: make data && python -m        ← runs solve with --workers 64
  │        tmd.areas.solve_weights ...
  ├─ SCP: download results              ← weight files + logs
  └─ Terminate instance                 ← nothing persists
```

### Cost Estimates (2026 pricing)
| Instance | vCPUs | Spot $/hr | Est. time | Cost/run |
|----------|-------|-----------|-----------|----------|
| c7a.8xlarge | 32 | ~$0.30 | ~16 min | ~$0.08 |
| c7a.16xlarge | 64 | ~$0.60 | ~8 min | ~$0.08 |
| c7a.24xlarge | 96 | ~$0.90 | ~6 min | ~$0.09 |

Sweet spot is 64 vCPUs — diminishing returns beyond that due to memory bandwidth.

### Implementation Plan (when ready)
1. **AMI setup:** Create a base AMI with Python environment, repo dependencies, and `make install`. Update AMI when deps change.
2. **Wrapper script:** `cloud_solve.sh` in repo root handles launch/upload/solve/download/terminate. Configurable via env vars (`AWS_INSTANCE_TYPE`, `AWS_REGION`, `AWS_KEY_PAIR`).
3. **User setup doc:** One-time steps: install AWS CLI, configure credentials, create key pair, create security group. ~30 minutes.
4. **Makefile integration:** `make cloud-solve-counties` as a convenience target.
5. **Fallback:** If spot instance is interrupted (rare, ~5% chance), script detects failure and re-launches. Solve is idempotent — partial results are overwritten.

### Alternatives Considered
| Option | Speed | Cost | Setup | Notes |
|--------|-------|------|-------|-------|
| **EC2 Spot** | ~8 min | ~$0.10 | Medium | **Recommended.** Existing code unchanged. |
| GitHub Actions (64-core) | ~10 min | ~$0.64 | Low | More expensive; needs org billing for large runners. |
| Modal (serverless) | ~3 min | ~$0.20 | Medium | Requires code refactoring for their API. |
| AWS Batch | ~10-15 min | ~$0.15 | High | Overkill for single-user workflow. |

### Open Questions
- Should the AMI be public (so any external user can find it) or should users build their own from a script?
- Should we use S3 as an intermediate staging area instead of direct SCP? (Simpler for large file sets, but adds S3 cost and permissions.)
- For PUF users: would additional controls be needed (e.g., encrypted AMI, no internet egress)?

## Data Storage
- County CSV and docguide committed on `county-data` branch, pushed to origin fork.
- Location: `tmd/areas/targets/prepare/prepare_counties/data/data_raw/`
- Not merged into CD or master branches — kept separate for now.
