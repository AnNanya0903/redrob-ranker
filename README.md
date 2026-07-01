# Redrob Ranker — Intelligent Candidate Discovery & Ranking Challenge

A transparent, rule-based ranking system that scores all 100,000 candidates
against the "AI Engineer, Redrob AI" job description and outputs the top 100
as a ranked CSV with grounded, per-candidate reasoning.

## Why rule-based, not embeddings/LLM

The compute budget (5 min, 16GB, CPU-only, no network) rules out a
per-candidate LLM call at 100K scale, and rules out fetching hosted
embedding models. A hand-built feature scorer over the structured schema:

- Runs in **~35-50 seconds** for all 100K candidates on a single CPU core.
- Makes every reasoning-column claim traceable to a specific field in the
  candidate's own record — nothing is generated freeform, so it can't
  hallucinate skills/employers that aren't in the profile.
- Is auditable: every scoring decision can be inspected and defended
  (see "Design rationale" below), which matters for Stage 4/5 review.

## Quick start

```bash
pip install -r requirements.txt   # stdlib only — this is a no-op today
gunzip -k candidates.jsonl.gz     # if you were given the gzipped pool
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv
```

Single command to reproduce the submission from the released pool:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

No pre-computation step is required — everything runs in the single
ranking pass.

## Files

| File | Purpose |
|---|---|
| `rank.py` | Entry point. Loads candidates, scores, ranks, writes CSV. |
| `features.py` | Feature extraction, honeypot detection, composite scoring. |
| `reasoning.py` | Builds the 1-2 sentence reasoning string from actual profile fields. |
| `requirements.txt` | Dependencies (stdlib only — no third-party packages). |
| `submission_metadata.yaml` | Portal metadata mirror (fill in team details before submitting). |
| `outputs/team_submission.csv` | Generated top-100 ranking. |

## Design rationale

### Honeypot detection

The dataset contains ~80 "subtly impossible" honeypot profiles. Three
independent internal-consistency checks were calibrated against the full
100K pool and together flag **70 candidates with zero overlap between
checks** — strong evidence they're catching the intended synthetic traps:

1. **Tenure overlap** — sum of `career_history[].duration_months` exceeds
   `years_of_experience * 12` by >30% (more employment than years lived
   professionally).
2. **Experience predates history** — `years_of_experience` implies a
   career start earlier than the candidate's earliest documented
   `start_date` (e.g. "8 years experience" but earliest job started 2
   years ago — the "company founded 3 years ago" trap from the honeypot doc).
3. **Zero-usage expert skills** — 3+ skills rated `expert`/`advanced` with
   `duration_months == 0` (claims mastery of something never actually used).

Honeypots are capped at score 0.02 rather than hard-deleted, so they can
never surface in a top-100 ranking regardless of how strong their other
signals look. Measured honeypot rate in the submitted top 100: **0%**.

### Skill-match scoring (anti keyword-stuffing)

Each required JD capability (embeddings, vector DB/hybrid search, Python,
ranking-evaluation frameworks) is scored via `_skill_strength()`, which
blends proficiency level, endorsement count, and months of actual use.
A skill claimed at "expert" with 0 endorsements and 0 duration is
discounted to ~25% trust — this is the core defense against candidates
who list every AI buzzword as a skill without evidence of real use.
A weaker fallback (plain keyword mention in career-history text) picks up
candidates who *did* the work without using the JD's specific vocabulary
(e.g. built a recommender system but never wrote the word "embeddings").

### "Gap between what the JD says and what it means"

The JD explicitly warns against ranking by keyword density. The scorer
reflects that by weighting **title relevance** and **company-type**
heavily alongside skills:

- Non-technical titles (Marketing Manager, Recruiter, etc.) are scored
  near zero regardless of listed skills.
- Candidates whose entire career is at pure IT-services/consulting firms
  (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, and similar) are
  penalized, per the JD's explicit statement on this.
- Pure-research-only backgrounds (no production-deployment language
  anywhere in career history) are penalized.
- Recent (<12mo), LangChain/API-wrapper-only "AI experience" without any
  pre-LLM-era production ML signal is heavily discounted.
- Computer-vision/speech/robotics-only backgrounds without NLP/IR
  exposure are discounted, per the JD's explicit exclusion.

### Behavioral signal modifier

Per `redrob_signals_doc.md`, all 23 signals are treated as a **multiplier**
on top of the skill/fit score, not an additive component — a perfect
skills match who hasn't logged in for 6 months shouldn't outrank a good
match who is active and responsive right now. The modifier blends
recency of `last_active_date`, `open_to_work_flag`,
`recruiter_response_rate`, `interview_completion_rate`,
`offer_acceptance_rate`, and profile/verification completeness.

### Location & notice period

Weighted per the JD's explicit statements: Pune/Noida scores highest,
other JD-named welcome cities (Hyderabad, Mumbai, Delhi NCR, Bangalore)
next, other Indian cities lower (relocation plausible but unconfirmed
Tier-1 origin), non-India candidates lowest (JD: case-by-case, no visa
sponsorship). Notice period follows the JD's stated sub-30-day preference
with a graduated falloff rather than a hard cutoff, since the JD says
30+ day candidates remain in scope.

## Pushing this repo to GitHub

This repo is git-initialized locally. To publish it:

```bash
# 1. Create an empty repo named "redrob-ranker" at github.com/AnNanya0903
#    (via the GitHub web UI: New repository -> do NOT initialize with
#    a README, since this repo already has one)

# 2. From inside this project folder:
git remote add origin https://github.com/AnNanya0903/redrob-ranker.git
git branch -M main
git push -u origin main
```

If you use the `gh` CLI instead, step 1+2 collapse to:

```bash
gh repo create AnNanya0903/redrob-ranker --public --source=. --remote=origin --push
```

## Deploying the sandbox (Section 10.5)

**Streamlit Community Cloud** (fastest path):
1. Push this repo to GitHub (above).
2. Go to https://share.streamlit.io -> "New app" -> select the
   `AnNanya0903/redrob-ranker` repo, branch `main`, main file path `app.py`.
3. Deploy. Copy the resulting `https://*.streamlit.app` URL into
   `sandbox_link` in `submission_metadata.yaml` and the portal form.

**HuggingFace Spaces** (alternative):
1. Create a new Space -> SDK: Streamlit.
2. Push this repo's contents to the Space's git remote (Spaces are git
   repos too), or upload the files via the Spaces UI.
3. The Space will build `requirements.txt` and run `app.py` automatically.

Either way, test the deployed sandbox yourself first: upload
`sample_candidates.json` (or a `.jsonl` slice of it, ≤100 rows) and
confirm it returns a ranked CSV within a few seconds.

## Compute environment this was tested on

- CPU-only, single process, stdlib-only Python 3.11
- Full 100,000-candidate pool: **~35-50 seconds** wall-clock, well inside
  the 5-minute budget
- No GPU, no network calls at any point in `rank.py`

## Known limitations

- The honeypot detector is heuristic and internal-consistency based; it
  does not catch fabrications that would require external verification
  (e.g. checking a named company's actual founding date against a real
  database), so it likely misses a subset of the ~80 stated honeypots.
- Location tiering uses a fixed city list; it does not distinguish
  candidates who are *currently* in a non-preferred city but explicitly
  state relocation intent beyond the `willing_to_relocate` flag already
  folded into notice/location context.
- This is a single-pass linear-weighted scorer, not a learned model — no
  supervised signal (hidden ground truth) was available to tune weights,
  so weights are hand-set from a close reading of the JD and validated
  qualitatively against the top-ranked output, not cross-validated.
