# Walkthrough Video — Script & Storyboard

I can't record or render an actual video file, but here's a tight script
you can read straight through while screen-recording (Loom, OBS, or even
a phone pointed at a screen works — the spec doesn't require production
value, just a real walkthrough of real work). Total runtime: ~4-5 minutes.

Record your screen showing this repo open in an editor/terminal, plus
the running sandbox app, while narrating the beats below.

---

## Beat 1 — Architecture (60-75s)

**Show:** README.md, then `rank.py` / `features.py` / `reasoning.py` side by side.

**Say:**
> "This ranks 100,000 candidates against the Redrob AI job description.
> I went with a rule-based scorer over the structured schema instead of
> embeddings or an LLM ranker, because the compute budget is CPU-only,
> no network, 5 minutes for the full pool — an LLM call per candidate
> just doesn't fit at that scale. The tradeoff is I don't get free
> semantic matching, so I had to encode what the JD actually cares
> about as explicit features: required skills with a trust discount
> against keyword stuffing, title relevance, company type — consulting
> vs product company — experience band, location, and notice period."

## Beat 2 — Honeypot detection (45-60s)

**Show:** `detect_honeypot()` in `features.py`, and the README section
on honeypots.

**Say:**
> "The dataset has about 80 honeypot profiles with subtly impossible
> data. I found three internal-consistency checks that catch this:
> career-history duration exceeding stated years of experience, years
> of experience predating the earliest documented job, and skills
> marked 'expert' with zero months of actual use. I calibrated these
> against the real 100K-candidate file — they flag 70 candidates
> total with zero overlap between the three checks, which is strong
> evidence they're catching three distinct synthetic trap types."

## Beat 3 — Run it end-to-end (60s)

**Show:** Terminal. Run:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv
```

**Say:**
> "Here's the single command from the README producing the submission
> CSV from the raw candidate pool — full 100K candidates, and it
> finishes in under a minute, well inside the 5-minute budget. Then I
> run the official validator against it — passes clean."

## Beat 4 — Sample output + reasoning grounding (45-60s)

**Show:** `head submission.csv` or open it in a spreadsheet viewer,
scroll through the top few rows.

**Say:**
> "Rank 1 here is a Lead AI Engineer at Razorpay — 6.7 years experience,
> Information Retrieval and Vector Search both rated expert with 90+
> months of actual use behind them, 30-day notice period. The reasoning
> column is built entirely from that candidate's own record — nothing
> is generated freeform, so every claim you see there traces back to an
> actual field in their profile. That matters for the honesty check at
> Stage 4."

## Beat 5 — Sandbox demo (45-60s)

**Show:** The running `streamlit run app.py`, upload
`sample_candidates.json` (or a small `.jsonl` slice), show the ranked
table and the download button.

**Say:**
> "And this is the sandbox — same ranking logic, running on a small
> uploaded sample, so you can verify the code actually runs without
> needing the full 100K file. It's deployed at [your Streamlit/HF URL]."

## Beat 6 — Close (15-20s)

**Say:**
> "That's the full pipeline — feature extraction, honeypot filtering,
> composite scoring, grounded reasoning, all CPU-only and reproducible
> from a single command. Repo and sandbox links are in the submission."

---

## Recording checklist

- [ ] Screen recording tool ready (Loom / OBS / QuickTime)
- [ ] Terminal font size bumped up so it's readable on camera
- [ ] `candidates.jsonl` present locally so the live run in Beat 3 works
- [ ] Sandbox app already deployed so Beat 5 shows the real hosted URL,
      not just `localhost`
- [ ] Export/upload the video, get a shareable link, add it wherever the
      portal asks for it (the spec doesn't list video as a required
      upload field in Section 10.2 — double check the actual portal
      form for where this goes)
